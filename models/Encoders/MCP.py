import os
import math
import json
import torch
import torch.nn as nn


# MCP resource types the encoder understands
RESOURCE_TYPES = {
    "file":        0,
    "directory":   1,
    "terminal":    2,
    "diagnostic":  3,
    "diff":        4,
    "search":      5,
    "symbol":      6,
    "reference":   7,
    "unknown":     8,
}
NUM_RESOURCE_TYPES = len(RESOURCE_TYPES)


def createMCPEncoder(vocab_size=32000, d_model=512, tokenizer=None):
    """
    Factory matching the project convention.

    Args:
        vocab_size : must match your SentencePiece vocab
        d_model    : must match the latent state width
        tokenizer  : an InputTextTokenizer instance (required)
    """
    if tokenizer is None:
        raise ValueError(
            "MCPEncoder requires a tokenizer. "
            "Pass createTokenizer() from TextEncoder."
        )
    return MCPEncoder(
        vocab_size=vocab_size,
        d_model=d_model,
        tokenizer=tokenizer,
    )


class MCPEncoder(nn.Module):
    """
    Encodes MCP (Model Context Protocol) tool responses into the shared
    latent representation [1, seq, d_model] for the routing loop.

    MCP tools return structured data from a coding environment — file
    contents, directory listings, terminal output, LSP diagnostics, git
    diffs, symbol lookups, etc.  This encoder:

        1. Parses the MCP response into a text body + resource type
        2. Tokenizes the text through the shared SentencePiece tokenizer
        3. Adds a learned resource-type embedding so the model can
           distinguish a file from a diagnostic from a terminal output
        4. Runs a transformer encoder for contextualised representations

    Compatible with EncoderRegistry:
        reg.add_encoder("mcp", createMCPEncoder(tokenizer=tokenizer))
        encodings = reg.encode(mcp_responses, "mcp")

    Input formats accepted:
        - str         : raw text, treated as resource type "unknown"
        - dict        : {"type": "file", "content": "...", ...}
        - list[dict]  : multiple MCP responses in one call
        - list[str]   : multiple raw text inputs
    """

    def __init__(
        self,
        vocab_size: int = 32000,
        d_model: int = 512,
        nhead: int = 8,
        num_layers: int = 3,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        max_seq_len: int = 4096,
        tokenizer=None,
    ):
        super().__init__()
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        self.tokenizer = tokenizer

        # Token embedding (shared vocab with the text encoder)
        self.token_embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=d_model,
            padding_idx=0,
        )

        # Resource-type embedding: distinguishes file vs terminal vs diff etc.
        self.type_embedding = nn.Embedding(
            num_embeddings=NUM_RESOURCE_TYPES,
            embedding_dim=d_model,
        )

        # Section embeddings: distinguishes metadata header from body content
        self.section_embedding = nn.Embedding(
            num_embeddings=2,       # 0 = metadata, 1 = body
            embedding_dim=d_model,
        )

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers
        )

    # ------------------------------------------------------------------
    # Positional encoding (matches project convention)
    # ------------------------------------------------------------------

    def _sincos_pos_embed(self, seq_len, device):
        pos = torch.arange(seq_len, dtype=torch.float32, device=device)
        omega = torch.exp(
            -torch.arange(0, self.d_model, 2, dtype=torch.float32,
                          device=device)
            * (math.log(10000.0) / self.d_model)
        )
        out = pos[:, None] * omega[None, :]
        emb = torch.stack([torch.sin(out), torch.cos(out)], dim=2).flatten(1)
        return emb.unsqueeze(0)

    # ------------------------------------------------------------------
    # MCP response parsing
    # ------------------------------------------------------------------

    def _parse_mcp_response(self, response):
        """
        Normalise an MCP response into (resource_type, metadata_str, body_str).

        Handles the common MCP tool output shapes:
            - Raw string
            - Dict with "type" and "content" / "text" / "body"
            - Dict with nested "result" from tool_call responses
        """
        if isinstance(response, str):
            return "unknown", "", response

        if not isinstance(response, dict):
            return "unknown", "", str(response)

        # Handle nested MCP tool_call result wrappers
        if "result" in response and isinstance(response["result"], dict):
            response = response["result"]
        if "content" in response and isinstance(response["content"], list):
            # MCP content blocks: [{"type": "text", "text": "..."}]
            blocks = response["content"]
            body_parts = []
            for block in blocks:
                if isinstance(block, dict) and "text" in block:
                    body_parts.append(block["text"])
                elif isinstance(block, str):
                    body_parts.append(block)
            body = "\n".join(body_parts)
            rtype = response.get("type", response.get("resourceType", "unknown"))
            return str(rtype), "", body

        # Standard dict with known fields
        rtype = response.get("type", response.get("resourceType", "unknown"))

        # Extract body from common field names
        body = ""
        for key in ("content", "text", "body", "output", "stdout"):
            if key in response and isinstance(response[key], str):
                body = response[key]
                break

        # Build metadata from everything that isn't the body
        meta_parts = []
        for key in ("uri", "path", "name", "language", "languageId",
                     "severity", "message", "range", "source", "command",
                     "exitCode", "stderr"):
            if key in response:
                val = response[key]
                if isinstance(val, (dict, list)):
                    val = json.dumps(val, separators=(",", ":"))
                meta_parts.append(f"{key}: {val}")

        metadata = " | ".join(meta_parts)

        return str(rtype), metadata, body

    def _tokenize_text(self, text, device):
        """Tokenize a string, truncate to max_seq_len, return [1, T] ids."""
        ids, _ = self.tokenizer.tokenize_single(text, device=device)
        if ids.size(1) > self.max_seq_len:
            ids = ids[:, :self.max_seq_len]
        return ids

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, mcp_input):
        """
        Encode one or more MCP responses.

        Args:
            mcp_input : str, dict, list[str], or list[dict]
                        See class docstring for accepted formats.

        Returns:
            output : [1, seq, d_model] per input item.
                     When called through EncoderRegistry.encode(), each
                     item in a list input is processed separately to
                     preserve unique sequence lengths.
        """
        # Normalise to single item (registry iterates for us)
        if isinstance(mcp_input, list):
            # If registry passed a single-element list, unwrap
            if len(mcp_input) == 1:
                mcp_input = mcp_input[0]
            else:
                raise ValueError(
                    "MCPEncoder.forward() expects a single MCP response. "
                    "Pass a list to EncoderRegistry.encode() instead."
                )

        device = next(self.parameters()).device
        rtype, metadata, body = self._parse_mcp_response(mcp_input)

        # --- Resource type index ---
        type_idx = RESOURCE_TYPES.get(rtype, RESOURCE_TYPES["unknown"])
        type_tensor = torch.tensor(
            [type_idx], dtype=torch.long, device=device
        )
        type_emb = self.type_embedding(type_tensor)             # [1, D]

        # --- Tokenize metadata and body separately ---
        combined_text = f"{metadata}\n{body}" if metadata else body
        input_ids = self._tokenize_text(combined_text, device)  # [1, T]
        T = input_ids.size(1)

        # Figure out where metadata ends and body begins
        if metadata:
            meta_ids = self._tokenize_text(metadata, device)
            meta_len = meta_ids.size(1)
        else:
            meta_len = 0

        # --- Build section ids: 0 for metadata tokens, 1 for body tokens ---
        section_ids = torch.ones(1, T, dtype=torch.long, device=device)
        if meta_len > 0 and meta_len < T:
            section_ids[:, :meta_len] = 0

        # --- Embed and combine ---
        x = self.token_embedding(input_ids) * math.sqrt(self.d_model)
        x = x + self._sincos_pos_embed(T, device)
        x = x + type_emb.unsqueeze(1)                          # broadcast
        x = x + self.section_embedding(section_ids)

        # --- Encode ---
        x = self.transformer(x)                                 # [1, T, D]

        return x

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def store_weights(self, path, filename="mcp_encoder.pt"):
        os.makedirs(path, exist_ok=True)
        torch.save(self.state_dict(), os.path.join(path, filename))
        print(f"Saved MCPEncoder to {path}/{filename}")

    def load_weights(self, filepath):
        state_dict = torch.load(filepath, map_location="cpu")
        self.load_state_dict(state_dict)
        print(f"Loaded MCPEncoder from {filepath}")