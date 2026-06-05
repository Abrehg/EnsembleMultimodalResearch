import os
import torch
import torch.nn as nn

def createShortMemory(dim=512, max_entries=16):
    return ShortMemory(d_model=dim, max_entries=max_entries)

class ShortMemory(nn.Module):
    """
    Short-term episodic memory block.

    Stores the latent state matrix at the end of each sequence (conversation
    turn) and makes it available both to the overall encoding pipeline and to
    individual experts.

    Based on the compression scheme from "Out of sight but not out of mind."

    Storage & compression
    ---------------------
    - Up to max_entries latent state matrices, each [K, D].
    - Cell 0 (the anchor) is always the first state ever stored — the original
      user-prompt context — and is never compressed away.
    - When the list is full a single compression pass runs over the non-anchor
      cells, merging consecutive pairs:
          (1, 2) → merged_1
          (3, 4) → merged_2
          ...
      Each merge is implemented as cross-attention: cell_n queries over the
      concatenation of both cells [K_n + K_{n+1}, D], then layer-norm with a
      residual.  An unpaired trailing cell is kept as-is.
    - After compression the list is roughly half the size, making room for the
      new entry.

    Outputs
    -------
    Encoder (full context)
        forward() → [1, N_tokens, D]
        All stored cells are concatenated along the sequence dimension and
        processed by a read transformer (FFN = 4*D).  Compatible with
        EncoderRegistry / EncodingCombination.

    Expert (recent context)
        get_last_3() → list of up to 3 [K, D] tensors
        Experts receive only the 3 most recent cells to keep cross-attention
        compute bounded.

    Dimension rules (identical to the rest of the model)
    -------------------------------------------------------
    - d_model = D    (operates on the full latent dimension; no D/2 pre-stage
                      because these tensors are already latent states)
    - dim_feedforward = 4 * D   (read transformer FFN)
    """

    def __init__(
        self,
        d_model: int = 512,
        max_entries: int = 16,
        nhead: int = 8,
        num_read_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model
        self.max_entries = max_entries

        # ── Compression ────────────────────────────────────────────────
        # Cross-attention that merges two adjacent cells into one.
        self.compress_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=nhead,
            dropout=dropout,
            batch_first=True,
        )
        self.compress_norm = nn.LayerNorm(d_model)

        # ── Full-memory read (encoder output) ──────────────────────────
        # TransformerEncoder over all cells concatenated.  FFN = 4*D.
        read_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.read_transformer = nn.TransformerEncoder(read_layer, num_layers=num_read_layers)

        # ── Runtime memory cells ───────────────────────────────────────
        # Plain Python list of [K, D] tensors.  Not a parameter — cleared
        # on reset() or load_weights().
        self._cells: list = []

    def reset(self):
        """Clear all stored cells (call at the start of a new conversation)."""
        self._cells = []

    def _merge_pair(self, cell_a: torch.Tensor, cell_b: torch.Tensor) -> torch.Tensor:
        """
        Merge two cells [K_a, D] and [K_b, D] into one [K_a, D] cell.

        cell_a acts as the query; both cells concatenated serve as key/value.
        The residual + norm keeps the output in the same space as cell_a.
        """
        kv = torch.cat([cell_a, cell_b], dim=0).unsqueeze(0)   # [1, K_a+K_b, D]
        q  = cell_a.unsqueeze(0)                                # [1, K_a, D]
        attn_out, _ = self.compress_attn(query=q, key=kv, value=kv)
        return self.compress_norm(q + attn_out).squeeze(0)      # [K_a, D]

    def _compress(self):
        """
        One compression pass when memory is full.

        Cell 0 (anchor) is always preserved unchanged.
        Non-anchor cells are merged in consecutive pairs:
            (cells[1], cells[2]) → merged
            (cells[3], cells[4]) → merged  ...
        A trailing unpaired cell is kept as-is.
        """
        anchor = self._cells[0]
        rest   = self._cells[1:]

        merged = [anchor]
        i = 0
        while i + 1 < len(rest):
            merged.append(self._merge_pair(rest[i], rest[i + 1]))
            i += 2
        if i < len(rest):
            merged.append(rest[i])

        self._cells = merged

    def add_state(self, state: torch.Tensor):
        """
        Add a new latent state to memory.

        Compresses the existing cells first if the list is full.

        Args:
            state : [1, K, D] or [K, D]
        """
        if state.dim() == 3:
            state = state.squeeze(0)    # strip batch dim → [K, D]

        if len(self._cells) >= self.max_entries:
            self._compress()

        self._cells.append(state.detach())

    # ──────────────────────────────────────────────────────────────────
    # Read interfaces
    # ──────────────────────────────────────────────────────────────────

    def get_last_3(self) -> list:
        """
        Return the (up to) 3 most recent cells as a list of [K, D] tensors.
        Intended for expert modules that cross-attend into recent memory.
        """
        return self._cells[-3:]

    def _encode_memory(self) -> torch.Tensor:
        """
        Concatenate all cells and run the read transformer.

        Returns:
            [1, N_tokens, D]  — full memory context for EncodingCombination.
            Returns a zero tensor [1, 1, D] when memory is empty so that the
            encoder registry pipeline can always receive a valid tensor.
        """
        if not self._cells:
            device = next(self.parameters()).device
            return torch.zeros(1, 1, self.d_model, device=device)

        memory_seq = torch.cat(self._cells, dim=0).unsqueeze(0)     # [1, N, D]
        return self.read_transformer(memory_seq)                     # [1, N, D]

    # ──────────────────────────────────────────────────────────────────
    # Forward — called by EncoderRegistry
    # ──────────────────────────────────────────────────────────────────

    def forward(self, state=None) -> torch.Tensor:
        """
        Optionally write a new state, then return the full memory encoding.

        Passing state=None is a read-only call — useful when retrieving
        memory context at the start of a turn without yet knowing the new
        state.  To write after a turn, either pass the state here or call
        add_state() directly.

        Args:
            state : [1, K, D] to store, or None for read-only.

        Returns:
            [1, N_tokens, D]
        """
        if state is not None:
            self.add_state(state)
        return self._encode_memory()

    # ──────────────────────────────────────────────────────────────────
    # Persistence (only learned weights; runtime cells are ephemeral)
    # ──────────────────────────────────────────────────────────────────

    def store_weights(self, path, filename="short_memory.pt"):
        os.makedirs(path, exist_ok=True)
        torch.save(self.state_dict(), os.path.join(path, filename))

    def load_weights(self, filepath):
        state_dict = torch.load(filepath, map_location="cpu")
        self.load_state_dict(state_dict)
        self._cells = []    # runtime cells are not persisted
