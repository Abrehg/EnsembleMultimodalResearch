import os
import json
import math
from pathlib import Path
from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

from Encoders.TextEncoder import createTokenizer

PAD_ID = 0
BOS_ID = 2
EOS_ID = 3


def createInternetSearchExpert(d_model=512, max_steps=6, top_k_query=6, n_results=5):
    return InternetSearchExpert(
        d_model=d_model,
        dim_feedforward=d_model * 4,
        max_steps=max_steps,
        top_k_query=top_k_query,
        n_results=n_results,
    )


# ── Shared expert primitives ──────────────────────────────────────────────────

class ACTModule(nn.Module):
    """Adaptive Computational Time halting (Graves 2016)."""
    def __init__(self, d_model: int, epsilon: float = 0.01):
        super().__init__()
        self.halt_proj = nn.Linear(d_model, 1)
        self.epsilon = epsilon

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """h: [B, S, D] → halt_prob: [B]"""
        return torch.sigmoid(self.halt_proj(h.mean(dim=1))).squeeze(-1)


class BlockAttentionResidual(nn.Module):
    """BAR: q_k = W_q · mean(h_k), attends to stacked prior residuals."""
    def __init__(self, d_model: int, nhead: int, dropout: float = 0.1):
        super().__init__()
        self.query_proj = nn.Linear(d_model, d_model)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=nhead, dropout=dropout, batch_first=True
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, h: torch.Tensor, residual_stack: list) -> torch.Tensor:
        if not residual_stack:
            return h
        kv = torch.cat(residual_stack, dim=1)
        q  = self.query_proj(h.mean(dim=1, keepdim=True))
        bar_out, _ = self.cross_attn(q, kv, kv)
        return self.norm(h + bar_out)


class MemoryLookback(nn.Module):
    """Cross-attention over 1-3 recent external memory vectors."""
    def __init__(self, d_model: int, nhead: int, dropout: float = 0.1):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=nhead, dropout=dropout, batch_first=True
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, h: torch.Tensor, memory_vectors: torch.Tensor) -> torch.Tensor:
        mem_out, _ = self.cross_attn(h, memory_vectors, memory_vectors)
        return self.norm(h + mem_out)


# ── Search connectors ─────────────────────────────────────────────────────────

class SearchConnector:
    """
    Abstract base for all search backends.

    Subclasses are fully interchangeable — swap the connector on
    InternetSearchExpert.set_connector() to change the data source
    without touching any model code.  search() must return a list of
    plain-text result strings (passages or article summaries).
    """

    def search(self, query: str, top_k: int = 5) -> list[str]:
        raise NotImplementedError

    @property
    def name(self) -> str:
        return self.__class__.__name__


class WikipediaConnector(SearchConnector):
    """
    Offline Wikipedia search connector — the default for training.

    Two modes, tried in order:
        1. Local dump  — a directory of WikiExtractor-formatted files
                         (one JSON per line: {id, title, text, url}).
                         Load once with WikipediaConnector.load_dump(path).
                         Retrieval uses weighted keyword overlap (title × 3 +
                         body × 1).
        2. Live API    — the `wikipedia` Python package (pip install wikipedia).
                         Falls back silently when unavailable or offline.

    Training methodology (LRAT-style, arXiv:2604.04949):
        1. Download a Wikipedia dump from https://dumps.wikimedia.org/
        2. Extract with WikiExtractor:
               python -m wikiextractor.WikiExtractor enwiki-*.xml.bz2 -o wiki_out/
        3. Call connector.load_dump("wiki_out/") at startup.
        InternetSearchExpert.last_retrieval_score [B] is the LRAT auxiliary signal:
        maximising it trains the QueryDecoder to surface relevant passages.
    """

    def __init__(self):
        self._articles: list[dict] = []
        self._dump_loaded = False

    def load_dump(self, dump_dir: str, max_articles: int = 500_000):
        """
        Parse a WikiExtractor output directory (files named wiki_00, wiki_01, …).
        """
        self._articles.clear()
        for wiki_file in sorted(Path(dump_dir).rglob("wiki_*")):
            if wiki_file.suffix:            # skip .bz2 / .gz variants
                continue
            with open(wiki_file, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                        self._articles.append({
                            "title": obj.get("title", ""),
                            "text":  obj.get("text",  "")[:2000],
                        })
                    except json.JSONDecodeError:
                        continue
                    if len(self._articles) >= max_articles:
                        break
            if len(self._articles) >= max_articles:
                break
        self._dump_loaded = True
        print(f"[WikipediaConnector] Loaded {len(self._articles):,} articles from {dump_dir}")

    def search(self, query: str, top_k: int = 5) -> list[str]:
        if self._dump_loaded and self._articles:
            return self._keyword_search(query, top_k)
        try:
            import wikipedia                          # type: ignore
            titles  = wikipedia.search(query, results=top_k * 2)
            results = []
            for title in titles[:top_k]:
                try:
                    results.append(wikipedia.summary(title, sentences=4))
                except Exception:
                    continue
            return results
        except Exception:
            return []

    def _keyword_search(self, query: str, top_k: int) -> list[str]:
        q_words = set(query.lower().split())
        scored: list[tuple[float, str]] = []
        for art in self._articles:
            title_hits = len(q_words & set(art["title"].lower().split()))
            body_hits  = len(q_words & set(art["text"].lower().split()))
            score = title_hits * 3.0 + body_hits
            if score > 0:
                scored.append((score, art["text"]))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [text for _, text in scored[:top_k]]


# class DuckDuckGoConnector(SearchConnector):
#     """
#     Online DuckDuckGo search via SerpAPI.
#     API endpoint: https://serpapi.com/duckduckgo-light-api
#
#     Requirements:
#         pip install google-search-results
#         Set SERPAPI_KEY in environment.
#
#     Swap in with:  expert.set_connector(DuckDuckGoConnector())
#     """
#
#     def __init__(self, api_key: Optional[str] = None):
#         self._api_key = api_key or os.environ.get("SERPAPI_KEY", "")
#
#     def search(self, query: str, top_k: int = 5) -> list[str]:
#         try:
#             from serpapi import GoogleSearch      # type: ignore
#             results = GoogleSearch({
#                 "engine":  "duckduckgo",
#                 "q":       query,
#                 "api_key": self._api_key,
#                 "num":     top_k * 2,
#             }).get_dict()
#             return [
#                 r.get("snippet", r.get("title", ""))
#                 for r in results.get("organic_results", [])
#             ][:top_k]
#         except Exception:
#             return []


# ── Query decoder ─────────────────────────────────────────────────────────────

class QueryDecoder(nn.Module):
    """
    Lightweight autoregressive decoder that generates a short text query
    (max MAX_QUERY_TOKENS = 30 tokens) from the latent state matrix.

    Architecture:
        Self-attention over state → 2-layer TransformerDecoder → vocabulary.

    This is a stripped-down TextOutputExpert — no ACT, no recursion, no
    BAR — its sole purpose is to produce a concise search query string.

    The generated text is what the SearchConnector receives.  The original
    latent cells (captured *before* this decoder runs) are used separately
    for ColBERT reranking of the results.
    """

    MAX_QUERY_TOKENS = 30

    def __init__(
        self,
        vocab_size: int = 32000,
        d_model: int = 512,
        nhead: int = 8,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model

        self.state_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.state_norm = nn.LayerNorm(d_model)

        self.token_embedding = nn.Embedding(vocab_size, d_model, padding_idx=PAD_ID)

        layer = nn.TransformerDecoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout, activation="gelu", batch_first=True,
        )
        self.decoder = nn.TransformerDecoder(layer, num_layers=2)

        self.output_proj = nn.Linear(d_model, vocab_size, bias=False)
        self.output_proj.weight = self.token_embedding.weight   # weight tying

        self.tokenizer = createTokenizer()

    def _sincos_pos(self, seq_len: int, device) -> torch.Tensor:
        pos   = torch.arange(seq_len, dtype=torch.float32, device=device)
        omega = torch.exp(
            -torch.arange(0, self.d_model, 2, dtype=torch.float32, device=device)
            * (math.log(10000.0) / self.d_model)
        )
        out = pos[:, None] * omega[None, :]
        emb = torch.stack([torch.sin(out), torch.cos(out)], dim=2).flatten(1)
        return emb.unsqueeze(0)

    def forward(self, state: torch.Tensor, temperature: float = 0.7) -> list[str]:
        """
        state   : [B, K, D]
        Returns : list[str] of length B (one query string per batch item)
        """
        B = state.size(0)
        device = state.device

        att, _ = self.state_attn(state, state, state)
        memory = self.state_norm(state + att)              # [B, K, D]

        generated = torch.full((B, 1), BOS_ID, dtype=torch.long, device=device)
        finished  = torch.zeros(B, dtype=torch.bool, device=device)

        for _ in range(self.MAX_QUERY_TOKENS):
            T   = generated.size(1)
            tgt = self.token_embedding(generated) * math.sqrt(self.d_model)
            tgt = tgt + self._sincos_pos(T, device)
            mask = nn.Transformer.generate_square_subsequent_mask(T, device=device)

            out    = self.decoder(tgt=tgt, memory=memory, tgt_mask=mask)
            logits = self.output_proj(out[:, -1, :]) / max(temperature, 1e-8)
            token  = torch.multinomial(F.softmax(logits, dim=-1), num_samples=1)
            token  = token.masked_fill(finished.unsqueeze(-1), PAD_ID)
            generated = torch.cat([generated, token], dim=1)
            finished  = finished | (token.squeeze(-1) == EOS_ID)
            if finished.all():
                break

        return self.tokenizer.detokenize(generated, strip_special=True)


# ── Adaptive context compressor ───────────────────────────────────────────────

class AdaptiveContextCompressor(nn.Module):
    """
    Cross-attention pooling: compresses variable-length retrieved content
    into n_slots fixed context slots for state injection.

    n_slots learned query tokens cross-attend to the full concatenated
    retrieved token sequence, producing a dense [B, n_slots, D] tensor.
    """

    def __init__(self, d_model: int, n_slots: int = 16, nhead: int = 8, dropout: float = 0.1):
        super().__init__()
        self.query_tokens = nn.Parameter(torch.randn(1, n_slots, d_model) * 0.02)
        self.cross_attn   = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.norm         = nn.LayerNorm(d_model)

    def forward(self, retrieved: torch.Tensor) -> torch.Tensor:
        """retrieved: [B, T_total, D] → [B, n_slots, D]"""
        B = retrieved.size(0)
        q = self.query_tokens.expand(B, -1, -1)
        out, _ = self.cross_attn(q, retrieved, retrieved)
        return self.norm(q + out)


# ── Main expert ───────────────────────────────────────────────────────────────

class InternetSearchExpert(nn.Module):
    """
    Internet search expert: state → text query → SERP connector → ColBERT rank → state injection.

    Pipeline per recursive step:
        1. Extract top-K active cells by L2 norm (the ColBERT query; captured before decoding)
        2. QueryDecoder generates a short text query (≤ 30 tokens) from the state
        3. SearchConnector retrieves raw text passages
        4. TextEncoder encodes passages; ColBERT ranks them against the step-1 cells
        5. AdaptiveContextCompressor pools top-k passages to n_context_slots fixed tokens
        6. Cross-attention state injection (identical form to LocalMemExpert)
        7. MemoryLookback / BAR / ACT halting

    The injected state refines the query on the next step, making the loop recursive.

    Artifacts (hidden from user):
        Retrieved plain-text passages are returned as hidden artifacts and forwarded
        to the async memory manager for world-knowledge updates to LocalMemExpert.

    Training (LRAT-style, arXiv:2604.04949):
        External supervision — QueryDecoder trains end-to-end via downstream loss.
        Auxiliary signal: last_retrieval_score [B] = mean ColBERT sim of the best
        retrieved document.  Maximising this encourages the decoder to produce queries
        that surface relevant passages.

    Connector swapping:
        expert.set_connector(WikipediaConnector())    # default (offline, dump-based)
        # expert.set_connector(DuckDuckGoConnector()) # production (SerpAPI / online)

    Confidence signals stored after each call:
        self.last_ponder_cost          [B]
        self.last_geometric_confidence [B]
        self.last_retrieval_score      [B]  (LRAT training signal)
    """

    def __init__(
        self,
        d_model: int = 512,
        nhead: int = 8,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        max_steps: int = 6,
        top_k_query: int = 6,
        n_results: int = 5,
        n_colbert_keep: int = 3,
        n_context_slots: int = 16,
    ):
        super().__init__()
        self.d_model        = d_model
        self.max_steps      = max_steps
        self.top_k_query    = top_k_query
        self.n_results      = n_results
        self.n_colbert_keep = n_colbert_keep

        # ── Query generation ─────────────────────────────────────────────────
        self.query_decoder = QueryDecoder(
            d_model=d_model, nhead=nhead,
            dim_feedforward=dim_feedforward, dropout=dropout,
        )

        # ── Content compression ──────────────────────────────────────────────
        self.compressor = AdaptiveContextCompressor(
            d_model=d_model, n_slots=n_context_slots, nhead=nhead, dropout=dropout
        )

        # ── State injection ──────────────────────────────────────────────────
        self.inject_cross_attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=True
        )
        self.inject_norm = nn.LayerNorm(d_model)

        # ── Recursive processing ─────────────────────────────────────────────
        self.act             = ACTModule(d_model)
        self.bar             = BlockAttentionResidual(d_model, nhead, dropout)
        self.memory_lookback = MemoryLookback(d_model, nhead, dropout)

        # ── Search connector (swappable; Wikipedia offline by default) ────────
        self._connector: Optional[SearchConnector] = WikipediaConnector()

        # ── Shared encoder (set via set_encoders) ────────────────────────────
        self._encoders: dict = {"text": None}

        self.last_ponder_cost: Optional[torch.Tensor]          = None
        self.last_geometric_confidence: Optional[torch.Tensor] = None
        self.last_retrieval_score: Optional[torch.Tensor]      = None

    # ── Configuration ──────────────────────────────────────────────────────────

    def set_connector(self, connector: SearchConnector):
        """Swap the search backend without touching any model weights."""
        self._connector = connector

    def set_encoders(self, text_encoder=None, **_):
        """Attach the shared TextEncoder for encoding retrieved passages."""
        if text_encoder is not None:
            self._encoders["text"] = text_encoder

    # ── Helpers ────────────────────────────────────────────────────────────────

    def _extract_query_cells(self, h: torch.Tensor) -> torch.Tensor:
        """Top-K most active latent cells by L2 norm. [B, K_state, D] → [B, k, D]"""
        norms = h.norm(dim=-1)
        k     = min(self.top_k_query, h.size(1))
        _, idx = norms.topk(k, dim=-1)
        return h.gather(1, idx.unsqueeze(-1).expand(-1, -1, h.size(-1)))

    @staticmethod
    def _colbert_score(query_cells: torch.Tensor, doc_emb: torch.Tensor) -> float:
        """
        ColBERT max-sim score.
        query_cells : [K, D]  latent cells captured before query decoding
        doc_emb     : [T, D]  TextEncoder output for a retrieved passage
        """
        q = F.normalize(query_cells.float(), dim=-1)
        d = F.normalize(doc_emb.float().to(q.device), dim=-1)
        return float(torch.einsum("kd,td->kt", q, d).max(dim=-1).values.sum())

    @torch.no_grad()
    def _encode_passage(self, text: str) -> torch.Tensor:
        enc = self._encoders["text"]
        if enc is None:
            raise RuntimeError("Text encoder not set. Call set_encoders() first.")
        return enc(text).squeeze(0)   # [T, D]

    def _search_and_rank(
        self,
        query_texts: list[str],
        colbert_queries: torch.Tensor,   # [B, K, D]
    ) -> tuple[list[list[str]], torch.Tensor]:
        """
        Per batch item: search → encode → ColBERT rank → keep top n_colbert_keep.

        Returns:
            top_texts   : list[list[str]] of length B
            ret_score   : [B] mean ColBERT sim of the best passage (training signal)
        """
        B = colbert_queries.size(0)
        top_texts:  list[list[str]] = []
        ret_scores: list[float]     = []

        for b in range(B):
            q_text  = query_texts[b] if isinstance(query_texts, list) else query_texts
            q_cells = colbert_queries[b]                        # [K, D]
            raw     = self._connector.search(q_text, top_k=self.n_results) if self._connector else []

            if not raw:
                top_texts.append([])
                ret_scores.append(0.0)
                continue

            scored: list[tuple[float, str]] = []
            for passage in raw:
                try:
                    emb   = self._encode_passage(passage)
                    score = self._colbert_score(q_cells, emb)
                    scored.append((score, passage))
                except Exception:
                    scored.append((0.0, passage))

            scored.sort(key=lambda x: x[0], reverse=True)
            kept = scored[:self.n_colbert_keep]
            top_texts.append([t for _, t in kept])
            ret_scores.append(kept[0][0] if kept else 0.0)

        return top_texts, torch.tensor(ret_scores, device=colbert_queries.device)

    def _stack_passages(
        self,
        top_texts: list[list[str]],
        B: int,
        device,
    ) -> Optional[torch.Tensor]:
        """
        Encode all passages and pad-stack into [B, T_total, D] for the compressor.
        Returns None when no passages were retrieved.
        """
        batch_embs: list[Optional[torch.Tensor]] = []
        has_any = False
        for b in range(B):
            parts = []
            for passage in top_texts[b]:
                try:
                    parts.append(self._encode_passage(passage).to(device))
                except Exception:
                    continue
            if parts:
                batch_embs.append(torch.cat(parts, dim=0))
                has_any = True
            else:
                batch_embs.append(None)

        if not has_any:
            return None

        D = self.d_model
        max_len = max((e.size(0) for e in batch_embs if e is not None), default=1)
        padded = []
        for e in batch_embs:
            if e is None:
                padded.append(torch.zeros(max_len, D, device=device))
            elif e.size(0) < max_len:
                pad = torch.zeros(max_len - e.size(0), D, device=device)
                padded.append(torch.cat([e, pad], dim=0))
            else:
                padded.append(e)
        return torch.stack(padded, dim=0)   # [B, max_len, D]

    # ── Forward ────────────────────────────────────────────────────────────────

    def forward(
        self,
        state: torch.Tensor,
        memory_vectors: Optional[torch.Tensor] = None,
    ):
        """
        Args:
            state          : [B, K, D]  latent state matrix from router
            memory_vectors : [B, M, D]  1-3 recent memory entries (optional)

        Returns:
            artifacts : list[str] or None
                Hidden plain-text passages (not shown to user; sent to the
                memory manager to update world knowledge in LocalMemExpert).
            state     : [B, K, D]  updated latent state
        """
        B = state.size(0)
        device = state.device
        state_init = state.detach().clone()

        h = state
        residual_stack: list  = []
        all_artifacts: list[str] = []

        accumulated      = torch.zeros(B, device=device)
        remainder        = torch.ones(B, device=device)
        weighted_h       = torch.zeros_like(h)
        halted           = torch.zeros(B, dtype=torch.bool, device=device)
        ponder_cost      = torch.zeros(B, device=device)
        total_ret_score  = torch.zeros(B, device=device)

        for step in range(self.max_steps):
            h_prev = h

            # Capture latent cells BEFORE decoding (used for ColBERT, not the API)
            colbert_query = self._extract_query_cells(h)      # [B, K, D]

            # Generate text query for the search connector (≤ 30 tokens)
            with torch.no_grad():
                query_texts = self.query_decoder(h)            # list[str], len=B

            # Search → ColBERT rerank → hidden artifacts
            top_texts, ret_score = self._search_and_rank(query_texts, colbert_query)
            total_ret_score = total_ret_score + ret_score * (~halted).float()
            for passages in top_texts:
                all_artifacts.extend(passages)

            # Encode + compress top passages to fixed context slots
            stacked = self._stack_passages(top_texts, B, device)
            if stacked is not None:
                compressed = self.compressor(stacked)          # [B, n_slots, D]
                update, _  = self.inject_cross_attn(query=h, key=compressed, value=compressed)
                h = self.inject_norm(h + update)

            # Past-state context: external memory lookback
            if memory_vectors is not None:
                h = self.memory_lookback(h, memory_vectors)

            # Inter-iteration context: BAR
            residual = h - h_prev
            h = self.bar(h, residual_stack)
            residual_stack.append(residual)

            # ACT halting
            halt_prob = self.act(h)
            active    = (~halted).float()
            new_accum = accumulated + halt_prob * active
            is_last   = (step == self.max_steps - 1)

            halts_here = (new_accum >= 1.0 - self.act.epsilon) & ~halted
            weight = torch.where(
                halts_here | (is_last & ~halted), remainder, halt_prob * active
            )

            weighted_h  = weighted_h + weight.view(B, 1, 1) * h
            ponder_cost = ponder_cost + active
            accumulated = torch.where(halts_here, torch.ones_like(accumulated), new_accum)
            remainder   = 1.0 - accumulated
            halted      = halted | halts_here

            if halted.all():
                break

        geometric_conf = 1.0 - F.cosine_similarity(
            state_init.mean(dim=1), weighted_h.mean(dim=1), dim=-1
        )

        self.last_ponder_cost          = ponder_cost.detach()
        self.last_geometric_confidence = geometric_conf.detach()
        self.last_retrieval_score      = total_ret_score.detach()

        return (all_artifacts if all_artifacts else None), weighted_h

    # ── Persistence ────────────────────────────────────────────────────────────

    def store_weights(self, path, filename="internet_search_expert"):
        os.makedirs(path, exist_ok=True)
        torch.save(self.state_dict(), os.path.join(path, filename))

    def load_weights(self, filepath):
        self.load_state_dict(torch.load(filepath, map_location="cpu"))
