import os
import json
from pathlib import Path
from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


def createLocalMemExpert(d_model=512, top_k_query=6, max_steps=8):
    return LocalMemExpert(
        d_model=d_model,
        top_k_query=top_k_query,
        max_steps=max_steps,
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
        q  = self.query_proj(h.mean(dim=1, keepdim=True))    # [B, 1, D]
        bar_out, _ = self.cross_attn(q, kv, kv)
        return self.norm(h + bar_out)


class MemoryLookback(nn.Module):
    """Cross-attention over 1–3 recent external memory vectors."""
    def __init__(self, d_model: int, nhead: int, dropout: float = 0.1):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=nhead, dropout=dropout, batch_first=True
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, h: torch.Tensor, memory_vectors: torch.Tensor) -> torch.Tensor:
        mem_out, _ = self.cross_attn(h, memory_vectors, memory_vectors)
        return self.norm(h + mem_out)


# ── Data stores (pure Python containers — no learned parameters) ──────────────

class WorldKnowledgeStore:
    """
    World knowledge indexed from .md files.

    Evolves during usage: new information (e.g., from internet search output)
    is written to the .md file system and indexed here. The async manager
    (Qwen 1.5B) decides whether new info extends an existing file or creates a
    new one; LocalMemExpert.get_pending_tasks() surfaces those decisions.

    Each entry stores:
        text   : str          — full document text
        emb    : Tensor[T, D] — token-level embeddings (TextEncoder output)
        source : str          — path to the source .md file
    """

    def __init__(self):
        self._docs: list[dict] = []

    def add(self, text: str, emb: torch.Tensor, source: str):
        self._docs.append({"text": text, "emb": emb.cpu(), "source": source})

    def get_all_embs(self) -> list[torch.Tensor]:
        return [d["emb"] for d in self._docs]

    def __len__(self):
        return len(self._docs)

    def save_index(self, path: str):
        os.makedirs(path, exist_ok=True)
        for i, doc in enumerate(self._docs):
            with open(os.path.join(path, f"doc_{i:05d}_meta.json"), "w") as f:
                json.dump({"text": doc["text"], "source": doc["source"]}, f)
            torch.save(doc["emb"], os.path.join(path, f"doc_{i:05d}_emb.pt"))

    def load_index(self, path: str):
        self._docs.clear()
        for meta_file in sorted(Path(path).glob("doc_*_meta.json")):
            with open(meta_file) as f:
                meta = json.load(f)
            emb = torch.load(str(meta_file).replace("_meta.json", "_emb.pt"), map_location="cpu")
            self._docs.append({"text": meta["text"], "emb": emb, "source": meta["source"]})


class LocalFileStore:
    """
    Multimodal local file index with a double graph structure.

    Files (text and images) are chunked and encoded using the in-house
    TextEncoder / VisionEncoder. Two adjacency graphs track relationships:

        source graph  : connects chunks from the same file (structural relation)
        content graph : connects semantically similar chunks across files
                        (cosine sim > CONTENT_SIM_THRESHOLD)

    Retrieval: rough seed from ColBERT → BFS graph expansion → ColBERT reranking.
    Graph topology maintenance (adding edges, merging clusters) is flagged to the
    async manager via LocalMemExpert.get_pending_tasks().
    """

    CONTENT_SIM_THRESHOLD = 0.70

    def __init__(self):
        self._chunks: list[dict] = []                     # {path, text, emb: [T,D], modality}
        self._source_adj: dict[int, list[int]] = {}       # same-file neighbors
        self._content_adj: dict[int, list[int]] = {}      # semantic neighbors

    def add_chunk(self, path: str, text: str, emb: torch.Tensor, modality: str = "text"):
        idx = len(self._chunks)
        self._chunks.append({"path": path, "text": text, "emb": emb.cpu(), "modality": modality})
        self._source_adj[idx] = []
        self._content_adj[idx] = []

        new_vec = F.normalize(emb.mean(dim=0).float(), dim=-1)   # [D]

        for j, existing in enumerate(self._chunks[:-1]):
            # Source graph: same file → connect
            if existing["path"] == path:
                self._source_adj[idx].append(j)
                self._source_adj[j].append(idx)

            # Content graph: semantically similar → connect
            other_vec = F.normalize(existing["emb"].mean(dim=0).float(), dim=-1)
            if float(torch.dot(new_vec, other_vec)) > self.CONTENT_SIM_THRESHOLD:
                self._content_adj[idx].append(j)
                self._content_adj[j].append(idx)

    def graph_expand(self, seed_indices: list[int], n_hops: int = 1) -> list[int]:
        """BFS over both adjacency graphs from seed indices."""
        visited  = set(seed_indices)
        frontier = list(seed_indices)
        for _ in range(n_hops):
            next_frontier = []
            for node in frontier:
                neighbors = (
                    self._source_adj.get(node, []) +
                    self._content_adj.get(node, [])
                )
                for neighbor in neighbors:
                    if neighbor not in visited:
                        visited.add(neighbor)
                        next_frontier.append(neighbor)
            frontier = next_frontier
        return list(visited)

    def get_all_embs(self) -> list[torch.Tensor]:
        return [c["emb"] for c in self._chunks]

    def __len__(self):
        return len(self._chunks)


class MemCellStore:
    """
    MemCell and MemScene storage following EverMemOS (arXiv:2601.02163).

    MemCell  : atomic unit of memory — atomic facts, user preferences, and
               action items extracted from a conversation transcript.
    MemScene : a cluster of related MemCells with a shared summary.

    New MemCells are consolidated into existing MemScenes (cosine sim > threshold)
    or start a fresh MemScene. Scene summarisation and MemCell extraction from raw
    transcripts are delegated to the async manager via LocalMemExpert.get_pending_tasks().

    Foresight generation allows proactive MemCell seeding for anticipated topics:
        memcell_store.flag_for_foresight("quantum computing basics")
    """

    SCENE_SIM_THRESHOLD = 0.65

    def __init__(self):
        self._cells: list[dict]       = []    # {content, emb: [T, D], scene_id}
        self._scenes: list[dict]      = []    # {summary, member_ids, centroid: [D]}
        self._pending_tasks: list[dict] = []

    # ── MemCell management ───────────────────────────────────────────────────

    def add_cell(self, content: str, emb: torch.Tensor):
        cell_vec = F.normalize(emb.mean(dim=0).float(), dim=-1)    # [D]
        scene_id = self._find_or_create_scene(cell_vec, content)
        cell_idx = len(self._cells)
        self._cells.append({"content": content, "emb": emb.cpu(), "scene_id": scene_id})
        self._scenes[scene_id]["member_ids"].append(cell_idx)
        self._update_centroid(scene_id)

    def _find_or_create_scene(self, cell_vec: torch.Tensor, content: str) -> int:
        best_sim, best_id = -1.0, -1
        for i, scene in enumerate(self._scenes):
            sim = float(torch.dot(cell_vec, scene["centroid"]))
            if sim > best_sim:
                best_sim, best_id = sim, i
        if best_sim >= self.SCENE_SIM_THRESHOLD:
            return best_id
        # New scene — flag async manager to generate a proper summary
        self._pending_tasks.append({
            "type": "summarize_scene",
            "content": content,
            "scene_idx": len(self._scenes),
        })
        self._scenes.append({
            "summary": content[:120],
            "member_ids": [],
            "centroid": cell_vec.cpu(),
        })
        return len(self._scenes) - 1

    def _update_centroid(self, scene_id: int):
        ids  = self._scenes[scene_id]["member_ids"]
        vecs = torch.stack([
            F.normalize(self._cells[i]["emb"].mean(dim=0).float(), dim=-1) for i in ids
        ])
        self._scenes[scene_id]["centroid"] = F.normalize(vecs.mean(dim=0), dim=-1).cpu()

    def flag_transcript(self, transcript_text: str):
        """Queue a raw transcript for MemCell extraction by the async manager."""
        self._pending_tasks.append({"type": "extract_memcells", "transcript": transcript_text})

    def flag_for_foresight(self, topic: str):
        """Proactively seed MemCells for an anticipated topic (foresight generation)."""
        self._pending_tasks.append({"type": "foresight", "topic": topic})

    def pop_pending_tasks(self) -> list[dict]:
        tasks = self._pending_tasks.copy()
        self._pending_tasks.clear()
        return tasks

    def get_all_embs(self) -> list[torch.Tensor]:
        return [c["emb"] for c in self._cells]

    def __len__(self):
        return len(self._cells)


# ── Main expert ───────────────────────────────────────────────────────────────

class LocalMemExpert(nn.Module):
    """
    Offline local-filesystem RAG expert with ColBERT retrieval.

    Three independently switchable memory subsystems support incremental training:

        Stage 1  world_knowledge   World .md files, seeded and evolved by
                                   internet search output. Each new fact is
                                   written to a .md file and indexed here.

        Stage 2  transcripts       MemCells / MemScenes from past conversation
                                   transcripts (EverMemOS, arXiv:2601.02163).
                                   Async construction by Qwen 1.5B.

        Stage 3  local_files       Multimodal indexed files (text + images) with
                                   a double graph (source + content adjacency).

    Enable/disable via set_training_stage(1|2|3) or the individual methods.
    All stages start disabled; at least one must be enabled for retrieval to fire.

    Query: top-K most active latent cells by L2 norm (K ≈ 4–8, tunable).
    Retrieval: Graph search (rough) → ColBERT max-sim reranking (fine).
    State injection (cross-attention, as specified):
        update, _ = retrieval_cross_attn(query=state, key=retrieved, value=retrieved)
        state = retrieval_norm(state + update)
    The updated state drives re-retrieval in the next recursive step (ACT-halted).

    Encoders are attached after construction:
        expert.set_encoders(text_encoder=..., vision_encoder=...)

    Async task queue (for Qwen 1.5B memory manager):
        tasks = expert.get_pending_tasks()   # drain after each turn

    Confidence signals after each call:
        self.last_ponder_cost          [B]
        self.last_geometric_confidence [B]
    """

    DOCS_PER_STORE  = 5    # candidates retrieved per enabled store
    GRAPH_HOPS      = 1    # BFS expansion depth for local file graph

    def __init__(
        self,
        d_model: int = 512,
        nhead: int = 8,
        dropout: float = 0.1,
        max_steps: int = 8,
        top_k_query: int = 6,
    ):
        super().__init__()
        self.d_model     = d_model
        self.max_steps   = max_steps
        self.top_k_query = top_k_query    # active-cell query budget (range 4–8)

        # ── State injection ──────────────────────────────────────────────────
        self.retrieval_cross_attn = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=nhead, dropout=dropout, batch_first=True
        )
        self.retrieval_norm = nn.LayerNorm(d_model)

        # ── Recursive processing ─────────────────────────────────────────────
        self.act             = ACTModule(d_model)
        self.bar             = BlockAttentionResidual(d_model, nhead, dropout)
        self.memory_lookback = MemoryLookback(d_model, nhead, dropout)

        # ── Data stores (no learned parameters) ──────────────────────────────
        self.world_store   = WorldKnowledgeStore()
        self.local_store   = LocalFileStore()
        self.memcell_store = MemCellStore()

        # ── Training-stage flags (all off by default) ─────────────────────────
        self._world_enabled       = False
        self._transcripts_enabled = False
        self._local_files_enabled = False

        # ── Shared encoder references (attached via set_encoders) ─────────────
        # Plain dict avoids double-registering shared parameters.
        self._encoders: dict = {"text": None, "vision": None}

        self.last_ponder_cost: Optional[torch.Tensor]          = None
        self.last_geometric_confidence: Optional[torch.Tensor] = None

    # ── Training-stage control ─────────────────────────────────────────────────

    def set_training_stage(self, stage: int):
        """
        Convenience setter for incremental training:
            stage 1 → world knowledge only
            stage 2 → world + transcripts
            stage 3 → all three subsystems
        """
        self._world_enabled       = stage >= 1
        self._transcripts_enabled = stage >= 2
        self._local_files_enabled = stage >= 3

    def enable_world_knowledge(self):   self._world_enabled       = True
    def disable_world_knowledge(self):  self._world_enabled       = False
    def enable_transcripts(self):       self._transcripts_enabled = True
    def disable_transcripts(self):      self._transcripts_enabled = False
    def enable_local_files(self):       self._local_files_enabled = True
    def disable_local_files(self):      self._local_files_enabled = False

    @property
    def active_stages(self) -> list[str]:
        return [s for s, on in [
            ("world",       self._world_enabled),
            ("transcripts", self._transcripts_enabled),
            ("local_files", self._local_files_enabled),
        ] if on]

    # ── Encoder attachment ─────────────────────────────────────────────────────

    def set_encoders(self, text_encoder=None, vision_encoder=None):
        """
        Attach shared encoder instances (called from OverallModel.__init__).
        Required before add_world_document() / index_local_file() / add_memcell().
        """
        if text_encoder  is not None: self._encoders["text"]   = text_encoder
        if vision_encoder is not None: self._encoders["vision"] = vision_encoder

    # ── ColBERT scoring ────────────────────────────────────────────────────────

    @staticmethod
    def _colbert_score(query_embs: torch.Tensor, doc_emb: torch.Tensor) -> float:
        """
        ColBERT late-interaction max-sim score.

        query_embs : [K, D]  top-K active latent cells
        doc_emb    : [T, D]  document token embeddings (TextEncoder output)

        Score = Σ_{q} max_{d} cosine(q, d)
        """
        q   = F.normalize(query_embs.float(), dim=-1)                  # [K, D]
        d   = F.normalize(doc_emb.float().to(q.device), dim=-1)        # [T, D]
        sim = torch.einsum("kd,td->kt", q, d)                          # [K, T]
        return float(sim.max(dim=-1).values.sum())

    def _rank_by_colbert(
        self,
        query_embs: torch.Tensor,
        doc_embs: list[torch.Tensor],
        top_k: int,
    ) -> list[int]:
        """Return indices of the top-k documents by ColBERT score."""
        if not doc_embs:
            return []
        scores = [self._colbert_score(query_embs, d) for d in doc_embs]
        return sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]

    # ── Query extraction ───────────────────────────────────────────────────────

    def _extract_query(self, h: torch.Tensor) -> torch.Tensor:
        """
        Select the top-K most active latent cells by L2 norm.
        K = min(top_k_query, sequence_length); default K ≈ 4–8 (tunable).

        h       : [B, K_state, D]
        Returns : [B, k, D]
        """
        norms = h.norm(dim=-1)                                   # [B, K_state]
        k     = min(self.top_k_query, h.size(1))
        _, idx = norms.topk(k, dim=-1)                           # [B, k]
        return h.gather(1, idx.unsqueeze(-1).expand(-1, -1, h.size(-1)))

    # def _max_pool_query(self, h: torch.Tensor) -> torch.Tensor:
    #     """
    #     Alternative single-vector query via max pooling. [B, K, D] → [B, 1, D].
    #     Uncomment to use as the query vector instead of top-K active cells.
    #     """
    #     return h.max(dim=1, keepdim=True).values

    # ── Retrieval pipeline ─────────────────────────────────────────────────────

    def _retrieve(self, query: torch.Tensor) -> Optional[torch.Tensor]:
        """
        Full retrieval pass over all enabled stores.

        Step 1 – Graph search (local files): rough seed → BFS expansion.
        Step 2 – ColBERT reranking: fine scoring over candidates.
        Step 3 – Compression: mean-pool each doc to a single vector.

        Uses the mean-batch query for store lookup (single retrieval per call).

        query   : [B, K, D]
        Returns : [B, n, D] compressed content, or None if stores are empty.
        """
        mean_q = query.mean(dim=0)             # [K, D]  — mean over batch
        device = query.device

        candidate_embs: list[torch.Tensor] = []

        # ── World knowledge ──────────────────────────────────────────────────
        if self._world_enabled and len(self.world_store) > 0:
            all_embs = self.world_store.get_all_embs()
            top_idx  = self._rank_by_colbert(mean_q, all_embs, self.DOCS_PER_STORE)
            candidate_embs.extend(all_embs[i] for i in top_idx)

        # ── Conversation transcripts (MemCells) ──────────────────────────────
        if self._transcripts_enabled and len(self.memcell_store) > 0:
            all_embs = self.memcell_store.get_all_embs()
            top_idx  = self._rank_by_colbert(mean_q, all_embs, self.DOCS_PER_STORE)
            candidate_embs.extend(all_embs[i] for i in top_idx)

        # ── Local files (graph search → ColBERT rerank) ──────────────────────
        if self._local_files_enabled and len(self.local_store) > 0:
            all_embs = self.local_store.get_all_embs()
            # Rough pass: initial ColBERT seed
            seed_idx  = self._rank_by_colbert(mean_q, all_embs, self.DOCS_PER_STORE)
            # Graph expansion for structural / semantic neighbors
            expanded  = self.local_store.graph_expand(seed_idx, n_hops=self.GRAPH_HOPS)
            exp_embs  = [all_embs[i] for i in expanded]
            # Fine reranking over expanded set
            top_idx   = self._rank_by_colbert(mean_q, exp_embs, self.DOCS_PER_STORE)
            candidate_embs.extend(exp_embs[i] for i in top_idx)

        if not candidate_embs:
            return None

        # Compress: mean-pool each doc [T, D] → [D], then stack → [B, n, D]
        compressed = torch.stack([d.mean(dim=0).to(device) for d in candidate_embs])
        B = query.size(0)
        return compressed.unsqueeze(0).expand(B, -1, -1)      # [B, n, D]

    # ── Public document/memory management ─────────────────────────────────────

    @torch.no_grad()
    def add_world_document(self, text: str, source_path: str):
        """
        Encode a world knowledge .md document and add it to the world store.
        Call this whenever the internet search output produces new information.
        """
        enc = self._encoders["text"]
        if enc is None:
            raise RuntimeError("Text encoder not set. Call set_encoders() first.")
        emb = enc(text).squeeze(0)              # [T, D]
        self.world_store.add(text, emb, source_path)
        # Flag async manager to decide whether to merge with an existing .md file
        self.memcell_store._pending_tasks.append({
            "type": "world_merge_or_create",
            "source": source_path,
            "text_preview": text[:200],
        })

    @torch.no_grad()
    def index_local_file(self, file_path: str, chunk_size: int = 512):
        """
        Chunk and index a local text file using the TextEncoder.
        Vision file support (images → VisionEncoder) can be added here
        by checking the file extension and calling self._encoders["vision"].
        """
        enc = self._encoders["text"]
        if enc is None:
            raise RuntimeError("Text encoder not set. Call set_encoders() first.")
        with open(file_path, "r", encoding="utf-8", errors="replace") as f:
            raw = f.read()
        chunks = [
            raw[i:i + chunk_size]
            for i in range(0, len(raw), chunk_size)
            if raw[i:i + chunk_size].strip()
        ]
        for chunk in chunks:
            emb = enc(chunk).squeeze(0)         # [T, D]
            self.local_store.add_chunk(file_path, chunk, emb, modality="text")
        # Flag async manager to update graph topology for this file
        self.memcell_store._pending_tasks.append({
            "type": "update_file_graph",
            "path": file_path,
            "n_chunks": len(chunks),
        })

    @torch.no_grad()
    def add_memcell(self, content: str):
        """
        Encode and store a MemCell (atomic facts, user preferences, action items).
        Typically called after the async manager processes a raw transcript.
        """
        enc = self._encoders["text"]
        if enc is None:
            raise RuntimeError("Text encoder not set. Call set_encoders() first.")
        emb = enc(content).squeeze(0)           # [T, D]
        self.memcell_store.add_cell(content, emb)

    def flag_transcript_for_processing(self, transcript_text: str):
        """
        Queue a raw transcript for MemCell extraction by the async manager.
        The manager (Qwen 1.5B) reads pending tasks via get_pending_tasks().
        """
        self.memcell_store.flag_transcript(transcript_text)

    def flag_for_foresight(self, topic: str):
        """Flag the async manager to proactively generate MemCells for a topic."""
        self.memcell_store.flag_for_foresight(topic)

    def get_pending_tasks(self) -> list[dict]:
        """
        Drain the async task queue. Call after each turn and dispatch to the
        Qwen 1.5B memory management model:
            - extract_memcells  : build MemCells from a raw transcript
            - summarize_scene   : generate a summary for a new MemScene
            - world_merge_or_create : decide how to file new world knowledge
            - update_file_graph : maintain local file graph topology
            - foresight         : proactively generate MemCells for a topic
        """
        return self.memcell_store.pop_pending_tasks()

    # ── Forward ────────────────────────────────────────────────────────────────

    def forward(
        self,
        state: torch.Tensor,
        memory_vectors: Optional[torch.Tensor] = None,
    ):
        """
        Recursive retrieval loop: extract query → retrieve → inject → halting check.

        The state injection makes the loop recursive: the updated state on each
        step refines the query for the next retrieval pass (re-retrieve as needed).
        ACT determines when the context is sufficiently enriched to stop.

        Args:
            state          : [B, K, D]  latent state matrix from router
            memory_vectors : [B, M, D]  1–3 recent memory entries (optional)

        Returns:
            output : None      — no text artifact; retrieved knowledge is in the state
            state  : [B, K, D] — updated latent state enriched with retrieved context
        """
        B = state.size(0)
        device = state.device
        state_init = state.detach().clone()

        h = state
        residual_stack: list = []

        accumulated  = torch.zeros(B, device=device)
        remainder    = torch.ones(B, device=device)
        weighted_h   = torch.zeros_like(h)
        halted       = torch.zeros(B, dtype=torch.bool, device=device)
        ponder_cost  = torch.zeros(B, device=device)

        for step in range(self.max_steps):
            h_prev = h

            # Query: top-K most active latent cells by L2 norm
            query = self._extract_query(h)               # [B, K, D]

            # Retrieve from all enabled stores; compress to fixed-size context
            retrieved = self._retrieve(query)            # [B, n, D] or None

            # State injection (cross-attention as specified)
            if retrieved is not None:
                update, _ = self.retrieval_cross_attn(
                    query=h,
                    key=retrieved,
                    value=retrieved,
                )
                h = self.retrieval_norm(h + update)

            # Past-state context: external memory lookback
            if memory_vectors is not None:
                h = self.memory_lookback(h, memory_vectors)

            # Inter-iteration context: BAR over accumulated residuals
            residual = h - h_prev
            h = self.bar(h, residual_stack)
            residual_stack.append(residual)

            # ACT halting
            halt_prob = self.act(h)                      # [B]
            active    = (~halted).float()
            new_accum = accumulated + halt_prob * active
            is_last   = (step == self.max_steps - 1)

            halts_here = (new_accum >= 1.0 - self.act.epsilon) & ~halted
            weight = torch.where(
                halts_here | (is_last & ~halted),
                remainder,
                halt_prob * active,
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

        return None, weighted_h

    # ── Persistence ────────────────────────────────────────────────────────────

    def store_weights(self, path, filename="local_mem_expert"):
        """Save learned parameters (nn.Module weights only; use save_stores for data)."""
        os.makedirs(path, exist_ok=True)
        torch.save(self.state_dict(), os.path.join(path, filename))

    def load_weights(self, filepath):
        state_dict = torch.load(filepath, map_location="cpu")
        self.load_state_dict(state_dict)

    def save_stores(self, path: str):
        """Persist all three data stores (document embeddings + metadata) to disk."""
        self.world_store.save_index(os.path.join(path, "world"))
        # LocalFileStore and MemCellStore persistence: extend save_index pattern above

    def load_stores(self, path: str):
        """Restore persisted data stores from disk."""
        world_path = os.path.join(path, "world")
        if os.path.isdir(world_path):
            self.world_store.load_index(world_path)
