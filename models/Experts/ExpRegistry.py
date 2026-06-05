import torch
import torch.nn as nn
import torch.nn.functional as F
import os

from Encoders.TextEncoder import TextEncoder
from Encoders.EncCombine import EncodingCombination
from Logic import RoutingLogic


def create_registry(dim=512):
    return ExpertRegistry(route_dim=dim // 4)


def compare(route_vec, expert_embeddings):
    normed = F.normalize(expert_embeddings, dim=-1)
    return route_vec @ normed.T


class ExpertRegistry(nn.Module):
    """
    Manages expert modules alongside a learnable embedding matrix for routing.

    Each expert has:
        - a unique string name
        - an nn.Module that processes the latent state
        - a row in the shared embedding matrix (nn.Parameter, receives gradients)

    Embeddings can be cold-initialized (random) via add_expert(), or warm-initialized
    from an experts.md description via add_expert_from_md(). The warm path seeds the
    embedding by running the description through the shared model pipeline:

        MD text → tokenize → TextEncoder → EncCombine → Router → L2 norm

    The seeded embedding remains an nn.Parameter and continues to adapt during
    training, mirroring collaborative-filtering item embeddings. experts.md quality
    directly determines how accurately the initial embedding places the expert in
    routing space — richer, more specific descriptions yield better cold-start routing.

    Call set_pipeline() once (from OverallModel.__init__) to attach the shared
    TextEncoder, EncodingCombination, and RoutingLogic instances before calling
    add_expert_from_md().

    The main routing action loop is fuse_route() (also callable via forward()).

    Dimension rules:
        route_dim = D // 4   (routing embedding vector size)
    """

    def __init__(self, route_dim: int = 128, compare_fn=None):
        super().__init__()
        self.route_dim = route_dim
        self.compare_fn = compare_fn or compare

        self._embeddings = nn.ParameterList()
        self._name_to_idx: dict[str, int] = {}
        self._idx_to_name: dict[int, str] = {}
        self.experts = nn.ModuleDict()
        self._pipeline: dict = {
            "text_encoder": None,
            "combiner": None,
            "router": None,
        }

    # ─────────────────────────────────────────────────────────────────────────
    # Pipeline attachment
    # ─────────────────────────────────────────────────────────────────────────

    def set_pipeline(
        self,
        text_encoder: TextEncoder,
        combiner: EncodingCombination,
        router: RoutingLogic,
    ):
        """
        Attach the shared pipeline components needed for MD-based embedding
        initialization. Call this once from OverallModel.__init__ after all
        components have been created.
        """
        self._pipeline["text_encoder"] = text_encoder
        self._pipeline["combiner"] = combiner
        self._pipeline["router"] = router

    # ─────────────────────────────────────────────────────────────────────────
    # Registration
    # ─────────────────────────────────────────────────────────────────────────

    def add_expert(self, name: str, module: nn.Module):
        """Register an expert with a random (cold) initial embedding."""
        if name in self._name_to_idx:
            raise ValueError(
                f"Expert '{name}' already registered. "
                f"Remove it first with remove_expert('{name}')."
            )

        idx = len(self._embeddings)
        embedding = nn.Parameter(
            F.normalize(torch.randn(1, self.route_dim), dim=-1)
        )
        self._embeddings.append(embedding)
        self._name_to_idx[name] = idx
        self._idx_to_name[idx] = name
        self.experts[name] = module

        print(f"Registered expert '{name}' at index {idx}")

    @torch.no_grad()
    def add_expert_from_md(
        self,
        name: str,
        module: nn.Module,
        md_path: str,
        device=None,
    ):
        """
        Register a new expert with a warm embedding seeded from an experts.md file.

        Reads the description, runs it through the shared pipeline, and uses the
        resulting L2-normalized routing vector as the initial embedding. The
        embedding then updates through normal gradient descent during training.

        set_pipeline() must be called before this method. TextEncoder handles
        tokenization internally — plain text is passed directly.

        Args:
            name    : unique expert name
            module  : the expert nn.Module
            md_path : path to an experts.md description file
            device  : target device; inferred from module parameters if None
        """
        if name in self._name_to_idx:
            raise ValueError(
                f"Expert '{name}' already registered. "
                f"Remove it first with remove_expert('{name}')."
            )

        text_encoder: TextEncoder = self._pipeline["text_encoder"]
        combiner: EncodingCombination = self._pipeline["combiner"]
        router: RoutingLogic = self._pipeline["router"]

        if text_encoder is None or combiner is None or router is None:
            raise RuntimeError(
                "Pipeline not set. Call set_pipeline(text_encoder, combiner, router) "
                "before using add_expert_from_md()."
            )

        if device is None:
            try:
                device = next(module.parameters()).device
            except StopIteration:
                device = torch.device("cpu")

        with open(md_path, "r", encoding="utf-8") as f:
            text = f.read()

        # Temporarily set eval mode so dropout/BN don't pollute the seed pass
        prev_enc  = text_encoder.training
        prev_comb = combiner.training
        prev_rtr  = router.training
        text_encoder.eval()
        combiner.eval()
        router.eval()

        encoded = text_encoder(text)               # [1, T, D]
        latent, _, _ = combiner([encoded])          # [1, K, D]
        route_vec, _ = router(latent)               # [1, route_dim] — already L2-normed
        route_vec = F.normalize(route_vec, dim=-1)  # guard: router.normalize may be off

        text_encoder.train(prev_enc)
        combiner.train(prev_comb)
        router.train(prev_rtr)

        idx = len(self._embeddings)
        embedding = nn.Parameter(route_vec.detach().clone())
        self._embeddings.append(embedding)
        self._name_to_idx[name] = idx
        self._idx_to_name[idx] = name
        self.experts[name] = module

        print(f"Registered expert '{name}' at index {idx} "
              f"with warm embedding from '{md_path}'")

    def remove_expert(self, name: str):
        if name not in self._name_to_idx:
            raise ValueError(f"Expert '{name}' is not registered.")

        idx = self._name_to_idx[name]
        del self._embeddings[idx]
        del self.experts[name]

        self._name_to_idx.clear()
        self._idx_to_name.clear()
        for i, key in enumerate(self.experts.keys()):
            self._name_to_idx[key] = i
            self._idx_to_name[i] = key

        print(f"Removed expert '{name}'. {len(self)} expert(s) remaining.")

    # ─────────────────────────────────────────────────────────────────────────
    # Properties
    # ─────────────────────────────────────────────────────────────────────────

    def __len__(self):
        return len(self._embeddings)

    @property
    def num_experts(self):
        return len(self)

    @property
    def embedding_matrix(self):
        if len(self._embeddings) == 0:
            raise RuntimeError("No experts registered.")
        return torch.cat(list(self._embeddings), dim=0)   # [N, route_dim]

    @property
    def expert_names(self):
        return [self._idx_to_name[i] for i in range(len(self))]

    # ─────────────────────────────────────────────────────────────────────────
    # Routing
    # ─────────────────────────────────────────────────────────────────────────

    def score(self, route_vec):
        """
        Compute similarity scores between the routing vector and all experts.

        Args:
            route_vec : [B, route_dim]  (L2-normalized)

        Returns:
            scores : [B, N]
        """
        return self.compare_fn(route_vec, self.embedding_matrix)

    def select(self, route_vec):
        """
        Hard-select the highest-scoring expert per batch item.

        Args:
            route_vec : [B, route_dim]

        Returns:
            expert_indices : [B]
            expert_names   : list[str]
            scores         : [B, N]  — full score vector for auxiliary losses
        """
        scores = self.score(route_vec)
        expert_indices = scores.argmax(dim=-1)
        names = [self._idx_to_name[i.item()] for i in expert_indices]
        return expert_indices, names, scores

    def fuse_route(self, route_vec):
        """
        Main action loop: score all experts against the routing vector,
        hard-select the best match, and return the chosen name(s) with
        the full score vector.

        Args:
            route_vec : [B, route_dim]  (L2-normalized)

        Returns:
            chosen : str (B=1) or list[str] (B>1)
            scores : [B, N]
        """
        _, names, scores = self.select(route_vec)
        return (names[0] if route_vec.size(0) == 1 else names), scores

    def forward(self, route_vec):
        return self.fuse_route(route_vec)

    def get_expert(self, name: str):
        return self.experts[name]

    # ─────────────────────────────────────────────────────────────────────────
    # Persistence
    # ─────────────────────────────────────────────────────────────────────────

    def store_weights(self, path, filename="expert_registry"):
        os.makedirs(path, exist_ok=True)
        checkpoint = {
            "state_dict": self.state_dict(),
            "expert_names": self.expert_names,
            "route_dim": self.route_dim,
        }
        torch.save(checkpoint, os.path.join(path, filename))

    def load_weights(self, filepath, strict=True):
        checkpoint = torch.load(filepath, map_location="cpu")

        saved_names = checkpoint["expert_names"]
        saved_dim = checkpoint["route_dim"]

        if saved_dim != self.route_dim:
            raise ValueError(
                f"Route dim mismatch: checkpoint has {saved_dim}, "
                f"registry has {self.route_dim}."
            )

        if strict and saved_names != self.expert_names:
            raise ValueError(
                f"Expert name/order mismatch.\n"
                f"  Checkpoint : {saved_names}\n"
                f"  Current    : {self.expert_names}\n"
                f"Use strict=False to load partial weights."
            )

        missing, unexpected = self.load_state_dict(
            checkpoint["state_dict"], strict=strict
        )

        status = "strict" if strict else "partial"
        print(f"Loaded registry weights ({status}) from {filepath}")

        if not strict:
            if missing:
                print(f"  New (randomly initialised): {len(missing)} params")
            if unexpected:
                print(f"  Skipped (removed experts):  {len(unexpected)} params")
