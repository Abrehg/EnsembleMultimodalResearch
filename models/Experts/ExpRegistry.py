import torch
import torch.nn as nn
import torch.nn.functional as F
import os

def create_registry():
    return ExpertRegistry()

# Can be changed as needed
def compare(route_vec, expert_embeddings):
    normed = F.normalize(expert_embeddings, dim=-1)
    return route_vec @ normed.T

class ExpertRegistry(nn.Module):
    """
    Manages a set of expert modules alongside a learnable embedding
    matrix used for collaborative-filtering-style routing.

    Each expert has:
        - a unique string name
        - an nn.Module that processes [B, K, d_model] -> [B, K, d_model]
        - a row in the shared embedding matrix (nn.Parameter, gets gradients)

    The registry handles addition, removal, and selection of experts.
    Selection returns both the chosen expert and the full score vector
    so callers can implement soft mixing / Gumbel-softmax during training.

    Usage:
        registry = ExpertRegistry(route_dim=256)
        registry.add_expert("code",  code_model)
        registry.add_expert("math",  math_model)

        expert, scores = registry(route_vec)        # hard select
        state = expert(state)

        # -- or during training, soft-mix --
        scores = registry.score(route_vec)           # [B, N]
        weights = F.softmax(scores / temperature, dim=-1)
    """

    def __init__(self, route_dim: int = 256, compare_fn=None):
        super().__init__()
        self.route_dim = route_dim
        self.compare_fn = compare_fn or compare

        self._embeddings = nn.ParameterList()

        self._name_to_idx: dict[str, int] = {}

        self._idx_to_name: dict[int, str] = {}

        self.experts = nn.ModuleDict()

    def add_expert(self, name: str, module: nn.Module):
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

    def __len__(self):
        return len(self._embeddings)

    @property
    def num_experts(self):
        return len(self)

    @property
    def embedding_matrix(self):
        if len(self._embeddings) == 0:
            raise RuntimeError("No experts registered.")
        return torch.cat([e for e in self._embeddings], dim=0)

    @property
    def expert_names(self):
        return [self._idx_to_name[i] for i in range(len(self))]

    def score(self, route_vec):
        """
        Compute similarity scores between the routing vector and every
        registered expert.

        Args:
            route_vec : [B, route_dim]  (L2-normalized)

        Returns:
            scores : [B, N]  — one score per expert
        """
        return self.compare_fn(route_vec, self.embedding_matrix)

    def select(self, route_vec):
        """
        Hard-select the best expert for each sample in the batch.

        Args:
            route_vec : [B, route_dim]

        Returns:
            expert_indices : [B]        — integer indices
            expert_names   : list[str]  — corresponding names
            scores         : [B, N]     — full score vector (for logging
                                          or auxiliary losses)
        """
        scores = self.score(route_vec)
        expert_indices = scores.argmax(dim=-1)

        names = [self._idx_to_name[i.item()] for i in expert_indices]

        return expert_indices, names, scores

    def forward(self, route_vec):
        _, names, scores = self.select(route_vec)

        if route_vec.size(0) == 1:
            return names[0], scores

        return [n for n in names], scores
    
    def get_expert(self, name):
        return self.experts[name]
    
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
        current_names = self.expert_names

        if saved_dim != self.route_dim:
            raise ValueError(
                f"Route dim mismatch: checkpoint has {saved_dim}, "
                f"registry has {self.route_dim}."
            )

        if strict and saved_names != current_names:
            raise ValueError(
                f"Expert name/order mismatch.\n"
                f"  Checkpoint : {saved_names}\n"
                f"  Current    : {current_names}\n"
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