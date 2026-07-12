"""
Core model: Multitask MLP with Cultural Prototype Gating.

Architecture:
    Input: [CLIP_512 | XLM-R_768] = 1280-d
        ↓
    Shared Trunk: 1280 → 512 → 256 (LayerNorm + Dropout)
        ↓
    3 sigmoid heads: india_label, western_label, china_label
        ↓
    (After prototype construction) Cultural gating applied

Design rationale:
- Shared trunk learns culture-invariant features
- Separate heads allow per-culture decision boundaries
- Gating mechanism modulates between model confidence and cultural prototype signal
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class MisogynyCoreModel(nn.Module):
    """
    Base model: shared MLP trunk + 3 independent binary heads.
    No cultural prototypes yet — those are added post-training in CulturalModel.
    """

    def __init__(self, input_dim=1280, hidden_dim=512, latent_dim=256, dropout=0.3):
        super().__init__()

        # Shared trunk
        self.trunk = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, latent_dim),
            nn.LayerNorm(latent_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # 3 independent binary classification heads
        self.head_india = nn.Linear(latent_dim, 1)
        self.head_western = nn.Linear(latent_dim, 1)
        self.head_china = nn.Linear(latent_dim, 1)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def get_latent(self, x):
        """Get 256-d latent representation (for prototype construction)."""
        return self.trunk(x)

    def forward(self, x):
        """
        Args:
            x: (B, 1280) concatenated embeddings

        Returns:
            logits: (B, 3) — raw logits for [india, western, china]
            latent: (B, 256) — latent representation for prototype use
        """
        latent = self.trunk(x)

        logits = torch.cat([
            self.head_india(latent),
            self.head_western(latent),
            self.head_china(latent),
        ], dim=-1)  # (B, 3)

        return logits, latent


class CulturalPrototypeLayer(nn.Module):
    """
    Post-training module that adds cultural prototype similarity as a decision signal.

    For each culture:
    - Computes similarity to positive (misogyny) prototypes
    - Computes similarity to negative (not-misogyny) prototypes
    - delta = sim_pos - sim_neg (decision-aligned)
    - Gates between base model prediction and cultural signal

    Must call set_prototypes() before forward().
    """

    def __init__(self, latent_dim=256, num_cultures=3):
        super().__init__()
        self.latent_dim = latent_dim
        self.num_cultures = num_cultures

        # Gating network: learns when to rely on cultural prototypes vs base model
        # Input: latent_dim + 2 (sim_pos, sim_neg per culture)
        self.gates = nn.ModuleList([
            nn.Sequential(
                nn.Linear(latent_dim + 2, 64),
                nn.ReLU(),
                nn.Linear(64, 1),
                nn.Sigmoid(),
            )
            for _ in range(num_cultures)
        ])

        # Refinement heads: take gated representation → final logit
        self.refinement_heads = nn.ModuleList([
            nn.Linear(latent_dim + 2, 1)
            for _ in range(num_cultures)
        ])

        # Prototype storage (set after prototype construction)
        self.pos_prototypes = None  # list of (K, 256) tensors per culture
        self.neg_prototypes = None  # list of (K, 256) tensors per culture

    def set_prototypes(self, pos_prototypes, neg_prototypes):
        """
        Args:
            pos_prototypes: list of 3 numpy arrays, each (K, 256) — misogyny centroids
            neg_prototypes: list of 3 numpy arrays, each (K, 256) — not-misogyny centroids
        """
        self.pos_prototypes = [torch.from_numpy(p).float() for p in pos_prototypes]
        self.neg_prototypes = [torch.from_numpy(p).float() for p in neg_prototypes]

    def _compute_prototype_features(self, latent, culture_idx):
        """
        Compute max similarity to pos and neg prototypes for one culture.

        Returns: (B, 2) tensor — [sim_pos, sim_neg]
        """
        pos = self.pos_prototypes[culture_idx]  # (K, 256)
        neg = self.neg_prototypes[culture_idx]  # (K, 256)

        # Normalize for cosine similarity
        latent_norm = F.normalize(latent, dim=-1)  # (B, 256)
        pos_norm = F.normalize(pos, dim=-1)  # (K, 256)
        neg_norm = F.normalize(neg, dim=-1)  # (K, 256)

        sim_pos = torch.mm(latent_norm, pos_norm.t()).max(dim=-1)[0]  # (B,)
        sim_neg = torch.mm(latent_norm, neg_norm.t()).max(dim=-1)[0]  # (B,)

        return torch.stack([sim_pos, sim_neg], dim=-1)  # (B, 2)

    def forward(self, latent, base_logits):
        """
        Args:
            latent: (B, 256) from trunk
            base_logits: (B, 3) from base heads

        Returns:
            refined_logits: (B, 3) — culturally refined predictions
        """
        if self.pos_prototypes is None:
            return base_logits  # no prototypes set yet, passthrough

        outputs = []
        for c in range(self.num_cultures):
            proto_feats = self._compute_prototype_features(latent, c)  # (B, 2)
            combined = torch.cat([latent, proto_feats], dim=-1)  # (B, 258)

            # Gate: how much to rely on cultural signal
            g = self.gates[c](combined)  # (B, 1)

            # Refined prediction
            cultural_logit = self.refinement_heads[c](combined)  # (B, 1)
            base_logit = base_logits[:, c:c+1]  # (B, 1)

            # Gated combination
            refined = g * cultural_logit + (1 - g) * base_logit  # (B, 1)
            outputs.append(refined)

        return torch.cat(outputs, dim=-1)  # (B, 3)


class FullModel(nn.Module):
    """
    Complete model combining core MLP + cultural prototype layer.

    Usage:
        1. Train core model first (stage 4)
        2. Build prototypes (stage 5)
        3. Attach prototype layer and fine-tune gates (stage 5b)
    """

    def __init__(self, input_dim=1280, hidden_dim=512, latent_dim=256,
                 dropout=0.3, num_cultures=3):
        super().__init__()
        self.core = MisogynyCoreModel(input_dim, hidden_dim, latent_dim, dropout)
        self.cultural = CulturalPrototypeLayer(latent_dim, num_cultures)
        self.use_prototypes = False

    def enable_prototypes(self, pos_prototypes, neg_prototypes):
        """Activate cultural prototype layer after stage 5."""
        self.cultural.set_prototypes(pos_prototypes, neg_prototypes)
        self.use_prototypes = True

    def get_latent(self, x):
        return self.core.get_latent(x)

    def forward(self, x):
        """
        Returns:
            logits: (B, 3) — final predictions
            latent: (B, 256) — for prototype construction
        """
        base_logits, latent = self.core(x)

        if self.use_prototypes:
            logits = self.cultural(latent, base_logits)
        else:
            logits = base_logits

        return logits, latent
