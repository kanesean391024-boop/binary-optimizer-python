f.fc1(x))
        x = F.relu(sel"""ai_heuristics.py

Lightweight scoring module combining:
- SimpleAIHeuristic (linear, explainable)
- TinyBinaryModel (16‑bit neural scorer)

Step 2 update: fully fix and integrate TinyBinaryModel + hybrid scoring.
"""

from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------- Tiny Neural Model -------------------
class TinyBinaryModel(nn.Module):
    """
    A tiny neural net that takes a 16‑bit binary array and predicts
    an "efficiency score" between 0 and 1.
    """
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(16, 32)
        self.fc2 = nn.Linear(32, 16)
        self.fc3 = nn.Linear(16, 1)

    def forward(self, x):
        x = F.relu(sef.fc2(x))
        return torch.sigmoid(self.fc3(x))


def score_binary_chunk(chunk: list[int], model: TinyBinaryModel) -> float:
    """Score a 16‑bit binary chunk using the neural model."""
    tensor = torch.tensor(chunk, dtype=torch.float32)
    with torch.no_grad():
        out = model(tensor)
    return float(out.item())


# ---------------- Simple Heuristic -------------------
class SimpleAIHeuristic:
    def __init__(self, w_freq: float = 1.0, w_len: float = 0.5, w_savings: float = 1.0, nn_weight: float = 0.25):
        """
        A linear scoring heuristic + optional neural bonus.
        nn_weight scales the neural model's contribution.
        """
        self.w_freq = w_freq
        self.w_len = w_len
        self.w_savings = w_savings
        self.nn_weight = nn_weight
        self.model = TinyBinaryModel()

    def _nn_bonus(self, pattern: str) -> float:
        if len(pattern) < 16:
            padded = pattern + "0" * (16 - len(pattern))
        else:
            padded = pattern[:16]
        chunk = [1 if c == "1" else 0 for c in padded]
        return score_binary_chunk(chunk, self.model)

    def score(self, pattern: str, occurrences: int, estimated_saved: int) -> float:
        base = (
            self.w_freq * occurrences
            + self.w_len * len(pattern)
            + self.w_savings * max(0, estimated_saved)
        )
        bonus = self.nn_weight * self._nn_bonus(pattern)
        return base + bonus

    def rank_candidates(self, candidates):
        scored = [(self.score(p, occ, est), (p, occ, est)) for (p, occ, est) in candidates]
        scored.sort(key=lambda x: x[0], reverse=True)
        return [c for _, c in scored]
```

"""ai_heuristics.py

Very small "AI-like" scoring module. This is intentionally lightweight and
explainable: it scor
es candidate patterns by frequency, length, and an
optionally learned weight vector.

Replace or extend this module with an actual ML model later.
"""

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# --- New Tiny Neural Model ---
class TinyBinaryModel(nn.Module):
    """
    A tiny neural net that takes a 16-bit binary array and predicts an
    "efficiency score".
    """
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(16, 32)
        self.fc2 = nn.Linear(32, 16)
        self.fc3 = nn.Linear(16, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return torch.sigmoid(self.fc3(x))


def score_binary_chunk(chunk: list[int], model: TinyBinaryModel) -> float:
    """
    Score a 16‑bit binary chunk using the neural model.
    """
    tensor = torch.tensor(chunk, dtype=torch.float32)
    with torch.no_grad():
        out = model(tensor)
    return float(out.item()) self.w_freq * occurrences + self.w_len * len(pattern) + self.w_savings * max(0, estimated_saved)

    def rank_candidates(self, candidates):
        # candidates: list of (pattern, occurrences, estimated_saved)
        scored = [(self.score(p, occ, est), (p, occ, est)) for (p, occ, est) in candidates]
        scored.sort(key=lambda x: x[0], reverse=True)
        return [c for _, c in scored]
```
