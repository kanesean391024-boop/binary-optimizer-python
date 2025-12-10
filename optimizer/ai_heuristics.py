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
