"""ai_heuristics.py


Very small "AI-like" scoring module. This is intentionally lightweight and
explainable: it scores candidate patterns by frequency, length, and an
optionally learned weight vector.


Replace or extend this module with an actual ML model later.
"""


from typing import Tuple


class SimpleAIHeuristic:
def __init__(self, w_freq: float = 1.0, w_len: float = 0.5, w_savings: float = 1.0):
# weights for a linear scoring function
self.w_freq = w_freq
self.w_len = w_len
self.w_savings = w_savings


def score(self, pattern: str, occurrences: int, estimated_saved: int) -> float:
# score: higher is better
return self.w_freq * occurrences + self.w_len * len(pattern) + self.w_savings * max(0, estimated_saved)


def rank_candidates(self, candidates):
# candidates: list of (pattern, occurrences, estimated_saved)
scored = [(self.score(p, occ, est), (p, occ, est)) for (p, occ, est) in candidates]
scored.sort(key=lambda x: x[0], reverse=True)
return [c for _, c in scored]
```
