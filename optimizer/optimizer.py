"""optimizer.py


High-level optimizer that combines PatternFinder and SimpleAIHeuristic to
select patterns and perform a simple replacement scheme.
"""


from .pattern_finder import PatternFinder
from .ai_heuristics import SimpleAIHeuristic


class Optimizer:
def __init__(self, heuristic: SimpleAIHeuristic = None):
self.finder = PatternFinder()
self.heuristic = heuristic or SimpleAIHeuristic()


def optimize(self, bitstring: str, max_patterns: int = 5):
"""Return an optimized representation and some metrics.


This optimizer performs a simple "tokenize top k patterns" replacement
and returns the transformed bitstring plus stats.
"""
candidates = self.finder.find_patterns(bitstring)
ranked = self.heuristic.rank_candidates(candidates)
selected = ranked[:max_patterns]


# Create a mapping token -> pattern and perform replacements.
# Tokens are ASCII characters starting at 128 to avoid '0' and '1' conflict.
token_base = 128
mapping = {}
transformed = bitstring
for idx, (pat, occ, saved) in enumerate(selected):
token = chr(token_base + idx)
mapping[token] = pat
transformed = transformed.replace(pat, token)


original_len = len(bitstring)
transformed_len = len(transformed)
# rough savings: account for mapping table size
mapping_overhead = sum(len(v) + 1 for v in mapping.values())
net_saved = original_len - transformed_len - mapping_overhead


return {
"original": bitstring,
"transformed": transformed,
"mapping": mapping,
"original_len": original_len,
"transformed_len": transformed_len,
"mapping_overhead": mapping_overhead,
"net_saved": net_saved,
}
```
