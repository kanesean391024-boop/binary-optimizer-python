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
        """
        Step 3: Neural‑integrated pipeline
        - PatternFinder finds candidates
        - SimpleAIHeuristic ranks them (now hybrid linear + neural)
        - Optimizer selects top‑K and tokenizes
        """
        # --- 1. find candidates ---
        candidates = self.finder.find_patterns(bitstring)

        # --- 2. score & rank using hybrid heuristic (neural + linear) ---
        ranked = self.heuristic.rank_candidates(candidates)
        selected = ranked[:max_patterns]

        # --- 3. tokenization / replacement ---
        token_base = 128
        mapping = {}
        transformed = bitstring
        for idx, (pat, occ, saved) in enumerate(selected):
            token = chr(token_base + idx)
            mapping[token] = pat
            transformed = transformed.replace(pat, token)

        original_len = len(bitstring)
        transformed_len = len(transformed)
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
