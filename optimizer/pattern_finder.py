"""pattern_finder.py


Utilities to find repeating binary patterns inside byte/bit sequences.
This is intentionally simple: it scans for repeated substrings and records
frequency and total-savings heuristics.
"""


from collections import defaultdict
from typing import List, Tuple


class PatternFinder:
"""Find repeating binary patterns in a bitstring.


Methods
-------
find_patterns(bitstring, min_len=2, max_len=16)
Returns a list of (pattern, occurrences, total_saved_bits) tuples.
"""


def __init__(self):
pass


def find_patterns(self, bitstring: str, min_len: int = 2, max_len: int = 16) -> List[Tuple[str,int,int]]:
"""Naive search for repeating substrings.


Parameters
----------
bitstring : str
String containing '0' and '1' characters.
min_len, max_len : int
Minimum and maximum pattern length to consider.


Returns
-------
List of (pattern, occurrences, total_saved_bits), sorted by total_saved_bits desc
"""
counts = defaultdict(int)
n = len(bitstring)
for L in range(min_len, min(max_len, n) + 1):
for i in range(0, n - L + 1):
pat = bitstring[i:i+L]
counts[pat] += 1


results = []
for pat, occ in counts.items():
if occ > 1:
# If we replaced occurrences with a shorter token, estimate savings.
# Suppose token length = ceil(log2(unique_tokens)) — but naive: use 8 bits for token.
token_overhead = 8
saved = (len(pat) - 1) * occ - token_overhead
# saved is a loose heuristic (can be negative)
results.append((pat, occ, saved))


# filter only positive savings and sort
filtered = [r for r in results if r[2] > 0]
filtered.sort(key=lambda x: x[2], reverse=True)
return filtered
```
