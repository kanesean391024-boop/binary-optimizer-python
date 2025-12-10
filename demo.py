"""demo.py


Small demo showing how to use the optimizer on synthetic data.
"""
from optimizer import Optimizer, SimpleAIHeuristic




def make_test_sequence():
# Build a synthetic binary string with repeated motifs
motifs = ["1011001", "000111", "1011001", "111000111", "1011001"]
filler = "0101010101010"
s = ("".join(motifs) + filler) * 3
# add more noise
s += "011001100110011001"
return s




def main():
s = make_test_sequence()
print("Original length:", len(s))


opt = Optimizer()
res = opt.optimize(s, max_patterns=3)


print("Transformed length:", res["transformed_len"])
print("Mapping overhead:", res["mapping_overhead"])
print("Net saved bits (heuristic):", res["net_saved"])
print("Mapping:")
for t, p in res["mapping"].items():
print(f" token {ord(t)} -> {p}")


print("Transformed sample (first 200 chars):\n", res["transformed"][0:200])


if __name__ == '__main__':
main()
```
