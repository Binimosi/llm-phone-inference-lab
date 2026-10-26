"""E1 main sample: 20 banks from cases_20221231.csv, fixed before registration.
The 3 banks that failed in 2023 are always included; 17 more are drawn by stratum (assets >= $100bn or not x
uninsured share > 0.5 or not), proportional to stratum size with at least 2 per stratum, seed 20261025.
Masked labels "Bank NN" are drawn without replacement from 10-99 with the same seed.
Output: sample_banks.csv (cert, label, stratum; names stay only in the cases file)."""
import csv, os, random
import e1_lib as L

SEED, N = 20261025, 20
cases = L.load_cases()
fails = [r for r in cases if r["failed"]]
rest = [r for r in cases if not r["failed"]]
stratum = lambda r: ("big" if float(r["assets_bn"]) >= 100 else "small") + ("_runprone" if float(r["uninsured_share"]) > 0.5 else "_stable")
groups = {}
for r in rest:
    groups.setdefault(stratum(r), []).append(r)
rng = random.Random(SEED)
need = N - len(fails)
alloc = {s: 2 for s in groups}
for _ in range(need - sum(alloc.values())):  # largest remaining share gets the next slot
    s = max(groups, key=lambda s: len(groups[s]) / len(rest) * need - alloc[s])
    alloc[s] += 1
picked = list(fails)
for s in sorted(groups):
    g = sorted(groups[s], key=lambda r: int(r["cert"]))
    picked += rng.sample(g, min(alloc[s], len(g)))
labels = rng.sample(range(10, 100), len(picked))
rows = [dict(cert=r["cert"], label=f"Bank {n}", stratum=stratum(r), failed_2023=bool(r["failed"])) for r, n in zip(picked, labels)]
with open(os.path.join(L.HERE, "sample_banks.csv"), "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
print("strata in population:", {s: len(g) for s, g in sorted(groups.items())}, "alloc:", alloc)
for r in rows:
    print(r)
