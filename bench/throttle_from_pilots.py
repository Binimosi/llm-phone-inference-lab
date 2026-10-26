"""Sustained-run latency vs CPU clock cap, from the three E1 pilot logs (23 Oct 2026; 240 llama-server calls).
Each record holds the wall time of one grammar-constrained call (`sec`) and the lowest ratio of current to maximum
CPU frequency across cores 0, 4 and 7, read from sysfs just before the call (`cap_before`; 1.0 = unthrottled).
Usage: python3 bench/throttle_from_pilots.py  ->  bench/results/throttle_pilots.txt"""
import json, os, statistics as st

HERE = os.path.dirname(os.path.abspath(__file__))
EXP = os.path.join(HERE, "..", "experiment")
LOGS = ["pilot_log.jsonl", "pilot2_log.jsonl", "pilot3_log.jsonl"]
BINS = [(0.0, 0.35), (0.35, 0.5), (0.5, 0.75), (0.75, 1.01)]

rows = []
for fn in LOGS:
    for line in open(os.path.join(EXP, fn)):
        r = json.loads(line)
        if r.get("sec") is not None and r.get("cap_before") is not None:
            rows.append((fn, r["cap_before"], r["sec"], r.get("prompt_tokens")))

out = [f"calls with timing and clock cap: {len(rows)} (of the 3 pilot logs)", "",
       "per pilot: n, median s/call, median cap"]
for fn in LOGS:
    s = [x for x in rows if x[0] == fn]
    if s:
        out.append(f"  {fn:18s} {len(s):4d}  {st.median(x[2] for x in s):6.1f}  {st.median(x[1] for x in s):.2f}")
out += ["", "by clock cap at call start: n, median s/call, median prompt tokens"]
for lo, hi in BINS:
    s = [x for x in rows if lo <= x[1] < hi]
    if s:
        pt = [x[3] for x in s if x[3]]
        out.append(f"  cap [{lo:.2f}, {min(hi, 1.0):.2f}]  {len(s):4d}  {st.median(x[2] for x in s):6.1f}  "
                   f"{st.median(pt) if pt else float('nan'):.0f}")
txt = "\n".join(out)
os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
open(os.path.join(HERE, "results", "throttle_pilots.txt"), "w").write(txt + "\n")
print(txt)
