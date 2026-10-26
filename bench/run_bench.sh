#!/usr/bin/env bash
# llama-bench pp512 / tg128, 6 threads, 3 repetitions; records the CPU clock cap before and after.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"; OUT="$HERE/results"; D=$(date +%F)
BIN=${LLAMA_BIN:-$HOME/llama.cpp/build/bin}; MODEL=${MODEL:-$HOME/models/gemma-4-E2B-it-Q4_0.gguf}
cap() { python3 -c "import sys; sys.path.insert(0, '$HERE/../experiment'); import e1_lib as L; print(L.clock_cap())"; }
mkdir -p "$OUT"
echo "cap_before $(cap)" > "$OUT/bench_$D.stderr"
"$BIN/llama-bench" -m "$MODEL" -t 6 -p 512 -n 128 -r 3 -o json > "$OUT/bench_$D.json" 2>> "$OUT/bench_$D.stderr"
echo "cap_after $(cap)" >> "$OUT/bench_$D.stderr"
lscpu > "$OUT/lscpu_$D.txt"
python3 - "$OUT/bench_$D.json" <<'PY'
import json, sys
for r in json.load(open(sys.argv[1])):
    t = f"pp{r['n_prompt']}" if r["n_prompt"] else f"tg{r['n_gen']}"
    print(f"{t:6s} {r['avg_ts']:7.2f} ± {r['stddev_ts']:.2f} tok/s  (threads {r['n_threads']}, commit {r['build_commit']})")
PY
