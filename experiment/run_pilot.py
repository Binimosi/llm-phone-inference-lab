"""E1 pilot (Section 7 of E1_DESIGN_PREREG_v0.md): 2 arms (omega 0, 1) x 5 banks x 5 questions x 2 supervisors
= 100 calls, plus a masking check. Requires llama-server on 127.0.0.1:8089 (see README in this folder).
Outputs: corpus_manifest.json, pilot_log.jsonl, pilot_summary.txt."""
import json, os, random, statistics as st, sys
from collections import Counter, defaultdict
import e1_lib as L

N_SUP, K, TOP, TEMP = 8, 30, 3, 0.7
BLOCK_SEED, CASE_SEED = 20261023, 20261024
OMEGAS, PILOT_SUPS = [0.0, 1.0], [0, 1]

pool, manifest = L.build_pool()
json.dump(dict(pool_chunks=len(pool), files=manifest), open(os.path.join(L.HERE, "corpus_manifest.json"), "w"), indent=1)
print("pool chunks:", len(pool), {s: sum(c["source"] == s for c in pool) for s in L.SOURCES})
shared, private = L.make_blocks(pool, N_SUP, K, BLOCK_SEED)

# Pilot banks: two 2023 failures (masking check), one >= $100bn survivor, one low-margin bank, one random.
cases = L.load_cases()
rng = random.Random(CASE_SEED)
fail = [r for r in cases if r["failed"] and r["name"] in ("SILICON VALLEY BANK", "SIGNATURE BANK")]
rest = [r for r in cases if not r["failed"]]
big = rng.choice([r for r in rest if float(r["assets_bn"]) >= 100])
thin = rng.choice([r for r in rest if float(r["leverage"]) - 5 < 2])
other = rng.choice([r for r in rest if r not in (big, thin)])
banks = fail + [big, thin, other]
labels = rng.sample(range(10, 100), len(banks))
banks = [dict(r, label=f"Bank {n}") for r, n in zip(banks, labels)]

log = open(os.path.join(L.HERE, "pilot_log.jsonl"), "w")
mask = []
for b in banks:
    v = L.card_values(b)
    out, sec, tm, _ = L.ask(L.card(b["label"], v) + "\n\nBased only on these figures, which US bank is this? "
                            "Give the bank's name, or answer 'unknown' if you cannot tell.", None, 1, 0.0,
                            system="You are a careful analyst.", max_tokens=40)
    mask.append((b["label"], b["name"], out))
    log.write(json.dumps(dict(kind="mask", bank=b["label"], cert=b["cert"], answer=out, sec=round(sec, 2))) + "\n")
    print("mask", b["label"], "|", out, flush=True)

n = 0
for om in OMEGAS:
    for s in PILOT_SUPS:
        docs = L.corpus_for(om, s, shared, private, K)
        for b in banks:
            v = L.card_values(b)
            for qid, q in L.QUESTIONS.items():
                hits = L.bm25_top(q["text"], docs, pool, TOP)
                out, sec, tm, us = L.ask(L.prompt(L.card(b["label"], v), [pool[i]["text"] for i in hits], q),
                                         q["labels"], 1000 + s, TEMP)
                truth = q["truth"](v)
                rec = dict(kind="call", omega=om, sup=s, bank=b["label"], cert=b["cert"], q=qid, cls=q["cls"],
                           answer=out, truth=truth, miss=int(out != truth), retrieved=[pool[i]["id"] for i in hits],
                           seed=1000 + s, temp=TEMP, sec=round(sec, 2), prompt_tokens=us.get("prompt_tokens"),
                           prompt_ms=tm.get("prompt_ms"), predicted_ms=tm.get("predicted_ms"))
                log.write(json.dumps(rec) + "\n"); log.flush()
                n += 1
                print(n, om, s, b["label"], qid, out, truth, f"{sec:.1f}s", flush=True)
log.close()

# Summary
recs = [json.loads(x) for x in open(os.path.join(L.HERE, "pilot_log.jsonl")) if '"call"' in x]
secs = [r["sec"] for r in recs]
toks = [r["prompt_tokens"] for r in recs if r["prompt_tokens"]]
lines = [f"calls {len(recs)}; seconds per call mean {st.mean(secs):.1f}, median {st.median(secs):.1f}, max {max(secs):.1f}",
         f"prompt tokens mean {st.mean(toks):.0f}, max {max(toks)}",
         f"pool chunks {len(pool)}; K {K}; top {TOP}; temperature {TEMP}", "",
         "miss rate by omega x class:"]
g = defaultdict(list)
for r in recs:
    g[(r["omega"], r["cls"])].append(r["miss"])
for k in sorted(g):
    lines.append(f"  omega {k[0]:.1f} {k[1]:8s} {sum(g[k])}/{len(g[k])}")
lines.append("\nmiss rate by question:")
for qid in L.QUESTIONS:
    m = [r["miss"] for r in recs if r["q"] == qid]
    lines.append(f"  {qid} {sum(m)}/{len(m)}  answers {dict(Counter(r['answer'] for r in recs if r['q'] == qid))}")
lines.append("\nsupervisor agreement (same answer, sup 0 vs 1) by omega:")
for om in OMEGAS:
    a = {(r["bank"], r["q"], r["sup"]): r["answer"] for r in recs if r["omega"] == om}
    pairs = [(a[(b, q, 0)], a[(b, q, 1)]) for (b, q, s) in a if s == 0]
    lines.append(f"  omega {om:.1f}: {sum(x == y for x, y in pairs)}/{len(pairs)}")
lines.append("\nmasking check (label | true name | model answer):")
lines += [f"  {a} | {b} | {c}" for a, b, c in mask]
open(os.path.join(L.HERE, "pilot_summary.txt"), "w").write("\n".join(lines) + "\n")
print("\n".join(lines))
