"""E1 pilot 2 (after pilot 1, 23 Oct 2026). Changes: computed ratios on the card; rule-extraction items R1-R5;
C3 dropped; supervisor = fixed analyst instruction + own seed + own label order; temperature 1.0; clock cap logged.
Part A: 2 arms (omega 0, 1) x 2 supervisors x 13 items = 52 calls (grammar-constrained, no thinking).
Part B: 10 calls with the model's thinking mode on (omega 1, supervisor 0), answer parsed from the final text.
Outputs: pilot2_log.jsonl, pilot2_summary.txt."""
import json, os, random, re, statistics as st
from collections import defaultdict
import e1_lib as L

N_SUP, K, TOP, TEMP = 8, 30, 3, 1.0
BLOCK_SEED = 20261023
OMEGAS, SUPS = [0.0, 1.0], [0, 1]
BANKS = {"SILICON VALLEY BANK": "Bank 49", "SYNCHRONY BANK": "Bank 77"}  # same masked labels as pilot 1

pool, _ = L.build_pool()
shared, private = L.make_blocks(pool, N_SUP, K, BLOCK_SEED)
banks = [dict(r, label=BANKS[r["name"]]) for r in L.load_cases() if r["name"] in BANKS]
items = [(q, None) for q, d in L.QUESTIONS_V2.items() if not d["card"]] + \
        [(q, b) for b in banks for q, d in L.QUESTIONS_V2.items() if d["card"]]


def order(labels, sup, qid):
    lab = list(labels)
    random.Random(f"{sup}:{qid}").shuffle(lab)
    return lab


def run(om, sup, qid, b, think=False):
    q = L.QUESTIONS_V2[qid]
    v = L.card_values_v2(b) if b else None
    docs = L.corpus_for(om, sup, shared, private, K)
    hits = L.bm25_top(q["text"], docs, pool, TOP)
    labels = order(q["labels"], sup, qid)
    user = L.prompt_v2(L.card_v2(b["label"], v) if b else "", [pool[i]["text"] for i in hits], q, labels)
    cap0 = L.clock_cap()
    if think:
        user += "\nThink it through, then end with a final line of the form 'ANSWER: <label>'."
        raw, sec, tm, us = L.ask_think(user, 1000 + sup, TEMP, L.ANALYSTS[sup] + L.TAIL)
        m = re.findall(r"ANSWER:\s*(.+)", raw)
        last = m[-1].strip().strip("*.'\" ") if m else ""
        out = next((l for l in sorted(labels, key=len, reverse=True) if l.lower() == last.lower()), "UNPARSED")
    else:
        out, sec, tm, us = L.ask(user, labels, 1000 + sup, TEMP, system=L.ANALYSTS[sup] + L.TAIL)
        raw = out
    truth = q["truth"](v)
    return dict(kind="think" if think else "call", omega=om, sup=sup, bank=b["label"] if b else "", cert=b["cert"] if b else "",
                q=qid, cls=q["cls"], answer=out, truth=truth, miss=int(out != truth), labels=labels,
                retrieved=[pool[i]["id"] for i in hits], seed=1000 + sup, temp=TEMP, sec=round(sec, 2),
                prompt_tokens=us.get("prompt_tokens"), completion_tokens=us.get("completion_tokens"),
                cap_before=cap0, cap_after=L.clock_cap(), raw=raw[-400:] if think else None)


log = open(os.path.join(L.HERE, "pilot2_log.jsonl"), "w")
recs, n = [], 0
for om in OMEGAS:
    for sup in SUPS:
        for qid, b in items:
            r = run(om, sup, qid, b); recs.append(r); log.write(json.dumps(r) + "\n"); log.flush(); n += 1
            print(n, om, sup, r["bank"], qid, r["answer"], r["truth"], f"{r['sec']:.1f}s cap {r['cap_before']}", flush=True)
think_items = [("R1", None), ("R2", None), ("R3", None), ("R4", None), ("R5", None)] + \
              [(q, b) for b in banks for q in ("C1", "C2")] + [("N2", banks[0])]
thinks = []
for qid, b in think_items[:10]:
    r = run(1.0, 0, qid, b, think=True); thinks.append(r); log.write(json.dumps(r) + "\n"); log.flush()
    print("think", r["bank"], qid, r["answer"], r["truth"], f"{r['sec']:.1f}s tokens {r['completion_tokens']}", flush=True)
log.close()

ok = [r for r in recs if (r["cap_before"] or 0) >= 0.9 and (r["cap_after"] or 0) >= 0.9]
lines = [f"part A calls {len(recs)}; temperature {TEMP}; label order shuffled per supervisor",
         f"seconds per call, all: mean {st.mean(r['sec'] for r in recs):.1f}, median {st.median(r['sec'] for r in recs):.1f}",
         f"seconds per call, clock cap >= 0.9 at both ends ({len(ok)} calls): "
         + (f"mean {st.mean(r['sec'] for r in ok):.1f}, median {st.median(r['sec'] for r in ok):.1f}" if ok else "none"),
         f"prompt tokens mean {st.mean(r['prompt_tokens'] for r in recs):.0f}", "", "misses by omega x class:"]
g = defaultdict(list)
for r in recs:
    g[(r["omega"], r["cls"])].append(r["miss"])
lines += [f"  omega {k[0]:.1f} {k[1]:6s} {sum(x)}/{len(x)}" for k, x in sorted(g.items())]
lines.append("\nmisses by question (answers):")
for qid in L.QUESTIONS_V2:
    rs = [r for r in recs if r["q"] == qid]
    lines.append(f"  {qid} {sum(r['miss'] for r in rs)}/{len(rs)}  " + ", ".join(f"{r['bank'] or '-'}:{r['answer']}" for r in rs))
lines.append("\nagreement sup 0 vs 1 (answers; both miss; one miss) by omega:")
for om in OMEGAS:
    a = {(r["bank"], r["q"], r["sup"]): r for r in recs if r["omega"] == om}
    pr = [(a[(bk, q, 0)], a[(bk, q, 1)]) for (bk, q, s) in a if s == 0]
    lines.append(f"  omega {om:.1f}: same answer {sum(x['answer'] == y['answer'] for x, y in pr)}/{len(pr)}; "
                 f"both miss {sum(x['miss'] and y['miss'] for x, y in pr)}; one miss {sum(x['miss'] != y['miss'] for x, y in pr)}")
lines.append(f"\npart B, thinking on (omega 1, sup 0): misses {sum(r['miss'] for r in thinks)}/{len(thinks)}; "
             f"unparsed {sum(r['answer'] == 'UNPARSED' for r in thinks)}; seconds mean {st.mean(r['sec'] for r in thinks):.1f}; "
             f"completion tokens mean {st.mean(r['completion_tokens'] or 0 for r in thinks):.0f}")
lines += [f"  {r['q']} {r['bank'] or '-'}: {r['answer']} (truth {r['truth']}), {r['sec']:.0f}s" for r in thinks]
nt = {(r["q"], r["bank"]): r for r in recs if r["omega"] == 1.0 and r["sup"] == 0}
lines.append("  same items without thinking (omega 1, sup 0): misses "
             f"{sum(nt[(r['q'], r['bank'])]['miss'] for r in thinks)}/{len(thinks)}")
open(os.path.join(L.HERE, "pilot2_summary.txt"), "w").write("\n".join(lines) + "\n")
print("\n".join(lines))
