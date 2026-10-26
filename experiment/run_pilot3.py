"""E1 pilot 3 (after pilot 2, 23 Oct 2026): gold-passage allocation with fixed coverage.
Each corpus = 30 filler chunks (no chunk that states a PCA or tailoring threshold) + exactly one rule passage per
family (pca, tailoring) from gold_passages.json. Arms:
  shared  - both supervisors: same filler block, same rule wording
  private - own filler block; supervisor 1 gets a different source wording of the same rule than supervisor 0
  nogold  - shared filler, no rule passage (coverage baseline, H1)
Supervisor 0 holds the same wording in shared and private, so the arms differ only in what supervisor 1 shares.
The wording assigned to supervisor 0 rotates across items. Supervisors as in pilot 2: analyst instruction, seed,
label order; temperature 1.0; no thinking. Items: R1-R5 + C1, C2, N1, N2 x 2 banks = 13; 3 arms x 2 sups = 78 calls.
Outputs: pilot3_log.jsonl, pilot3_summary.txt."""
import json, os, random, re, statistics as st
from collections import defaultdict
import e1_lib as L

K, TOP, TEMP, BLOCK_SEED = 30, 3, 1.0, 20261023
SUPS, ARMS = [0, 1], ["shared", "private", "nogold"]
WORDINGS = {"pca": ["ecfr", "fdic_rms", "fed_cbem"], "tailoring": ["fr", "ecfr", "fdic_rms"]}
BANKS = {"SILICON VALLEY BANK": "Bank 49", "SYNCHRONY BANK": "Bank 77"}
RULE = re.compile(r"well[- ]?capitalized|adequately capitalized|prompt corrective|Category (I{1,3}|IV)\b|"
                  r"\$(100|250|700) billion|\b(10|8|6\.5|5|4\.5|4)(\.0)? ?(percent|%)", re.I)


def setup():
    pool, _ = L.build_pool()
    filler = [c for c in pool if not RULE.search(c["text"])]
    G = json.load(open(os.path.join(L.HERE, "gold_passages.json")))
    gold = [dict(id=f"gold:{fam}:{src}", source=f"gold_{src}", fam=fam, text=g["text"])
            for fam, d in G["gold"].items() for src, g in d.items()]
    docs = filler + gold  # gold chunks appended after filler; indices stable
    gidx = {(g["fam"], g["id"].split(":")[2]): len(filler) + i for i, g in enumerate(gold)}
    shared, private = L.make_blocks(filler, len(SUPS), K, BLOCK_SEED)
    return docs, gidx, shared, private, G["family"]


def corpus(arm, sup, item_no, docs, gidx, shared, private):
    fill = private[sup] if arm == "private" else shared
    if arm == "nogold":
        return list(fill), {}
    w = {}
    for fam, ws in WORDINGS.items():
        k0 = item_no % len(ws)
        w[fam] = ws[(k0 + (1 if (arm == "private" and sup == 1) else 0)) % len(ws)]
    return list(fill) + [gidx[(fam, src)] for fam, src in w.items()], w


def order(labels, sup, qid):
    lab = list(labels)
    random.Random(f"{sup}:{qid}").shuffle(lab)
    return lab


def main():
    docs, gidx, shared, private, family = setup()
    banks = [dict(r, label=BANKS[r["name"]]) for r in L.load_cases() if r["name"] in BANKS]
    items = [(q, None) for q, d in L.QUESTIONS_V2.items() if not d["card"]] + \
            [(q, b) for b in banks for q, d in L.QUESTIONS_V2.items() if d["card"]]
    log = open(os.path.join(L.HERE, "pilot3_log.jsonl"), "w")
    recs = []
    for arm in ARMS:
        for sup in SUPS:
            for n, (qid, b) in enumerate(items):
                q = L.QUESTIONS_V2[qid]
                v = L.card_values_v2(b) if b else None
                cdocs, wording = corpus(arm, sup, n, docs, gidx, shared, private)
                hits = L.bm25_top(q["text"], cdocs, docs, TOP)
                labels = order(q["labels"], sup, qid)
                user = L.prompt_v2(L.card_v2(b["label"], v) if b else "", [docs[i]["text"] for i in hits], q, labels)
                cap0 = L.clock_cap()
                out, sec, tm, us = L.ask(user, labels, 1000 + sup, TEMP, system=L.ANALYSTS[sup] + L.TAIL)
                fam = family.get(qid)
                gold_id = f"gold:{fam}:{wording[fam]}" if fam and wording else None
                r = dict(kind="call", arm=arm, sup=sup, item=n, bank=b["label"] if b else "", cert=b["cert"] if b else "",
                         q=qid, cls=q["cls"], answer=out, truth=q["truth"](v), labels=labels,
                         wording=wording.get(fam) if fam else None, gold_in_corpus=bool(gold_id),
                         gold_retrieved=bool(gold_id and gold_id in [docs[i]["id"] for i in hits]),
                         retrieved=[docs[i]["id"] for i in hits], seed=1000 + sup, temp=TEMP, sec=round(sec, 2),
                         prompt_tokens=us.get("prompt_tokens"), cap_before=cap0, cap_after=L.clock_cap())
                r["miss"] = int(r["answer"] != r["truth"])
                recs.append(r); log.write(json.dumps(r) + "\n"); log.flush()
                print(len(recs), arm, sup, r["bank"] or "-", qid, r["wording"], "gold-ret" if r["gold_retrieved"] else "",
                      out, r["truth"], f"{sec:.1f}s cap {cap0}", flush=True)
    log.close()
    summarize(recs)


def summarize(recs):
    L_ = [f"calls {len(recs)}; temperature {TEMP}; K {K} filler + 1 rule passage per family; top {TOP}",
          f"seconds per call: mean {st.mean(r['sec'] for r in recs):.1f}, median {st.median(r['sec'] for r in recs):.1f}; "
          f"clock cap at start: median {st.median(r['cap_before'] or 0 for r in recs):.2f}", "",
          "misses by arm x class:"]
    g = defaultdict(list)
    for r in recs:
        g[(r["arm"], r["cls"])].append(r["miss"])
    L_ += [f"  {k[0]:8s} {k[1]:7s} {sum(x)}/{len(x)}" for k, x in sorted(g.items())]
    ruled = [r for r in recs if r["gold_in_corpus"]]
    L_.append(f"\nrule passage retrieved (top {TOP}) when in corpus: {sum(r['gold_retrieved'] for r in ruled)}/{len(ruled)}")
    for ok in (True, False):
        s = [r for r in ruled if r["gold_retrieved"] == ok]
        if s:
            L_.append(f"  misses when {'retrieved' if ok else 'not retrieved'}: {sum(r['miss'] for r in s)}/{len(s)}")
    L_.append("\nmisses by wording (rule and covered items, shared+private arms):")
    for w in sorted({r["wording"] for r in ruled}):
        s = [r for r in ruled if r["wording"] == w]
        L_.append(f"  {w:9s} {sum(r['miss'] for r in s)}/{len(s)}")
    L_.append("\nmisses by question and arm:")
    for qid in L.QUESTIONS_V2:
        L_.append(f"  {qid} " + "  ".join(f"{a}: {sum(r['miss'] for r in recs if r['q'] == qid and r['arm'] == a)}/"
                                          f"{sum(1 for r in recs if r['q'] == qid and r['arm'] == a)}" for a in ARMS))
    L_.append("\nsupervisor pairs by arm (same answer; both miss; exactly one miss):")
    for a in ARMS:
        d = {(r["item"], r["sup"]): r for r in recs if r["arm"] == a}
        pr = [(d[(i, 0)], d[(i, 1)]) for (i, s) in d if s == 0]
        L_.append(f"  {a:8s} same {sum(x['answer'] == y['answer'] for x, y in pr)}/{len(pr)}; "
                  f"both miss {sum(x['miss'] and y['miss'] for x, y in pr)}; one miss {sum(x['miss'] != y['miss'] for x, y in pr)}")
    open(os.path.join(L.HERE, "pilot3_summary.txt"), "w").write("\n".join(L_) + "\n")
    print("\n".join(L_))


if __name__ == "__main__":
    main()
