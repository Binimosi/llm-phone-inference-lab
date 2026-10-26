"""E1 main run (pre-registered design, E1_OSF_PREREG.md). Do NOT run before the OSF registration is filed.
Arms (filler x wording) + no-rule:
  SS shared filler, same wording      SD shared filler, different wordings
  PS private filler, same wording     PD private filler, different wordings
  N0 private filler, no rule passage
8 supervisors (analyst instruction, seed 1000+s, own label order), temperature 1.0, no thinking.
Items: R1-R25 + C1, C2, N1, N2 x 20 banks (sample_banks.csv) = 105; 5 x 8 x 105 = 4,200 calls.
Call order: items in sequence; within an item, the 40 arm-supervisor cells in a seeded random order (so heat and
time drift do not line up with arms). Resumable: cells already in main_log.jsonl are skipped; every record is
fsynced, and a partial last line left by a power loss is cut on start (that cell is re-run). resume_e1.sh restarts
the server and this script after a crash or reboot.
Step 0 (masking check, temperature 0) runs once per bank and is logged to main_mask.jsonl."""
import csv, json, os, random, sys
import e1_lib as L
import run_pilot3 as P3

N_SUP, K, TOP, TEMP = 8, 30, 3, 1.0
BLOCK_SEED, ORDER_SEED = 20261023, 20261026
ARMS = ["SS", "SD", "PS", "PD", "N0"]
WORDINGS = P3.WORDINGS
LOG = os.path.join(L.HERE, "main_log.jsonl")
MASK = os.path.join(L.HERE, "main_mask.jsonl")
GENERIC = {"BANK", "NATIONAL", "ASSOCIATION", "THE", "OF", "AND", "TRUST", "COMPANY", "N.A.", "NA", "FSB", "&"}


def corpus(arm, sup, item_no, docs, gidx, shared, private):
    fill = shared if arm in ("SS", "SD") else private[sup]
    if arm == "N0":
        return list(fill), {}
    w = {}
    for fam, ws in WORDINGS.items():
        k0 = item_no % len(ws)
        w[fam] = ws[(k0 + (sup if arm in ("SD", "PD") else 0)) % len(ws)]
    return list(fill) + [gidx[(fam, src)] for fam, src in w.items()], w


def items_and_banks():
    cases = {r["cert"]: r for r in L.load_cases()}
    banks = [dict(cases[s["cert"]], label=s["label"]) for s in csv.DictReader(open(os.path.join(L.HERE, "sample_banks.csv")))]
    items = [(q, None) for q, d in L.QUESTIONS_V2.items() if not d["card"]] + \
            [(q, b) for b in banks for q, d in L.QUESTIONS_V2.items() if d["card"]]
    return items, banks


def mask_check(banks):
    L.repair_jsonl(MASK)
    done = {json.loads(x)["cert"] for x in open(MASK)} if os.path.exists(MASK) else set()
    with open(MASK, "a") as f:
        for b in banks:
            if b["cert"] in done:
                continue
            out, sec, _, _ = L.ask(L.card_v2(b["label"], L.card_values_v2(b)) + "\n\nBased only on these figures, which US bank "
                                   "is this? Give the bank's name, or answer 'unknown' if you cannot tell.", None, 1, 0.0,
                                   system="You are a careful analyst.", max_tokens=40)
            words = [w for w in b["name"].upper().replace(",", " ").split() if w not in GENERIC]
            hit = any(w in out.upper() for w in words)
            L.append_jsonl(f, dict(cert=b["cert"], label=b["label"], answer=out, identified=hit))
            print("mask", b["label"], "|", out, "| identified" if hit else "", flush=True)


def main():
    docs, gidx, _, _, family = P3.setup()
    filler_n = min(gidx.values())
    shared, private = L.make_blocks(docs[:filler_n], N_SUP, K, BLOCK_SEED)
    items, banks = items_and_banks()
    mask_check(banks)
    cut = L.repair_jsonl(LOG)
    if cut:
        print(f"cut a partial last line ({cut} bytes) from {LOG}; that cell is re-run", flush=True)
    done = {(r["arm"], r["sup"], r["item_id"]) for r in map(json.loads, open(LOG))} if os.path.exists(LOG) else set()
    cells = [(a, s) for a in ARMS for s in range(N_SUP)]
    total = len(items) * len(cells)
    with open(LOG, "a") as log:
        for n, (qid, b) in enumerate(items):
            item_id = f"{qid}@{b['cert']}" if b else qid
            order = list(cells)
            random.Random(f"{ORDER_SEED}:{item_id}").shuffle(order)
            q = L.QUESTIONS_V2[qid]
            v = L.card_values_v2(b) if b else None
            for arm, sup in order:
                if (arm, sup, item_id) in done:
                    continue
                cdocs, wording = corpus(arm, sup, n, docs, gidx, shared, private)
                hits = L.bm25_top(q["text"], cdocs, docs, TOP)
                labels = P3.order(q["labels"], sup, qid)
                user = L.prompt_v2(L.card_v2(b["label"], v) if b else "", [docs[i]["text"] for i in hits], q, labels)
                cap0 = L.clock_cap()
                out, sec, _, us = L.ask(user, labels, 1000 + sup, TEMP, system=L.ANALYSTS[sup] + L.TAIL)
                fam = family.get(qid)
                gold_id = f"gold:{fam}:{wording[fam]}" if fam and wording else None
                r = dict(arm=arm, sup=sup, item=n, item_id=item_id, q=qid, cls=q["cls"], bank=b["label"] if b else "",
                         cert=b["cert"] if b else "", answer=out, truth=q["truth"](v), miss=int(out != q["truth"](v)),
                         wording=wording.get(fam) if fam else None, gold_in_corpus=bool(gold_id),
                         gold_retrieved=bool(gold_id and gold_id in [docs[i]["id"] for i in hits]),
                         retrieved=[docs[i]["id"] for i in hits], labels=labels, seed=1000 + sup, temp=TEMP,
                         sec=round(sec, 2), prompt_tokens=us.get("prompt_tokens"), cap_before=cap0, cap_after=L.clock_cap())
                L.append_jsonl(log, r)
                done.add((arm, sup, item_id))
                print(f"{len(done)}/{total}", arm, sup, item_id, out, r["truth"], f"{sec:.1f}s", flush=True)


if __name__ == "__main__":
    if "--i-have-registered" not in sys.argv:
        sys.exit("Refusing to start: file the OSF pre-registration first, then run with --i-have-registered.")
    main()
