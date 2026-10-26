"""E1 shared code: corpus pool, supervisor corpora, BM25 retrieval, masked bank cards, questions,
rule-based ground truth, and the llama-server call. Design: E1_DESIGN_PREREG_v0.md."""
import csv, hashlib, json, math, os, random, re, subprocess, time, urllib.request
import xml.etree.ElementTree as ET
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.join(HERE, "corpus_raw")
SERVER = "http://127.0.0.1:8089/v1/chat/completions"

# Source files (all dated on or before the 2023 failures; see corpus_manifest.json for dates and hashes).
SOURCES = {
    "ecfr": ["ecfr_324H_PCA.xml", "ecfr_324_10_minimums.xml", "ecfr_324_22_adjustments.xml",
             "ecfr_252_5_categories.xml", "ecfr_329_LCR_scope.xml"],
    "fr_tailoring": ["fr_84FR59230.txt"],
    "fdic_rms_capital": ["fdic_rms_2-1_wb20221204.pdf"],
    "fdic_rms_liquidity": ["fdic_rms_6-1_wb20221210.pdf"],
    "occ_liquidity": ["occ_ch_liquidity_wb20221016.pdf"],
    "fed_cbem": ["fed_cbem_wb20230120.pdf"],
    "ffiec_rcr": ["ffiec_031_041_instr_202212.pdf"],
}
SNAPSHOT = {"ecfr": "eCFR point-in-time 2022-12-31", "fr_tailoring": "84 FR 59230, 1 Nov 2019",
            "fdic_rms_capital": "Wayback 2022-12-04", "fdic_rms_liquidity": "Wayback 2022-12-10",
            "occ_liquidity": "Wayback 2022-10-16", "fed_cbem": "Wayback 2023-01-20",
            "ffiec_rcr": "Dec 2022 edition (Wayback 2023-05-22)"}
# Large books: keep only the capital and liquidity parts (PDF page numbers, inclusive).
PAGES = {"fed_cbem_wb20230120.pdf": [(751, 770), (800, 853)],  # 3000 Assessment of Capital Adequacy; 3200.1 Liquidity Risk
         "ffiec_031_041_instr_202212.pdf": [(439, 623)]}       # Schedule RC-R, Regulatory Capital
TOPIC = re.compile(r"capital|leverage|well capitalized|adequately capitalized|prompt corrective|tier 1|"
                   r"liquidity|uninsured|deposit|unrealized|available-for-sale|held-to-maturity|"
                   r"category (ii|iii|iv)|\$(100|250|700) billion|risk-based", re.I)
CHUNK_WORDS = 300


def _text(path):
    if path.endswith(".xml"):
        return " ".join(ET.parse(path).getroot().itertext())
    if path.endswith(".pdf"):
        # Reading order (no -layout): the RMS and OCC PDFs are two-column.
        ranges = PAGES.get(os.path.basename(path), [(None, None)])
        return " ".join(subprocess.run(["pdftotext"] + (["-f", str(a), "-l", str(b)] if a else []) + [path, "-"],
                                       capture_output=True, text=True).stdout for a, b in ranges)
    t = open(path, encoding="utf-8", errors="replace").read()
    return re.sub(r"<[^>]+>", " ", t)


def build_pool():
    """Split every available source into ~300-word chunks and keep chunks with >= 3 topic hits."""
    pool, manifest = [], {}
    for src, files in SOURCES.items():
        for fn in files:
            p = os.path.join(RAW, fn)
            if not os.path.exists(p):
                continue
            words = _text(p).split()
            manifest[fn] = dict(source=src, snapshot=SNAPSHOT[src], words=len(words),
                                sha256=hashlib.sha256(open(p, "rb").read()).hexdigest())
            for k in range(0, len(words), CHUNK_WORDS):
                txt = " ".join(words[k:k + CHUNK_WORDS])
                if len(TOPIC.findall(txt)) >= 3:
                    pool.append(dict(id=f"{src}:{fn}:{k // CHUNK_WORDS}", source=src, text=txt))
    for fn, m in manifest.items():
        m["chunks_kept"] = sum(1 for c in pool if c["id"].split(":")[1] == fn)
    return pool, manifest


def make_blocks(pool, n_sup, k, seed):
    """Disjoint blocks: one shared block and n_sup private blocks of k chunks each."""
    idx = list(range(len(pool)))
    random.Random(seed).shuffle(idx)
    need = (n_sup + 1) * k
    if need > len(idx):
        raise ValueError(f"pool has {len(idx)} chunks, need {need}")
    shared = idx[:k]
    private = [idx[k * (i + 1): k * (i + 2)] for i in range(n_sup)]
    return shared, private


def corpus_for(omega, sup, shared, private, k):
    n_sh = round(omega * k)
    return shared[:n_sh] + private[sup][:k - n_sh]


_tok = lambda s: re.findall(r"[a-z0-9$.%]+", s.lower())


def bm25_top(query, docs, pool, top=3, k1=1.5, b=0.75):
    """Plain BM25 over the given chunk indices; ties broken by chunk index (deterministic)."""
    toks = {i: _tok(pool[i]["text"]) for i in docs}
    avg = sum(len(t) for t in toks.values()) / len(toks)
    df = Counter(w for t in toks.values() for w in set(t))
    n = len(toks)
    q = _tok(query)
    scores = []
    for i, t in toks.items():
        tf = Counter(t)
        s = sum(math.log(1 + (n - df[w] + 0.5) / (df[w] + 0.5)) * tf[w] * (k1 + 1) /
                (tf[w] + k1 * (1 - b + b * len(t) / avg)) for w in q if w in tf)
        scores.append((-s, i))
    return [i for _, i in sorted(scores)[:top]]


def load_cases():
    return list(csv.DictReader(open(os.path.join(HERE, "cases_20221231.csv"))))


def card_values(r):
    """Values exactly as shown on the card; ground truth is computed from these rounded values."""
    f = lambda k, d: round(float(r[k]), d)
    return dict(assets=f("assets_bn", 1), equity=f("equity_bn", 1), dep_dom=f("dep_dom_bn", 1),
                dep_unins=f("dep_unins_bn", 1), unreal=f("unreal_loss_bn", 1),
                leverage=f("leverage", 2), cet1=f("cet1", 2), tier1=f("tier1_rbc", 2), total=f("total_rbc", 2))


def card(label, v):
    return (f"{label}: call report data at 31 December 2022 (US$ billions unless stated)\n"
            f"- Total assets: {v['assets']}\n- Total equity capital: {v['equity']}\n"
            f"- Deposits in domestic offices: {v['dep_dom']}; of which estimated uninsured: {v['dep_unins']}\n"
            f"- Unrealized losses on securities (amortized cost minus fair value, held-to-maturity plus available-for-sale): {v['unreal']}\n"
            f"- Leverage ratio: {v['leverage']}%\n- Common equity tier 1 capital ratio: {v['cet1']}%\n"
            f"- Tier 1 risk-based capital ratio: {v['tier1']}%\n- Total risk-based capital ratio: {v['total']}%")


def _pca(v):
    if v["total"] >= 10 and v["tier1"] >= 8 and v["cet1"] >= 6.5 and v["leverage"] >= 5:
        return "well_capitalized"
    if v["total"] >= 8 and v["tier1"] >= 6 and v["cet1"] >= 4.5 and v["leverage"] >= 4:
        return "adequately_capitalized"
    if v["total"] >= 6 and v["tier1"] >= 4 and v["cet1"] >= 3 and v["leverage"] >= 3:
        return "undercapitalized"
    return "significantly_undercapitalized"  # critically: tangible equity not on the card; no such bank in sample


def _size_cat(v):
    a = v["assets"]
    return "category_II" if a >= 700 else "category_III" if a >= 250 else "category_IV" if a >= 100 else "none"


YN = ["yes", "no"]
QUESTIONS = {
    "C1": dict(cls="covered", labels=["well_capitalized", "adequately_capitalized", "undercapitalized",
                                       "significantly_undercapitalized", "critically_undercapitalized"],
               text="Under the prompt corrective action capital categories, which capital category is this bank in?",
               truth=_pca),
    "C2": dict(cls="covered", labels=["category_II", "category_III", "category_IV", "none"],
               text="Considering total assets only, which category of the 2019 tailoring framework for capital and "
                    "liquidity requirements would apply to a banking organization of this size?",
               truth=_size_cat),
    "C3": dict(cls="covered", labels=YN,
               text="Does the bank's leverage ratio exceed the minimum leverage ratio required to be well capitalized "
                    "by at least 2 percentage points?",
               truth=lambda v: "yes" if v["leverage"] - 5 >= 2 else "no"),
    "N1": dict(cls="novel", labels=YN,
               text="Do estimated uninsured deposits exceed 50% of deposits in domestic offices?",
               truth=lambda v: "yes" if v["dep_unins"] / v["dep_dom"] > 0.5 else "no"),
    "N2": dict(cls="novel", labels=YN,
               text="Would recognizing all unrealized losses on securities reduce total equity capital by more than 25%?",
               truth=lambda v: "yes" if v["unreal"] / v["equity"] > 0.25 else "no"),
}
SYSTEM = ("You are a bank supervisor. Use the bank data and the supervisory reference passages provided. "
          "Answer with exactly one of the allowed labels.")


def prompt(card_txt, passages, q):
    ref = "\n\n".join(f"[Passage {j + 1}] {p}" for j, p in enumerate(passages)) if passages else "(none)"
    return (f"Supervisory reference passages:\n{ref}\n\n{card_txt}\n\nQuestion: {q['text']}\n"
            f"Allowed labels: {', '.join(q['labels'])}")


def grammar(labels):
    return "root ::= " + " | ".join(f'"{l}"' for l in labels)


def ask(user, labels, seed, temperature, system=SYSTEM, max_tokens=16):
    body = dict(messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
                temperature=temperature, seed=seed, max_tokens=max_tokens, cache_prompt=False,
                chat_template_kwargs={"enable_thinking": False})
    if labels:
        body["grammar"] = grammar(labels)
    req = urllib.request.Request(SERVER, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    t0 = time.time()
    d = json.load(urllib.request.urlopen(req, timeout=600))
    return d["choices"][0]["message"]["content"].strip(), time.time() - t0, d.get("timings", {}), d.get("usage", {})


# ---- v2 (after pilot 1, 23 Oct 2026): computed ratios on the card, rule-extraction items, analyst variants ----

def card_values_v2(r):
    v = card_values(r)
    v["unins_pct"] = round(100 * v["dep_unins"] / v["dep_dom"], 1)
    v["unreal_pct"] = round(100 * v["unreal"] / v["equity"], 1)
    return v


def card_v2(label, v):
    return (card(label, v) + f"\n- Estimated uninsured deposits as % of domestic deposits: {v['unins_pct']}%"
            f"\n- Unrealized losses on securities as % of total equity capital: {v['unreal_pct']}%")


# R items: the answer must come from the rulebook (the retrieval channel); identical for every bank, asked without a card.
QUESTIONS_V2 = {
    "R1": dict(cls="rule", card=False, labels=["3%", "4%", "5%", "6%", "8%"], truth=lambda v: "5%",
               text="Under the prompt corrective action rules, what minimum leverage ratio must a bank have to be well capitalized?"),
    "R2": dict(cls="rule", card=False, labels=["6%", "8%", "10%", "12%"], truth=lambda v: "10%",
               text="Under the prompt corrective action rules, what minimum total risk-based capital ratio must a bank have to be well capitalized?"),
    "R3": dict(cls="rule", card=False, labels=["4.5%", "5%", "6.5%", "7%", "8%"], truth=lambda v: "6.5%",
               text="Under the prompt corrective action rules, what minimum common equity tier 1 capital ratio must a bank have to be well capitalized?"),
    "R4": dict(cls="rule", card=False, labels=["$50 billion", "$100 billion", "$250 billion", "$700 billion"],
               truth=lambda v: "$100 billion",
               text="Under the 2019 tailoring framework, what is the minimum total consolidated assets for a banking organization to fall in Category IV?"),
    "R5": dict(cls="rule", card=False, labels=["$50 billion", "$100 billion", "$250 billion", "$700 billion"],
               truth=lambda v: "$250 billion",
               text="Under the 2019 tailoring framework, what level of total consolidated assets places a banking organization in Category III regardless of other risk indicators?"),
}

# R6-R25 (24 Oct 2026, before registration): more rule items to raise power. Each answer is stated in all three gold
# wordings of its family (gold_passages.json). Scenario items carry their figures in the question; their ground truth
# is computed by the same rule functions as the card items.
PCA3 = ["well_capitalized", "adequately_capitalized", "undercapitalized"]
TAILOR4 = ["category_II", "category_III", "category_IV", "none"]
_PCA_Q = "Under the prompt corrective action rules, what minimum {} must a bank have to be {}?"
_PCA_S = ("Under the prompt corrective action rules, a bank that is not subject to any written agreement, order or capital "
          "directive, is not a subsidiary of a global systemically important bank holding company and has not elected the "
          "community bank leverage ratio framework has a total risk-based capital ratio of {}%, a tier 1 risk-based capital "
          "ratio of {}%, a common equity tier 1 capital ratio of {}% and a leverage ratio of {}%. Which capital category is it in?")
_TAIL_S = ("Under the 2019 tailoring framework, a U.S. bank holding company that is not a global systemically important bank "
           "has average total consolidated assets of ${} billion, cross-jurisdictional activity of ${} billion, weighted "
           "short-term wholesale funding of ${} billion, nonbank assets of ${} billion and off-balance-sheet exposure of "
           "${} billion. Which category of the framework applies?")
_TAIL_IND = ("Under the 2019 tailoring framework, a banking organization with at least $100 billion but less than $250 billion "
             "in total consolidated assets is placed in Category III if which of the following is $75 billion or more?")


def _tailor(a, cja, stwf, nb, obs):
    if a >= 700 or cja >= 75:
        return "category_II"
    if a >= 250 or (a >= 100 and max(stwf, nb, obs) >= 75):
        return "category_III"
    return "category_IV" if a >= 100 else "none"


def _thr(measure, cat, labels, truth):
    return dict(cls="rule", card=False, labels=labels, truth=lambda v, t=truth: t, text=_PCA_Q.format(measure, cat))


def _pca_item(total, tier1, cet1, lev):
    x = dict(total=total, tier1=tier1, cet1=cet1, leverage=lev)
    return dict(cls="rule", card=False, labels=PCA3, truth=lambda v, x=x: _pca(x), text=_PCA_S.format(total, tier1, cet1, lev))


def _tail_item(*x):
    return dict(cls="rule", card=False, labels=TAILOR4, truth=lambda v, x=x: _tailor(*x), text=_TAIL_S.format(*x))


QUESTIONS_V2.update({
    "R6": _thr("tier 1 risk-based capital ratio", "well capitalized", ["4%", "6%", "8%", "10%"], "8%"),
    "R7": _thr("total risk-based capital ratio", "adequately capitalized", ["6%", "8%", "10%", "12%"], "8%"),
    "R8": _thr("tier 1 risk-based capital ratio", "adequately capitalized", ["4%", "6%", "8%", "10%"], "6%"),
    "R9": _thr("common equity tier 1 capital ratio", "adequately capitalized", ["3%", "4.5%", "6.5%", "7%"], "4.5%"),
    "R10": _thr("leverage ratio", "adequately capitalized (for a bank not rated composite 1 under CAMELS)",
                ["3%", "4%", "5%", "6%"], "4%"),
    "R11": _pca_item(12.0, 10.0, 9.0, 7.0),   # well capitalized
    "R12": _pca_item(11.0, 9.0, 7.5, 4.6),    # adequately: leverage below 5
    "R13": _pca_item(10.5, 7.5, 7.0, 6.0),    # adequately: tier 1 below 8
    "R14": _pca_item(11.0, 9.0, 6.0, 6.0),    # adequately: CET1 below 6.5
    "R15": _pca_item(7.5, 6.5, 5.0, 4.5),     # undercapitalized: total below 8
    "R16": _pca_item(10.0, 8.0, 6.5, 5.0),    # well capitalized: every ratio exactly at its threshold
    "R17": dict(cls="rule", card=False, labels=["$25 billion", "$50 billion", "$75 billion", "$100 billion"],
                truth=lambda v: "$75 billion",
                text="Under the 2019 tailoring framework, a banking organization with at least $100 billion but less than "
                     "$250 billion in total consolidated assets is placed in Category III if its weighted short-term "
                     "wholesale funding is at least what amount?"),
    "R18": dict(cls="rule", card=False, labels=["nonbank assets", "uninsured deposits", "unrealized securities losses",
                                                "tier 1 capital"], truth=lambda v: "nonbank assets", text=_TAIL_IND),
    "R19": dict(cls="rule", card=False, labels=["off-balance-sheet exposure", "retail deposits", "loan loss allowance",
                                                "common equity tier 1 capital"],
                truth=lambda v: "off-balance-sheet exposure", text=_TAIL_IND),
    "R20": _tail_item(180, 10, 20, 15, 30),   # IV
    "R21": _tail_item(180, 10, 90, 15, 30),   # III: weighted short-term wholesale funding
    "R22": _tail_item(320, 10, 20, 20, 20),   # III: assets
    "R23": _tail_item(95, 5, 10, 10, 10),     # none
    "R24": _tail_item(140, 10, 20, 15, 80),   # III: off-balance-sheet exposure
    "R25": _tail_item(240, 10, 60, 60, 60),   # IV: every indicator below $75 billion
    "C1": dict(QUESTIONS["C1"], card=True),
    "C2": dict(QUESTIONS["C2"], card=True),
    "N1": dict(cls="novel", card=True, labels=YN, truth=lambda v: "yes" if v["unins_pct"] > 50 else "no",
               text="Do estimated uninsured deposits exceed 50% of deposits in domestic offices?"),
    "N2": dict(cls="novel", card=True, labels=YN, truth=lambda v: "yes" if v["unreal_pct"] > 25 else "no",
               text="Do unrealized losses on securities exceed 25% of total equity capital?"),
})


def repair_jsonl(path):
    """Cut a partial last line left by a power loss mid-write (that cell is simply re-run). Returns bytes removed."""
    if not os.path.exists(path):
        return 0
    with open(path, "rb+") as f:
        data = f.read()
        if not data or data.endswith(b"\n"):
            return 0
        keep = data.rfind(b"\n") + 1
        f.truncate(keep)
        return len(data) - keep


def append_jsonl(f, rec):
    """Write one record and force it to disk, so a sudden power loss loses at most the call in progress."""
    f.write(json.dumps(rec) + "\n"); f.flush(); os.fsync(f.fileno())

# Fixed analyst instructions (D2b): neutral wording variants, one per supervisor.
ANALYSTS = [
    "You are a bank supervisor.",
    "You are a bank examiner in a regional supervisory office.",
    "You are a senior analyst on a bank supervision team.",
    "You review capital and liquidity for a banking supervisor.",
    "You are a prudential supervisor responsible for a portfolio of banks.",
    "You are an off-site monitoring analyst at a banking agency.",
    "You are a supervisory staff member preparing a bank assessment.",
    "You are a risk specialist working for a bank regulator.",
]
TAIL = " Use the supervisory reference passages provided. Answer with exactly one of the allowed labels."


def prompt_v2(card_txt, passages, q, labels):
    ref = "\n\n".join(f"[Passage {j + 1}] {p}" for j, p in enumerate(passages)) if passages else "(none)"
    body = f"{card_txt}\n\n" if card_txt else ""
    return f"Supervisory reference passages:\n{ref}\n\n{body}Question: {q['text']}\nAllowed labels: {', '.join(labels)}"


def clock_cap():
    """Lowest ratio of current frequency cap to hardware maximum across cores 0, 4 and 7 (1.0 = unthrottled)."""
    out = []
    for c in (0, 4, 7):
        d = f"/sys/devices/system/cpu/cpu{c}/cpufreq/"
        try:
            out.append(int(open(d + "scaling_max_freq").read()) / int(open(d + "cpuinfo_max_freq").read()))
        except OSError:
            pass
    return round(min(out), 3) if out else None


def ask_think(user, seed, temperature, system, max_tokens=1024):
    """Thinking mode on, no grammar; returns the final text (falls back to the reasoning text if the final is empty)."""
    body = dict(messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
                temperature=temperature, seed=seed, max_tokens=max_tokens, cache_prompt=False,
                chat_template_kwargs={"enable_thinking": True})
    req = urllib.request.Request(SERVER, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    t0 = time.time()
    d = json.load(urllib.request.urlopen(req, timeout=1200))
    m = d["choices"][0]["message"]
    return (m.get("content") or m.get("reasoning_content") or "").strip(), time.time() - t0, d.get("timings", {}), d.get("usage", {})
