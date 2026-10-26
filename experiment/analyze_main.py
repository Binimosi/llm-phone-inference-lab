"""E1 pre-registered analysis (E1_OSF_PREREG.md, Section "Analysis plan"). Usage: python3 analyze_main.py [log] [mask]
Per arm and item class: miss rate, tau = -Phi^-1(miss rate), r by tetrachoric inversion of the pooled pairwise
co-miss rate (28 supervisor pairs), the empirical systemic-miss rate (share of items where more than half of the
8 supervisors miss) and the model-implied P_sys (q = 0.5). Inference: cluster bootstrap over banks (each of the 25
rule items is its own cluster: they share no bank), 2,000 draws, seed 7; one-sided percentile p-values; Holm over H1-H4 (five tests)."""
import json, os, sys
from itertools import combinations
import numpy as np
from scipy.optimize import brentq
from scipy.stats import multivariate_normal, norm

HERE = os.path.dirname(os.path.abspath(__file__))
ARMS = ["SS", "SD", "PS", "PD", "N0"]
B, SEED, N_SUP = 2000, 7, 8


def load(log, mask):
    R = [json.loads(x) for x in open(log)]
    drop = set()
    if mask and os.path.exists(mask):
        drop = {m["cert"] for m in map(json.loads, open(mask)) if m["identified"]}
    return [r for r in R if r["cert"] not in drop], drop


def matrix(R, arm, classes):
    """items x supervisors miss matrix for one arm, plus the cluster id of each item."""
    cell = {(r["item_id"], r["sup"]): r["miss"] for r in R if r["arm"] == arm and r["cls"] in classes}
    ids = sorted({i for i, _ in cell})
    ids = [i for i in ids if all((i, s) in cell for s in range(N_SUP))]  # complete items only
    M = np.array([[cell[(i, s)] for s in range(N_SUP)] for i in ids], dtype=float).reshape(len(ids), N_SUP)
    clus = np.array([i.split("@")[1] if "@" in i else i for i in ids])
    return M, clus


def tetra_r(p, p11):
    """Solve Phi2(-tau, -tau; r) = p11 with tau = -Phi^-1(p). Boundary values when the co-miss rate is outside the range."""
    if p <= 0 or p >= 1:
        return np.nan
    t = norm.ppf(p)
    f = lambda r: multivariate_normal(mean=[0, 0], cov=[[1, r], [r, 1]]).cdf([t, t]) - p11
    lo, hi = -0.99, 0.999
    if f(hi) <= 0:
        return 1.0
    if f(lo) >= 0:
        return lo
    return brentq(f, lo, hi, xtol=1e-5)


def stats(M):
    if M.size == 0:
        return dict(n=0, miss=np.nan, tau=np.nan, r=np.nan, sys_emp=np.nan, sys_model=np.nan)
    p = M.mean()
    p11 = np.mean([np.mean(M[:, a] * M[:, b]) for a, b in combinations(range(M.shape[1]), 2)])
    r = tetra_r(p, p11)
    tau = -norm.ppf(p) if 0 < p < 1 else np.nan
    sys_emp = np.mean(M.sum(1) > M.shape[1] / 2)
    sys_model = norm.cdf((-tau - np.sqrt(1 - r) * norm.ppf(0.5)) / np.sqrt(r)) if np.isfinite(tau) and 0 < r < 1 else \
        (norm.cdf(-tau) if r == 1.0 and np.isfinite(tau) else np.nan)
    return dict(n=M.shape[0], miss=p, tau=tau, r=r, sys_emp=sys_emp, sys_model=sys_model)


def contrasts(S):
    """The five pre-registered contrasts (positive = hypothesis direction). S[(arm, classset)] -> stats."""
    RC, NV = "rule+covered", "novel"
    return {
        "H1 coverage: miss(N0) - miss(PD)": S[("N0", RC)]["miss"] - S[("PD", RC)]["miss"],
        "H2a wording: r(same) - r(different)": (S[("SS", RC)]["r"] + S[("PS", RC)]["r"] - S[("SD", RC)]["r"] - S[("PD", RC)]["r"]) / 2,
        "H2b filler: r(shared) - r(private)": (S[("SS", RC)]["r"] + S[("SD", RC)]["r"] - S[("PS", RC)]["r"] - S[("PD", RC)]["r"]) / 2,
        "H3 systemic: sys_emp(SS) - sys_emp(PD)": S[("SS", RC)]["sys_emp"] - S[("PD", RC)]["sys_emp"],
        "H4 novel: r(novel) - r(covered), gold arms": np.nanmean([S[(a, NV)]["r"] - S[(a, "covered")]["r"] for a in ARMS[:4]]),
    }


CLASSES = {"rule+covered": ("rule", "covered"), "covered": ("covered",), "novel": ("novel",), "rule": ("rule",)}


def all_stats(mats, pick=None):
    S = {}
    for (arm, cs), (M, clus) in mats.items():
        if pick is not None:
            M = np.vstack([M[clus == c] for c in pick]) if len(pick) else M[:0]
        S[(arm, cs)] = stats(M)
    return S


def holm(p):
    order = sorted(p, key=p.get)
    out, m, run = {}, len(p), 0.0
    for k, h in enumerate(order):
        run = max(run, min(1.0, (m - k) * p[h]))
        out[h] = run
    return out


def main(log=os.path.join(HERE, "main_log.jsonl"), mask=os.path.join(HERE, "main_mask.jsonl"), out=None):
    R, drop = load(log, mask)
    mats = {(a, cs): matrix(R, a, cl) for a in ARMS for cs, cl in CLASSES.items()}
    S = all_stats(mats)
    est = contrasts(S)
    clusters = sorted({c for (_, cl) in mats.values() for c in cl})
    rng = np.random.default_rng(SEED)
    boot = {k: [] for k in est}
    for _ in range(B):
        pick = list(rng.choice(clusters, size=len(clusters), replace=True))
        for k, v in contrasts(all_stats(mats, pick)).items():
            boot[k].append(v)
    p = {k: float(np.mean(np.array(v)[np.isfinite(v)] <= 0)) for k, v in boot.items()}
    ph = holm(p)
    lines = [f"records {len(R)}; banks dropped by masking check: {sorted(drop) or 'none'}; bootstrap {B} over {len(clusters)} clusters", "",
             "arm  class         n   miss   tau     r    sys_emp sys_model"]
    for a in ARMS:
        for cs in CLASSES:
            s = S[(a, cs)]
            lines.append(f"{a:4s} {cs:12s} {s['n']:3d}  {s['miss']:.3f} {s['tau']:6.3f} {s['r']:6.3f}  {s['sys_emp']:.3f}   {s['sys_model']:.3f}")
    lines += ["", "contrast                                       estimate  95% CI              p(1-sided)  p(Holm)"]
    for k, v in est.items():
        b = np.array(boot[k]); b = b[np.isfinite(b)]
        lo, hi = np.percentile(b, [2.5, 97.5]) if b.size else (np.nan, np.nan)
        lines.append(f"{k:46s} {v:8.3f}  [{lo:7.3f}, {hi:7.3f}]   {p[k]:.4f}      {ph[k]:.4f}")
    txt = "\n".join(lines)
    if out:
        open(out, "w").write(txt + "\n")
    print(txt)
    return S, est, p, ph


if __name__ == "__main__":
    main(*(sys.argv[1:3]))
