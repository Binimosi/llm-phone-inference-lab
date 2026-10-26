"""Power and size of the pre-registered tests at the registered size (20 banks), by simulation from the one-factor
model. 100 replications per scenario, bootstrap B = 200 inside each (the registered analysis uses 2,000).
Scenarios: null (all arms tau 1.0, r 0.4); planted effects as in sim_check.py; and 'pilot-like' (shared arms r 0.9,
private-different r 0.4) reflecting pilot 3. Output: power_check.txt."""
import io, contextlib, os, sys
import numpy as np
import analyze_main as A
import sim_check as S

HERE = os.path.dirname(os.path.abspath(__file__))
S.SCEN["pilot_like"] = {"SS": (1.0, 0.9), "SD": (1.0, 0.6), "PS": (1.0, 0.7), "PD": (1.0, 0.4), "N0": (0.2, 0.4)}
REPS = int(sys.argv[1]) if len(sys.argv) > 1 else 100
A.B = 200
out = []
for name in ("null", "effects", "pilot_like"):
    rej = {}
    for k in range(REPS):
        path = S.simulate(name, 1000 + k)
        with contextlib.redirect_stdout(io.StringIO()):
            _, _, _, ph = A.main(path, None)
        os.remove(path)
        for h, v in ph.items():
            rej[h] = rej.get(h, 0) + (v < 0.05)
    out.append(f"{name}: rejection rate at Holm-adjusted 5% over {REPS} replications")
    out += [f"  {h:46s} {rej[h] / REPS:.2f}" for h in rej]
    print("\n".join(out[-6:]), flush=True)
open(os.path.join(HERE, "power_check.txt"), "w").write("\n".join(out) + "\n")
