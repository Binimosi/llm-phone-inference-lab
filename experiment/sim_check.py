"""Check analyze_main.py on simulated logs drawn from the one-factor model with known (tau, r) per arm.
Same shape as the main run: 5 arms x 8 supervisors x 105 items (25 rule items + 4 items x 20 banks).
Scenario 'effects' should recover the planted r ordering; scenario 'null' (all arms equal) should not reject."""
import json, os, sys
import numpy as np
import analyze_main as A

HERE = os.path.dirname(os.path.abspath(__file__))
ITEMS = [(f"R{k}", "rule") for k in range(1, 26)] + [(f"{q}@{b}", c) for b in range(20) for q, c in
                                                      (("C1", "covered"), ("C2", "covered"), ("N1", "novel"), ("N2", "novel"))]
SCEN = {  # arm: (tau, r) for rule/covered items; novel items use r + 0.15 (capped) and tau - 0.3 in gold arms
    "effects": {"SS": (1.0, 0.8), "SD": (1.0, 0.5), "PS": (1.0, 0.6), "PD": (1.0, 0.3), "N0": (0.2, 0.3)},
    "null": {a: (1.0, 0.4) for a in A.ARMS},
}


def simulate(name, seed):
    rng = np.random.default_rng(seed)
    path = os.path.join(HERE, f"_sim_{name}.jsonl")
    with open(path, "w") as f:
        for item_id, cls in ITEMS:
            for arm, (tau, r) in SCEN[name].items():
                if cls == "novel" and name == "effects" and arm != "N0":
                    tau, r = tau - 0.3, min(r + 0.15, 0.95)
                c = rng.standard_normal()
                for s in range(8):
                    e = np.sqrt(r) * c + np.sqrt(1 - r) * rng.standard_normal()
                    f.write(json.dumps(dict(arm=arm, sup=s, item_id=item_id, cls=cls, cert=item_id.split("@")[1] if "@" in item_id else "",
                                            miss=int(e < -tau))) + "\n")
    return path


if __name__ == "__main__":
    A.B = int(sys.argv[1]) if len(sys.argv) > 1 else 300
    for name in SCEN:
        print(f"\n===== scenario: {name} (B = {A.B})")
        A.main(simulate(name, 11), None)
        os.remove(os.path.join(HERE, f"_sim_{name}.jsonl"))
