import json, os

files = {
    "C2ST + C2ST-L"  : "result/c2st_HDGM_d2_type1.json",
    "MMD-D"          : "result/mmd_d_HDGM_d2_type1.json",
    "RL-C2ST"        : "result/c2st_semi_HDGM_d2_type1.json",
    "RL-MMD-D"       : "result/rl_mmd_d_HDGM_d2_type1.json",
}

N_vals = [1000, 2000, 4000, 6000, 8000, 10000]

for method, path in files.items():
    if not os.path.exists(path):
        print(f"{method}: FILE MISSING")
        continue
    d = json.load(open(path))
    print(f"\n{method} | Type-I error | Fig 4b")
    results = d["result"]
    if isinstance(results[0], dict):
        for r in results:
            n = r.get("N", "?")
            s = r.get("RL-C2ST-S", r.get("C2ST-S", "?"))
            l = r.get("RL-C2ST-L", r.get("C2ST-L", "?"))
            flag = " <--" if (float(s) > 0.06 or float(l) > 0.06) else ""
            print(f"  N={n:>6}: S={float(s):.3f}  L={float(l):.3f}{flag}")
    else:
        for n, r in zip(d["N"], results):
            flag = " <--" if float(r) > 0.06 else ""
            print(f"  N={n:>6}: {float(r):.3f}{flag}")
