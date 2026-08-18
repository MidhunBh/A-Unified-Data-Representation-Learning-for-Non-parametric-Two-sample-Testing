import pickle, json, numpy as np, time

with open("result/c2st_HDGM_baseline_d2_t1.pkl", "rb") as f:
    r = pickle.load(f)

n_list = [125, 250, 500, 750, 1000, 1250]

print("C2ST + C2ST-L Type-I | HDGM-S d=2")
for n, (s, l) in zip(n_list, r):
    print(f"N={8*n:>6}: C2ST-S={np.mean(s):.3f}  C2ST-L={np.mean(l):.3f}")

meta = {
    "method"  : "C2ST + C2ST-L",
    "dataset" : "HDGM-S",
    "level"   : "hard",
    "metric"  : "type-I error",
    "panel"   : "Fig 4b",
    "d"       : 2,
    "N"       : [8*n for n in n_list],
    "N_TRAIL" : 100,
    "note"    : "first run, before logging system. sampler was _t1.",
    "result"  : [
        {"N": 8*n, "C2ST-S": float(np.mean(s)), "C2ST-L": float(np.mean(l))}
        for n, (s, l) in zip(n_list, r)
    ],
    "ts"      : time.strftime("%Y-%m-%d %H:%M"),
}
with open("result/c2st_HDGM_d2_type1.json", "w") as f:
    json.dump(meta, f, indent=2)
print("saved to result/c2st_HDGM_d2_type1.json")
