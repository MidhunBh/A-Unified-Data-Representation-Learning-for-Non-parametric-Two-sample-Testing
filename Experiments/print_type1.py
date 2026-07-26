import json
d = json.load(open("result/mmd_d_HDGM_d2_type1.json"))
print("MMD-D Type-I error | HDGM-S hard d=2 | Fig 4b")
print("N        Type-I  (nominal alpha=0.05)")
for n, r in zip(d["N"], d["result"]):
    flag = " <-- ABOVE NOMINAL" if r > 0.05 else ""
    print(f"N={n:>6}: {r:.3f}{flag}")
