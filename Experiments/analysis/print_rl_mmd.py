import json
d = json.load(open("result/rl_mmd_d_HDGM_d2_power.json"))
ref_mmd = [0.073, 0.270, 0.628, 0.919, 0.955, 0.996]
print("RL-MMD-D vs plain MMD-D | d=2 | Fig 4a")
print("N        RL-MMD-D  MMD-D")
for n, r, ref in zip(d["N"], d["result"], ref_mmd):
    print(f"N={n:>6}: {r:.3f}     {ref:.3f}")
