import json

d = json.load(open("result/rl_c2st_tabpfn_HDGM_d2_power.json"))
c2 = json.load(open("result/c2st_HDGM_d2_power.json"))
semi = json.load(open("result/c2st_semi_HDGM_d2_power.json"))

print("Fig 3b (d=2) — full comparison")
print(f"{'N':>7} | {'C2ST-S':>7} {'C2ST-L':>7} | {'RL-S(AE)':>9} {'RL-L(AE)':>9} | {'RL-S(TabPFN)':>13} {'RL-L(TabPFN)':>13}")
print("-" * 78)

for r_c2, r_semi, r_tp in zip(c2["result"], semi["result"], d["result"]):
    print(f"{r_c2['N']:>7} | "
          f"{r_c2['C2ST-S']:>7.3f} {r_c2['C2ST-L']:>7.3f} | "
          f"{r_semi['RL-C2ST-S']:>9.3f} {r_semi['RL-C2ST-L']:>9.3f} | "
          f"{r_tp['RL-C2ST-S']:>13.3f} {r_tp['RL-C2ST-L']:>13.3f}")