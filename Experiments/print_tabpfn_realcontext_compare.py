import json

d = json.load(open("result/rl_c2st_tabpfn_realcontext_HDGM_d2_power.json"))
c2 = json.load(open("result/c2st_HDGM_d2_power.json"))
semi = json.load(open("result/c2st_semi_HDGM_d2_power.json"))
dummy = json.load(open("result/rl_c2st_tabpfn_HDGM_d2_power.json"))

print("Fig 3b (d=2) — full comparison, all TabPFN variants")
print(f"{'N':>7} | {'C2ST-S':>7} {'C2ST-L':>7} | {'RL-S(AE)':>9} {'RL-L(AE)':>9} | {'RL-S(TabPFN-dummy)':>19} {'RL-L(TabPFN-dummy)':>19} | {'RL-S(TabPFN-real)':>18} {'RL-L(TabPFN-real)':>18}")
print("-" * 155)

for r_c2, r_semi, r_dummy, r_real in zip(c2["result"], semi["result"], dummy["result"], d["result"]):
    print(f"{r_c2['N']:>7} | "
          f"{r_c2['C2ST-S']:>7.3f} {r_c2['C2ST-L']:>7.3f} | "
          f"{r_semi['RL-C2ST-S']:>9.3f} {r_semi['RL-C2ST-L']:>9.3f} | "
          f"{r_dummy['RL-C2ST-S']:>19.3f} {r_dummy['RL-C2ST-L']:>19.3f} | "
          f"{r_real['RL-C2ST-S']:>18.3f} {r_real['RL-C2ST-L']:>18.3f}")

print("\nDelta at each N: RL-L(TabPFN-real) minus RL-L(AE)  [positive = TabPFN wins]")
for r_semi, r_real in zip(semi["result"], d["result"]):
    delta = r_real["RL-C2ST-L"] - r_semi["RL-C2ST-L"]
    print(f"  N={r_semi['N']:>6}: {delta:+.3f}")