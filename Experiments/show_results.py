import json

d2 = json.load(open('result/c2st_HDGM_d2_power.json'))
d10 = json.load(open('result/c2st_HDGM_d10_power.json'))
semi_d2 = json.load(open('result/c2st_semi_HDGM_d2_power.json'))
semi_d10 = json.load(open('result/c2st_semi_HDGM_d10_power.json'))
mmd = json.load(open('result/mmd-d_HDGM_baseline_0.00005_d2.json'))

print('=== Fig 3b (d=2) ===')
print('N        C2ST-S  C2ST-L  RL-S    RL-L')
for r2, rs2 in zip(d2['result'], semi_d2['result']):
    print(f"N={r2['N']:>6}: {r2['C2ST-S']:.3f}   {r2['C2ST-L']:.3f}   {rs2['RL-C2ST-S']:.3f}   {rs2['RL-C2ST-L']:.3f}")

print()
print('=== Fig 3c (d=10) ===')
print('N        C2ST-S  C2ST-L  RL-S    RL-L')
for r10, rs10 in zip(d10['result'], semi_d10['result']):
    print(f"N={r10['N']:>6}: {r10['C2ST-S']:.3f}   {r10['C2ST-L']:.3f}   {rs10['RL-C2ST-S']:.3f}   {rs10['RL-C2ST-L']:.3f}")

print()
print('=== Fig 4a MMD-D (d=2) ===')
for r, n in zip(mmd['result'], [8*n for n in mmd['n_list']]):
    print(f"N={n:>6}: {r:.3f}")
