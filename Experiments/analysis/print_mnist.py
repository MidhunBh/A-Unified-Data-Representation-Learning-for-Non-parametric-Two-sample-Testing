import json
d = json.load(open("result/c2st_semi_mnist.json"))
print("RL-C2ST (joint AE) | MNIST vs Fake MNIST")
for r in d["result"]:
    print(f"M={r['M']}: C2ST-S={r['C2ST-S']:.3f}  C2ST-L={r['C2ST-L']:.3f}")
