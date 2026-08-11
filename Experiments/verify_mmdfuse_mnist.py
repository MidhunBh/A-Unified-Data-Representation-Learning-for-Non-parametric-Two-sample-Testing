import numpy as np
import torch
import time
import os
import pickle
import torchvision.transforms as transforms
from torchvision import datasets
from mmd_fuse import *
from utils import sample_mnist_semi

print("Loading MNIST...")
os.makedirs("./data/mnist", exist_ok=True)
dataloader_real_all = torch.utils.data.DataLoader(
    datasets.MNIST("./data/mnist", train=True, download=False,
        transform=transforms.Compose([
            transforms.Resize(32),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])])),
    batch_size=60000, shuffle=True)
for i, (imgs, labels) in enumerate(dataloader_real_all):
    data_real_all = imgs

data_fake_all = pickle.load(open('./data/Fake_MNIST_data_EP100_N10000.pckl', 'rb'))[0]
data_fake_all = torch.from_numpy(data_fake_all).float()

M = 500   # worst-case size in the planned sweep
s1_tr, s1_te, s2_tr, s2_te = sample_mnist_semi(data_real_all, data_fake_all, M, M, kk=0)
S1 = torch.cat([s1_tr, s1_te], dim=0).numpy().reshape(2*M, -1).astype(np.float32)
S2 = torch.cat([s2_tr, s2_te], dim=0).numpy().reshape(2*M, -1).astype(np.float32)
print(f"S1 shape: {S1.shape}, S2 shape: {S2.shape}")

key = random.PRNGKey(42)
key, subkey = random.split(key)
t0 = time.time()
out = mmdfuse(S1, S2, subkey)
print(f"first call (includes JIT compile): out={out}, time={time.time()-t0:.2f}s")

key, subkey = random.split(key)
t0 = time.time()
out2 = mmdfuse(S1, S2, subkey)
print(f"second call (compiled): out={out2}, time={time.time()-t0:.3f}s")
print(f"estimated time for 10 trials at M=500: {(time.time()-t0)*10:.1f}s")