import os
import numpy as np
import pickle
import torch
import torchvision.transforms as transforms
from torchvision import datasets
import torch.nn as nn
import tqdm
from utils import *

# Device
if torch.cuda.is_available():
    device = torch.device("cuda:0")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
print("device:", device)

# Seeds
np.random.seed(819)
torch.manual_seed(819)
torch.cuda.manual_seed(819)
torch.backends.cudnn.deterministic = True

dtype = torch.float
alpha = 0.05
batch_size = 100
N_TRAIL = 10
N_TEST = 100
N_TEST_F = 100.0
N_PER = 100
N_EPOCH = 1000
channels = 1
img_size = 32

# Conv feature extractor — same architecture as c2st_mnist.py
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        def discriminator_block(in_filters, out_filters, bn=True):
            block = [nn.Conv2d(in_filters, out_filters, 3, 2, 1),
                     nn.LeakyReLU(0.2, inplace=True), nn.Dropout2d(0)]
            if bn:
                block.append(nn.BatchNorm2d(out_filters, 0.8))
            return block
        self.model = nn.Sequential(
            *discriminator_block(channels, 8, bn=False),
            *discriminator_block(8, 16),
            *discriminator_block(16, 32),
        )
        ds_size = img_size // 2 ** 3
        self.feature_layer = nn.Sequential(
            nn.Linear(32 * ds_size ** 2, 100),
            nn.ReLU(),
        )

    def forward(self, img):
        out = self.model(img)
        out = out.view(out.shape[0], -1)
        return self.feature_layer(out)


# Load data
os.makedirs("./data/mnist", exist_ok=True)
dataloader_real_all = torch.utils.data.DataLoader(
    datasets.MNIST("./data/mnist", train=True, download=False,
        transform=transforms.Compose([
            transforms.Resize(img_size), transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])])),
    batch_size=60000, shuffle=True)

for i, (imgs, labels) in enumerate(dataloader_real_all):
    data_real_all = imgs

data_fake_all = pickle.load(open('./data/Fake_MNIST_data_EP100_N10000.pckl', 'rb'))[0]
data_fake_all = torch.from_numpy(data_fake_all).float()

n_list = [100, 200, 300, 400, 500]
feat_dim = 100   # output dim of feature_layer above

summary = []
for n in n_list:
    n_train = n
    n_test = n
    summary_n = []

    for kk in tqdm.trange(N_TRAIL):
        s1_tr, s1_te, s2_tr, s2_te = sample_mnist_semi(
            data_real_all, data_fake_all, n_train, n_test, kk=kk)

        S_tr = torch.cat([s1_tr, s2_tr], dim=0).to(device, dtype)
        S_te = torch.cat([s1_te, s2_te], dim=0).to(device, dtype)

        # Train conv feature extractor with cross-entropy first
        torch.random.manual_seed(1102)
        disc = Discriminator().to(device, dtype)
        head = nn.Sequential(nn.Linear(feat_dim, 2), nn.Softmax(dim=1)).to(device, dtype)

        y = torch.cat([torch.zeros(n_train), torch.ones(n_train)]).to(device, dtype).long()
        optimizer = torch.optim.Adam(
            list(disc.parameters()) + list(head.parameters()),
            lr=0.0005, weight_decay=0.0001)
        criterion = nn.CrossEntropyLoss().to(device)
        dataset = torch.utils.data.TensorDataset(S_tr, y)
        loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size*2, shuffle=True)

        for epoch in range(2*n):
            for S_b, y_b in loader:
                loss = criterion(head(disc(S_b)), y_b)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        # Freeze feature extractor, extract features
        for param in disc.parameters():
            param.requires_grad = False

        with torch.no_grad():
            Fea_tr = disc(S_tr)
            Fea_te = disc(S_te)

        # Phase 2 — train deep kernel on extracted features
        model_mmd, sigma, sigma0, ep = MMD_D_fit(
            Fea_tr, feat_dim, 30, 30, N_EPOCH, device, dtype, lr_mmd=0.00005)

        # Phase 3 — permutation test
        H_MMD = np.zeros(N_TEST)
        for k in range(N_TEST):
            H_MMD[k], _, _ = TST_MMD_u(
                model_mmd(Fea_te), n_test, N_PER, Fea_te,
                sigma, sigma0, ep, alpha, k * kk + 2024)

        print(f"Test Power of MMD-D at M={n}: ", H_MMD.sum() / N_TEST_F)
        summary_n.append(H_MMD.sum() / N_TEST_F)

    summary.append(summary_n)
    print(f"Average Test Power of MMD-D at M={n}: ", np.mean(summary_n))

with open("result/mmd_d_mnist_power.pkl", "wb") as f:
    pickle.dump(summary, f)

import json, subprocess, time
meta = {
    "commit"  : subprocess.check_output(
                    ["git", "rev-parse", "--short", "HEAD"],
                    cwd=r"C:\Users\midhu\Documents\GitHub\A-Unified-Data-Representation-Learning-for-Non-parametric-Two-sample-Testing"
                ).decode().strip(),
    "method"  : "MMD-D",
    "dataset" : "MNIST vs Fake MNIST",
    "metric"  : "test power",
    "panel"   : "Table 3",
    "M"       : n_list,
    "N_TRAIL" : N_TRAIL,
    "N_EPOCH" : N_EPOCH,
    "note"    : "conv feature extractor pretrained with cross-entropy, frozen, features fed into MMD_D_fit deep kernel",
    "result"  : [float(np.mean(r)) for r in summary],
    "ts"      : time.strftime("%Y-%m-%d %H:%M"),
}
with open("result/mmd_d_mnist_power.json", "w") as f:
    json.dump(meta, f, indent=2)
print("logged to result/mmd_d_mnist_power.json")