import os
import numpy as np
import pickle
import torch
import torchvision.transforms as transforms
from torchvision import datasets
import torch.nn as nn
import tqdm
from utils import *

if torch.cuda.is_available():
    device = torch.device("cuda:0")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
print("device:", device)

np.random.seed(819)
torch.manual_seed(819)
torch.cuda.manual_seed(819)
torch.backends.cudnn.deterministic = True

dtype = torch.float
alpha = 0.05
batch_size = 100

N_TEST = 100
N_TEST_F = 100.0
N_PER = 100
channels = 1
img_size = 32

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        def discriminator_block(in_filters, out_filters, bn=True):
            block = [nn.Conv2d(in_filters, out_filters, 3, 2, 1), nn.LeakyReLU(0.2, inplace=True), nn.Dropout2d(0)]
            if bn:
                block.append(nn.BatchNorm2d(out_filters, 0.8))
            return block
        self.model = nn.Sequential(
            *discriminator_block(channels, 8, bn=False),
            *discriminator_block(8, 16),
            *discriminator_block(16, 32),
        )
        ds_size = img_size // 2 ** 3
        self.adv_layer = nn.Sequential(
            nn.Linear(32 * ds_size ** 2, 100), nn.ReLU(),
            nn.Linear(100, 20), nn.ReLU(),
            nn.Linear(20, 2), nn.Softmax(dim=1))
    def forward(self, img):
        out = self.model(img)
        out = out.view(out.shape[0], -1)
        return self.adv_layer(out)

torch.manual_seed(819)
os.makedirs("./data/mnist", exist_ok=True)
dataloader_real_all = torch.utils.data.DataLoader(
    datasets.MNIST("./data/mnist", train=True, download=False,
        transform=transforms.Compose([transforms.Resize(img_size), transforms.ToTensor(), transforms.Normalize([0.5], [0.5])])),
    batch_size=60000, shuffle=True)
for i, (imgs, labels) in enumerate(dataloader_real_all):
    data_real_all = imgs
data_fake_all = pickle.load(open('./data/Fake_MNIST_data_EP100_N10000.pckl', 'rb'))[0]
data_fake_all = torch.from_numpy(data_fake_all).float()

n_list = [200, 400, 600, 800, 1000]   # paper's actual Table 3 grid, was [100,200,300,400,500]

summary = []
for n in n_list:
    summary_s = []
    summary_l = []
    for kk in tqdm.trange(100, desc=f"M={n}"):
        n_train = n
        n_test = n

        s1_tr, s1_te, s2_tr, s2_te = sample_mnist_semi(data_real_all, data_fake_all, n_train, n_test, kk=kk)

        S_encoder = torch.cat([s1_tr, s1_te, s2_tr, s2_te], dim=0).to(device, dtype)
        epoch = 1000
        torch.random.manual_seed(1102)
        model = Autoencoder_Img(channels, img_size, 100).to(device, dtype)
        criterion = torch.nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.002)
        dataset = torch.utils.data.TensorDataset(S_encoder)
        dataloader_autoencoder = torch.utils.data.DataLoader(dataset, batch_size=200, shuffle=True)

        for ep in range(epoch):
            for input_data in dataloader_autoencoder:
                outputs = model(input_data[0])
                loss = criterion(outputs, input_data[0])
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        S = torch.cat([s1_tr, s2_tr], dim=0).to(device, dtype)
        y = torch.concat((torch.zeros(n_train), torch.ones(n_train))).to(device, dtype).long()
        S_test = torch.cat([s1_te, s2_te], dim=0).to(device, dtype)

        torch.random.manual_seed(1102)
        discriminator = Discriminator().to(device, dtype)

        def c_model(x):
            return discriminator(model(x))

        optimizer_D = torch.optim.Adam(list(discriminator.parameters()) + list(model.parameters()), lr=0.0004)
        criterion = torch.nn.CrossEntropyLoss().to(device)
        dataset = torch.utils.data.TensorDataset(S, y)
        dataloader_C2ST = torch.utils.data.DataLoader(dataset, batch_size=batch_size*2, shuffle=True)

        for epoch in range(2*n):
            for S_b, y_b in dataloader_C2ST:
                loss_C2ST = criterion(c_model(S_b), y_b)
                optimizer_D.zero_grad()
                loss_C2ST.backward(retain_graph=True)
                optimizer_D.step()

        H_C2ST_S = np.zeros([N_TEST])
        H_C2ST_L = np.zeros([N_TEST])
        for k in range(N_TEST):
            H_C2ST_S[k], _, _ = TST_C2ST_D(S_test, n_train, N_PER, alpha, c_model, device, dtype)
            H_C2ST_L[k], _, _ = TST_LCE_D(S_test, n_train, N_PER, alpha, c_model, device, dtype)

        summary_s.append(H_C2ST_S.sum() / N_TEST_F)
        summary_l.append(H_C2ST_L.sum() / N_TEST_F)

    print(f"M={n}: C2ST-S={np.mean(summary_s):.3f}  C2ST-L={np.mean(summary_l):.3f}", flush=True)
    summary.append((summary_s, summary_l))

    with open("result/c2st_semi_mnist_papergrid.pkl", "wb") as f:
        pickle.dump(summary, f)

    import json, subprocess, time
    try:
        commit = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"],
            cwd=r"C:\Users\midhu\Documents\GitHub\A-Unified-Data-Representation-Learning-for-Non-parametric-Two-sample-Testing").decode().strip()
    except Exception:
        commit = "unknown"
    meta = {
        "commit": commit, "method": "RL-C2ST (joint AE)", "dataset": "MNIST vs Fake MNIST",
        "metric": "test power", "panel": "Fig 3a + Table 3 - paper grid",
        "M": n_list[:len(summary)], "N_TRAIL": 100,
        "note": "Rerun at paper's actual Table 3 grid [200,400,600,800,1000], was [100,200,300,400,500]. Joint fine-tuning, matches original c2st_semi_mnist.py exactly except n_list.",
        "result": [{"M": n_list[i], "C2ST-S": float(np.mean(s)), "C2ST-L": float(np.mean(l))} for i, (s, l) in enumerate(summary)],
        "ts": time.strftime("%Y-%m-%d %H:%M"),
    }
    with open("result/c2st_semi_mnist_papergrid.json", "w") as f:
        json.dump(meta, f, indent=2)

print("=== DONE ===")
