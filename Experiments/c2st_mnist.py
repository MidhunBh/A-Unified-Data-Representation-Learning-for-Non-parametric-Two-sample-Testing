import math
import os
import numpy as np
import pickle
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torchvision import datasets
import matplotlib.pyplot as plt
import torch.nn as nn
import tqdm
import copy
from utils import *

# Set up device
if torch.cuda.is_available():
    device = torch.device("cuda:0")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
print("device:", device)

# Setup seeds
np.random.seed(819)
torch.manual_seed(819)
torch.cuda.manual_seed(819)
torch.backends.cudnn.deterministic = True

# Setup for experiments
dtype = torch.float
alpha = 0.05 # test threshold
batch_size = 100 # size of the batches

N_EPOCHS = 2000
N_TRAIL = 10  # number of trails
N_TEST = 100  # number of test sets
N_TEST_F = 100.0  # number of test sets (float)0
N_PER = 100  # permutation times

# Fetch data
channels = 1 # number of image channels
img_size = 32 # size of each image dimension
    
# Define the deep network for C2ST-S and C2ST-L
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        def discriminator_block(in_filters, out_filters, bn=True):
            block = [nn.Conv2d(in_filters, out_filters, 3, 2, 1), nn.LeakyReLU(0.2, inplace=True), nn.Dropout2d(0)]
            if bn:
                block.append(nn.BatchNorm2d(out_filters, 0.8))
            return block

        self.model = nn.Sequential(
            # *discriminator_block(channels, 16, bn=False),
            *discriminator_block(channels, 8, bn=False),
            *discriminator_block(8, 16),
            *discriminator_block(16, 32),
            # *discriminator_block(32, 64),
            # *discriminator_block(64, 128),
        )

        # The height and width of downsampled image
        # ds_size = img_size // 2 ** 4
        ds_size = img_size // 2 ** 3
        self.adv_layer = nn.Sequential(
            # nn.Linear(128 * ds_size ** 2, 100),
            nn.Linear(32 * ds_size ** 2, 100),
            nn.ReLU(),
            nn.Linear(100, 20),
            nn.ReLU(),
            nn.Linear(20, 2),
            nn.Softmax(dim=1))

    def forward(self, img):
        out = self.model(img)
        out = out.view(out.shape[0], -1)
        validity = self.adv_layer(out)

        return validity


torch.manual_seed(819)
# Configure data loader
os.makedirs("./data/mnist", exist_ok=True)
dataloader_real_all = torch.utils.data.DataLoader(
    datasets.MNIST(
        "./data/mnist",
        train=True,
        download=False,
        transform=transforms.Compose(
            [transforms.Resize(img_size), transforms.ToTensor(), transforms.Normalize([0.5], [0.5])]
        ),
    ),
    batch_size=60000,
    shuffle=True,
)

# Obtain MNIST images
for i, (imgs, labels) in enumerate(dataloader_real_all):
    data_real_all = imgs
    label_real_all = labels

# Obtain fake MNIST images
data_fake_all = pickle.load(open('./data/Fake_MNIST_data_EP100_N10000.pckl', 'rb'))[0]
data_fake_all = torch.from_numpy(data_fake_all).float()


n_list = [100, 200, 300, 400, 500]
# n = 500

summary = []
for n in n_list:
    # summary_s = []
    # summary_l = []
    summary_d = []
    for kk in tqdm.trange(100):
        n_train = n
        n_test = n

        s1_tr, s1_te, s2_tr, s2_te = sample_mnist_semi(data_real_all, data_fake_all, n_train, n_test, kk=kk)
        # s1_tr, s1_te, s2_tr, s2_te = sample_mnist_semi(data_real_all, data_real_all, n_train, n_test, kk=kk)
        
        S = torch.cat([s1_tr, s2_tr], dim=0).to(device, dtype)
        y = torch.concat(
                (torch.zeros(n_train), torch.ones(n_train))).to(device, dtype).long()
        
        S_test = torch.cat([s1_te, s2_te], dim=0).to(device, dtype)

        # random seed for control model init, weight, bias and shuffle
        torch.random.manual_seed(1102)
        discriminator = Discriminator().to(device, dtype)

        optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=0.0005, weight_decay=0.0001)
        # optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=0.001)
        criterion = torch.nn.CrossEntropyLoss().to(device)
        dataset = torch.utils.data.TensorDataset(S, y)
        
        dataloader_C2ST = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size*2, shuffle=True
        )

        for epoch in range(2*n):
            for S_b, y_b in dataloader_C2ST:
                loss_C2ST = criterion(discriminator(S_b), y_b)
                optimizer_D.zero_grad()
                loss_C2ST.backward(retain_graph=True)

                optimizer_D.step()
                
            if (epoch + 1) % 200 == 0:
                print(criterion(discriminator(S_b), y_b).item())      

        new_layer = nn.Sequential(*list(discriminator.adv_layer.children())[:1])
        discriminator.adv_layer = new_layer
        # print(new_layer)
        # break

        h_d, _, _ = TST_MMD_l(discriminator(S), n_train, N_PER, alpha, kernel="poly")

        print("test power on training set", h_d)

        H_C2ST_D = np.zeros([N_TEST])
        for k in range(N_TEST):
            H_C2ST_D[k], _, _ = TST_MMD_l(discriminator(S_test), n_train, N_PER, alpha, k, "poly")
        print(f"Test Power of C2ST-D at n = {n}: ", H_C2ST_D.sum() / N_TEST_F)

        summary_d.append(H_C2ST_D.sum() / N_TEST_F)
        
    summary.append(summary_d)
    print(f"Average Test Power of C2ST-D at n = {n}: ", np.mean(summary_d))
    
with open("result/c2st_mnist_baseline_mmd.pkl", "wb") as f:
    pickle.dump(summary, f)


import json, subprocess, time
meta = {
    "commit"  : subprocess.check_output(
                    ["git", "rev-parse", "--short", "HEAD"],
                    cwd=r"C:\Users\midhu\Documents\GitHub\A-Unified-Data-Representation-Learning-for-Non-parametric-Two-sample-Testing"
                ).decode().strip(),
    "method"  : "C2ST-MMD (poly kernel)",
    "dataset" : "MNIST vs Fake MNIST",
    "metric"  : "test power",
    "panel"   : "Table 3",
    "M"       : n_list,
    "N_TRAIL" : N_TRAIL,
    "note"    : "strips classification head, runs MMD with poly kernel on features",
    "result"  : [float(np.mean(r)) for r in summary],
    "ts"      : time.strftime("%Y-%m-%d %H:%M"),
}
with open("result/c2st_mnist_baseline.json", "w") as f:
    json.dump(meta, f, indent=2)
print("logged to result/c2st_mnist_baseline.json")

        # Run two-sample test (C2STs) on the training set
    #     h_s, _, _ = TST_C2ST_D(S, n_train, N_PER, alpha, discriminator, device, dtype)
    #     h_l, _, _ = TST_LCE_D(S, n_train, N_PER, alpha, discriminator, device, dtype)
    #     print(h_s, h_l)
                
    #     H_C2ST_S = np.zeros([N_TEST])
    #     H_C2ST_L = np.zeros([N_TEST])

    #     for k in range(N_TEST):
    #         H_C2ST_S[k], _, _ = TST_C2ST_D(S_test, n_train, N_PER, alpha, discriminator, device, dtype)
    #         H_C2ST_L[k], _, _ = TST_LCE_D(S_test, n_train, N_PER, alpha, discriminator, device, dtype)

        # print(f"Test Power of C2ST-S at n = {n}: ", H_C2ST_S.sum() / N_TEST_F)
        # print(f"Test Power of C2ST-L at n = {n}: ", H_C2ST_L.sum() / N_TEST_F)
        # print(f"Type-I error of C2ST-S at n = {n}: ", H_C2ST_S.sum() / N_TEST_F)
        # print(f"Type-I error of C2ST-L at n = {n}: ", H_C2ST_L.sum() / N_TEST_F)
        
    #     summary_s.append(H_C2ST_S.sum() / N_TEST_F)
    #     summary_l.append(H_C2ST_L.sum() / N_TEST_F)

        # print("Average Test Power of C2ST-S: ", np.mean(summary_s))
        # print("Average Test Power of C2ST-L: ", np.mean(summary_l))
    # print("Average Type-I error of C2ST-S: ", np.mean(summary_s))
    # print("Average Type-I error of C2ST-L: ", np.mean(summary_l))
#     summary.append((summary_s, summary_l))
    
# with open("result/c2st_mnist_baseline_lr001.pkl", "wb") as f:
#     pickle.dump(summary, f)