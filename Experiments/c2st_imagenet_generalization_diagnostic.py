import pickle
import numpy as np
import torch
import torch.nn as nn

from utils import sample_mnist_semi, TST_C2ST_D, TST_LCE_D

# ------------------------------------------------------------
# Exact author settings
# ------------------------------------------------------------
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
dtype = torch.float

np.random.seed(819)
torch.manual_seed(819)
torch.cuda.manual_seed(819)
torch.backends.cudnn.deterministic = True

alpha = 0.05
N_PER = 100
batch_size = 100

channels = 3
img_size = 128

# ------------------------------------------------------------
# Exact author ImageNet discriminator
# ------------------------------------------------------------
class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()

        def block(in_filters, out_filters, bn=True):
            x = [
                nn.Conv2d(in_filters, out_filters, 3, 2, 1),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Dropout2d(0),
            ]
            if bn:
                x.append(nn.BatchNorm2d(out_filters, 0.8))
            return x

        self.model = nn.Sequential(
            *block(channels, 8, bn=False),
            *block(8, 16),
            *block(16, 32),
        )

        ds_size = img_size // 2**3

        self.adv_layer = nn.Sequential(
            nn.Linear(32 * ds_size**2, 100),
            nn.ReLU(),
            nn.Linear(100, 20),
            nn.ReLU(),
            nn.Linear(20, 2),
            nn.Softmax(dim=1),
        )

    def forward(self, x):
        x = self.model(x)
        x = x.view(x.shape[0], -1)
        return self.adv_layer(x)

# ------------------------------------------------------------
# Our reconstructed author inputs
# ------------------------------------------------------------
real = torch.from_numpy(
    pickle.load(open(
        "/system/user/publicwork/mbhaskar/imagenet_reproduction/processed/val_data.pkl",
        "rb"
    ))[:10000]
)

fake = torch.from_numpy(
    pickle.load(open(
        "/system/user/publicwork/mbhaskar/imagenet_reproduction/processed/images_data.pkl",
        "rb"
    ))
)

# Largest Table 3 point:
# paper M=1000 -> author n_train=n_test=500
n = 500
kk = 0

r_tr, r_te, f_tr, f_te = sample_mnist_semi(
    real, fake, n_train=n, n_test=n, kk=kk
)

S_train = torch.cat([r_tr, f_tr], dim=0).to(device, dtype)
S_test  = torch.cat([r_te, f_te], dim=0).to(device, dtype)

y_train = torch.cat([
    torch.zeros(n),
    torch.ones(n)
]).long().to(device)

y_test = torch.cat([
    torch.zeros(n),
    torch.ones(n)
]).long().to(device)

# ------------------------------------------------------------
# Exact author training
# ------------------------------------------------------------
torch.random.manual_seed(1102)

model = Discriminator().to(device, dtype)

optimizer = torch.optim.Adam(model.parameters(), lr=0.0002)
criterion = nn.CrossEntropyLoss().to(device)

loader = torch.utils.data.DataLoader(
    torch.utils.data.TensorDataset(S_train, y_train),
    batch_size=batch_size * 2,
    shuffle=True,
)

for epoch in range(1000):
    model.train()

    for xb, yb in loader:
        loss = criterion(model(xb), yb)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

model.eval()

# ------------------------------------------------------------
# Direct generalization diagnostics
# ------------------------------------------------------------
with torch.no_grad():
    out_tr = model(S_train)
    out_te = model(S_test)

    pred_tr = out_tr.argmax(1)
    pred_te = out_te.argmax(1)

    train_acc = (pred_tr == y_train).float().mean().item()
    test_acc  = (pred_te == y_test).float().mean().item()

    real_test_acc = (pred_te[:n] == 0).float().mean().item()
    fake_test_acc = (pred_te[n:] == 1).float().mean().item()

    train_hard_gap = abs(
        pred_tr[:n].float().mean() -
        pred_tr[n:].float().mean()
    ).item()

    test_hard_gap = abs(
        pred_te[:n].float().mean() -
        pred_te[n:].float().mean()
    ).item()

    train_soft_gap = abs(
        out_tr[:n, 0].mean() -
        out_tr[n:, 0].mean()
    ).item()

    test_soft_gap = abs(
        out_te[:n, 0].mean() -
        out_te[n:, 0].mean()
    ).item()

print()
print("===== GENERALIZATION =====")
print(f"train accuracy:    {train_acc:.4f}")
print(f"test accuracy:     {test_acc:.4f}")
print(f"real test accuracy:{real_test_acc:.4f}")
print(f"fake test accuracy:{fake_test_acc:.4f}")

print()
print("===== RAW SEPARATION =====")
print(f"train hard gap: {train_hard_gap:.6f}")
print(f"test hard gap:  {test_hard_gap:.6f}")
print(f"train soft gap: {train_soft_gap:.6f}")
print(f"test soft gap:  {test_soft_gap:.6f}")

# ------------------------------------------------------------
# Exact author's statistical tests
# ------------------------------------------------------------
h_s, threshold_s, stat_s = TST_C2ST_D(
    S_test, n, N_PER, alpha, model, device, dtype
)

h_l, threshold_l, stat_l = TST_LCE_D(
    S_test, n, N_PER, alpha, model, device, dtype
)

print()
print("===== AUTHOR TEST =====")
print(
    "C2ST-S:",
    "h =", h_s,
    "stat =", float(stat_s),
    "threshold =", float(threshold_s),
)

print(
    "C2ST-L:",
    "h =", h_l,
    "stat =", float(stat_l),
    "threshold =", float(threshold_l),
)

print()
print("DIAGNOSTIC COMPLETE")
