import torch
import numpy as np
from torch.optim.lr_scheduler import StepLR
import torch.nn as nn


class ModelLatentF(torch.nn.Module):

    """Latent space for both domains."""

    def __init__(self, x_in, H, x_out):
        """Init latent features."""
        super(ModelLatentF, self).__init__()
        self.restored = False
        self.latent = torch.nn.Sequential(
            torch.nn.Linear(x_in, H, bias=True),
            torch.nn.Softplus(),
            torch.nn.Linear(H, H, bias=True),
            torch.nn.Softplus(),
            torch.nn.Linear(H, H, bias=True),
            torch.nn.Softplus(),
            torch.nn.Linear(H, x_out, bias=True),
        )

    def forward(self, input):
        """Forward the LeNet."""
        fealant = self.latent(input)
        return fealant
    

class AutoEncoder(torch.nn.Module):
    def __init__(self, x_in, H, x_out):
        super(AutoEncoder, self).__init__()

        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(x_in, H, bias=True),
            torch.nn.Softplus(),
            torch.nn.Linear(H, H, bias=True),
            torch.nn.Softplus(),
            torch.nn.Linear(H, H, bias=True),
            torch.nn.Softplus(),
            torch.nn.Linear(H, x_out, bias=True),
        )

        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(x_out, H, bias=True),
            torch.nn.Softplus(),
            torch.nn.Linear(H, H, bias=True),
            torch.nn.Softplus(),
            torch.nn.Linear(H, H, bias=True),
            torch.nn.Softplus(),
            torch.nn.Linear(H, x_in, bias=True),
        )

    def forward(self, x):
        x = self.encoder(x)
        # print(x.shape)
        x = self.decoder(x)
        return x


class ExtendedModel(torch.nn.Module):
    def __init__(self, encoder, H, x_out):
        super(ExtendedModel, self).__init__()

        self.encoder = encoder

        self.additional_layers = torch.nn.Sequential(
            torch.nn.Linear(H, H, bias=True),
            torch.nn.Softplus(),
            torch.nn.Linear(H, H, bias=True),
            torch.nn.Softplus(),
            torch.nn.Linear(H, x_out, bias=True),
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.additional_layers(x)
        return x


##############################################################################################################
#   helper functions
##############################################################################################################


def get_item(x, is_accelerate):
    """get the numpy value from a torch tensor."""
    if is_accelerate:
        x = x.cpu().detach().numpy()
    else:
        x = x.detach().numpy()
    return x


def MatConvert(x, device, dtype):
    """convert the numpy to a torch tensor."""
    x = torch.from_numpy(x).to(device, dtype)
    return x


def Pdist2(x, y):
    """compute the paired distance between x and y."""
    x_norm = (x**2).sum(1).view(-1, 1)
    if y is not None:
        y_norm = (y**2).sum(1).view(1, -1)
    else:
        y = x
        y_norm = x_norm.view(1, -1)
    Pdist = x_norm + y_norm - 2.0 * torch.mm(x, torch.transpose(y, 0, 1))
    Pdist[Pdist < 0] = 0
    return Pdist


def h1_mean_var_gram(Kx, Ky, Kxy, is_var_computed, use_1sample_U=True):
    """compute value of MMD and std of MMD using kernel matrix."""
    Kxxy = torch.cat((Kx, Kxy), 1)
    Kyxy = torch.cat((Kxy.transpose(0, 1), Ky), 1)
    Kxyxy = torch.cat((Kxxy, Kyxy), 0)
    nx = Kx.shape[0]
    ny = Ky.shape[0]
    is_unbiased = True
    if is_unbiased:
        xx = torch.div((torch.sum(Kx) - torch.sum(torch.diag(Kx))), (nx * (nx - 1)))
        yy = torch.div((torch.sum(Ky) - torch.sum(torch.diag(Ky))), (ny * (ny - 1)))
        # one-sample U-statistic.
        if use_1sample_U:
            xy = torch.div(
                (torch.sum(Kxy) - torch.sum(torch.diag(Kxy))), (nx * (ny - 1))
            )
        else:
            xy = torch.div(torch.sum(Kxy), (nx * ny))
        mmd2 = xx - 2 * xy + yy
    else:
        xx = torch.div((torch.sum(Kx)), (nx * nx))
        yy = torch.div((torch.sum(Ky)), (ny * ny))
        # one-sample U-statistic.
        if use_1sample_U:
            xy = torch.div((torch.sum(Kxy)), (nx * ny))
        else:
            xy = torch.div(torch.sum(Kxy), (nx * ny))
        mmd2 = xx - 2 * xy + yy
    if not is_var_computed:
        return mmd2, None

    hh = Kx + Ky - Kxy - Kxy.transpose(0, 1)
    V1 = torch.dot(hh.sum(1) / ny, hh.sum(1) / ny) / ny
    V2 = (hh).sum() / (nx) / nx
    varEst = 4 * (V1 - V2**2)
    if varEst == 0.0:
        print("error!!" + str(V1))

    return mmd2, varEst, Kxyxy


def MMDu(Fea, len_s, Fea_org, sigma, sigma0=0.1, epsilon=10 ** (-10), is_smooth=True, is_var_computed=True, use_1sample_U=True):
    """compute value of deep-kernel MMD and std of deep-kernel MMD using merged data."""
    X = Fea[0:len_s, :]  # fetch the sample 1 (features of deep networks)
    Y = Fea[len_s:, :]  # fetch the sample 2 (features of deep networks)
    X_org = Fea_org[0:len_s, :]  # fetch the original sample 1
    Y_org = Fea_org[len_s:, :]  # fetch the original sample 2
    L = 1  # generalized Gaussian (if L>1)
    Dxx = Pdist2(X, X)
    Dyy = Pdist2(Y, Y)
    Dxy = Pdist2(X, Y)
    Dxx_org = Pdist2(X_org, X_org)
    Dyy_org = Pdist2(Y_org, Y_org)
    Dxy_org = Pdist2(X_org, Y_org)
    if is_smooth:
        Kx = (1-epsilon) * torch.exp(-(Dxx / sigma0) - (Dxx_org / sigma)
                                     )**L + epsilon * torch.exp(-Dxx_org / sigma)
        Ky = (1-epsilon) * torch.exp(-(Dyy / sigma0) - (Dyy_org / sigma)
                                     )**L + epsilon * torch.exp(-Dyy_org / sigma)
        Kxy = (1-epsilon) * torch.exp(-(Dxy / sigma0) - (Dxy_org / sigma)
                                      )**L + epsilon * torch.exp(-Dxy_org / sigma)
    else:
        Kx = torch.exp(-Dxx / sigma0)
        Ky = torch.exp(-Dyy / sigma0)
        Kxy = torch.exp(-Dxy / sigma0)
    return h1_mean_var_gram(Kx, Ky, Kxy, is_var_computed, use_1sample_U)

def MMDl(Fea, len_s, kernel="linear"):
    X = Fea[0:len_s, :]  # fetch the sample 1 (features of deep networks)
    Y = Fea[len_s:, :]  # fetch the sample 2 (features of deep networks)

    def rbf_kernel(A, B, sigma=None):
        # Compute the pairwise squared Euclidean distances
        dists = torch.cdist(A, B) ** 2

        # Determine sigma using the median heuristic
        if sigma is None:
            sigma = torch.median(dists)

        # Compute the RBF kernel
        gamma = 1.0 / (2 * sigma**2)
        kernel_matrix = torch.exp(-gamma * dists)

        return kernel_matrix
    if kernel == "linear":
        Kx = torch.mm(X, X.t())
        Ky = torch.mm(Y, Y.t())
        Kxy = torch.mm(X, Y.t())
    elif kernel == "poly":
        c, d = 1, 3
        Kx = (X @ X.t() + c) ** d
        Ky = (Y @ Y.t() + c) ** d
        Kxy = (X @ Y.t() + c) ** d
    elif kernel == "rbf":
        Kx = rbf_kernel(X, X)
        Ky = rbf_kernel(Y, Y)
        Kxy = rbf_kernel(X, Y)


    return h1_mean_var_gram(Kx, Ky, Kxy, True, True)


##############################################################################################################
#   c2st training and testing
##############################################################################################################


def C2ST_NN_fit(
    S,
    y,
    x_in,
    H,
    x_out,
    N_epoch,
    batch_size,
    device,
    dtype,
    model=None,
    lr_c2st=0.001,
    lr_scheduler=False
):
    """Train a deep network for C2STs."""

    # random seed for control model init, weight, bias and shuffle
    torch.random.manual_seed(1102)

    if model is None:
        model = ModelLatentF(x_in, H, x_out)
    else:
        model = model

    N = S.shape[0]
    model_C2ST = model.to(device, dtype)

    w_C2ST = torch.randn([x_out, 2]).to(device, dtype)
    b_C2ST = torch.randn([1, 2]).to(device, dtype)
    w_C2ST.requires_grad = True
    b_C2ST.requires_grad = True

    optimizer_C2ST = torch.optim.Adam(
        list(model_C2ST.parameters()) + [w_C2ST] + [b_C2ST], lr=lr_c2st
    )
    if lr_scheduler:
        scheduler = StepLR(optimizer_C2ST, step_size=500, gamma=0.5)
    criterion = torch.nn.CrossEntropyLoss()
    dataset = torch.utils.data.TensorDataset(S, y)

    dataloader_C2ST = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=True
    )
    len_dataloader = len(dataloader_C2ST)
    for epoch in range(N_epoch):
        data_iter = iter(dataloader_C2ST)
        tt = 0
        while tt < len_dataloader:
            # training model using source data
            data_source = next(data_iter)
            S_b, y_b = data_source
            output_b = model_C2ST(S_b).mm(w_C2ST) + b_C2ST
            loss_C2ST = criterion(output_b, y_b)
            optimizer_C2ST.zero_grad()
            loss_C2ST.backward(retain_graph=True)
            # Update sigma0 using gradient descent
            optimizer_C2ST.step()
            tt = tt + 1
        if lr_scheduler:
            scheduler.step()

        if (epoch+1) % 200 == 0:
            print(criterion(model_C2ST(S).mm(w_C2ST) + b_C2ST, y).item())
            if lr_scheduler:
                print(f"Epoch {epoch+1}/{N_epoch}, Current LR: {scheduler.get_last_lr()[0]}")

    return model_C2ST, w_C2ST, b_C2ST

def TST_C2ST(S, N1, N_per, alpha, model_C2ST, w_C2ST, b_C2ST, rd_seed=0):
    """run C2ST-L on non-image datasets."""
    N = S.shape[0]
    f = torch.nn.Softmax(dim=1)
    output = f(model_C2ST(S).mm(w_C2ST) + b_C2ST)
    pred_C2ST = output.max(1, keepdim=True)[1]

    STAT = abs(
        pred_C2ST[:N1].type(torch.FloatTensor).mean()
        - pred_C2ST[N1:].type(torch.FloatTensor).mean()
    )
    STAT_vector = np.zeros(N_per)
    for r in range(N_per):
        np.random.seed(1102 + r * 3 + rd_seed)
        ind = np.random.choice(N, N, replace=False)

        # divide into new X, Y
        ind_X = ind[:N1]
        ind_Y = ind[N1:]
        STAT_vector[r] = abs(
            pred_C2ST[ind_X].type(torch.FloatTensor).mean()
            - pred_C2ST[ind_Y].type(torch.FloatTensor).mean()
        )

    S_vector = np.sort(STAT_vector)
    threshold = S_vector[np.int64(np.ceil(N_per * (1 - alpha)))]
    # threshold_lower = S_vector[np.int64(np.ceil(N_per *  alpha))]
    h = 0
    if STAT.item() > threshold:
        h = 1
    return h, threshold, STAT

def TST_LCE(S, N1, N_per, alpha, model_C2ST, w_C2ST, b_C2ST, rd_seed=0):
    """run C2ST-L on non-image datasets."""
    N = S.shape[0]
    f = torch.nn.Softmax(dim=1)
    output = f(model_C2ST(S).mm(w_C2ST) + b_C2ST)

    STAT = abs(
        output[:N1, 0].type(torch.FloatTensor).mean()
        - output[N1:, 0].type(torch.FloatTensor).mean()
    )
    STAT_vector = np.zeros(N_per)
    for r in range(N_per):
        np.random.seed(1102 + r * 3 + rd_seed)
        ind = np.random.choice(N, N, replace=False)

        # divide into new X, Y
        ind_X = ind[:N1]
        ind_Y = ind[N1:]
        STAT_vector[r] = abs(
            output[ind_X, 0].type(torch.FloatTensor).mean()
            - output[ind_Y, 0].type(torch.FloatTensor).mean()
        )

    S_vector = np.sort(STAT_vector)
    threshold = S_vector[np.int64(np.ceil(N_per * (1 - alpha)))]
    # threshold_lower = S_vector[np.int64(np.ceil(N_per *  alpha))]
    h = 0
    if STAT.item() > threshold:
        h = 1
    return h, threshold, STAT

def TST_C2ST_D(S,N1,N_per,alpha,discriminator,device,dtype):
    """run C2ST-S on MNIST and CIFAR datasets."""
    np.random.seed(seed=1102)
    torch.manual_seed(1102)
    torch.cuda.manual_seed(1102)
    N = S.shape[0]
    f = torch.nn.Softmax(dim=1)
    output = discriminator(S)
    pred_C2ST = output.max(1, keepdim=True)[1]
    STAT = abs(pred_C2ST[:N1].type(torch.FloatTensor).mean() - pred_C2ST[N1:].type(torch.FloatTensor).mean())
    STAT_vector = np.zeros(N_per)
    for r in range(N_per):
        ind = np.random.choice(N, N, replace=False)
        # divide into new X, Y
        ind_X = ind[:N1]
        ind_Y = ind[N1:]
        STAT_vector[r] = abs(pred_C2ST[ind_X].type(torch.FloatTensor).mean() - pred_C2ST[ind_Y].type(torch.FloatTensor).mean())
    S_vector = np.sort(STAT_vector)
    threshold = S_vector[np.int64(np.ceil(N_per * (1 - alpha)))]
    threshold_lower = S_vector[np.int64(np.ceil(N_per *  alpha))]
    h = 0
    if STAT.item() > threshold:
        h = 1
    return h, threshold, STAT

def TST_LCE_D(S,N1,N_per,alpha,discriminator,device,dtype):
    """run C2ST-L on MNIST and CIFAR datasets."""
    np.random.seed(seed=1102)
    torch.manual_seed(1102)
    torch.cuda.manual_seed(1102)
    N = S.shape[0]
    f = torch.nn.Softmax(dim=1)
    output = discriminator(S)
    STAT = abs(output[:N1,0].type(torch.FloatTensor).mean() - output[N1:,0].type(torch.FloatTensor).mean())
    STAT_vector = np.zeros(N_per)
    for r in range(N_per):
        ind = np.random.choice(N, N, replace=False)
        # divide into new X, Y
        ind_X = ind[:N1]
        ind_Y = ind[N1:]
        # print(indx)
        STAT_vector[r] = abs(output[ind_X,0].type(torch.FloatTensor).mean() - output[ind_Y,0].type(torch.FloatTensor).mean())
    S_vector = np.sort(STAT_vector)
    threshold = S_vector[np.int64(np.ceil(N_per * (1 - alpha)))]
    h = 0
    if STAT.item() > threshold:
        h = 1
    return h, threshold, STAT

def permutate_on_data(S,N1,N_per,discriminator,inds):
    """run C2ST-L on MNIST and CIFAR datasets."""
    np.random.seed(seed=1102)
    torch.manual_seed(1102)
    torch.cuda.manual_seed(1102)
    # N = S.shape[0]
    f = torch.nn.Softmax(dim=1)
    output = discriminator(S)
    STAT = abs(output[:N1,0].type(torch.FloatTensor).mean() - output[N1:,0].type(torch.FloatTensor).mean())
    STAT_vector = np.zeros(N_per + 1)
    for r in range(N_per):
        # ind = np.random.choice(N, N, replace=False)
        ind = inds[r]
        # divide into new X, Y
        ind_X = ind[:N1]
        ind_Y = ind[N1:]
        # print(indx)
        STAT_vector[r] = abs(output[ind_X,0].type(torch.FloatTensor).mean() - output[ind_Y,0].type(torch.FloatTensor).mean())
    # S_vector = np.sort(STAT_vector)
    # threshold = S_vector[np.int64(np.ceil(N_per * (1 - alpha)))]
    # h = 0
    # if STAT.item() > threshold:
    #     h = 1
    # return h, threshold, STAT
    STAT_vector[-1] = STAT
    return STAT_vector # (N_EPOCH+1,)

##############################################################################################################
#   MMD-D training and testing
##############################################################################################################


def MMD_D_fit(
    S,
    x_in,
    H,
    x_out,
    N_epoch,
    device,
    dtype,
    model=None,
    lr_mmd=0.001,
):
    """Train a deep network for MMD-D."""
    torch.random.manual_seed(1102)

    if model is None:
        model = ModelLatentF(x_in, H, x_out)
    else:
        model = model

    N, d = S.shape
    model_mmd = model.to(device, dtype)

    epsilonOPT = torch.log(MatConvert(
        np.random.rand(1) * 10 ** (-10), device, dtype))
    sigmaOPT = MatConvert(np.ones(1) * np.sqrt(2 * d), device, dtype)
    sigma0OPT = MatConvert(np.ones(1) * np.sqrt(0.005), device, dtype)

    optimizer_mmd = torch.optim.Adam(
        list(model_mmd.parameters()) + [epsilonOPT, sigmaOPT, sigma0OPT],
        lr=lr_mmd,
    )

    for epoch in range(N_epoch):
        sigma = sigmaOPT ** 2
        sigma0 = sigma0OPT ** 2
        ep = torch.exp(epsilonOPT) / (1 + torch.exp(epsilonOPT))

        modelmmd_output = model_mmd(S)

        mmd_value_tmp, mmd_std_tmp, _ = MMDu(
            modelmmd_output, np.int64(N/2), S, sigma, sigma0, ep)

        mmd_value_tmp = -1 * (mmd_value_tmp + 10**(-8))
        mmd_std_tmp = torch.sqrt(mmd_std_tmp + 10**(-8))

        if mmd_std_tmp.item() == 0:
            print("error!!")
        if np.isnan(mmd_std_tmp.item()):
            print("error!!")

        J = torch.div(mmd_value_tmp, mmd_std_tmp)
        optimizer_mmd.zero_grad()
        J.backward(retain_graph=True)
        # Update weights using gradient descent
        optimizer_mmd.step()

        if (epoch + 1) % 100 == 0:
            print("mmd: ", -1 * mmd_value_tmp.item(), "std: ",
                  mmd_std_tmp.item(), "Stat J: ", -1 * J.item())

    sigma = sigmaOPT ** 2
    sigma0 = sigma0OPT ** 2
    ep = torch.exp(epsilonOPT) / (1 + torch.exp(epsilonOPT))

    return model_mmd, sigma, sigma0, ep


def mmd2_permutations(K, n_X, k=0, permutations=500):
    """
        Fast implementation of permutations using kernel matrix.
    """
    K = torch.as_tensor(K)
    n = K.shape[0]
    assert K.shape[0] == K.shape[1]
    n_Y = n_X
    assert n == n_X + n_Y
    w_X = 1
    w_Y = -1
    ws = torch.full((permutations + 1, n), w_Y, dtype=K.dtype, device=K.device)
    ws[-1, :n_X] = w_X
    torch.random.manual_seed(802 * k + n_X)
    for i in range(permutations):
        ws[i, torch.randperm(n)[:n_X].numpy()] = w_X
    biased_ests = torch.einsum("pi,ij,pj->p", ws, K, ws)
    is_unbiased = True
    if is_unbiased:  # u-stat estimator
        # need to subtract \sum_i k(X_i, X_i) + k(Y_i, Y_i) + 2 k(X_i, Y_i)
        # first two are just trace, but last is harder:
        is_X = ws > 0
        X_inds = is_X.nonzero()[:, 1].view(permutations + 1, n_X)
        Y_inds = (~is_X).nonzero()[:, 1].view(permutations + 1, n_Y)
        del is_X, ws
        cross_terms = K.take(Y_inds * n + X_inds).sum(1)
        del X_inds, Y_inds
        ests = (biased_ests - K.trace() + 2 * cross_terms) / (n_X * (n_X - 1))
    else:  # biased estimator
        ests = biased_ests / (n * n)
    est = ests[-1]
    rest = ests[:-1]
    p_val = (rest > est).float().mean()
    return est.item(), p_val.item(), rest


def TST_MMD_u(Fea, N1, N_per, Fea_org, sigma, sigma0, ep, alpha, k=0, is_smooth=True):
    """run two-sample test (TST) using deep kernel kernel."""
    TEMP = MMDu(Fea, N1, Fea_org, sigma, sigma0, ep, is_smooth)
    Kxyxy = TEMP[2]
    nx = N1
    mmd_value, p_val, rest = mmd2_permutations(
        Kxyxy, nx, k, permutations=N_per)
    if p_val > alpha:
        h = 0
    else:
        h = 1
    return h, p_val, mmd_value

def TST_MMD_l(Fea, N1, N_per, alpha, k=0, is_smooth=True, kernel="linear"):
    """run two-sample test (TST) using deep kernel kernel."""
    TEMP = MMDl(Fea, N1, kernel)
    Kxyxy = TEMP[2]
    nx = N1
    mmd_value, p_val, rest = mmd2_permutations(
        Kxyxy, nx, k, permutations=N_per)
    if p_val > alpha:
        h = 0
    else:
        h = 1
    return h, p_val, mmd_value


##############################################################################################################
#   Generate HDGM data
##############################################################################################################

def generate_hdgm_cov_matrix(n_clusters, d, cluster_gap):
    mu_mx = np.zeros([n_clusters, d])
    for i in range(n_clusters):
        mu_mx[i] = mu_mx[i] + cluster_gap*i
    sigma_mx_1 = np.eye(d)
    sigma_mx_2 = [np.eye(d), np.eye(d)]
    sigma_mx_2[0][0, 1] = 0.5
    sigma_mx_2[0][1, 0] = 0.5
    sigma_mx_2[1][0, 1] = -0.5
    sigma_mx_2[1][1, 0] = -0.5

    return mu_mx, sigma_mx_1, sigma_mx_2


def sample_hdgm_semi_t2(n_train, n_test, d=10, n_clusters=2, kk=0, level="hard"):
    if level == "hard":
        mu_mx_1, sigma_mx_1, sigma_mx_2 = generate_hdgm_cov_matrix(n_clusters, d, 0.5)
        mu_mx_2 = mu_mx_1
    elif level == "medium":
        mu_mx_1, sigma_mx_1, sigma_mx_2 = generate_hdgm_cov_matrix(n_clusters, d, 10)
        mu_mx_2 = mu_mx_1
    else:
        mu_mx_1, sigma_mx_1, sigma_mx_2 = generate_hdgm_cov_matrix(n_clusters, d, 10)
        mu_mx_2 = mu_mx_1 - 1.5
    n = np.int64(n_train + n_test)

    s1 = np.zeros([n*n_clusters, d])
    s2 = np.zeros([n*n_clusters, d])

    np.random.seed(seed=1102*kk)
    # tr_idx = np.random.choice(n, n_train, replace=False)
    # tr_idx = np.tile(tr_idx, n_clusters)
    # for i in range(n_clusters):
    #     tr_idx[i*n_train:(i+1)*n_train] = tr_idx[i*n_train:(i+1)*n_train] + i*n

    tr_idx = np.random.choice(n*n_clusters, np.int64(n_train*n_clusters), replace=False)

    te_idx = np.delete(np.arange(n*n_clusters), tr_idx)

    for i in range(n_clusters):
        np.random.seed(seed=1102*kk + i + n)
        s1[i*n:(i+1)*n, :] = np.random.multivariate_normal(mu_mx_1[i], sigma_mx_1, n)
        np.random.seed(seed=819*kk + i + n + 1)
        s2[i*n:(i+1)*n, :] = np.random.multivariate_normal(mu_mx_2[i],
                                                           sigma_mx_2[i], n)

    return s1[tr_idx], s1[te_idx], s2[tr_idx], s2[te_idx]


##############################################################################################################
#   sample CIFAR10 and CIFAR10.1
##############################################################################################################

def sample_cifar10_semi_t2(data_real_all, data_fake_all, n_train, n_test, kk=0):
    n_total = n_train + n_test

    # Collect CIFAR10 images
    np.random.seed(seed=1102 * (kk + 19) + n_train) 
    idx_real_total = np.random.choice(len(data_real_all), n_total, replace=False)
    idx_real_tr = idx_real_total[:n_train]
    idx_real_te = idx_real_total[n_train:]
    
    s1_tr = data_real_all[idx_real_tr]
    s1_te = data_real_all[idx_real_te]

    # Collect CIFAR10.1 images
    np.random.seed(seed=819 * (kk + 9) + n_train)
    idx_fake_total = np.random.choice(len(data_fake_all), n_total, replace=False)
    idx_fake_tr = idx_fake_total[:n_train]
    idx_fake_te = idx_fake_total[n_train:]

    s2_tr = data_fake_all[idx_fake_tr]
    s2_te = data_fake_all[idx_fake_te]

    return s1_tr, s1_te, s2_tr, s2_te

##############################################################################################################
#   sample MNIST and fake MNIST
##############################################################################################################

def sample_mnist_semi(data_all_p, data_all_q, n_train, n_test, kk=0):
    n_total = n_train + n_test

    # Collect CIFAR10 images
    np.random.seed(seed=1102 * (kk + 19) + n_train) 
    idx_p_total = np.random.choice(len(data_all_p), n_total, replace=False)
    idx_p_tr = idx_p_total[:n_train]
    idx_p_te = idx_p_total[n_train:]
    
    s1_tr = data_all_p[idx_p_tr]
    s1_te = data_all_p[idx_p_te]

    # Collect CIFAR10.1 images
    np.random.seed(seed=819 * (kk + 9) + n_train)
    idx_q_total = np.random.choice(len(data_all_q), n_total, replace=False)
    idx_q_tr = idx_q_total[:n_train]
    idx_q_te = idx_q_total[n_train:]

    s2_tr = data_all_q[idx_q_tr]
    s2_te = data_all_q[idx_q_te]

    return s1_tr, s1_te, s2_tr, s2_te

##############################################################################################################
#   Semi-supervised learning algorithm
##############################################################################################################

def train_autoencoder(S, epoch, x_in, H, x_out, batch_size, device, dtype, lr=0.002):
    # Setup seeds
    torch.random.manual_seed(1102)

    model = AutoEncoder(x_in, H, x_out).to(device, dtype)
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    dataset = torch.utils.data.TensorDataset(S)
    dataloader_autoencoder = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=True
    )

    len_dataloader = len(dataloader_autoencoder)

    # print(next(iter(dataloader_autoencoder))[0].shape)

    for ep in range(epoch):
        iterloader = iter(dataloader_autoencoder)
        tt = 0
        while tt < len_dataloader:
            # training model using source data
            data_source = next(iterloader)
            input_data = data_source[0]
            # print(input_data)
            outputs = model(input_data)
            loss = criterion(outputs, input_data)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            tt = tt + 1

        if (ep + 1) % 60 == 0:
            print(f"Epoch [{ep+1}/{epoch}], Loss: {loss.item():.4f}")

    return model.encoder


class Encoder_Img(nn.Module):
    def __init__(self, channels, img_size, z_size=300):
        super(Encoder_Img, self).__init__()

        def encoder_block(in_filters, out_filters, bn=True):
            block = [nn.Conv2d(in_filters, out_filters, 3, 2, 1), nn.LeakyReLU(0.2, inplace=True), nn.Dropout2d(0)]
            if bn:
                block.append(nn.BatchNorm2d(out_filters, 0.8))
            return block
        
        self.model = nn.Sequential(
            *encoder_block(channels, 8, bn=False),
            *encoder_block(8, 16),
            *encoder_block(16, 32),
            # *encoder_block(channels, 16, bn=False),
            # *encoder_block(16, 32),
            # *encoder_block(32, 64),
            # *encoder_block(64, 128),
        )

        # The height and width of downsampled image
        # ds_size = img_size // 2 ** 4
        ds_size = img_size // 2 ** 3
        self.adv_layer  = nn.Sequential(
            nn.Linear(32 * ds_size ** 2, z_size))
            # nn.Linear(64 * ds_size ** 2, 300))
        
    def forward(self, img):
        out = self.model(img)
        out = out.view(out.shape[0], -1)
        z = self.adv_layer(out)

        return z

class Decoder_Img(nn.Module):
    def __init__(self, channels, img_size, z_size=300):
        super(Decoder_Img, self).__init__()

        # self.ds_size = img_size // 2 ** 4
        self.ds_size = img_size // 2 ** 3

        # self.init_size = 128 * self.ds_size ** 2
        self.init_size = 32 * self.ds_size ** 2
        self.l1 = nn.Sequential(nn.Linear(z_size, self.init_size))

        def transposed_block(in_filters, out_filters, bn=True):
            block = [nn.ConvTranspose2d(in_filters, out_filters, 3, stride=2, padding=1, output_padding=1), nn.LeakyReLU(0.2, inplace=True)]
            if bn:
                block.append(nn.BatchNorm2d(out_filters, 0.8))
            return block
        
        self.conv_blocks = nn.Sequential(
            # *transposed_block(128, 64),
            # *transposed_block(64, 32),
            # *transposed_block(32, 16),
            # nn.ConvTranspose2d(16, channels, 3, stride=2, padding=1, output_padding=1),  # No batch normalization here
            *transposed_block(32, 16),
            *transposed_block(16, 8),
            nn.ConvTranspose2d(8, channels, 3, stride=2, padding=1, output_padding=1),  # No batch normalization here
            nn.Tanh()
        )
    
    def forward(self, z):
        out = self.l1(z)
        out = out.view(out.shape[0], 32, self.ds_size, self.ds_size)
        # out = out.view(out.shape[0], 64, self.ds_size, self.ds_size)
        img = self.conv_blocks(out)
        return img
    
class Autoencoder_Img(nn.Module):
    def __init__(self, channels, img_size, z_size=100):
        super(Autoencoder_Img, self).__init__()
        self.encoder = Encoder_Img(channels, img_size, z_size)
        self.decoder = Decoder_Img(channels, img_size, z_size)

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x