import matplotlib
matplotlib.use("Agg")  # headless – no display needed

# ────────────────────────────────────────────────────────────
# Cell 1  [msd-cell-01]
# ────────────────────────────────────────────────────────────
# ── Imports ──────────────────────────────────────────────────────────────────
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.preprocessing import StandardScaler

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

import gpytorch
from gpytorch.models import ExactGP
from gpytorch.means import ConstantMean
from gpytorch.kernels import RBFKernel, ScaleKernel
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.distributions import MultivariateNormal
from gpytorch import variational

import psutil, os, sys, resource, gc, multiprocessing, threading, time, random
from tqdm import tqdm

print('PyTorch version:', torch.__version__)
print('CUDA available:', torch.cuda.is_available())

# ────────────────────────────────────────────────────────────
# Cell 2  [msd-cell-02]
# ────────────────────────────────────────────────────────────
# ── Data Loading ──────────────────────────────────────────────────────────────
# Load the full MSD dataset. Splitting into folds is done in Section 2.

data_path = '../../data/YearPredictionMSD/YearPredictionMSD.txt'
df_full   = pd.read_csv(data_path, header=None)

X_full = df_full.iloc[:, 1:].values.astype(np.float32)   # (515345, 90)
y_full = df_full.iloc[:, 0 ].values.astype(np.float32)   # (515345,) — release year

print(f'Full MSD dataset: {X_full.shape[0]:,} samples × {X_full.shape[1]} features')
print(f'Year range: {int(y_full.min())}–{int(y_full.max())}')

# ────────────────────────────────────────────────────────────
# Cell 3  [msd-cell-03]
# ────────────────────────────────────────────────────────────
# ── Global hyperparameters ────────────────────────────────────────────────────
INPUT_DIM   = 90    # number of features in MSD
HIDDEN_DIM1 = 256
HIDDEN_DIM2 = 128
HIDDEN_DIM3 = 64

BATCH_SIZE        = 256
MAX_EPOCHS        = 1000
EARLY_STOP_WINDOW = 50
LR                = 1e-4
MOMENTUM          = 0.95
WEIGHT_DECAY      = 1e-3
GRAD_CLIP         = 1.0   # gradient clipping threshold (prevents explosion with SGD+momentum)
SEED              = 7199

DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('Using device:', DEVICE)

# ────────────────────────────────────────────────────────────
# Cell 4  [msd-cell-04]
# ────────────────────────────────────────────────────────────
# ── Dataset helper ────────────────────────────────────────────────────────────
class mydataset(Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y
    def __len__(self):
        return len(self.x)
    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

# ────────────────────────────────────────────────────────────
# Cell 5  [msd-cell-05]
# ────────────────────────────────────────────────────────────
import tracemalloc, ctypes

# ── Reproducibility & measurement utilities ───────────────────────────────────
def set_global_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def measure_time(func, *args, **kwargs):
    """Wall-clock time of func(*args, **kwargs). Returns (result, seconds)."""
    t0 = time.time()
    result = func(*args, **kwargs)
    return result, time.time() - t0


def _release_memory():
    """
    Best-effort memory release between measurements.
    Clears Python reference cycles, Python caches, and PyTorch's CPU pool.
    Does NOT guarantee the OS reclaims pages — RSS may remain high on macOS.
    """
    gc.collect(); gc.collect(); gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    # On Linux: ask glibc to release freed pages back to OS
    try:
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except Exception:
        pass   # macOS / Windows: no-op


def measure_memory_cpu(func, *args, **kwargs):
    """
    Estimate peak CPU memory increase caused by func().

    ── Three complementary strategies ──────────────────────────────────────────
    1. RSS polling  (10 ms interval, system-level)
       Peak = max(RSS during func) − RSS just before func.
       ⚠ Can read ZERO for models run after a heavier model, because PyTorch's
         malloc cache retains pages.  The RSS doesn't grow if the allocator
         satisfies the request from its own pool.

    2. OS High-Water Mark (HWM) delta  — most reliable for C-level allocs
       hwm = resource.getrusage().ru_maxrss  (bytes on macOS, KB on Linux)
       HWM is the lifetime maximum RSS and never decreases.
       delta = hwm_after − hwm_before.
       ⚠ Also reads ZERO when models share the same pool (same reason as RSS).

    3. tracemalloc  (Python allocator only)
       Tracks numpy arrays, Python lists, etc.  Misses PyTorch C-tensor allocs.

    ── Why memory can appear near-zero for later models ────────────────────────
    PyTorch maintains a CPU malloc cache.  Once DNN runs and allocates, say,
    728 MB, those pages stay in the process RSS.  When MLKM runs next, it draws
    from the same pool without requesting new pages from the OS.  All three
    metrics above see near-zero increase — even though MLKM is genuinely using
    ~150–200 MB of working memory.

    For isolated measurements, call _release_memory() before this function
    and allow a brief gc settle.  For a true isolated reading, the only reliable
    method is to run the function in a fresh subprocess.

    Returns
    -------
    result      : return value of func
    delta_mb    : net RSS change after − before (can be negative)
    peak_mb     : best estimate of peak memory increase above pre-call baseline
                  = max(rss_poll_peak, tracemalloc_peak, hwm_delta)
    """
    _release_memory()

    process    = psutil.Process(os.getpid())
    mem_before = process.memory_info().rss

    # OS high-water mark (lifetime max RSS; never decreases)
    _hwm_factor = 1 if sys.platform == 'darwin' else 1024   # macOS: bytes, Linux: KB
    hwm_before  = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss

    peak_rss  = [mem_before]
    stop_evt  = threading.Event()

    def _monitor():
        while not stop_evt.is_set():
            try:
                cur = process.memory_info().rss
                if cur > peak_rss[0]:
                    peak_rss[0] = cur
            except Exception:
                pass
            stop_evt.wait(0.01)

    tracemalloc.start()
    mon_thread = threading.Thread(target=_monitor, daemon=True)
    mon_thread.start()

    result    = func(*args, **kwargs)

    mem_after  = process.memory_info().rss
    hwm_after  = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    stop_evt.set()
    mon_thread.join()
    if mem_after > peak_rss[0]:
        peak_rss[0] = mem_after

    _, peak_bytes_tm = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    delta_mb        = (mem_after  - mem_before) / 1024**2
    peak_rss_mb     = max((peak_rss[0] - mem_before) / 1024**2, 0.0)
    peak_tm_mb      = peak_bytes_tm / 1024**2
    peak_hwm_mb     = max((hwm_after - hwm_before) / _hwm_factor / 1024**2, 0.0)

    # Best estimate: take the maximum of all three approaches.
    # Note: when a model reuses an earlier model's memory pool all three
    # may read near-zero.  That is expected, not a bug in this function.
    peak_mb = max(peak_rss_mb, peak_tm_mb, peak_hwm_mb)

    return result, delta_mb, peak_mb


def measure_memory_gpu(func, *args, **kwargs):
    """Peak GPU memory during func(). Returns (result, peak_MB or None)."""
    if not torch.cuda.is_available():
        return func(*args, **kwargs), None
    torch.cuda.reset_peak_memory_stats()
    result  = func(*args, **kwargs)
    peak_mb = torch.cuda.max_memory_allocated() / 1024**2
    return result, peak_mb

# ────────────────────────────────────────────────────────────
# Cell 6  [msd-cell-06]
# ────────────────────────────────────────────────────────────
# ── Conformal Prediction (batched for large test sets) ────────────────────────
def conformal_prediction_split_batch(
    net, device,
    calibration_x, calibration_y,
    test_x, test_y,
    alpha=0.05,
    batch_size=1024
):
    """
    Split conformal prediction for regression (batched for efficiency).

    Scores: R_i = |y_i - f(x_i)|  for calibration points.
    Quantile: q_hat = ceil((m+1)(1-alpha))-th smallest score.
    Intervals: [f(x) - q_hat, f(x) + q_hat].

    Returns
    -------
    coverage : float
    avg_interval_length : float   (= 2 * q_hat for homoskedastic)
    q_hat : float
    """
    net.eval()

    # ── Step 1: Calibration scores (batched) ─────────────────────────────────
    cal_preds = []
    with torch.no_grad():
        for i in range(0, len(calibration_x), batch_size):
            xb = torch.from_numpy(calibration_x[i:i+batch_size]).float().to(device)
            cal_preds.append(net(xb).squeeze().cpu().numpy())
    cal_preds = np.concatenate(cal_preds)
    scores = np.abs(calibration_y - cal_preds)

    # ── Step 2: Conformal quantile ────────────────────────────────────────────
    m = len(scores)
    sorted_scores = np.sort(scores)
    k = int(np.ceil((m + 1) * (1 - alpha)))
    k = min(max(k, 1), m)
    q_hat = float(sorted_scores[k - 1])

    # ── Step 3: Test coverage (batched) ──────────────────────────────────────
    test_preds = []
    with torch.no_grad():
        for i in range(0, len(test_x), batch_size):
            xb = torch.from_numpy(test_x[i:i+batch_size]).float().to(device)
            test_preds.append(net(xb).squeeze().cpu().numpy())
    test_preds = np.concatenate(test_preds)

    lower = test_preds - q_hat
    upper = test_preds + q_hat
    coverage = float(np.mean((lower <= test_y) & (test_y <= upper)))
    avg_interval_length = 2.0 * q_hat

    return coverage, avg_interval_length, q_hat

# ────────────────────────────────────────────────────────────
# Cell 7  [msd-cell-07]
# ────────────────────────────────────────────────────────────
# ── Random Fourier Feature ────────────────────────────────────────────────────
def _sample_1d(pdf, gamma):
    if pdf == 'G':
        return torch.randn(1) * gamma
    elif pdf == 'L':
        return torch.distributions.Laplace(torch.tensor([0.0]), torch.tensor([1.0])).sample() * gamma
    elif pdf == 'C':
        return torch.distributions.Cauchy(torch.tensor([0.0]), torch.tensor([1.0])).sample() * gamma

def _sample(pdf, gamma, d):
    return torch.tensor([_sample_1d(pdf, gamma) for _ in range(d)])

class RandomFourierFeature:
    """Random Fourier Feature mapping x (n,d) → phi(x) (n,D)."""
    def __init__(self, d, D, kernel='G', gamma=1.0):
        self.d, self.D, self.gamma = d, D, gamma
        kernel = kernel.upper()
        if kernel not in ('G', 'L', 'C'):
            raise ValueError('kernel must be G, L, or C')
        self.kernel = kernel
        self.b = torch.rand(D) * 2 * torch.pi
        self.W = _sample(kernel, gamma, d * D).reshape(D, d)

    def transform(self, x):
        # x: (n, d)  →  result: (n, D)
        ones = torch.ones(len(x), device=x.device).reshape(1, -1)   # (1, n)
        W = self.W.to(x.device)
        b = self.b.to(x.device)
        result = torch.sqrt(torch.tensor(2.0 / self.D, device=x.device)) * \
                 torch.cos(W @ x.T + b.reshape(-1, 1) * ones)
        return result.T

# ────────────────────────────────────────────────────────────
# Cell 8  [msd-cell-08]
# ────────────────────────────────────────────────────────────
# ── Weight initialisation ─────────────────────────────────────────────────────
def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.uniform_(m.weight, -0.1, 0.1)
        if m.bias is not None:
            nn.init.zeros_(m.bias)

# ────────────────────────────────────────────────────────────
# Cell 9  [msd-cell-09]
# ────────────────────────────────────────────────────────────
# ── DNN ───────────────────────────────────────────────────────────────────────
class Net(nn.Module):
    def __init__(self, h1=HIDDEN_DIM1, h2=HIDDEN_DIM2, h3=HIDDEN_DIM3):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(INPUT_DIM, h1), nn.ReLU(),
            nn.Linear(h1, h2),        nn.ReLU(),
            nn.Linear(h2, h3),        nn.ReLU(),
            nn.Linear(h3, 1)
        )
    def forward(self, x):
        return self.layers(x)


def run_dnn(train_x, train_y, test_x, test_y,
            h1=HIDDEN_DIM1, h2=HIDDEN_DIM2, h3=HIDDEN_DIM3,
            batch_size=BATCH_SIZE, max_epochs=MAX_EPOCHS,
            early_stop_window=EARLY_STOP_WINDOW,
            lr=LR, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY,
            device=None, verbose=False, seed=SEED):
    set_global_seed(seed)
    if device is None:
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    nntrain_x = torch.from_numpy(train_x).float()
    nntrain_y = torch.from_numpy(train_y).float().squeeze()
    train_loader = DataLoader(mydataset(nntrain_x, nntrain_y),
                              batch_size=batch_size, shuffle=True)

    net = Net(h1, h2, h3).to(device)
    net.apply(init_weights)
    criterion = nn.MSELoss()
    optimizer = optim.SGD(net.parameters(), lr=lr,
                          momentum=momentum, weight_decay=weight_decay)

    trainloss, testloss = [], []
    for epoch in range(max_epochs):
        net.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(net(x).squeeze(), y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), GRAD_CLIP)
            optimizer.step()

        net.eval()
        with torch.no_grad():
            p_tr = net(torch.from_numpy(train_x).float().to(device)).squeeze().cpu().numpy()
            trainloss.append(mean_squared_error(train_y, p_tr))
            p_te = net(torch.from_numpy(test_x).float().to(device)).squeeze().cpu().numpy()
            testloss.append(mean_squared_error(test_y, p_te))

        if epoch > early_stop_window:
            if trainloss[-1] > max(trainloss[-early_stop_window:-1]):
                if verbose: print(f'DNN early stop @ epoch {epoch}')
                break
        if verbose and epoch % 100 == 0:
            print(f'epoch {epoch:4d}  train={trainloss[-1]:.4f}  test={testloss[-1]:.4f}')

    return trainloss[-1], testloss[-1], len(trainloss), net, optimizer, trainloss, testloss, device

# ────────────────────────────────────────────────────────────
# Cell 10  [msd-cell-10]
# ────────────────────────────────────────────────────────────
# ── ResNet ────────────────────────────────────────────────────────────────────
class ResidualBlock(nn.Module):
    """Parallel-branch residual block: ReLU(fc1(x) + fc2(x))."""
    def __init__(self, in_f, out_f):
        super().__init__()
        self.fc1 = nn.Linear(in_f, out_f)
        self.fc2 = nn.Linear(in_f, out_f)
    def forward(self, x):
        return F.relu(F.relu(self.fc1(x)) + self.fc2(x))


class ResNet(nn.Module):
    def __init__(self, h1=HIDDEN_DIM1, h2=HIDDEN_DIM2, h3=HIDDEN_DIM3):
        super().__init__()
        self.blocks = nn.Sequential(
            ResidualBlock(INPUT_DIM, h1),
            ResidualBlock(h1, h2),
            ResidualBlock(h2, h3)
        )
        self.out = nn.Linear(h3, 1)
    def forward(self, x):
        return self.out(self.blocks(x))


def run_resnet(train_x, train_y, test_x, test_y,
               h1=HIDDEN_DIM1, h2=HIDDEN_DIM2, h3=HIDDEN_DIM3,
               batch_size=BATCH_SIZE, max_epochs=MAX_EPOCHS,
               early_stop_window=EARLY_STOP_WINDOW,
               lr=LR, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY,
               device=None, verbose=False, seed=SEED):
    set_global_seed(seed)
    if device is None:
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    nntrain_x = torch.from_numpy(train_x).float()
    nntrain_y = torch.from_numpy(train_y).float().squeeze()
    train_loader = DataLoader(mydataset(nntrain_x, nntrain_y),
                              batch_size=batch_size, shuffle=True)

    net = ResNet(h1, h2, h3).to(device)
    net.apply(init_weights)
    criterion = nn.MSELoss()
    optimizer = optim.SGD(net.parameters(), lr=lr,
                          momentum=momentum, weight_decay=weight_decay)

    trainloss, testloss = [], []
    for epoch in range(max_epochs):
        net.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(net(x).squeeze(), y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), GRAD_CLIP)
            optimizer.step()

        net.eval()
        with torch.no_grad():
            p_tr = net(torch.from_numpy(train_x).float().to(device)).squeeze().cpu().numpy()
            trainloss.append(mean_squared_error(train_y, p_tr))
            p_te = net(torch.from_numpy(test_x).float().to(device)).squeeze().cpu().numpy()
            testloss.append(mean_squared_error(test_y, p_te))

        if epoch > early_stop_window:
            if trainloss[-1] > max(trainloss[-early_stop_window:-1]):
                if verbose: print(f'ResNet early stop @ epoch {epoch}')
                break
        if verbose and epoch % 100 == 0:
            print(f'epoch {epoch:4d}  train={trainloss[-1]:.4f}  test={testloss[-1]:.4f}')

    return trainloss[-1], testloss[-1], len(trainloss), net, optimizer, trainloss, testloss, device

# ────────────────────────────────────────────────────────────
# Cell 11  [msd-cell-11]
# ────────────────────────────────────────────────────────────
# ── Deep Kernel Learning (DKL) ────────────────────────────────────────────────
class DKLFeatureExtractor(nn.Module):
    def __init__(self, in_dim=INPUT_DIM, h1=HIDDEN_DIM1, h2=HIDDEN_DIM2,
                 h3=HIDDEN_DIM3, out_dim=2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, h1), nn.ReLU(),
            nn.Linear(h1, h2),    nn.ReLU(),
            nn.Linear(h2, h3),    nn.ReLU(),
            nn.Linear(h3, out_dim)
        )
    def forward(self, x):
        return self.net(x)


class DKLExactGP(ExactGP):
    def __init__(self, train_x, train_y, likelihood, feature_extractor):
        super().__init__(train_x, train_y, likelihood)
        self.feature_extractor = feature_extractor
        self.mean_module  = ConstantMean()
        self.covar_module = ScaleKernel(RBFKernel())

    def forward(self, x):
        feat = self.feature_extractor(x)
        mean  = self.mean_module(feat)
        covar = self.covar_module(feat)
        return MultivariateNormal(mean, covar)


def run_dkl(train_x, train_y, test_x, test_y,
            h1=HIDDEN_DIM1, h2=HIDDEN_DIM2, h3=HIDDEN_DIM3,
            training_iter=300, pred_batch=2048, seed=SEED):
    """Deep Kernel Learning. Returns (train_mse, test_mse).

    ExactGP complexity is O(n^3) in training and O(n) per test point.
    dkl_max_n caps the training set so Cholesky factorisation stays feasible.
    pred_batch batches test-set prediction to avoid OOM on large test sets.
    """
    set_global_seed(seed)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # ── Cap training data (ExactGP is O(n^3)) ────────────────────────────────
    # if len(train_x) > dkl_max_n:
    #     rng = np.random.default_rng(seed)
    #     idx = rng.choice(len(train_x), dkl_max_n, replace=False)
    #     train_x_dkl, train_y_dkl = train_x[idx], train_y[idx]
    #     print(f'  DKL: subsampled {len(train_x)} → {dkl_max_n} training points (ExactGP cap)')
    # else:
    #     
    train_x_dkl, train_y_dkl = train_x, train_y

    train_x_t = torch.from_numpy(train_x_dkl).float().to(device)
    train_y_t = torch.from_numpy(train_y_dkl).float().squeeze().to(device)

    feat       = DKLFeatureExtractor(INPUT_DIM, h1, h2, h3).to(device)
    likelihood = GaussianLikelihood().to(device)
    model      = DKLExactGP(train_x_t, train_y_t, likelihood, feat).to(device)

    model.train(); likelihood.train()
    optimizer = torch.optim.Adam([
        {'params': model.feature_extractor.parameters(), 'lr': 1e-3},
        {'params': model.covar_module.parameters()},
        {'params': model.mean_module.parameters()},
        {'params': likelihood.parameters()},
    ], lr=0.05)
    mll = ExactMarginalLogLikelihood(likelihood, model)

    for _ in range(training_iter):
        optimizer.zero_grad()
        loss = -mll(model(train_x_t), train_y_t)
        loss.backward()
        optimizer.step()

    model.eval(); likelihood.eval()
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        pred_tr = likelihood(model(train_x_t)).mean.cpu().numpy()
        # Batched test prediction (test set can be very large)
        chunks = []
        for i in range(0, len(test_x), pred_batch):
            xb = torch.from_numpy(test_x[i:i+pred_batch]).float().to(device)
            chunks.append(likelihood(model(xb)).mean.cpu())
        pred_te = torch.cat(chunks).numpy()

    train_mse = mean_squared_error(train_y_dkl, pred_tr)
    test_mse  = mean_squared_error(test_y,  pred_te)
    return train_mse, test_mse


# ────────────────────────────────────────────────────────────
# Cell 12  [msd-cell-12]
# ────────────────────────────────────────────────────────────
# ── Deep Gaussian Process (DGP) ───────────────────────────────────────────────
class DGPLayer(gpytorch.models.deep_gps.DeepGPLayer):
    def __init__(self, input_dims, output_dims, inducing_points):
        if output_dims is not None:
            batch_shape = torch.Size([output_dims])
        else:
            batch_shape = torch.Size([])

        var_strat = variational.CholeskyVariationalDistribution(
            inducing_points.size(-2), batch_shape=batch_shape)
        var_strat = variational.VariationalStrategy(
            self, inducing_points, var_strat, learn_inducing_locations=True)

        super().__init__(var_strat, input_dims, output_dims)
        self.mean_module  = ConstantMean(batch_shape=batch_shape)
        self.covar_module = ScaleKernel(
            RBFKernel(batch_shape=batch_shape, ard_num_dims=input_dims),
            batch_shape=batch_shape
        )

    def forward(self, x):
        return MultivariateNormal(self.mean_module(x), self.covar_module(x))


def run_dgp(train_x, train_y, test_x, test_y,
            training_iter=1000, batch_size=256,
            num_inducing=128, hidden_dim=5, pred_batch=1024, seed=SEED):
    """Deep Gaussian Process. Returns (train_mse, test_mse).

    pred_batch batches prediction to avoid OOM on large datasets
    (DGP with num_likelihood_samples=16 multiplies memory by 16x).
    """
    set_global_seed(seed)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    train_x_t = torch.from_numpy(train_x).float().to(device)
    train_y_t = torch.from_numpy(train_y).float().squeeze().to(device)

    class TwoLayerDGP(gpytorch.models.deep_gps.DeepGP):
        def __init__(self):
            super().__init__()
            ind1 = train_x_t[torch.randperm(train_x_t.size(0))[:num_inducing]]
            ind2 = torch.randn(num_inducing, hidden_dim).to(device)
            self.hidden_layer = DGPLayer(train_x_t.shape[1], hidden_dim, ind1)
            self.output_layer = DGPLayer(hidden_dim, None, ind2)
            self.likelihood   = GaussianLikelihood().to(device)

        def forward(self, x):
            return self.output_layer(self.hidden_layer(x))

        def predict(self, x):
            self.eval(); self.likelihood.eval()
            with torch.no_grad(), gpytorch.settings.num_likelihood_samples(16):
                return self.likelihood(self(x)).mean.mean(dim=0)

        def predict_batched(self, x_t, bs=pred_batch):
            """Batched prediction for large datasets."""
            chunks = []
            for i in range(0, x_t.size(0), bs):
                chunks.append(self.predict(x_t[i:i+bs]).cpu())
            return torch.cat(chunks)

    dgp = TwoLayerDGP().to(device)
    dgp.train(); dgp.likelihood.train()
    optimizer = torch.optim.Adam(dgp.parameters(), lr=0.01)
    elbo = gpytorch.mlls.DeepApproximateMLL(
        gpytorch.mlls.VariationalELBO(
            dgp.likelihood, dgp, num_data=train_x_t.size(0)))

    for _ in range(training_iter):
        perm = torch.randperm(train_x_t.size(0), device=device)
        for j in range(0, train_x_t.size(0), batch_size):
            idx = perm[j:j+batch_size]
            optimizer.zero_grad()
            loss = -elbo(dgp(train_x_t[idx]), train_y_t[idx])
            loss.backward()
            optimizer.step()

    # Use batched prediction to avoid OOM on large test sets
    train_mse = mean_squared_error(train_y, dgp.predict_batched(train_x_t).numpy())
    test_x_t  = torch.from_numpy(test_x).float().to(device)
    test_mse  = mean_squared_error(test_y,  dgp.predict_batched(test_x_t).numpy())
    return train_mse, test_mse


# ────────────────────────────────────────────────────────────
# Cell 13  [msd-cell-13]
# ────────────────────────────────────────────────────────────
# ── RF + Ridge ────────────────────────────────────────────────────────────────
def run_rf_ridge(train_x, train_y, test_x, test_y,
                 D=500, gamma=0.4, kernel='G', seed=SEED):
    """Random Fourier Features + Ridge regression. Returns (train_mse, test_mse)."""
    set_global_seed(seed)
    rff = RandomFourierFeature(train_x.shape[1], D, kernel=kernel, gamma=gamma)

    x_tr = torch.from_numpy(train_x).float()
    x_te = torch.from_numpy(test_x).float()
    with torch.no_grad():
        phi_tr = rff.transform(x_tr).numpy()
        phi_te = rff.transform(x_te).numpy()

    ridge = RidgeCV(alphas=[1e-4, 1e-3, 1e-2, 1e-1, 1.0],
                    cv=KFold(n_splits=5, shuffle=True, random_state=seed))
    ridge.fit(phi_tr, train_y)

    train_mse = mean_squared_error(train_y, ridge.predict(phi_tr))
    test_mse  = mean_squared_error(test_y,  ridge.predict(phi_te))
    return train_mse, test_mse

# ────────────────────────────────────────────────────────────
# Cell 14  [msd-cell-14]
# ────────────────────────────────────────────────────────────
# ── MLKM (Multi-Layer Kernel Machine) ─────────────────────────────────────────
def run_mlkm(train_x, train_y, test_x, test_y,
             D1=500, D2=500, D3=500,
             h1=HIDDEN_DIM1, h2=HIDDEN_DIM2, h3=HIDDEN_DIM3,
             batch_size=BATCH_SIZE, max_epochs=MAX_EPOCHS,
             early_stop_window=EARLY_STOP_WINDOW,
             lr=LR, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY,
             device=None, verbose=False, seed=SEED):
    """
    MLKM: each layer applies RFF then a learned linear map.
    Architecture: INPUT --rff1--> D1 --fc1--> h1 --rff2--> D2 --fc2--> h2
                       --rff3--> D3 --fc3--> h3 --out--> 1
    RFF objects are local to this function call (no closure rebinding risk).
    """
    set_global_seed(seed)
    if device is None:
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    nntrain_x = torch.from_numpy(train_x).float()
    nntrain_y = torch.from_numpy(train_y).float().squeeze()
    train_loader = DataLoader(mydataset(nntrain_x, nntrain_y),
                              batch_size=batch_size, shuffle=True)

    rff1 = RandomFourierFeature(INPUT_DIM, D1, kernel='G', gamma=0.1)
    rff2 = RandomFourierFeature(h1, D2, kernel='G', gamma=0.4)
    rff3 = RandomFourierFeature(h2, D3, kernel='G', gamma=0.4)

    class KernelNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(D1, h1)
            self.fc2 = nn.Linear(D2, h2)
            self.fc3 = nn.Linear(D3, h3)
            self.out = nn.Linear(h3, 1)
        def forward(self, x):
            x = self.fc1(rff1.transform(x))
            x = self.fc2(rff2.transform(x))
            x = self.fc3(rff3.transform(x))
            return self.out(x)

    net = KernelNet().to(device)
    net.apply(init_weights)
    criterion = nn.MSELoss()
    optimizer = optim.SGD(net.parameters(), lr=lr,
                          momentum=momentum, weight_decay=weight_decay)

    trainloss, testloss = [], []
    for epoch in range(max_epochs):
        net.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(net(x).squeeze(), y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), GRAD_CLIP)
            optimizer.step()

        net.eval()
        with torch.no_grad():
            p_tr = net(torch.from_numpy(train_x).float().to(device)).squeeze().cpu().numpy()
            trainloss.append(mean_squared_error(train_y, p_tr))
            p_te = net(torch.from_numpy(test_x).float().to(device)).squeeze().cpu().numpy()
            testloss.append(mean_squared_error(test_y, p_te))

        if epoch > early_stop_window:
            if trainloss[-1] > max(trainloss[-early_stop_window:-1]):
                if verbose: print(f'MLKM early stop @ epoch {epoch}')
                break
        if verbose and epoch % 100 == 0:
            print(f'epoch {epoch:4d}  train={trainloss[-1]:.4f}  test={testloss[-1]:.4f}')

    return trainloss[-1], testloss[-1], len(trainloss), net, optimizer, trainloss, testloss, device

# ────────────────────────────────────────────────────────────
# Cell 15  [msd-cell-15]
# ────────────────────────────────────────────────────────────
# ── ResKernelNet (Residual Kernel Machine) ─────────────────────────────────────
def run_reskernelnet(train_x, train_y, test_x, test_y,
                     D1=500,
                     h1=HIDDEN_DIM1, h2=HIDDEN_DIM2, h3=HIDDEN_DIM3,
                     batch_size=BATCH_SIZE, max_epochs=MAX_EPOCHS,
                     early_stop_window=EARLY_STOP_WINDOW,
                     lr=LR, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY,
                     device=None, verbose=False, seed=SEED):
    """
    ResKernelNet:
      block_l(x) = A_l x + Delta_l x + W_l phi_l(x)
    Architecture:
      INPUT --block1(D1,h1)--> h1 --block2(D2=h1,h2)--> h2
            --block3(D3=h2,h3)--> h3 --out--> 1
    Note: D2=h1 and D3=h2 so that RFF dimensions match hidden dims.
    """
    set_global_seed(seed)
    if device is None:
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    D2 = h1   # must match for residual connections
    D3 = h2

    nntrain_x = torch.from_numpy(train_x).float()
    nntrain_y = torch.from_numpy(train_y).float().squeeze()
    train_loader = DataLoader(mydataset(nntrain_x, nntrain_y),
                              batch_size=batch_size, shuffle=True)

    rff1 = RandomFourierFeature(INPUT_DIM, D1, kernel='G', gamma=0.1)
    rff2 = RandomFourierFeature(h1, D2, kernel='G', gamma=0.4)
    rff3 = RandomFourierFeature(h2, D3, kernel='G', gamma=0.4)

    class ResKernelBlock(nn.Module):
        def __init__(self, d_in, d_phi, d_out, rff):
            super().__init__()
            self.rff   = rff
            self.W     = nn.Linear(d_phi, d_out)
            self.A     = nn.Identity() if d_in == d_out else nn.Linear(d_in, d_out, bias=False)
            self.Delta = nn.Linear(d_in, d_out, bias=False)
            nn.init.zeros_(self.Delta.weight)
        def forward(self, x):
            return self.A(x) + self.Delta(x) + self.W(self.rff.transform(x))

    class ResKernelNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.b1  = ResKernelBlock(INPUT_DIM, D1, h1, rff1)
            self.b2  = ResKernelBlock(h1, D2, h2, rff2)
            self.b3  = ResKernelBlock(h2, D3, h3, rff3)
            self.out = nn.Linear(h3, 1)
        def forward(self, x):
            return self.out(self.b3(self.b2(self.b1(x))))

    net = ResKernelNet().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.SGD(net.parameters(), lr=lr,
                          momentum=momentum, weight_decay=weight_decay)

    trainloss, testloss = [], []
    for epoch in range(max_epochs):
        net.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(net(x).squeeze(), y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), GRAD_CLIP)
            optimizer.step()

        net.eval()
        with torch.no_grad():
            p_tr = net(torch.from_numpy(train_x).float().to(device)).squeeze().cpu().numpy()
            trainloss.append(mean_squared_error(train_y, p_tr))
            p_te = net(torch.from_numpy(test_x).float().to(device)).squeeze().cpu().numpy()
            testloss.append(mean_squared_error(test_y, p_te))

        if epoch > early_stop_window:
            if trainloss[-1] > max(trainloss[-early_stop_window:-1]):
                if verbose: print(f'ResKernelNet early stop @ epoch {epoch}')
                break
        if verbose and epoch % 100 == 0:
            print(f'epoch {epoch:4d}  train={trainloss[-1]:.4f}  test={testloss[-1]:.4f}')

    return trainloss[-1], testloss[-1], len(trainloss), net, optimizer, trainloss, testloss, device

# ────────────────────────────────────────────────────────────
# Cell 17  [5wp9ckaaa43]
# ────────────────────────────────────────────────────────────
# ── Section 2 helper: run & visualise one fold ────────────────────────────────

def _plot_fold(fold_idx, row, curves, conf_rows,
               save_prefix='msd_fold', save_dir='msd_results'):
    """
    Produce three figures for one fold — mirrors the original single-run plots:
      Fig 1 · Training curves  (train & test MSE vs epoch, 4 neural models)
      Fig 2 · Method comparison bar charts  (test MSE / train MSE / runtime / CPU peak)
      Fig 3 · Conformal prediction  (coverage & interval length, 4 neural models)
    All figures are saved to save_dir/.
    """
    import matplotlib.pyplot as plt
    import numpy as np

    os.makedirs(save_dir, exist_ok=True)
    prefix = os.path.join(save_dir, f'{save_prefix}{fold_idx:02d}')

    # ── Fig 1: Training curves ─────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    palette = {'DNN':'tab:blue','ResNet':'tab:green',
               'MLKM':'tab:red','RKN':'tab:purple'}
    for name, (trl, tel) in curves.items():
        c = palette.get(name, 'gray')
        axes[0].plot(trl, color=c, label=name)
        axes[1].plot(tel, color=c, label=name)
    axes[0].set_title('Training Loss (MSE)'); axes[0].set_xlabel('Epoch')
    axes[1].set_title('Test Loss (MSE)');     axes[1].set_xlabel('Epoch')
    for ax in axes:
        ax.set_ylabel('MSE'); ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
    plt.suptitle(f'Fold {fold_idx} — Training Curves  '
                 f'(n_train={row["n_train"]}, n_test={row["n_test"]})', fontsize=11)
    plt.tight_layout()
    plt.savefig(f'{prefix}_training_curves.png', dpi=120)
    plt.show()

    # ── Fig 2: Method comparison bar charts ───────────────────────────────────
    all_methods = ['RF+Ridge','DNN','ResNet','DKL','DGP','MLKM','ResKernelNet']
    key_map     = {'RF+Ridge':     ('RF_train',     'RF_test',     'RF_time',     'RF_cpu_peak'),
                   'DNN':          ('DNN_train',    'DNN_test',    'DNN_time',    'DNN_cpu_peak'),
                   'ResNet':       ('ResNet_train', 'ResNet_test', 'ResNet_time', 'ResNet_cpu_peak'),
                   'DKL':          ('DKL_train',    'DKL_test',    'DKL_time',    'DKL_cpu_peak'),
                   'DGP':          ('DGP_train',    'DGP_test',    'DGP_time',    'DGP_cpu_peak'),
                   'MLKM':         ('MLKM_train',   'MLKM_test',   'MLKM_time',   'MLKM_cpu_peak'),
                   'ResKernelNet': ('RKN_train',    'RKN_test',    'RKN_time',    'RKN_cpu_peak')}
    colors = plt.cm.tab10(np.linspace(0, 0.9, len(all_methods)))
    x = np.arange(len(all_methods))
    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    for ax, (tr_or_te, label, logy) in zip(
        axes.flatten(),
        [('test', 'Test MSE', True), ('train', 'Train MSE', True),
         ('time', 'Runtime (s)', True), ('cpu', 'CPU Peak (MB)', False)]
    ):
        idx = {'test':1,'train':0,'time':2,'cpu':3}[tr_or_te]
        vals = [row[key_map[m][idx]] for m in all_methods]
        bars = ax.bar(x, vals, color=colors)
        ax.set_xticks(x); ax.set_xticklabels(all_methods, rotation=30, ha='right', fontsize=9)
        ax.set_title(label); ax.set_ylabel(label)
        if logy and min(v for v in vals if v > 0) > 0:
            ax.set_yscale('log')
        ax.grid(True, alpha=0.25, axis='y')
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                    f'{v:.3f}' if v < 10 else f'{v:.1f}',
                    ha='center', va='bottom', fontsize=7)
    plt.suptitle(f'Fold {fold_idx} — Method Comparison', fontsize=12)
    plt.tight_layout()
    plt.savefig(f'{prefix}_method_comparison.png', dpi=120)
    plt.show()

    # ── Fig 3: Conformal prediction ────────────────────────────────────────────
    if conf_rows:
        conf_names = [r['Method'] for r in conf_rows]
        coverages  = [r['Coverage'] for r in conf_rows]
        intervals  = [r['Interval'] for r in conf_rows]
        x3 = np.arange(len(conf_names))
        fig, axes = plt.subplots(1, 2, figsize=(11, 4))
        axes[0].bar(x3, coverages, color='steelblue', alpha=0.8)
        axes[0].axhline(0.95, color='red', ls='--', label='Target 95%')
        axes[0].set_xticks(x3); axes[0].set_xticklabels(conf_names, rotation=20)
        axes[0].set_title('Conformal Coverage'); axes[0].set_ylabel('Coverage')
        axes[0].set_ylim(0.8, 1.05); axes[0].legend()
        axes[1].bar(x3, intervals, color='coral', alpha=0.8)
        axes[1].set_xticks(x3); axes[1].set_xticklabels(conf_names, rotation=20)
        axes[1].set_title('Conformal Avg Interval Length'); axes[1].set_ylabel('Interval')
        plt.suptitle(f'Fold {fold_idx} — Conformal Prediction (α=0.05)', fontsize=12)
        plt.tight_layout()
        plt.savefig(f'{prefix}_conformal.png', dpi=120)
        plt.show()

    print(f'  Plots saved → {save_dir}/{save_prefix}{fold_idx:02d}_*.png')


def run_one_fold(fold_idx, X_fold, y_fold, D_mlkm=128, seed=SEED,
                 plot=True, save_dir='msd_results'):
    """
    Run all 7 methods on one data fold, save per-fold CSV + plots, return metrics dict.

    Split: 70% train · 15% calibration · 15% test  (sequential, no shuffle).
    Scalers (X and y) are fitted on the training set only to prevent leakage.
    Results CSV and all plots are written to save_dir/.

    Parameters
    ----------
    fold_idx          : int   fold number (1-based)
    X_fold, y_fold    : arrays  raw (unscaled) fold data
    D_mlkm            : int   RFF dimension for MLKM and ResKernelNet
    seed              : int   random seed
    plot              : bool  save per-fold figures
    save_dir          : str   directory for plots and per-fold CSV

    Returns
    -------
    dict with all metrics (flat)
    """
    os.makedirs(save_dir, exist_ok=True)

    n_fold  = len(X_fold)
    n_train = int(n_fold * 0.70)
    n_calib = int(n_fold * 0.15)

    X_tr, y_tr = X_fold[:n_train],                  y_fold[:n_train]
    X_c,  y_c  = X_fold[n_train:n_train+n_calib],   y_fold[n_train:n_train+n_calib]
    X_te, y_te = X_fold[n_train+n_calib:],           y_fold[n_train+n_calib:]

    # Normalise — scaler fitted on train only
    sx = StandardScaler()
    X_tr = sx.fit_transform(X_tr).astype(np.float32)
    X_c  = sx.transform(X_c).astype(np.float32)
    X_te = sx.transform(X_te).astype(np.float32)
    sy = StandardScaler()
    y_tr_n = sy.fit_transform(y_tr.reshape(-1,1)).ravel().astype(np.float32)
    y_c_n  = sy.transform(y_c.reshape(-1,1)).ravel().astype(np.float32)
    y_te_n = sy.transform(y_te.reshape(-1,1)).ravel().astype(np.float32)

    row = {'fold': fold_idx,
           'n_train': len(X_tr), 'n_cal': len(X_c), 'n_test': len(X_te)}
    curves = {}

    # ── RF + Ridge ────────────────────────────────────────────────────────────
    _release_memory()
    def _rf(): return run_rf_ridge(X_tr, y_tr_n, X_te, y_te_n, seed=seed)
    ((rf_r, _), _, rf_p), rf_t = measure_time(
        lambda: measure_memory_cpu(lambda: measure_memory_gpu(_rf)))
    row['RF_train'], row['RF_test'] = rf_r
    row['RF_time'],  row['RF_cpu_peak'] = rf_t, rf_p

    # ── DNN ───────────────────────────────────────────────────────────────────
    _release_memory()
    def _dnn(): return run_dnn(X_tr, y_tr_n, X_te, y_te_n, seed=seed)
    ((dnn_r, _), _, dnn_p), dnn_t = measure_time(
        lambda: measure_memory_cpu(lambda: measure_memory_gpu(_dnn)))
    dnn_train, dnn_test, _, dnn_net, _, dnn_trl, dnn_tel, dnn_dev = dnn_r
    row['DNN_train'], row['DNN_test'] = dnn_train, dnn_test
    row['DNN_time'],  row['DNN_cpu_peak'] = dnn_t, dnn_p
    curves['DNN'] = (dnn_trl, dnn_tel)

    # ── ResNet ────────────────────────────────────────────────────────────────
    _release_memory()
    def _res(): return run_resnet(X_tr, y_tr_n, X_te, y_te_n, seed=seed)
    ((res_r, _), _, res_p), res_t = measure_time(
        lambda: measure_memory_cpu(lambda: measure_memory_gpu(_res)))
    res_train, res_test, _, res_net, _, res_trl, res_tel, res_dev = res_r
    row['ResNet_train'], row['ResNet_test'] = res_train, res_test
    row['ResNet_time'],  row['ResNet_cpu_peak'] = res_t, res_p
    curves['ResNet'] = (res_trl, res_tel)

    # ── DKL ───────────────────────────────────────────────────────────────────
    _release_memory()
    def _dkl(): return run_dkl(X_tr, y_tr_n, X_te, y_te_n, seed=seed)
    ((dkl_r, _), _, dkl_p), dkl_t = measure_time(
        lambda: measure_memory_cpu(lambda: measure_memory_gpu(_dkl)))
    row['DKL_train'], row['DKL_test'] = dkl_r
    row['DKL_time'],  row['DKL_cpu_peak'] = dkl_t, dkl_p

    # ── DGP ───────────────────────────────────────────────────────────────────
    _release_memory()
    def _dgp(): return run_dgp(X_tr, y_tr_n, X_te, y_te_n, seed=seed)
    ((dgp_r, _), _, dgp_p), dgp_t = measure_time(
        lambda: measure_memory_cpu(lambda: measure_memory_gpu(_dgp)))
    row['DGP_train'], row['DGP_test'] = dgp_r
    row['DGP_time'],  row['DGP_cpu_peak'] = dgp_t, dgp_p

    # ── MLKM ─────────────────────────────────────────────────────────────────
    _release_memory()
    def _mlkm():
        return run_mlkm(X_tr, y_tr_n, X_te, y_te_n,
                        D1=D_mlkm, D2=D_mlkm, D3=D_mlkm, seed=seed)
    ((mlkm_r, _), _, mlkm_p), mlkm_t = measure_time(
        lambda: measure_memory_cpu(lambda: measure_memory_gpu(_mlkm)))
    mlkm_train, mlkm_test, _, mlkm_net, _, mlkm_trl, mlkm_tel, mlkm_dev = mlkm_r
    row['MLKM_train'], row['MLKM_test'] = mlkm_train, mlkm_test
    row['MLKM_time'],  row['MLKM_cpu_peak'] = mlkm_t, mlkm_p
    curves['MLKM'] = (mlkm_trl, mlkm_tel)

    # ── ResKernelNet ──────────────────────────────────────────────────────────
    _release_memory()
    def _rkn():
        return run_reskernelnet(X_tr, y_tr_n, X_te, y_te_n, D1=D_mlkm, seed=seed)
    ((rkn_r, _), _, rkn_p), rkn_t = measure_time(
        lambda: measure_memory_cpu(lambda: measure_memory_gpu(_rkn)))
    rkn_train, rkn_test, _, rkn_net, _, rkn_trl, rkn_tel, rkn_dev = rkn_r
    row['RKN_train'], row['RKN_test'] = rkn_train, rkn_test
    row['RKN_time'],  row['RKN_cpu_peak'] = rkn_t, rkn_p
    curves['RKN'] = (rkn_trl, rkn_tel)

    # ── Conformal prediction (4 neural models) ────────────────────────────────
    conf_rows = []
    for label, net, dev in [('DNN',    dnn_net,  dnn_dev),
                             ('ResNet', res_net,  res_dev),
                             ('MLKM',   mlkm_net, mlkm_dev),
                             ('RKN',    rkn_net,  rkn_dev)]:
        cov, length, q = conformal_prediction_split_batch(
            net, dev, X_c, y_c_n, X_te, y_te_n)
        conf_rows.append({'Method': label, 'Coverage': cov,
                          'Interval': length, 'q_hat': q})
        row[f'{label}_conf_cov']  = cov
        row[f'{label}_conf_intv'] = length

    # ── Save per-fold CSV immediately ─────────────────────────────────────────
    csv_path = os.path.join(save_dir, f'fold{fold_idx:02d}_metrics.csv')
    pd.DataFrame([row]).to_csv(csv_path, index=False)

    # ── Print fold summary ────────────────────────────────────────────────────
    print(f'\n  ── Fold {fold_idx} results ──')
    print(f'  {"Method":<14} {"Train MSE":>10} {"Test MSE":>10} {"Time(s)":>8} {"CPU MB":>8}')
    print(f'  {"-"*54}')
    for m, tk, tec, tc, cc in [
        ('RF+Ridge',     'RF_train',     'RF_test',     'RF_time',     'RF_cpu_peak'),
        ('DNN',          'DNN_train',    'DNN_test',    'DNN_time',    'DNN_cpu_peak'),
        ('ResNet',       'ResNet_train', 'ResNet_test', 'ResNet_time', 'ResNet_cpu_peak'),
        ('DKL',          'DKL_train',    'DKL_test',    'DKL_time',    'DKL_cpu_peak'),
        ('DGP',          'DGP_train',    'DGP_test',    'DGP_time',    'DGP_cpu_peak'),
        ('MLKM',         'MLKM_train',   'MLKM_test',   'MLKM_time',   'MLKM_cpu_peak'),
        ('ResKernelNet', 'RKN_train',    'RKN_test',    'RKN_time',    'RKN_cpu_peak'),
    ]:
        print(f'  {m:<14} {row[tk]:10.4f} {row[tec]:10.4f} '
              f'{row[tc]:8.1f} {row[cc]:8.1f}')
    print(f'\n  Conformal (α=0.05):')
    for r in conf_rows:
        print(f'  {r["Method"]:<12} coverage={r["Coverage"]:.4f}  '
              f'interval={r["Interval"]:.4f}  q_hat={r["q_hat"]:.4f}')
    print(f'  Metrics saved → {csv_path}')

    if plot:
        _plot_fold(fold_idx, row, curves, conf_rows, save_dir=save_dir)

    return row


print('run_one_fold() and _plot_fold() defined.')



# ════════════════════════════════════════════════════════════════════════════
# MAIN — Fold 1
# ════════════════════════════════════════════════════════════════════════════
import glob, sys

RESULTS_DIR  = 'msd_results'
N_FOLDS      = 10
D_FOLD       = 128
fold_size    = len(X_full) // N_FOLDS

os.makedirs(RESULTS_DIR, exist_ok=True)

k     = 1
start = 0
end   = fold_size

print('=' * 68, flush=True)
print(f'  Running Fold {k}/{N_FOLDS}  (rows {start:,}–{end-1:,})', flush=True)
print('=' * 68, flush=True)
set_global_seed(SEED + k - 1)
_release_memory()
row = run_one_fold(k, X_full[start:end], y_full[start:end],
                   D_mlkm=D_FOLD, seed=SEED + k - 1,
                   plot=True, save_dir=RESULTS_DIR)

pd.DataFrame([row]).to_csv(os.path.join(RESULTS_DIR, 'msd_fold_results_combined.csv'), index=False)
print(f'\nFold 1 complete.  Results → {RESULTS_DIR}/', flush=True)

# ────────────────────────────────────────────────────────────
# Cell 22  [5rqip7ee7l7]  (D-sensitivity)
# ────────────────────────────────────────────────────────────
# ── Section 3: D-Sensitivity Analysis ─────────────────────────────────────────
# Run MLKM and ResKernelNet on Fold 1 with different RFF dimensions D.
# Uses the same 70/15/15 train/calib/test split as run_one_fold.

D_SWEEP    = list(range(64, 513, 32))   # [64, 96, 128, ..., 512]
ALPHA_CONF = 0.05                       # 95 % conformal coverage
FOLD_SEED  = SEED                       # Fold 1 seed

# ── Reconstruct Fold 1 splits (identical to run_one_fold logic) ───────────────
fold_size_ = len(X_full) // N_FOLDS
X_fold1    = X_full[0 : fold_size_]
y_fold1    = y_full[0 : fold_size_]

n_fold1  = len(X_fold1)
n_train1 = int(n_fold1 * 0.70)
n_calib1 = int(n_fold1 * 0.15)

X_tr1 = X_fold1[:n_train1]
X_c1  = X_fold1[n_train1 : n_train1 + n_calib1]
X_te1 = X_fold1[n_train1 + n_calib1 :]
y_tr1 = y_fold1[:n_train1]
y_c1  = y_fold1[n_train1 : n_train1 + n_calib1]
y_te1 = y_fold1[n_train1 + n_calib1 :]

sx1 = StandardScaler()
X_tr1 = sx1.fit_transform(X_tr1).astype(np.float32)
X_c1  = sx1.transform(X_c1).astype(np.float32)
X_te1 = sx1.transform(X_te1).astype(np.float32)

sy1 = StandardScaler()
y_tr1_n = sy1.fit_transform(y_tr1.reshape(-1,1)).ravel().astype(np.float32)
y_c1_n  = sy1.transform(y_c1.reshape(-1,1)).ravel().astype(np.float32)
y_te1_n = sy1.transform(y_te1.reshape(-1,1)).ravel().astype(np.float32)

# Convert to tensors
tr_x  = torch.tensor(X_tr1);  tr_y  = torch.tensor(y_tr1_n)
cal_x = torch.tensor(X_c1);   cal_y = torch.tensor(y_c1_n)
te_x  = torch.tensor(X_te1);  te_y  = torch.tensor(y_te1_n)

print(f'Fold 1 — train: {len(tr_x):,}  calib: {len(cal_x):,}  test: {len(te_x):,}')
print(f'D sweep: {D_SWEEP}')
print()

# ── Helper: conformal half-width from a fitted net ────────────────────────────
def _conf_half(net, device, cal_x, cal_y, te_x, te_y, alpha=ALPHA_CONF):
    """Return mean conformal prediction interval half-width."""
    _, lo, hi = conformal_prediction_split_batch(
        net, device, cal_x, cal_y, te_x, te_y, alpha)
    return float((hi - lo).mean() / 2)

# ── D sweep ───────────────────────────────────────────────────────────────────
d_sweep_results = {'D': [], 'model': [],
                   'train_mse': [], 'test_mse': [],
                   'runtime': [], 'cpu_peak_mb': [], 'conf_half': []}

for D in D_SWEEP:
    print(f'─── D = {D} ───')
    for model_name, run_fn, run_kwargs in [
        ('MLKM',        run_mlkm,        {'D1': D, 'D2': D, 'D3': D}),
        ('ResKernelNet', run_reskernelnet, {'D1': D}),
    ]:
        _release_memory()
        set_global_seed(FOLD_SEED)
        try:
            (tr_mse, te_mse, n_ep, net, optimizer,
             trainloss, testloss, device), _, mem_mb = measure_memory_cpu(
                run_fn, tr_x, tr_y, cal_x, cal_y, te_x, te_y,
                seed=FOLD_SEED, **run_kwargs)
            rt = measure_time(
                run_fn, tr_x, tr_y, cal_x, cal_y, te_x, te_y,
                seed=FOLD_SEED, **run_kwargs)
            chalf = _conf_half(net, device, cal_x, cal_y, te_x, te_y)
        except Exception as e:
            print(f'  {model_name} D={D} FAILED: {e}')
            tr_mse = te_mse = rt = mem_mb = chalf = float('nan')

        d_sweep_results['D'].append(D)
        d_sweep_results['model'].append(model_name)
        d_sweep_results['train_mse'].append(tr_mse)
        d_sweep_results['test_mse'].append(te_mse)
        d_sweep_results['runtime'].append(rt)
        d_sweep_results['cpu_peak_mb'].append(mem_mb)
        d_sweep_results['conf_half'].append(chalf)
        print(f'  {model_name:12s}  train={tr_mse:.4f}  test={te_mse:.4f}'
              f'  time={rt:.1f}s  mem={mem_mb:.1f}MB  CI±={chalf:.4f}')
    print()

d_sweep_df = pd.DataFrame(d_sweep_results)
d_sweep_df.to_csv('msd_d_sweep.csv', index=False)
print('Saved: msd_d_sweep.csv')
print()
print(d_sweep_df.to_string(index=False))


# ────────────────────────────────────────────────────────────
# Cell 23  [etl1qedt9ku]  (D-sensitivity)
# ────────────────────────────────────────────────────────────
# ── Section 3: D-Sensitivity Visualisation ────────────────────────────────────
# Three-panel figure: Test MSE · Runtime · Conformal interval half-width vs D.

fig, axes = plt.subplots(1, 3, figsize=(15, 4))
colors  = {'MLKM': '#2196F3', 'ResKernelNet': '#FF5722'}
markers = {'MLKM': 'o',        'ResKernelNet': 's'}

for model_name in ['MLKM', 'ResKernelNet']:
    sub = d_sweep_df[d_sweep_df['model'] == model_name].sort_values('D')
    c, m = colors[model_name], markers[model_name]

    # ── Panel A: Test MSE vs D ─────────────────────────────────────────────────
    axes[0].plot(sub['D'], sub['test_mse'],
                 marker=m, color=c, linewidth=2, markersize=7, label=model_name)
    axes[0].plot(sub['D'], sub['train_mse'],
                 marker=m, color=c, linewidth=1.5, linestyle='--', markersize=5,
                 alpha=0.6, label=f'{model_name} (train)')

    # ── Panel B: Runtime vs D ──────────────────────────────────────────────────
    axes[1].plot(sub['D'], sub['runtime'],
                 marker=m, color=c, linewidth=2, markersize=7, label=model_name)

    # ── Panel C: Conformal interval half-width vs D ────────────────────────────
    axes[2].plot(sub['D'], sub['conf_half'],
                 marker=m, color=c, linewidth=2, markersize=7, label=model_name)

# ── formatting ─────────────────────────────────────────────────────────────────
axes[0].set_title('Test / Train MSE vs D', fontsize=13)
axes[0].set_xlabel('RFF dimension D');  axes[0].set_ylabel('MSE')
axes[0].legend(fontsize=8);            axes[0].grid(alpha=0.3)

axes[1].set_title('Runtime vs D', fontsize=13)
axes[1].set_xlabel('RFF dimension D');  axes[1].set_ylabel('Time (s)')
axes[1].legend(fontsize=9);            axes[1].grid(alpha=0.3)

axes[2].set_title('Conformal Interval Half-Width vs D\n(90 % coverage)', fontsize=13)
axes[2].set_xlabel('RFF dimension D');  axes[2].set_ylabel('CI half-width (year)')
axes[2].legend(fontsize=9);            axes[2].grid(alpha=0.3)

for ax in axes:
    ax.set_xticks(D_SWEEP)

plt.suptitle('D-Sensitivity Analysis — MSD Fold 1 (MLKM vs ResKernelNet)',
             fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig('msd_d_sweep.png', dpi=150, bbox_inches='tight')
plt.show()
print('Saved: msd_d_sweep.png')

# ── Print compact comparison table ────────────────────────────────────────────
print('\n── D-Sensitivity Summary Table ─────────────────────────────────────')
print(f'{"Model":14s} {"D":>5s}  {"TrainMSE":>10s}  {"TestMSE":>10s}'
      f'  {"Runtime(s)":>12s}  {"CPUPeak(MB)":>12s}  {"CI±":>8s}')
print('─' * 80)
for _, r in d_sweep_df.sort_values(['model', 'D']).iterrows():
    print(f'{r.model:14s} {int(r.D):>5d}  {r.train_mse:10.4f}  {r.test_mse:10.4f}'
          f'  {r.runtime:12.1f}  {r.cpu_peak_mb:12.1f}  {r.conf_half:8.4f}')

# ── D_eff* analysis: smallest D where test MSE within 5 % of best D ───────────
print('\n── Effective D* (within 5 % of best MSE) ─────────────────────────')
for model_name in ['MLKM', 'ResKernelNet']:
    sub = d_sweep_df[d_sweep_df['model'] == model_name].sort_values('D')
    best_mse  = sub['test_mse'].min()
    threshold = 1.05 * best_mse
    ok = sub[sub['test_mse'] <= threshold]
    d_eff = int(ok['D'].iloc[0]) if len(ok) > 0 else D_SWEEP[-1]
    print(f'  {model_name:12s}  best_MSE={best_mse:.4f}  5%-threshold={threshold:.4f}'
          f'  D_eff*={d_eff}')


print("\n\nAll done. Check msd_results/ for all outputs.", flush=True)
