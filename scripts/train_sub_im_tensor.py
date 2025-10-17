import os 
import sys
# Get the absolute path of the parent directory
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(parent_dir)  # Add parent directory to sys.path

# model
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric as tg
import torch_scatter
from typing import Dict, Union

palette = ['#43AA8B', '#F8961E', '#F94144']
sub = str.maketrans("0123456789", "₀₁₂₃₄₅₆₇₈₉")

# data pre-processing and visualization
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import pandas as pd
from scipy.interpolate import interp1d
import math
import e3nn.o3 as o3
from e3nn.util.jit import compile_mode
from e3nn.io import CartesianTensor

# supress error log from font
import logging
logging.getLogger('matplotlib.font_manager').setLevel(level=logging.CRITICAL)
# utilities
import time
from tqdm import tqdm
from utils.utils_data import (load_data, train_valid_test_split, save_or_load_onehot, build_data, plot_spherical_harmonics_comparison, plot_cartesian_tensor_comparison, compute_aniso_mae)
from utils.utils_model_sub_tensor import Network, train
import wandb

from utils.normalize_cart import (
    compute_norm_params, normalize_with_params, denormalize,
    cart_to_sph, sph_to_cart, save_norm_params, load_norm_params
)


plt.rcParams["mathtext.fontset"] = "cm"

bar_format = '{l_bar}{bar:10}{r_bar}{bar:-10b}'
default_dtype = torch.float64
torch.set_default_dtype(default_dtype)

# Create a colormap based on the number of unique symbols
datasets = ['g', 'y', 'r']
colors = dict(zip(datasets, palette))
cmap = mpl.colors.LinearSegmentedColormap.from_list('cmap', [palette[k] for k in [0,2,1]])

# Check device
device = "cuda:0" if torch.cuda.is_available() else "cpu"
print('torch device:' , device)

torch.manual_seed(3407)

## load data
data_file = '../dataset/symmetrized_dataset_with_bandgap.pkl'
df, species = load_data(data_file)
df = df.reset_index(drop=True)
print('data acquired')


energy_min = 0 #Unit of energy in eV
energy_max = 30 #Unit of energy in eV
nstep = 150 #Number of the energy points (F)
mode = "both"   # options: "0e", "2e", "both"
if mode in ["0e", "2e", ]:
    df = df[df['crystal_system'] != 'cubic'].reset_index(drop=True)

new_x = np.linspace(energy_min, energy_max, nstep)
def interpolate_matrix(matrix, omega):
    """Interpolates the full (3001, 3, 3) matrix along the energy axis."""
    interp = interp1d(omega, matrix, kind='linear', axis=0, fill_value=0, bounds_error=False)
    return interp(new_x)  # Shape: (F, 3, 3)

df['rel_permittivity_imag_interp'] = [
    interpolate_matrix(row['rel_permittivity_imag'], row['omega']) for _, row in df.iterrows()
]
df['energies_interp'] = df.apply(lambda x: new_x, axis=1)

stack_matrices_tensor = torch.tensor(np.stack(df['rel_permittivity_imag_interp'].values), dtype=torch.float64, device=device)  # Shape: (N, F, 3, 3)
sph_coefs_tensor = cart_to_sph(stack_matrices_tensor)   
# Separate
scalar_0e = sph_coefs_tensor[:, :, 0]     # (N, F)
tensor_2e = sph_coefs_tensor[:, :, 1:]    # (N, F, 5)

# Normalization
scale_0e = torch.mean(torch.max(torch.abs(scalar_0e), dim=1).values)
scale_2e = torch.median(torch.max(torch.norm(tensor_2e, dim=-1), dim=1).values)

scalar_0e /= (scale_0e + 1e-12)
tensor_2e /= (scale_2e.unsqueeze(-1) + 1e-12)

if mode == "0e":
    sph_coefs_tensor = scalar_0e.unsqueeze(-1)  # (N,F,1)
elif mode == "2e":
    sph_coefs_tensor = tensor_2e                # (N,F,5)
elif mode == "both":
    sph_coefs_tensor = torch.cat([scalar_0e.unsqueeze(-1), tensor_2e], dim=-1)  # (N,F,6)
else:
    raise ValueError(f"Unsupported mode={mode}")

df["sph_coefs"] = list(sph_coefs_tensor.cpu().numpy())

type_onehot, mass_onehot, dipole_onehot, radius_onehot, type_encoding = save_or_load_onehot()

r_max = 6. # cutoff radius
scale_data = 1
df['data'] = df.progress_apply(lambda x: build_data(x, 'sph_coefs', scale_data, type_onehot, mass_onehot, dipole_onehot, radius_onehot, type_encoding, r_max), axis=1)


# run_time = time.strftime('%y%m%d', time.localtime())
run_time = '250929'
# # train/valid/test split
idx_train, idx_valid, idx_test = train_valid_test_split(df, valid_size=.1, test_size=.1, seed=22, plot=True)
np.savetxt('../model/idx_train_'+ run_time +'.txt', idx_train, fmt='%i', delimiter='\t')
np.savetxt('../model/idx_valid_'+ run_time +'.txt', idx_valid, fmt='%i', delimiter='\t')
np.savetxt('../model/idx_test_'+ run_time +'.txt', idx_test, fmt='%i', delimiter='\t')
with open('../model/idx_train_'+run_time+'.txt', 'r') as f: idx_train = [int(i.split('\n')[0]) for i in f.readlines()]
with open('../model/idx_valid_'+run_time+'.txt', 'r') as f: idx_valid = [int(i.split('\n')[0]) for i in f.readlines()]
with open('../model/idx_test_'+run_time+'.txt', 'r') as f: idx_test = [int(i.split('\n')[0]) for i in f.readlines()]

# format dataloaders
batch_size = 64
dataloader_train = tg.loader.DataLoader(df.iloc[idx_train]['data'].values, batch_size=batch_size)
dataloader_valid = tg.loader.DataLoader(df.iloc[idx_valid]['data'].values, batch_size=batch_size)
dataloader_test = tg.loader.DataLoader(df.iloc[idx_test]['data'].values, batch_size=batch_size)

def get_neighbors(df, idx):
    n = []
    for entry in df.iloc[idx].itertuples():
        N = entry.data.pos.shape[0]
        for i in range(N):
            n.append(len((entry.data.edge_index[0] == i).nonzero()))
    return np.array(n)

n_train = get_neighbors(df, idx_train)
n_valid = get_neighbors(df, idx_valid)
n_test = get_neighbors(df, idx_test)

class NetWrapper(Network):
    def __init__(self, in_dim, em_dim, **kwargs):            
        # override the `reduce_output` keyword to instead perform an averge over atom contributions    
        self.pool = False
        if kwargs['reduce_output'] == True:
            kwargs['reduce_output'] = False
            self.pool = True
            
        super().__init__(**kwargs)

        self.em_z = nn.Linear(in_dim, em_dim)    
        self.em_x = nn.Linear(in_dim, em_dim)    

    def forward(self, data: Union[tg.data.Data, Dict[str, torch.Tensor]]) -> torch.Tensor:
        data.z = F.relu(self.em_z(data.z))
        data.x = F.relu(self.em_x(data.x))

        output = super().forward(data)
        
        if self.pool == True:
            output = torch_scatter.scatter_mean(output, data.batch, dim=0)  # take mean over atoms per example
        return output
    
class LearnableUncertaintyLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.log_sigma_0e = nn.Parameter(torch.tensor(0.0))
        self.log_sigma_2e = nn.Parameter(torch.tensor(0.0))

    def forward(self, loss_0e, loss_2e):
        precision_0e = torch.exp(-2 * self.log_sigma_0e)
        precision_2e = torch.exp(-2 * self.log_sigma_2e)
        weighted = 0.5 * (precision_0e * loss_0e + precision_2e * loss_2e)
        reg = self.log_sigma_0e + self.log_sigma_2e
        return weighted + reg


out_dim = len(df.iloc[0]['energies_interp']) 
if mode == "0e":
    irreps_out = f"{out_dim}x0e"
elif mode == "2e":
    irreps_out = f"{out_dim}x2e"
elif mode == "both":
    irreps_out = f"{out_dim}x0e + {out_dim}x2e"
em_dim = 64

use_batch_norm = False
dropout_prob=0.4
lr = 1e-2
lmax = 2
layers = 2
mul = 32 
model = NetWrapper(
    in_dim=118,
    em_dim=em_dim,
    irreps_in=str(em_dim)+"x0e",
    irreps_out=irreps_out,    
    irreps_node_attr=str(em_dim)+"x0e",
    layers=layers,
    mul=mul,
    lmax=lmax,
    max_radius=r_max,
    num_neighbors=n_train.mean(),
    reduce_output=True,
    dropout_prob=dropout_prob,
    use_batch_norm = use_batch_norm
)

model.to(device)
if mode == "both":
    loss_balancer = LearnableUncertaintyLoss().to(device)
    opt = torch.optim.AdamW(
        list(model.parameters()) + list(loss_balancer.parameters()),
        lr=lr, weight_decay=0.05
    )
else:
    loss_balancer = None
    opt = torch.optim.AdamW(list(model.parameters()), lr=lr, weight_decay=0.05)

total_params = sum(param.numel() for param in model.parameters())
trainable_params = sum(param.numel() for param in model.parameters() if param.requires_grad)

print(f"Total parameters: {total_params}")
print(f"Trainable parameters: {trainable_params}")

run_name = f'TSENN_ablation_{run_time}_Lmax_{lmax}_Lr_{lr}_layers_{layers}_mul_{mul}_{mode}'
max_iter = 10

scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
    opt,
    T_0=10, T_mult=1,
    eta_min=0     # Minimum learning rate (optional, default is 0)
) 

loss_fn = torch.nn.MSELoss()
loss_fn_mae = torch.nn.L1Loss()

loss_fn_eval = torch.nn.MSELoss()
loss_fn_mae_eval = torch.nn.L1Loss()

use_weighting = False
wandb.init(
    project="Tensor Predict Imaginary",  # Change this to your project name
    name=run_name,  # Unique identifier for this run
    config={
        "max_iter": max_iter,
        "lr": opt.param_groups[0]["lr"],  # Log learning rate
        "use_weighting": use_weighting,
        "r_max": r_max,
        "batch_size": batch_size,
        "dropout_prob": dropout_prob,
        "normalization": True if scale_data != 1 else False,
        "batch_norm": use_batch_norm,
        "energy_max": energy_max,
        "nstep": nstep,
        "scheduler": type(scheduler).__name__,  # Log scheduler type
        "loss_function": type(loss_fn).__name__  # Automatically log loss function type
    }
)

train(model, opt, dataloader_train, dataloader_valid,
      loss_fn, loss_fn_mae, loss_fn_eval, loss_fn_mae_eval,
      run_name, max_iter=max_iter, scheduler=scheduler,
      device=device, alpha=1.0, beta=1.0,
      loss_balancer=None if mode!="both" else loss_balancer,
      mode=mode)

# Output MSE.txt
history = torch.load('../model/' + run_name + '.torch', map_location=device)['history']
steps = [d['step'] + 1 for d in history]
loss_train = [d['train']['loss'] for d in history]
loss_valid = [d['valid']['loss'] for d in history]
fig, ax = plt.subplots(figsize=(4,4))
ax.plot(steps, loss_train, 'o-', label="Training", color='C0',markersize=3)
ax.plot(steps, loss_valid, 'o-', label="Validation", color='C3',markersize=3)
ax.set_xlabel('Epochs')
ax.set_ylabel('Loss')
ax.legend(frameon=False)
plt.tight_layout()
save_png_dir = "../pngs"
save_path = os.path.join(save_png_dir, run_name + '_loss.png')
fig.savefig(save_path,dpi=300)
wandb.log({"Loss Plot": wandb.Image(save_path)})

# predict on all data
model.load_state_dict(torch.load('../model/'+run_name + '_best.torch', map_location=device)['state'])
model.pool = True

dataloader = tg.loader.DataLoader(df['data'].values, batch_size=64)
df['mse_sph'] = 0.
df['y_pred_sph'] = np.empty((len(df), 0)).tolist()

model.to(device)
model.eval()

predictions = [] 
df['y_pred_sph'] = None
i0 = 0

with torch.no_grad():
    for i, d in tqdm(enumerate(dataloader), total=len(dataloader), bar_format=bar_format):
        d.to(device)
        output = model(d)

        if mode == "0e":
            irreps_0e = model.irreps_out.count(o3.Irrep("0e"))
            out_dim = irreps_0e
            output_0e = output[:, :irreps_0e] * scale_0e
            y_0e = d.y.view(d.y.shape[0], out_dim) * scale_0e
            loss = F.mse_loss(output_0e, y_0e)
            combined_output = output_0e.unsqueeze(2)  # (B,F,1)

        elif mode == "2e":
            irreps_2e = model.irreps_out.count(o3.Irrep("2e")) * 5
            out_dim = model.irreps_out.count(o3.Irrep("2e"))
            output_2e = output[:, :irreps_2e].view(output.shape[0], out_dim, 5) * scale_2e
            y_2e = d.y.view(d.y.shape[0], out_dim, 5) * scale_2e
            loss = F.mse_loss(output_2e, y_2e)
            combined_output = output_2e  # (B,F,5)

        elif mode == "both":
            irreps_0e = model.irreps_out.count(o3.Irrep("0e"))
            irreps_2e = model.irreps_out.count(o3.Irrep("2e")) * 5
            out_dim = model.irreps_out.count(o3.Irrep("0e"))

            output_0e = output[:, :irreps_0e] * scale_0e
            output_2e = output[:, irreps_0e:irreps_0e + irreps_2e].view(output.shape[0], out_dim, 5) * scale_2e
            y_0e = d.y[:, :, 0].view(d.y.shape[0], out_dim) * scale_0e
            y_2e = d.y[:, :, 1:].view(d.y.shape[0], out_dim, 5) * scale_2e
            loss = F.mse_loss(output_0e, y_0e) + F.mse_loss(output_2e, y_2e)
            combined_output = torch.cat([output_0e.unsqueeze(2), output_2e], dim=2)  # (B,F,6)

        for batch_idx in range(d.y.shape[0]):
            df.loc[i0 + batch_idx, 'y_pred_sph'] = [combined_output[batch_idx].cpu().numpy()]
            df.loc[i0 + batch_idx, 'mse_sph'] = loss.cpu().numpy() 

        # Update batch index counter
        i0 += d.y.shape[0]
    
    
def denormalize_sph_coefs(sph_coefs, scale_0e, scale_2e):
    """Denormalize spherical coefficients: col0 = ℓ=0, col1..5 = ℓ=2."""
    sph_denorm = sph_coefs.copy()
    sph_denorm[:, 0] *= (scale_0e.item() if torch.is_tensor(scale_0e) else scale_0e)
    sph_denorm[:, 1:] *= (scale_2e.item() if torch.is_tensor(scale_2e) else scale_2e)
    return sph_denorm

sph_pred_list = []
sph_true_list = []

for pred_arr, true_arr in zip(df['y_pred_sph'], df['sph_coefs']):
    pred = np.array(pred_arr[0])   # predicted spherical (F,1)/(F,5)/(F,6)
    true = np.array(true_arr)      # target spherical (F,1)/(F,5)/(F,6)

    # --- denormalize ground truth ---
    true = denormalize_sph_coefs(true, scale_0e, scale_2e)

    if mode == "0e":
        # Pad both to (F,6)
        pred_full = np.concatenate([pred, np.zeros((pred.shape[0], 5))], axis=1)
        true_full = np.concatenate([true, np.zeros((true.shape[0], 5))], axis=1)

    elif mode == "2e":
        pred_full = np.concatenate([np.zeros((pred.shape[0], 1)), pred], axis=1)
        true_full = np.concatenate([np.zeros((true.shape[0], 1)), true], axis=1)

    elif mode == "both":
        pred_full, true_full = pred, true

    sph_pred_list.append(pred_full)
    sph_true_list.append(true_full)

# Stack → tensors (N,F,6)
sph_pred = torch.tensor(np.stack(sph_pred_list), device=device)
sph_true = torch.tensor(np.stack(sph_true_list), device=device)

# Cartesian back-transform
cart_pred = sph_to_cart(sph_pred)
cart_true = sph_to_cart(sph_true)

# Compute errors
cart_pred_tensor = torch.tensor(cart_pred.cpu().numpy(), dtype=torch.float64)
cart_true_tensor = torch.tensor(cart_true.cpu().numpy(), dtype=torch.float64)

column = 'y_true_cart'

df['y_true_cart'] = list(cart_true_tensor.detach().cpu().numpy())
df['y_pred_cart'] = list(cart_pred_tensor.detach().cpu().numpy())

# Error metrics
def compute_symmetric_errors(pred, true):
    inds_diag = [(0, 0), (1, 1), (2, 2)]
    inds_off = [(0, 1), (0, 2), (1, 2)]
    diffs = []
    for i, j in inds_diag + inds_off:
        diff = pred[:, :, i, j] - true[:, :, i, j]
        diffs.append(diff)
    diffs = torch.stack(diffs, dim=0)  # (6,N,F)
    mse = torch.mean(diffs ** 2, dim=0)  # (N,F)
    mae = torch.mean(torch.abs(diffs), dim=0)  # (N,F)
    return mse, mae

mse_torch, mae_cart = compute_symmetric_errors(cart_pred_tensor, cart_true_tensor)
df['mse_cart'] = np.mean(mse_torch.cpu().numpy(), axis=1)
df['mae_cart'] = np.mean(mae_cart.cpu().numpy(), axis=1)


def get_random_sample_indices(idx, n):
    """Returns `n` randomly selected unique indices from `idx`."""
    if len(idx) < n:
        n = len(idx)  # Ensure we don't exceed available samples
    return np.random.choice(idx, size=n, replace=False)

def get_random_sample_indices(idx, n, df, system="cubic"):
    """Return `n` random unique indices from `idx`, restricted to a crystal system."""
    # Mask by crystal system
    mask = df.loc[idx, "crystal_system"] == system
    filtered_idx = np.array(idx)[mask.values]

    if len(filtered_idx) == 0:
        raise ValueError(f"No samples found for system='{system}' in provided indices.")

    if len(filtered_idx) < n:
        n = len(filtered_idx)

    return np.random.choice(filtered_idx, size=n, replace=False)

n_samples = 12  # Adjust as needed

# random_idx_train = get_random_sample_indices(idx_train, n_samples)
# random_idx_valid = get_random_sample_indices(idx_valid, n_samples)
# random_idx_test = get_random_sample_indices(idx_test, n_samples)
random_idx_train = get_random_sample_indices(idx_train, n_samples, df, system="cubic")
random_idx_valid = get_random_sample_indices(idx_valid, n_samples, df, system="cubic")
random_idx_test = get_random_sample_indices(idx_test, n_samples, df, system="cubic")
# Use the same indices for both functions
plot_spherical_harmonics_comparison(df, random_idx_train, column, title_prefix="training_set", n=n_samples)
plot_spherical_harmonics_comparison(df, random_idx_valid, column, title_prefix="validation_set", n=n_samples)
plot_spherical_harmonics_comparison(df, random_idx_test, column, title_prefix="testing_set", n=n_samples)


plot_cartesian_tensor_comparison(df, random_idx_train, column, title_prefix="training_set", n=n_samples)
plot_cartesian_tensor_comparison(df, random_idx_valid, column, title_prefix="validation_set", n=n_samples)
plot_cartesian_tensor_comparison(df, random_idx_test, column, title_prefix="testing_set", n=n_samples)

########################################################################################################################
# Log png to WandB
wandb.log({
    "Spherical Harmonics - Training": wandb.Image(f"../pngs/training_set_spectra.png"),
    "Spherical Harmonics - Validation": wandb.Image(f"../pngs/validation_set_spectra.png"),
    "Spherical Harmonics - Testing": wandb.Image(f"../pngs/testing_set_spectra.png"),
    "Cartesian Tensor - Training": wandb.Image(f"../pngs/training_set_cart_spectra.png"),
    "Cartesian Tensor - Validation": wandb.Image(f"../pngs/validation_set_cart_spectra.png"),
    "Cartesian Tensor - Testing": wandb.Image(f"../pngs/testing_set_cart_spectra.png"),
})
# =========================
# Compute losses per split
# =========================
def compute_split_losses(df, idx, name=""):
    subset = df.loc[idx].copy().reset_index()

    mse_mean = subset["mse_cart"].mean()
    mse_median = subset["mse_cart"].median()
    mae_mean = subset["mae_cart"].mean()
    mae_median = subset["mae_cart"].median()

    print(f"\n{name} Set:")
    print(f"  MSE mean   = {mse_mean:.4e}, median = {mse_median:.4e}")
    print(f"  MAE mean   = {mae_mean:.4e}, median = {mae_median:.4e}")

    return {
        f"{name}_MSE_mean": mse_mean,
        f"{name}_MSE_median": mse_median,
        f"{name}_MAE_mean": mae_mean,
        f"{name}_MAE_median": mae_median,
    }

# Evaluate splits
train_stats = compute_split_losses(df, idx_train, name="Train")
valid_stats = compute_split_losses(df, idx_valid, name="Validation")
test_stats  = compute_split_losses(df, idx_test,  name="Test")

# =========================
# Log to WandB
# =========================
wandb.log({**train_stats, **valid_stats, **test_stats})

model.eval()

# Ensure tensor indexing type/device
idx_test_t = torch.as_tensor(idx_test, dtype=torch.long, device=sph_pred.device)

with torch.no_grad():
    sph_pred_t = sph_pred[idx_test_t]   # (N_test, F, 6)
    sph_true_t = sph_true[idx_test_t]   # (N_test, F, 6)

    # ---- Leakage diagnostics (test only) ----
    norm_0e_pred = torch.norm(sph_pred_t[:, :, 0], dim=1)
    norm_2e_pred = torch.norm(sph_pred_t[:, :, 1:], dim=(1, 2))
    norm_0e_true = torch.norm(sph_true_t[:, :, 0], dim=1)
    norm_2e_true = torch.norm(sph_true_t[:, :, 1:], dim=(1, 2))

    print("\n=== Leakage Diagnostics (Test) ===")
    if mode == "0e":
        print(f"[Mode=0e] Median predicted l=0 norm: {norm_0e_pred.median().item():.3e}")
        print(f"[Mode=0e] Median *leakage* l=2 norm: {norm_2e_pred.median().item():.3e}")
    elif mode == "2e":
        print(f"[Mode=2e] Median predicted l=2 norm: {norm_2e_pred.median().item():.3e}")
        print(f"[Mode=2e] Median *leakage* l=0 norm: {norm_0e_pred.median().item():.3e}")
    else:  # both
        print(f"[Mode=both] Median l=0 norm: {norm_0e_pred.median().item():.3e}")
        print(f"[Mode=both] Median l=2 norm: {norm_2e_pred.median().item():.3e}")

with torch.no_grad():
    diff = torch.abs(sph_pred_t - sph_true_t)          # (N_test, F, 6)
    mae_per_comp = torch.nanmean(diff, dim=1)          # (N_test, 6)
    mae_l0 = mae_per_comp[:, 0]
    mae_l2_mean = torch.nanmean(mae_per_comp[:, 1:], dim=1)
    mae_total = torch.nanmean(mae_per_comp, dim=1)

    contrib_l0 = (1/6) * mae_l0
    contrib_l2 = (5/6) * mae_l2_mean

    recon_err = torch.abs(mae_total - (contrib_l0 + contrib_l2)).median().item()
    print(f"Reconstruction check (median abs diff): {recon_err:.3e}")

    total_mean, total_median = mae_total.mean().item(), mae_total.median().item()
    c0_mean, c0_median = contrib_l0.mean().item(), contrib_l0.median().item()
    c2_mean, c2_median = contrib_l2.mean().item(), contrib_l2.median().item()
    pct0_mean = 100.0 * c0_mean / (c0_mean + c2_mean + 1e-30)
    pct2_mean = 100.0 - pct0_mean
    pct0_med  = 100.0 * c0_median / (c0_median + c2_median + 1e-30)
    pct2_med  = 100.0 - pct0_med

    print(f"TOTAL MAE (Test): mean={total_mean:.4f}, median={total_median:.4f}")
    print(f"Contrib l=0: mean={c0_mean:.4f} ({pct0_mean:.1f}%), median={c0_median:.4f} ({pct0_med:.1f}%)")
    print(f"Contrib l=2: mean={c2_mean:.4f} ({pct2_mean:.1f}%), median={c2_median:.4f} ({pct2_med:.1f}%)")

wandb.finish()

# # # --- utilities: split iso/aniso, combine, norms ---
# def split_iso_aniso(t):  # t: (N, F, 3, 3)
#     tr = t[..., 0, 0] + t[..., 1, 1] + t[..., 2, 2]
#     iso = (tr / 3.0)[..., None, None] * np.eye(3)[None, None, :, :]
#     aniso = t - iso
#     return iso, aniso

# def frob_norm(x):  # (...,3,3)
#     return np.sqrt(np.sum(x**2, axis=(-1, -2)))

# def mae_over_freq(arr):  # (N,F,...) -> (N,)
#     return np.nanmean(np.abs(arr), axis=1)

# # --- build predicted tensors for each mode you'd trained ---
# # For each run, you already have df['y_pred_cart'] and df['y_true_cart'] as arrays (F,3,3) per row.
# def eval_ablation(df_subset, pred_cart_key="y_pred_cart", true_cart_key="y_true_cart"):
#     # Stack to arrays: (N,F,3,3)
#     y_pred = np.stack(df_subset[pred_cart_key].to_numpy())
#     y_true = np.stack(df_subset[true_cart_key].to_numpy())

#     pred_iso,  pred_aniso  = split_iso_aniso(y_pred)
#     true_iso,  true_aniso  = split_iso_aniso(y_true)

#     # --- isotropy metrics ---
#     # scalar iso as 3x3 identity scaled, but MAE_iso as scalar difference of traces /3
#     mae_iso_scalar = mae_over_freq((pred_iso - true_iso)[...,0,0])  # (N,) scalar channel MAE

#     # --- anisotropy metrics ---
#     # per-component aniso MAE (xx,yy,zz,xy,xz,yz)
#     comp = [(0,0), (1,1), (2,2), (0,1), (0,2), (1,2)]
#     labels = ["xx","yy","zz","xy","xz","yz"]
#     mae_aniso_comp = {}
#     for (i,j), lbl in zip(comp, labels):
#         mae_aniso_comp[lbl] = mae_over_freq(pred_aniso[..., i, j] - true_aniso[..., i, j])

#     # anisotropy-strength MAE via Frobenius norms
#     mae_aniso_strength = mae_over_freq(frob_norm(pred_aniso) - frob_norm(true_aniso))

#     # --- full-tensor MAE variants (averaged over 6 unique comps) ---
#     def full_mae(A, B):
#         # take 6 independent components
#         diffs = np.stack([A[...,i,j]-B[...,i,j] for (i,j) in comp], axis=-1)  # (N,F,6)
#         return mae_over_freq(diffs).mean(axis=-1)  # (N,) mean over 6 comps

#     # Oracle variants to isolate channels:
#     # (1) Oracle-aniso: use predicted iso + true aniso
#     mae_full_oracle_aniso = full_mae(pred_iso + true_aniso, y_true)
#     # (2) Oracle-iso:    use true iso + predicted aniso
#     mae_full_oracle_iso   = full_mae(true_iso + pred_aniso, y_true)
#     # (3) Actual full MAE from predictions
#     mae_full_actual       = full_mae(y_pred, y_true)

#     # --- leakage diagnostics ---
#     # normalized inner product between iso and aniso predictions (per sample, per freq)
#     num = np.sum(pred_iso * pred_aniso, axis=(-1,-2))               # (N,F)
#     den = frob_norm(pred_iso) * frob_norm(pred_aniso)               # (N,F)
#     rho = np.nanmedian(np.abs(num / np.clip(den, 1e-12, None)), axis=1)  # (N,)

#     # projections (should be ~0)
#     # leakage of aniso into iso: isotropic part of pred_aniso
#     tr_aniso = (pred_aniso[...,0,0] + pred_aniso[...,1,1] + pred_aniso[...,2,2])
#     iso_from_aniso = (tr_aniso/3.0)[..., None, None] * np.eye(3)[None, None, :, :]
#     lam_0_from_2 = np.nanmedian(frob_norm(iso_from_aniso) / np.clip(frob_norm(pred_aniso), 1e-12, None), axis=1)  # (N,)

#     # leakage of iso into aniso: traceless part of pred_iso (should be zero)
#     tr_iso = (pred_iso[...,0,0] + pred_iso[...,1,1] + pred_iso[...,2,2])
#     iso_clean = (tr_iso/3.0)[..., None, None] * np.eye(3)[None, None, :, :]
#     aniso_from_iso = pred_iso - iso_clean
#     lam_2_from_0 = np.nanmedian(frob_norm(aniso_from_iso) / np.clip(frob_norm(pred_iso), 1e-12, None), axis=1)

#     # Aggregate split-level stats (mean & median)
#     out = {
#         "MAE_iso_mean":   float(np.nanmean(mae_iso_scalar)),
#         "MAE_iso_median": float(np.nanmedian(mae_iso_scalar)),
#         "MAE_aniso_strength_mean":   float(np.nanmean(mae_aniso_strength)),
#         "MAE_aniso_strength_median": float(np.nanmedian(mae_aniso_strength)),
#         "MAE_full_actual_mean":      float(np.nanmean(mae_full_actual)),
#         "MAE_full_actual_median":    float(np.nanmedian(mae_full_actual)),
#         "MAE_full_oracle_iso_mean":  float(np.nanmean(mae_full_oracle_iso)),
#         "MAE_full_oracle_iso_median":float(np.nanmedian(mae_full_oracle_iso)),
#         "MAE_full_oracle_aniso_mean":float(np.nanmean(mae_full_oracle_aniso)),
#         "MAE_full_oracle_aniso_median":float(np.nanmedian(mae_full_oracle_aniso)),
#         "Leak_rho_median":           float(np.nanmedian(rho)),
#         "Leak_lam_0_from_2_median":  float(np.nanmedian(lam_0_from_2)),
#         "Leak_lam_2_from_0_median":  float(np.nanmedian(lam_2_from_0)),
#     }
#     # per-component medians (optional: add means too)
#     for lbl in labels:
#         out[f"MAE_aniso_{lbl}_median"] = float(np.nanmedian(mae_aniso_comp[lbl]))
#     return out

# # After each trained run ("0e", "2e", "both") you already populate df with y_pred_cart & y_true_cart.
# stats_train = eval_ablation(df.loc[idx_train])
# stats_valid = eval_ablation(df.loc[idx_valid])
# stats_test  = eval_ablation(df.loc[idx_test])

# # print or log to W&B
# import pprint, wandb
# print("\n[TRAIN]"); pprint.pprint(stats_train)
# print("\n[VALID]"); pprint.pprint(stats_valid)
# print("\n[TEST]");  pprint.pprint(stats_test)
# # wandb.log({**{f"Train/{k}":v for k,v in stats_train.items()},
# #            **{f"Valid/{k}":v for k,v in stats_valid.items()},
# #            **{f"Test/{k}":v  for k,v in stats_test.items()}})
# def error_energy_split(df, idx):
#     # stack (N,F,3,3)
#     Yp = np.stack(df.loc[idx, 'y_pred_cart'].to_numpy())
#     Yt = np.stack(df.loc[idx, 'y_true_cart'].to_numpy())
#     D  = Yp - Yt

#     # isotropic / anisotropic split
#     tr = D[...,0,0] + D[...,1,1] + D[...,2,2]            # (N,F)
#     I3 = np.eye(3)[None,None,:,:]
#     D_iso   = (tr/3.0)[...,None,None]*I3                 # (N,F,3,3)
#     D_aniso = D - D_iso

#     # Frobenius norms squared
#     f2   = lambda X: np.sum(X*X, axis=(-1,-2))           # (N,F)
#     E_iso   = f2(D_iso)                                  # (N,F)
#     E_aniso = f2(D_aniso)                                # (N,F)
#     E_tot   = E_iso + E_aniso                            # (N,F)

#     # Global fractions (sum over ω and samples)
#     num_iso   = np.sum(E_iso)
#     num_aniso = np.sum(E_aniso)
#     den       = num_iso + num_aniso + 1e-30
#     p_iso_global   = float(num_iso/den)
#     p_aniso_global = float(num_aniso/den)

#     # Per-sample fractions (median across samples for robustness)
#     Es_iso   = np.sum(E_iso,   axis=1)                   # (N,)
#     Es_aniso = np.sum(E_aniso, axis=1)                   # (N,)
#     Es_tot   = Es_iso + Es_aniso + 1e-30
#     p_iso_per_sample   = Es_iso/Es_tot
#     p_aniso_per_sample = Es_aniso/Es_tot
#     p_iso_median   = float(np.median(p_iso_per_sample))
#     p_aniso_median = float(np.median(p_aniso_per_sample))

#     return {
#         "p_iso_global": p_iso_global,
#         "p_aniso_global": p_aniso_global,
#         "p_iso_median": p_iso_median,
#         "p_aniso_median": p_aniso_median
#     }

# # Example on your test split:
# res = error_energy_split(df, idx_test)
# print("Error-energy split (L2/Frobenius):")
# print(f"Global: iso = {res['p_iso_global']*100:.1f}%, aniso = {res['p_aniso_global']*100:.1f}%")
# print(f"Median per-sample: iso = {res['p_iso_median']*100:.1f}%, aniso = {res['p_aniso_median']*100:.1f}%")
# sph_pred, sph_true: torch tensors (N, F, 6) already on device
# Column 0 = l=0 (m=0), columns 1..5 = l=2 (5 components)
