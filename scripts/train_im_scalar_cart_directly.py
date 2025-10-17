import os, sys, time, logging, math
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(parent_dir)  # Add parent directory to sys.path
from typing import Dict, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric as tg
import torch_scatter

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
from scipy.interpolate import interp1d
from tqdm import tqdm

import wandb
from utils.utils_data import (
    load_data, train_valid_test_split, save_or_load_onehot,
    build_data, plot_cartesian_tensor_comparison
)
from utils.utils_model_scalar_cart import Network, train, visualize_layers

mpl.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams["mathtext.fontset"] = "cm"
fontsize = 16
textsize = 16
sub = str.maketrans("0123456789", "₀₁₂₃₄₅₆₇₈₉")
plt.rcParams['font.sans-serif'] = ['Helvetica', 'Arial', 'Liberation Sans', 'sans-serif']
plt.rcParams['axes.linewidth'] = 1
plt.rcParams['mathtext.default'] = 'regular'
plt.rcParams['xtick.bottom'] = True
plt.rcParams['ytick.left'] = True
plt.rcParams['font.size'] = fontsize
plt.rcParams['axes.labelsize'] = textsize 
plt.rcParams['xtick.labelsize'] = textsize  
plt.rcParams['ytick.labelsize'] = textsize
plt.rcParams['legend.fontsize'] = fontsize
plt.rcParams['text.usetex'] = False
plt.rcParams['figure.dpi'] = 150       
plt.rcParams['savefig.dpi'] = 300   
# -------------------
# Settings
# -------------------
logging.getLogger('matplotlib.font_manager').setLevel(logging.CRITICAL)
plt.rcParams["mathtext.fontset"] = "cm"

# device = "cuda:0" if torch.cuda.is_available() else "cpu"
device = "cpu"
torch.manual_seed(3407)
torch.set_default_dtype(torch.float64)
print("torch device:", device)

# -------------------
# Component flexibility
# -------------------
COMPONENT_MAP = {"xx":0, "yy":1, "zz":2, "xy":3, "xz":4, "yz":5}
SYMM_INDS = [(0,0), (1,1), (2,2), (0,1), (0,2), (1,2)]

# pick subset here
selected_components = ["xx","yy","zz"]   # <--- change freely
selected_indices = [COMPONENT_MAP[c] for c in selected_components]
ncomp = len(selected_indices)
print(f"Training on components {selected_components} (indices {selected_indices})")

def six_to_mat(arr6):
    """(nstep, ncomp) -> (nstep, 3, 3) symmetric (fills only selected comps)."""
    nstep = arr6.shape[0]
    mat = np.zeros((nstep, 3, 3), dtype=arr6.dtype)
    for k, idx in enumerate(selected_indices):
        i, j = SYMM_INDS[idx]
        mat[:, i, j] = arr6[:, k]
        mat[:, j, i] = arr6[:, k]
    return mat

# -------------------
# Load + preprocess
# -------------------
data_file = '../dataset/symmetrized_dataset_with_bandgap.pkl'
df, species = load_data(data_file)
df = df.reset_index(drop=True)
print("data acquired")

energy_min, energy_max, nstep = 0, 30, 300
new_x = np.linspace(energy_min, energy_max, nstep)

def interpolate_matrix(matrix, omega):
    interp = interp1d(omega, matrix, kind='linear', axis=0,
                      fill_value=0, bounds_error=False)
    return interp(new_x)  # (nstep, 3, 3)

def extract_selected(matrix):
    full6 = np.stack([matrix[:, i, j] for i, j in SYMM_INDS], axis=-1)  # (nstep,6)
    return full6[:, selected_indices]                                   # (nstep,ncomp)

df['rel_permittivity_imags_interp'] = [
    interpolate_matrix(row['rel_permittivity_imag'], row['omega'])
    for _, row in df.iterrows()
]
df['energies_interp'] = [new_x] * len(df)
df['stack_matrices_tensor'] = [
    extract_selected(m) for m in df['rel_permittivity_imags_interp']
]

# scaling
tmp = np.array(df['stack_matrices_tensor'].tolist())  # (N,nstep,ncomp)
# scale_data = np.median([np.max(np.abs(sample)) for sample in tmp])
scale_data = 1
print("Scale factor:", scale_data)

# build dataset
type_onehot, mass_onehot, dipole_onehot, radius_onehot, type_encoding = save_or_load_onehot()
r_max = 6.0
df['data'] = df.progress_apply(
    lambda x: build_data(x, 'stack_matrices_tensor', scale_data,
                         type_onehot, mass_onehot, dipole_onehot,
                         radius_onehot, type_encoding, r_max),
    axis=1
)

# -------------------
# Train/valid/test split
# -------------------
run_time = '250909'
with open(f'../model/idx_train_{run_time}.txt') as f: idx_train = [int(i) for i in f]
with open(f'../model/idx_valid_{run_time}.txt') as f: idx_valid = [int(i) for i in f]
with open(f'../model/idx_test_{run_time}.txt')  as f: idx_test  = [int(i) for i in f]

batch_size = 12
dataloader_train = tg.loader.DataLoader(df.iloc[idx_train]['data'].values, batch_size=batch_size, shuffle=True)
dataloader_valid = tg.loader.DataLoader(df.iloc[idx_valid]['data'].values, batch_size=batch_size)
dataloader_test  = tg.loader.DataLoader(df.iloc[idx_test]['data'].values,  batch_size=batch_size)
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

# -------------------
# Define model
# -------------------
class PeriodicNetwork(Network):
    def __init__(self, in_dim, em_dim, **kwargs):
        self.pool = False
        if kwargs['reduce_output']:
            kwargs['reduce_output'] = False
            self.pool = True
        super().__init__(**kwargs)
        self.em_z = nn.Linear(in_dim, em_dim)
        self.em_x = nn.Linear(in_dim, em_dim)

    def forward(self, data: Union[tg.data.Data, Dict[str, torch.Tensor]]) -> torch.Tensor:
        data.z = F.relu(self.em_z(data.z))
        data.x = F.relu(self.em_x(data.x))
        output = super().forward(data)
        if self.pool:
            output = torch_scatter.scatter_mean(output, data.batch, dim=0)
        return output

out_dim = len(df.iloc[0]['energies_interp'])
em_dim = 64
model = PeriodicNetwork(
    in_dim=118,
    em_dim=em_dim,
    irreps_in=f"{em_dim}x0e",
    irreps_out=f"{out_dim * ncomp}x0e",
    irreps_node_attr=f"{em_dim}x0e",
    layers=2, mul=32, lmax=2, max_radius=r_max,
    num_neighbors=n_train.mean(),
    reduce_output=True, dropout_prob=0.0, use_batch_norm=False
)
model.to(device)
visualize_layers(model)

# -------------------
# Train
# -------------------
opt = torch.optim.AdamW(model.parameters(), lr=1e-2, weight_decay=0.05)
scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(opt, T_0=10, T_mult=1)

loss_fn, loss_fn_mae = nn.MSELoss(), nn.L1Loss()
loss_fn_eval, loss_fn_mae_eval = nn.MSELoss(), nn.L1Loss()

run_name = f"TSENN_cart_model_{run_time}_{'_'.join(selected_components)}"
wandb.init(project="Cartesian flexible model", name=run_name)

# train(model, opt, dataloader_train, dataloader_valid,
#       loss_fn, loss_fn_mae, loss_fn_eval, loss_fn_mae_eval,
#       run_name, max_iter=100, scheduler=scheduler,
#       device=device, out_dim=out_dim, ncomp=ncomp, selected_components=selected_components)

# -------------------
# Best model prediction
# -------------------
model.load_state_dict(torch.load(f'../model/{run_name}_best.torch', map_location=device)['state'])

model.pool = True
model.eval()

dataloader = tg.loader.DataLoader(df['data'].values, batch_size=64)
df['y_pred_cart'] = np.empty((len(df), 0)).tolist()

i0 = 0
with torch.no_grad():
    for d in tqdm(dataloader, total=len(dataloader), bar_format='{l_bar}{bar:10}{r_bar}'):
        d.to(device)
        output = model(d).view(d.y.size(0), -1, ncomp)  # (batch,nstep,ncomp)
        out_np = output.cpu().numpy() * scale_data
        out_full = [six_to_mat(out_np[b]) for b in range(out_np.shape[0])]
        df.loc[i0:i0+len(d.y)-1, 'y_pred_cart'] = out_full
        i0 += len(d.y)

# -------------------
# Error analysis
# -------------------
cart_true = np.stack(df['rel_permittivity_imags_interp'].values)
cart_pred = np.stack(df['y_pred_cart'].values)

cart_true_t = torch.tensor(cart_true, dtype=torch.float64)
cart_pred_t = torch.tensor(cart_pred, dtype=torch.float64)

diffs = cart_pred_t - cart_true_t
mse = (diffs**2).mean(dim=(1,2,3)).cpu().numpy()
mae = diffs.abs().mean(dim=(1,2,3)).cpu().numpy()
df['mse_cart'], df['mae_cart'] = mse, mae

print("Mean MSE:", mse.mean(), "Mean MAE:", mae.mean())

# -------------------
# Plot random samples (only selected comps)
# -------------------
def plot_selected_comps(df, idx_list, title_prefix="set", n=6):
    n = min(len(idx_list), n)
    idxs = np.random.choice(idx_list, size=n, replace=False)

    fig, axes = plt.subplots(n, ncomp, figsize=(3*ncomp, 2*n), squeeze=False)
    axes = np.atleast_2d(axes)  # ensure it's always 2D

    for row, idx in enumerate(idxs):
        omega = df.iloc[idx]['energies_interp']
        pred = df.iloc[idx]['y_pred_cart']
        true = df.iloc[idx]['rel_permittivity_imags_interp']

        for col, comp in enumerate(selected_components):
            i, j = SYMM_INDS[COMPONENT_MAP[comp]]
            ax = axes[row, col]
            ax.plot(omega, true[:, i, j], label="True", alpha=0.9)
            ax.plot(omega, pred[:, i, j], label="Pred", alpha=0.9)
            if row == 0:
                ax.set_title(comp)
            if col == 0:
                ax.set_ylabel("Value")
            x_ticks = np.linspace(omega[0], omega[-1], 4)
            ax.set_xticks(x_ticks)
    # Legend only once
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right", frameon=False)
    save_path = f"../pngs/{title_prefix}_pred_vs_true.png"
    fig.savefig(save_path, dpi=200)
    plt.close(fig)
    wandb.log({f"{title_prefix} comparison": wandb.Image(save_path)})

wandb.finish()

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

n_samples = 8
column = "rel_permittivity_imags_interp"  # make sure this matches df.columns

random_idx_train = get_random_sample_indices(idx_train, n_samples, df, system="cubic")
random_idx_valid = get_random_sample_indices(idx_valid, n_samples, df, system="cubic")
random_idx_test = get_random_sample_indices(idx_test, n_samples, df, system="cubic")

# Plot only cubic cases
plot_cartesian_tensor_comparison(df, random_idx_train, column, title_prefix="training_set_cubic", n=n_samples)
plot_cartesian_tensor_comparison(df, random_idx_valid, column, title_prefix="validation_set_cubic", n=n_samples)
plot_cartesian_tensor_comparison(df, random_idx_test, column, title_prefix="testing_set_cubic", n=n_samples)