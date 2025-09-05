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
from matplotlib.lines import Line2D
import seaborn as sns
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
from mendeleev import element
from tqdm import tqdm
from utils.utils_data import (load_data, train_valid_test_split, save_or_load_onehot, build_data, plot_spherical_harmonics_comparison, plot_cartesian_tensor_comparison)
from utils.utils_model_full_tensor import Network, train
import wandb
from torch_geometric.nn import GATv2Conv
from torch_geometric.nn import global_mean_pool as gmp
from torch.nn import Linear, BatchNorm1d, Dropout

plt.rcParams["mathtext.fontset"] = "cm"

bar_format = '{l_bar}{bar:10}{r_bar}{bar:-10b}'
default_dtype = torch.float64
torch.set_default_dtype(default_dtype)

# Create a colormap based on the number of unique symbols
datasets = ['g', 'y', 'r']
colors = dict(zip(datasets, palette))
cmap = mpl.colors.LinearSegmentedColormap.from_list('cmap', [palette[k] for k in [0,2,1]])

# Check device
device = "cuda:3" if torch.cuda.is_available() else "cpu"
print('torch device:' , device)

## load data
data_file = '../dataset/symmetrized_dataset.pkl'
df, species = load_data(data_file)
df = df.reset_index(drop=True)
print('data acquired')

energy_min = 0 #Unit of energy in eV
energy_max = 30 #Unit of energy in eV
nstep = 300 #Number of the energy points

new_x = np.linspace(energy_min, energy_max, nstep)
# Efficiently interpolate all matrices using list comprehension
def interpolate_matrix(matrix, omega):
    """Interpolates the full (3001, 3, 3) matrix along the energy axis."""
    interp = interp1d(omega, matrix, kind='linear', axis=0, fill_value=0, bounds_error=False)
    return interp(new_x)  # Shape: (301, 3, 3)


# Apply interpolation efficiently
df['imag_Permittivity_Matrices_interp'] = [
    interpolate_matrix(row['imag_symmetrized_permittivity'], row['omega']) for _, row in df.iterrows()
]
# Apply the custom function to create a new column
df['energies_interp'] = df.apply(lambda x: new_x, axis=1)

stack_matrices_tensor = torch.tensor(np.stack(df['imag_Permittivity_Matrices_interp'].values), dtype=torch.float64, device=device)  # Shape: (num_samples, 301, 3, 3)

# Transform Cartesian tensor to irreps
x = CartesianTensor("ij=ji")  # Symmetric rank-2 tensor
sph_coefs_tensor = x.from_cartesian(stack_matrices_tensor)  # Shape: (num_samples, 301, 6)
df['sph_coefs'] = list(sph_coefs_tensor.cpu().numpy())  # Move to CPU and store as list


type_onehot, mass_onehot, dipole_onehot, radius_onehot, type_encoding = save_or_load_onehot()

# Find the scaling value
tmp = np.array([df.iloc[i]['sph_coefs'] for i in range(len(df))])
print(tmp.shape)
scale_data = np.median(np.max(np.abs(tmp), axis=(1, 2)))
print(scale_data)


r_max = 6. # cutoff radius
df['data'] = df.progress_apply(lambda x: build_data(x, 'sph_coefs', scale_data, type_onehot, mass_onehot, dipole_onehot, radius_onehot, type_encoding, r_max), axis=1)


run_time = time.strftime('%y%m%d', time.localtime())
# # train/valid/test split
idx_train, idx_valid, idx_test = train_valid_test_split(df, valid_size=.1, test_size=.1, plot=True)
# # #Save train loss values sets
np.savetxt('../model/idx_train_'+ run_time +'.txt', idx_train, fmt='%i', delimiter='\t')
np.savetxt('../model/idx_valid_'+ run_time +'.txt', idx_valid, fmt='%i', delimiter='\t')
np.savetxt('../model/idx_test_'+ run_time +'.txt', idx_test, fmt='%i', delimiter='\t')
# load train/valid/test indices
with open('../model/idx_train_'+run_time+'.txt', 'r') as f: idx_train = [int(i.split('\n')[0]) for i in f.readlines()]
with open('../model/idx_valid_'+run_time+'.txt', 'r') as f: idx_valid = [int(i.split('\n')[0]) for i in f.readlines()]
with open('../model/idx_test_'+run_time+'.txt', 'r') as f: idx_test = [int(i.split('\n')[0]) for i in f.readlines()]

# format dataloaders
batch_size = 4
dataloader_train = tg.loader.DataLoader(df.iloc[idx_train]['data'].values, batch_size=batch_size, shuffle=True)
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


def calculateEdgeAttributes(dist, r_cutoff, dr=0.01):
    if dr == 0:
        return dist
    else:
        rgrid = np.arange(0, r_cutoff, dr)
        sigma = r_cutoff / 3
        attr = np.exp(-0.5 * (rgrid - dist)**2 /sigma**2) / np.sqrt(2 * np.pi) / sigma
        return attr
    
class GAT(torch.nn.Module):
    def __init__(self, hidden_channels, num_hidden_layers, num_edge_features, num_heads, device=torch.device('cuda'), r_cutoff=5.0, dr=0.1):
        super().__init__()
        self.r_cutoff = r_cutoff
        self.dr = dr

        self.Conv_list = nn.ModuleList()
        in_channels = hidden_channels

        for i in range(num_hidden_layers):
            out_channels = hidden_channels  # per head
            conv = GATv2Conv(
                in_channels,
                out_channels,
                heads=num_heads,
                edge_dim=num_edge_features,
                concat=True  # ensures output is out_channels * num_heads
            ).to(device)
            self.Conv_list.append(conv)

            # Update input size for next layer
            in_channels = out_channels * num_heads

        self.lin1 = Linear(in_channels, in_channels)
        self.bn = BatchNorm1d(in_channels)
        self.dropout = Dropout(p=0.2)
        self.lin2 = Linear(in_channels, 1800)
        
    def calculate_edge_attr(self, edge_vec, r_cutoff, dr=0.1):
        edge_length = edge_vec.norm(dim=-1)  # (E,)
        rgrid = torch.arange(0, r_cutoff, dr, device=edge_vec.device)
        sigma = r_cutoff / 3

        # Gaussian radial basis encoding
        edge_attr = torch.exp(-0.5 * (edge_length[:, None] - rgrid[None, :]) ** 2 / sigma**2)
        edge_attr = edge_attr / (sigma * (2 * math.pi) ** 0.5)  # optional normalization
        return edge_attr
    
    def forward(self, data):
        x = data.x
        edge_index = data.edge_index
        edge_vec = data.edge_vec
        batch = data.batch

        edge_attr = self.calculate_edge_attr(edge_vec, r_cutoff=self.r_cutoff, dr=self.dr)

        global_feature = 0
        for conv in self.Conv_list:
            x = conv(x, edge_index=edge_index, edge_attr=edge_attr)
            x = F.relu(x)
            global_feature += gmp(x, batch)

        global_feature = F.relu(self.bn(self.lin1(global_feature)))
        global_feature = self.lin2(global_feature)
        return global_feature.squeeze()
    
class GATv2wrapper(nn.Module):
    def __init__(self, in_dim, em_dim, gat_model):
        super().__init__()
        self.gat_model = gat_model
        self.em_z = nn.Linear(in_dim, em_dim)
        self.em_x = nn.Linear(in_dim, em_dim)

    def forward(self, data):
        data.z = F.relu(self.em_z(data.z))
        data.x = F.relu(self.em_x(data.x))
        return self.gat_model(data)    

INPUT_FEATURE_DIM = 118
EMBEDDING_DIM = 96

# Determine edge feature dimension
r_cutoff = 6.0
dr = 0.01
num_edge_features = int(r_cutoff / dr)  # Same as len(rgrid)

# Define model
model = GAT(
    hidden_channels=EMBEDDING_DIM,
    num_hidden_layers=4,
    num_edge_features=num_edge_features,  # <-- match with RBF size
    num_heads=8,
    r_cutoff=r_cutoff,
    dr=dr
)

model = GATv2wrapper(INPUT_FEATURE_DIM, EMBEDDING_DIM, model)



model.to(device)

total_params = sum(param.numel() for param in model.parameters())
trainable_params = sum(param.numel() for param in model.parameters() if param.requires_grad)

print(f"Total parameters: {total_params}")
print(f"Trainable parameters: {trainable_params}")

# for name, param in model.named_parameters():
#     print(f"{name}: {param.shape} requires_grad={param.requires_grad}")
        
run_name = f'gatv2_model_{run_time}'
# opt = torch.optim.AdamW(model.parameters(), lr=7e-3, weight_decay=0.05)
opt = torch.optim.Adam(model.parameters(), lr=7e-3, weight_decay=0.05)
# scheduler = torch.optim.lr_scheduler.ExponentialLR(opt, gamma=0.96)
# scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, 'min')
max_iter = 100

# scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
#     opt,
#     mode='min',
#     factor=0.5,       # instead of default 0.1
#     patience=5,      # wait longer before reducing
#     min_lr=0,
#     verbose=True
# )
scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
    opt,
    T_0=10, T_mult=1,
    eta_min=0     # Minimum learning rate (optional, default is 0)
)

loss_fn = torch.nn.L1Loss()
loss_fn_mae = torch.nn.L1Loss()

loss_fn_eval = torch.nn.L1Loss()
loss_fn_mae_eval = torch.nn.L1Loss()

use_weighting = False
wandb.init(
    project="Tensor Predict Imaginary GATv2",  # Change this to your project name
    name=run_name,  # Unique identifier for this run
    config={
        "max_iter": max_iter,
        "lr": opt.param_groups[0]["lr"],  # Log learning rate
        "use_weighting": use_weighting,
        "r_max": r_max,
        "batch_size": batch_size,
        "normalization": True if scale_data != 1 else False,
        "energy_max": energy_max,
        "nstep": nstep,
        "scheduler": type(scheduler).__name__,  # Log scheduler type
        "loss_function": type(loss_fn).__name__  # Automatically log loss function type
    }
)

train(model, opt, dataloader_train, dataloader_valid, loss_fn, loss_fn_mae, loss_fn_eval, loss_fn_mae_eval, run_name, 
    max_iter=max_iter, scheduler=scheduler, device=device)

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
        
        irreps_0e = 300
        irreps_2e = 1500
        out_dim = 300
        
        output_0e = output[:, :irreps_0e]  # Shape: (batch_size, irreps_0e)
        output_2e = output[:, irreps_0e:irreps_0e + irreps_2e].contiguous().view(output.shape[0], out_dim, 5)  # Shape: (batch_size, 301, 5)

        y_0e = d.y[:, :, 0].view(d.y.shape[0], out_dim) 
        y_2e = d.y[:, :, 1:].view(d.y.shape[0], out_dim, 5)  # Shape: (batch_size, 301, 5)

        loss_0e = F.mse_loss(output_0e, y_0e)   
        loss_2e = F.mse_loss(output_2e, y_2e)   
        loss = loss_0e + loss_2e
        
        combined_output = torch.cat([output_0e.unsqueeze(2), output_2e], dim=2)  # Shape: (batch_size, 301, 6)
        predictions.append(combined_output.cpu())

        for batch_idx in range(d.y.shape[0]):
            df.loc[i0 + batch_idx, 'y_pred_sph'] = [combined_output[batch_idx].cpu().numpy()]
            # df.loc[i0 + batch_idx, 'y_pred_cart'] = [realsphvec2cart(combined_output[batch_idx].cpu().numpy())]
            df.loc[i0 + batch_idx, 'mse_sph'] = loss.cpu().numpy() * scale_data

        # Update batch index counter
        i0 += d.y.shape[0]

column = 'imag_Permittivity_Matrices_interp'

df['y_pred_sph'] = df['y_pred_sph'].map(lambda x: x[0]) * scale_data

# Convert all spherical tensors to a batched tensor
sph_tensors = torch.tensor(np.stack(df['y_pred_sph'].values))  # Batch process
# Convert using x.to_cartesian in batch
cart_tensors = x.to_cartesian(sph_tensors)
# Assign back to the DataFrame
df['y_pred_cart'] = list(cart_tensors.numpy())  # Convert back to list of NumPy arrays

# Convert to NumPy arrays
cart_true = np.stack(df[column].values)  # Shape: (num_samples, 301, 3, 3)
cart_pred = np.stack(df['y_pred_cart'].values)  # Shape: (num_samples, 301, 3, 3)

# Convert to PyTorch tensors
cart_true_tensor = torch.tensor(cart_true, dtype=torch.float64)
cart_pred_tensor = torch.tensor(cart_pred, dtype=torch.float64)

# Define upper diagonal indices of symmetric 3x3 tensor
inds_diag = [(0, 0), (1, 1), (2, 2)]
inds_off = [(0, 1), (0, 2), (1, 2)]

# Compute symmetric-only MAE and MSE
def compute_symmetric_errors(pred, true):
    diffs = []
    for i, j in inds_diag + inds_off:
        diff = pred[:, :, i, j] - true[:, :, i, j]  # shape: (N, freq)
        diffs.append(diff)
    diffs = torch.stack(diffs, dim=0)  # shape: (6, N, freq)
    mse = torch.mean(diffs ** 2, dim=0)  # shape: (N, freq)
    mae = torch.mean(torch.abs(diffs), dim=0)  # shape: (N, freq)
    return mse, mae

# Compute and assign
mse_torch, mae_cart = compute_symmetric_errors(cart_pred_tensor, cart_true_tensor)
mse_torch = mse_torch.cpu().numpy()  # shape: (N, freq)
mae_cart = mae_cart.cpu().numpy()    # shape: (N, freq)

sph_true = np.stack(df['sph_coefs'].values)  # Shape: (num_samples, 301, 3, 3)
sph_pred = np.stack(df['y_pred_sph'].values)  # Shape: (num_samples, 301, 3, 3)
# Convert to PyTorch tensors

sph_true_tensor = torch.tensor(sph_true, dtype=torch.float64)
sph_pred_tensor = torch.tensor(sph_pred, dtype=torch.float64)


# Store the MSE values in the DataFrame
df['mse_cart'] = np.mean(mse_torch, axis=1)
df['mae_cart'] = np.mean(mae_cart, axis=1)
mae_sph = torch.mean(torch.abs(sph_pred_tensor - sph_true_tensor), dim=(1, 2)).cpu().numpy() 
df['mae_sph'] = mae_sph


mse_sph_mean = df['mse_sph'].mean()
mse_sph_std = df['mse_sph'].std()
mse_cart_mean = df['mse_cart'].mean()
mse_cart_std = df['mse_cart'].std()

mae_sph_mean = df['mae_sph'].mean()
mae_sph_std = df['mae_sph'].std()
mae_cart_mean = df['mae_cart'].mean()
mae_cart_std = df['mae_cart'].std()

wandb.log({
    "Mean MSE in cart": mse_cart_mean,
    "Std MSE in cart": mse_cart_std,
    "Mean MSE in sph": mse_sph_mean,
    "Std MSE in sph": mse_sph_std
})

def get_random_sample_indices(idx, n):
    """Returns `n` randomly selected unique indices from `idx`."""
    if len(idx) < n:
        n = len(idx)  # Ensure we don't exceed available samples
    return np.random.choice(idx, size=n, replace=False)

n_samples = 12  # Adjust as needed

random_idx_train = get_random_sample_indices(idx_train, n_samples)
random_idx_valid = get_random_sample_indices(idx_valid, n_samples)
random_idx_test = get_random_sample_indices(idx_test, n_samples)

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
wandb.finish()
