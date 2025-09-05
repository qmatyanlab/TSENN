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
from utils.utils_data import (load_data, train_valid_test_split, save_or_load_onehot, build_data, plot_spherical_harmonics_comparison, plot_cartesian_tensor_comparison)
from utils.utils_model_full_tensor import Network, train
import wandb

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

torch.manual_seed(3407)

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


# run_time = time.strftime('%y%m%d', time.localtime())
run_time = '250713'
# # train/valid/test split
# idx_train, idx_valid, idx_test = train_valid_test_split(df, valid_size=.1, test_size=.1, plot=True)
# # #Save train loss values sets
# np.savetxt('../model/idx_train_'+ run_time +'.txt', idx_train, fmt='%i', delimiter='\t')
# np.savetxt('../model/idx_valid_'+ run_time +'.txt', idx_valid, fmt='%i', delimiter='\t')
# np.savetxt('../model/idx_test_'+ run_time +'.txt', idx_test, fmt='%i', delimiter='\t')
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
em_dim = 64

use_batch_norm = False
dropout_prob=0.4

model = NetWrapper(
    in_dim=118,
    em_dim=em_dim,
    irreps_in=str(em_dim)+"x0e",
    irreps_out=str(out_dim)+"x0e +" + str(out_dim) + "x2e",
    irreps_node_attr=str(em_dim)+"x0e",
    layers=2,
    mul=32,
    lmax=1,
    max_radius=r_max,
    num_neighbors=n_train.mean(),
    reduce_output=True,
    dropout_prob=dropout_prob,
    use_batch_norm = use_batch_norm
)

model.to(device)
loss_balancer = LearnableUncertaintyLoss().to(device)
opt = torch.optim.AdamW(list(model.parameters()) + list(loss_balancer.parameters()), lr=7e-3, weight_decay=0.05)
total_params = sum(param.numel() for param in model.parameters())
trainable_params = sum(param.numel() for param in model.parameters() if param.requires_grad)

print(f"Total parameters: {total_params}")
print(f"Trainable parameters: {trainable_params}")

run_name = f'tensor_model_{run_time}_Lmax=1_with_b'
max_iter = 100

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

train(model, opt, dataloader_train, dataloader_valid, loss_fn, loss_fn_mae, loss_fn_eval, loss_fn_mae_eval, run_name, 
    max_iter=max_iter, scheduler=scheduler, device=device, alpha=1.0, beta= 1.0, loss_balancer= loss_balancer)

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
        
        irreps_0e = model.irreps_out.count(o3.Irrep("0e"))
        irreps_2e = model.irreps_out.count(o3.Irrep("2e")) * 5
        out_dim = model.irreps_out.count(o3.Irrep("0e")) 
        
        output_0e = output[:, :irreps_0e]  # Shape: (batch_size, irreps_0e)
        output_2e = output[:, irreps_0e:irreps_0e + irreps_2e].contiguous().view(output.shape[0], out_dim, 5)  # Shape: (batch_size, 300, 5)

        y_0e = d.y[:, :, 0].view(d.y.shape[0], out_dim) 
        y_2e = d.y[:, :, 1:].view(d.y.shape[0], out_dim, 5)  # Shape: (batch_size, 300, 5)

        loss_0e = F.mse_loss(output_0e, y_0e)   
        loss_2e = F.mse_loss(output_2e, y_2e)   
        loss = loss_0e + loss_2e
        
        combined_output = torch.cat([output_0e.unsqueeze(2), output_2e], dim=2)  # Shape: (batch_size, 300, 6)
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
cart_true = np.stack(df[column].values)  # Shape: (num_samples, 300, 3, 3)
cart_pred = np.stack(df['y_pred_cart'].values)  # Shape: (num_samples, 300, 3, 3)

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

sph_true = np.stack(df['sph_coefs'].values)  # Shape: (num_samples, 300, 3, 3)
sph_pred = np.stack(df['y_pred_sph'].values)  # Shape: (num_samples, 300, 3, 3)
# Convert to PyTorch tensors

sph_true_tensor = torch.tensor(sph_true, dtype=torch.float64)
sph_pred_tensor = torch.tensor(sph_pred, dtype=torch.float64)


# Store the MSE values in the DataFrame
df['mse_cart'] = np.mean(mse_torch, axis=1)
df['mae_cart'] = np.mean(mae_cart, axis=1)
mae_sph = torch.mean(torch.abs(sph_pred_tensor - sph_true_tensor), dim=(1, 2)).cpu().numpy() 
df['mae_sph'] = mae_sph

mae_sph_mean = df['mae_sph'].mean()
mae_sph_std = df['mae_sph'].std()
mae_cart_mean = df['mae_cart'].mean()
mae_cart_std = df['mae_cart'].std()

wandb.log({
    "Mean MAE in cart": mae_cart_mean,
    "Std MAE in cart": mae_cart_std,
    "Mean MAE in sph": mae_sph_mean,
    "Std MAE in sph": mae_sph_std
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
