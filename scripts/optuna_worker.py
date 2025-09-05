import os 
import sys
# Get the absolute path of the parent directory
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(parent_dir)  # Add parent directory to sys.path

gpu_id = int(sys.argv[1])  # You pass GPU id from main launcher
os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

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


bar_format = '{l_bar}{bar:10}{r_bar}{bar:-10b}'
default_dtype = torch.float64
torch.set_default_dtype(default_dtype)

# Create a colormap based on the number of unique symbols
datasets = ['g', 'y', 'r']
colors = dict(zip(datasets, palette))
cmap = mpl.colors.LinearSegmentedColormap.from_list('cmap', [palette[k] for k in [0,2,1]])

def get_neighbors(df, idx):
    n = []
    for entry in df.iloc[idx].itertuples():
        N = entry.data.pos.shape[0]
        for i in range(N):
            n.append(len((entry.data.edge_index[0] == i).nonzero()))
    return np.array(n)

# === get dataloaders and metadata ===
def get_dataloaders_and_metadata():
    data_file = '../dataset/symmetrized_dataset.pkl'
    df, species = load_data(data_file)
    df = df.reset_index(drop=True)

    energy_min = 0
    energy_max = 30
    nstep = 300
    new_x = np.linspace(energy_min, energy_max, nstep)

    def interpolate_matrix(matrix, omega):
        interp = interp1d(omega, matrix, kind='linear', axis=0, fill_value=0, bounds_error=False)
        return interp(new_x)

    df['imag_Permittivity_Matrices_interp'] = [
        interpolate_matrix(row['imag_symmetrized_permittivity'], row['omega']) for _, row in df.iterrows()
    ]
    df['energies_interp'] = df.apply(lambda x: new_x, axis=1)

    stack_matrices_tensor = torch.tensor(np.stack(df['imag_Permittivity_Matrices_interp'].values), dtype=torch.float64)
    x = CartesianTensor("ij=ji")
    sph_coefs_tensor = x.from_cartesian(stack_matrices_tensor)
    df['sph_coefs'] = list(sph_coefs_tensor.numpy())

    type_onehot, mass_onehot, dipole_onehot, radius_onehot, type_encoding = save_or_load_onehot()

    tmp = np.array([df.iloc[i]['sph_coefs'] for i in range(len(df))])
    scale_data = np.median(np.max(np.abs(tmp), axis=(1, 2)))

    r_max = 6.
    df['data'] = df.progress_apply(lambda x: build_data(x, 'sph_coefs', scale_data,
                                                        type_onehot, mass_onehot, dipole_onehot,
                                                        radius_onehot, type_encoding, r_max), axis=1)

    idx_train, idx_valid, idx_test = train_valid_test_split(df, valid_size=.1, test_size=.1, plot=False)
    dataloader_train = tg.loader.DataLoader(df.iloc[idx_train]['data'].values, batch_size=4, shuffle=True)
    dataloader_valid = tg.loader.DataLoader(df.iloc[idx_valid]['data'].values, batch_size=4)
    
    out_dim = len(df.iloc[0]['energies_interp'])
    torch.save({
    "df": df,
    "idx_train": idx_train,
    "idx_valid": idx_valid,
    "idx_test": idx_test,
    "scale_data": scale_data,
    "r_max": r_max,
    "out_dim": out_dim
    }, "../dataset/cached_preprocessed_data.pt")

def get_dataloaders_from_cache():
    cache = torch.load("../dataset/cached_preprocessed_data.pt")

    df = cache["df"]
    idx_train = cache["idx_train"]
    idx_valid = cache["idx_valid"]
    out_dim = cache["out_dim"]
    r_max = cache["r_max"]
    n_train = get_neighbors(df, idx_train)

    dataloader_train = tg.loader.DataLoader(df.iloc[idx_train]['data'].values, batch_size=4, shuffle=True)
    dataloader_valid = tg.loader.DataLoader(df.iloc[idx_valid]['data'].values, batch_size=4)

    return dataloader_train, dataloader_valid, out_dim, r_max, n_train


class PeriodicNetwork(Network):
    def __init__(self, in_dim, em_dim, **kwargs):            
        # override the `reduce_output` keyword to instead perform an averge over atom contributions    
        self.pool = False
        if kwargs['reduce_output'] == True:
            kwargs['reduce_output'] = False
            self.pool = True
            
        super().__init__(**kwargs)

        self.em_z = nn.Linear(in_dim, em_dim)    #Linear layer for atom type
        self.em_x = nn.Linear(in_dim, em_dim)    #Linear layer for atom type

    def forward(self, data: Union[tg.data.Data, Dict[str, torch.Tensor]]) -> torch.Tensor:
        data.z = F.relu(self.em_z(data.z))
        data.x = F.relu(self.em_x(data.x))

        output = super().forward(data)
        
        # if pool_nodes was set to True, use scatter_mean to aggregate
        if self.pool == True:
            output = torch_scatter.scatter_mean(output, data.batch, dim=0)  # take mean over atoms per example
            # output = torch_scatter.scatter_add(output, data.batch, dim=0)  # take mean over atoms per example
            # output, _ = torch_scatter.scatter_max(output, data.batch, dim=0)  # max over atoms per examples
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
    
        
loss_fn = torch.nn.L1Loss()
loss_fn_mse = torch.nn.MSELoss()

loss_fn_eval = torch.nn.L1Loss()
loss_fn_mse_eval = torch.nn.MSELoss()

import optuna
import shutil

# Optional: limit PyTorch reproducibility
# torch.manual_seed(0)

def objective(trial):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataloader_train, dataloader_valid, out_dim, r_max, n_train = get_dataloaders_from_cache()

    # === Sample Hyperparameters ===
    # em_dim = trial.suggest_categorical("em_dim", [32, 64, 96, 128])
    em_dim = 64
    # dropout_prob = trial.suggest_float("dropout_prob", 0.0, 0.5)
    dropout_prob = 0.4
    # lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
    lr = 1e-2
    # mul = trial.suggest_categorical("mul", [16, 32, 64]) # done with testing
    mul = 32
    layers = trial.suggest_int("layers", 2, 4)
    # layers = 2
    # use_batch_norm = trial.suggest_categorical("use_batch_norm", [True, False])
    use_batch_norm = False
    # scheduler_type = trial.suggest_categorical("scheduler", ["cosine", "plateau"])
    scheduler_type = "cosine"  

    # === Construct Model ===
    model = PeriodicNetwork(
        in_dim=118,
        em_dim=em_dim,
        irreps_in=f"{em_dim}x0e",
        irreps_out=f"{out_dim}x0e + {out_dim}x2e",
        irreps_node_attr=f"{em_dim}x0e",
        layers=layers,
        mul=mul,
        lmax=2,
        max_radius=r_max,
        num_neighbors=n_train.mean(),
        reduce_output=True,
        dropout_prob=dropout_prob,
        use_batch_norm=use_batch_norm
    ).to(device)

    # === Optimizer and Scheduler ===
    loss_balancer = LearnableUncertaintyLoss().to(device)

    # opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.05)
    opt = torch.optim.AdamW(list(model.parameters()) + list(loss_balancer.parameters()), lr=lr, weight_decay=0.05)

    if scheduler_type == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            opt, T_0=10, T_mult=1, eta_min=0
        )
    elif scheduler_type == "plateau":
        scheduler = torch.optim.lr_scheduler.ExponentialLR(opt, gamma=0.96)
    else:
        raise ValueError(f"Unknown scheduler: {scheduler_type}")
    # === Train ===
    trial_run_name = f"optuna_trial_{trial.number}"
    wandb.init(project="Tensor Optuna Sweep with learnable balancing", name=f"trail_{trial.number}_gpu{gpu_id}", config={
        "em_dim": em_dim,
        "dropout_prob": dropout_prob,
        "lr": lr,
        "layers": layers,
        "mul": mul,
        "scheduler": scheduler_type
    })

    train(
        model, opt, dataloader_train, dataloader_valid,
        loss_fn, loss_fn_mse, loss_fn_eval, loss_fn_mse_eval,
        run_name=trial_run_name,
        max_iter=100,  # Fewer iterations for quick optimization
        scheduler=scheduler,
        device=device,
        disable_tqdm=True,
        # loss_balancer=None
        loss_balancer=loss_balancer
    )

    wandb.finish()

    # === Evaluate Best Validation Loss ===
    try:
        hist = torch.load(f"../model/{trial_run_name}.torch", map_location=device)["history"]
        val_losses = [d["valid"]["loss"] for d in hist]
        best_val_loss = min(val_losses)
    except Exception as e:
        print(f"[Trial {trial.number}] Failed to load model history: {e}")
        best_val_loss = np.inf

    # # Optional: clean up to save space
    # try:
    #     os.remove(f"../model/{trial_run_name}.torch")
    #     os.remove(f"../model/{trial_run_name}_best.torch")
    # except:
    #     pass

    return best_val_loss

# # === Optuna Study w/ 1 GPU===
# study = optuna.create_study(
#     direction="minimize",
#     study_name="tensor_spectrum_tuning"
# )
# study.optimize(objective, n_trials=30, timeout=None)

# # Print best result
# print("Best trial:")
# best_trial = study.best_trial
# print("  Value (validation loss):", best_trial.value)
# print("  Params:")
# for k, v in best_trial.params.items():
#     print(f"    {k}: {v}")

def main():
    import optuna
    from optuna.exceptions import DuplicatedStudyError
    from optuna.samplers import GridSampler

    storage = "sqlite:///optuna_tensor_study.db"
    study_name = "tensor_spectrum_tuning"

    ## Option 1: Repeating old study
    try:
        optuna.create_study(
            direction="minimize",
            study_name=study_name,
            storage=storage
        )
    except DuplicatedStudyError:
        pass  # Already created

    # Load study
    study = optuna.load_study(study_name=study_name, storage=storage)
    study.optimize(objective, n_trials=10, timeout=None)

    print(f"Study '{study_name}' has {len(study.trials)} trials.")

    print("Best trial:")
    best_trial = study.best_trial
    print(f"  Value (validation loss): {best_trial.value}")
    for k, v in best_trial.params.items():
        print(f"    {k}: {v}")
    # ## Option 2: Create a new study
    # try:
    #     optuna.delete_study(study_name=study_name, storage=storage)
    #     print(f"Deleted existing study: {study_name}")
    # except KeyError:
    #     print(f"No existing study named {study_name} to delete.")

    # # Recreate fresh study
    # study = optuna.create_study(
    #     direction="minimize",
    #     study_name=study_name,
    #     storage=storage
    # )    
    # study.optimize(objective, n_trials=1, timeout=None)

# def main():
    # get_dataloaders_and_metadata()

if __name__ == "__main__":
    main()