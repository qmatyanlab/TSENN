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
from utils.utils_data import (load_data, train_valid_test_split, save_or_load_onehot, build_data, plot_spherical_harmonics_comparison, plot_cartesian_tensor_comparison, compute_aniso_mae)
from utils.utils_model_full_tensor import Network, train
import wandb
from utils.normalize_cart import (
    compute_norm_params, normalize_with_params, denormalize,
    cart_to_sph, sph_to_cart, save_norm_params, load_norm_params
)


bar_format = '{l_bar}{bar:10}{r_bar}{bar:-10b}'
default_dtype = torch.float64
torch.set_default_dtype(default_dtype)
torch.manual_seed(3407)

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
    data_file = '../dataset/symmetrized_dataset_with_bandgap.pkl'
    # data_file = '../dataset/auxiliary_dataset.pkl'
    df, species = load_data(data_file)
    df = df.reset_index(drop=True)

    energy_min = 0
    energy_max = 30
    nstep = 300
    new_x = np.linspace(energy_min, energy_max, nstep)

    def interpolate_matrix(matrix, omega):
        interp = interp1d(omega, matrix, kind='linear', axis=0, fill_value=0, bounds_error=False)
        return interp(new_x)

    df['rel_permittivity_imag_interp'] = [
        interpolate_matrix(row['rel_permittivity_imag'], row['omega']) for _, row in df.iterrows()
    ]
    df['energies_interp'] = df.apply(lambda x: new_x, axis=1)

    stack_matrices_tensor = torch.tensor(np.stack(df['rel_permittivity_imag_interp'].values), dtype=torch.float64)
    # Convert to spherical coefficients
    sph_coefs_tensor = cart_to_sph(stack_matrices_tensor)

    # Separate 0e and 2e
    scalar_0e = sph_coefs_tensor[:, :, 0]     # (N, F)
    tensor_2e = sph_coefs_tensor[:, :, 1:]    # (N, F, 5)

    # Normalize
    scale_0e = torch.mean(torch.max(torch.abs(scalar_0e), dim=1).values)
    scale_2e = torch.median(torch.max(torch.norm(tensor_2e, dim=-1), dim=1).values)
    scalar_0e /= (scale_0e + 1e-12)
    tensor_2e /= (scale_2e.unsqueeze(-1) + 1e-12)

    # Merge back normalized spherical coefficients
    sph_coefs_norm = torch.cat([scalar_0e.unsqueeze(-1), tensor_2e], dim=-1)  # (N,F,6)

    # Save spherical, not cartesian
    df['sph_coefs'] = list(sph_coefs_norm.cpu().numpy())

    type_onehot, mass_onehot, dipole_onehot, radius_onehot, type_encoding = save_or_load_onehot()

    scale_data = 1

    r_max = 6.
    df['data'] = df.progress_apply(lambda x: build_data(x, 'sph_coefs', scale_data,
                                                        type_onehot, mass_onehot, dipole_onehot,
                                                        radius_onehot, type_encoding, r_max), axis=1)

    idx_train, idx_valid, idx_test = train_valid_test_split(df, valid_size=.1, test_size=.1, seed=12,  plot=False)
    
    out_dim = len(df.iloc[0]['energies_interp'])
    torch.save({
    "df": df,
    "idx_train": idx_train,
    "idx_valid": idx_valid,
    "idx_test": idx_test,
    "scale_data": scale_data,
    "scale_0e": scale_0e.cpu().item(),
    "scale_2e": scale_2e.cpu().item(),
    "r_max": r_max,
    "out_dim": out_dim
    }, "../dataset/cached_preprocessed_data.pt")
    # torch.save({
    # "df": df,
    # "idx_train": idx_train,
    # "idx_valid": idx_valid,
    # "idx_test": idx_test,
    # "scale_data": scale_data,
    # "scale_0e": scale_0e.cpu().item(),
    # "scale_2e": scale_2e.cpu().item(),
    # "r_max": r_max,
    # "out_dim": out_dim
    # }, "../dataset/cached_auxiliary_preprocessed_data.pt")
def get_dataloaders_from_cache(batch_size=4):
    cache = torch.load("../dataset/cached_preprocessed_data.pt")
    cache = torch.load("../dataset/cached_auxiliary_preprocessed_data.pt")

    df = cache["df"]
    idx_train = cache["idx_train"]
    idx_valid = cache["idx_valid"]
    out_dim = cache["out_dim"]
    r_max = cache["r_max"]
    n_train = get_neighbors(df, idx_train)

    scale_0e = cache["scale_0e"]
    scale_2e = cache["scale_2e"]

    dataloader_train = tg.loader.DataLoader(
        df.iloc[idx_train]['data'].values,
        batch_size=batch_size, shuffle=True
    )
    dataloader_valid = tg.loader.DataLoader(
        df.iloc[idx_valid]['data'].values,
        batch_size=batch_size
    )

    return dataloader_train, dataloader_valid, out_dim, r_max, n_train, scale_0e, scale_2e

def evaluate_model(model, dataloader, df, column="rel_permittivity_imag_interp",
                   scale_0e=None, scale_2e=None, device="cpu"):
    """
    Run model on dataloader, attach predictions to df, and compute anisotropy metrics.
    Returns: updated df and perc_below (% of samples with anisotropy MAE < 0.1).
    """
    model.eval()

    df = df.copy()
    # df['y_pred_sph'] = np.empty((len(df), 0)).tolist()
    df['y_pred_sph'] = [None] * len(df)
    i0 = 0
    predictions = [] 
    with torch.no_grad():
        for i, d in tqdm(enumerate(dataloader), total=len(dataloader), bar_format=bar_format):
            d.to(device)
            output = model(d)
            
            irreps_0e = model.irreps_out.count(o3.Irrep("0e"))
            irreps_2e = model.irreps_out.count(o3.Irrep("2e")) * 5
            out_dim = model.irreps_out.count(o3.Irrep("0e")) 
            
            output_0e = output[:, :irreps_0e] * scale_0e # Shape: (batch_size, irreps_0e)
            output_2e = output[:, irreps_0e:irreps_0e + irreps_2e].contiguous().view(output.shape[0], out_dim, 5) * scale_2e  # Shape: (batch_size, 300, 5)

            y_0e = d.y[:, :, 0].view(d.y.shape[0], out_dim) * scale_0e
            y_2e = d.y[:, :, 1:].view(d.y.shape[0], out_dim, 5)  * scale_2e # Shape: (batch_size, 300, 5)

            loss_0e = F.mse_loss(output_0e, y_0e) 
            loss_2e = F.mse_loss(output_2e, y_2e) 
            loss = loss_0e + loss_2e
            
            combined_output = torch.cat([output_0e.unsqueeze(2), output_2e], dim=2)  # Shape: (batch_size, 300, 6)
            predictions.append(combined_output.cpu())

            for batch_idx in range(d.y.shape[0]):
                df.loc[i0 + batch_idx, 'y_pred_sph'] = [combined_output[batch_idx].cpu().numpy()]
                df.loc[i0 + batch_idx, 'mse_sph'] = loss.cpu().numpy() 

            i0 += d.y.shape[0]

    # Convert sph → cart
    df['y_pred_sph'] = df['y_pred_sph'].map(lambda x: x[0])
    sph_pred = torch.tensor(np.stack(df['y_pred_sph'].values), device=device)
    cart_pred = sph_to_cart(sph_pred)
    df['y_pred_cart'] = list(cart_pred.cpu().numpy())

    # === Compute anisotropy MAE ===
    df[["aniso_comp_mae", "aniso_norm_mae"]] = df.apply(compute_aniso_mae, axis=1)
    aniso_mae = df['aniso_norm_mae'].dropna().values
    threshold = 0.1
    perc_below = np.mean(aniso_mae < threshold) * 100

    return df, perc_below

class Netwrapper(Network):
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
    # batch_size = trial.suggest_categorical("batch_size", [4, 6, 8])
    batch_size =  trial.suggest_int("batch_size", 2, 12)

    dataloader_train, dataloader_valid, out_dim, r_max, n_train, scale_0e, scale_2e = \
        get_dataloaders_from_cache(batch_size=batch_size)
    # === Sample Hyperparameters ===

    em_dim = trial.suggest_categorical("em_dim", [32, 64, 96, 128])
    # em_dim = 64
    # dropout_prob = trial.suggest_float("dropout_prob", 0.0, 0.5)
    dropout_prob = 0.4
    lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
    # lr = 1e-2

    mul = trial.suggest_categorical("mul", [16, 32, 64]) # done with testing
    # mul = 32
    layers = trial.suggest_int("layers", 2, 4)
    
    lmax = 2
    # layers = 2
    # use_batch_norm = trial.suggest_categorical("use_batch_norm", [True, False])
    use_batch_norm = False
    # scheduler_type = trial.suggest_categorical("scheduler", ["cosine", "plateau"])
    scheduler_type = "cosine"  

    # === Construct Model ===
    model = Netwrapper(
        in_dim=118,
        em_dim=em_dim,
        irreps_in=f"{em_dim}x0e",
        irreps_out=f"{out_dim}x0e + {out_dim}x2e",
        irreps_node_attr=f"{em_dim}x0e",
        layers=layers,
        mul=mul,
        lmax=lmax,
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
        "scheduler": scheduler_type,
        "batch_size": batch_size,
        "lmax": lmax
    })

    train(
        model, opt, dataloader_train, dataloader_valid,
        loss_fn, loss_fn_mse, loss_fn_eval, loss_fn_mse_eval,
        run_name=trial_run_name,
        max_iter=100, 
        scheduler=scheduler,
        device=device,
        disable_tqdm=True,
        loss_balancer=loss_balancer
    )
    wandb.finish()

    # === Evaluate on validation set ===
    model.load_state_dict(torch.load('../model/' + trial_run_name + '_best.torch', map_location=device)['state'])
    cache = torch.load("../dataset/cached_preprocessed_data.pt")
    # cache = torch.load("../dataset/cached_auxiliary_preprocessed_data.pt")
    df = cache["df"]

    # Build a dataloader for the whole dataset
    dataloader_all = tg.loader.DataLoader(df['data'].values, batch_size=64)

    df, perc_below = evaluate_model(
        model, dataloader_all, df,
        scale_0e=scale_0e, scale_2e=scale_2e, device=device
    )

    # Optuna will MAXIMIZE % below threshold → set direction="maximize"
    return perc_below


def main():
    import optuna
    from optuna.exceptions import DuplicatedStudyError
    from optuna.samplers import GridSampler

    storage = "sqlite:///optuna_tensor_study.db"
    study_name = "tensor_spectrum_tuning"

    ## Option 1: Repeating old study
    try:
        optuna.create_study(
            direction="maximize",
            study_name=study_name,
            storage=storage
        )
    except DuplicatedStudyError:
        pass  # Already created

    # Load study
    study = optuna.load_study(study_name=study_name, storage=storage)
    
    study.optimize(objective, n_trials=15, timeout=None)

    print(f"Study '{study_name}' has {len(study.trials)} trials.")

    print("Best trial:")
    best_trial = study.best_trial
    print(f"  Value (validation loss): {best_trial.value}")
    print("Best trial number:", best_trial.number)
    print("Validation loss:", best_trial.value)
    print("Params:", best_trial.params)
    for k, v in best_trial.params.items():
        print(f"    {k}: {v}")
        
        
# GET DATASET!
# def main():
#     get_dataloaders_and_metadata()

if __name__ == "__main__":
    main()
