import os
import sys
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(parent_dir)  # Add parent directory to sys.path
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import torch_geometric as tg
import numpy as np
import optuna
import wandb
import torch_scatter
from e3nn.io import CartesianTensor
from scipy.interpolate import interp1d

from utils.utils_data import (load_data, train_valid_test_split, save_or_load_onehot, build_data, plot_spherical_harmonics_comparison, plot_cartesian_tensor_comparison)
from utils.eatgnn_ori import Network, train


def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)


def cleanup():
    dist.destroy_process_group()

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
    # dataloader_train = tg.loader.DataLoader(df.iloc[idx_train]['data'].values, batch_size=4, shuffle=True)
    # dataloader_valid = tg.loader.DataLoader(df.iloc[idx_valid]['data'].values, batch_size=4)
    
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

def get_dataloaders_from_cache(rank=None, world_size=None, batch_size=2):
    cache = torch.load("../dataset/cached_preprocessed_data.pt")
    df = cache["df"]
    idx_train = cache["idx_train"]
    idx_valid = cache["idx_valid"]
    out_dim = cache["out_dim"]
    r_max = cache["r_max"]
    n_train = get_neighbors(df, idx_train)

    if rank is not None and world_size is not None:
        train_dataset = df.iloc[idx_train]['data'].values
        train_sampler = torch.utils.data.distributed.DistributedSampler(
            train_dataset, num_replicas=world_size, rank=rank, shuffle=True
        )
        dataloader_train = tg.loader.DataLoader(train_dataset, sampler=train_sampler, batch_size=batch_size)
    else:
        dataloader_train = tg.loader.DataLoader(df.iloc[idx_train]['data'].values, batch_size=batch_size, shuffle=True)

    dataloader_valid = tg.loader.DataLoader(df.iloc[idx_valid]['data'].values, batch_size=batch_size)
    return dataloader_train, dataloader_valid, out_dim, r_max, n_train, idx_train

def get_neighbors(df, idx):
    n = []
    for entry in df.iloc[idx].itertuples():
        N = entry.data.pos.shape[0]
        for i in range(N):
            n.append(len((entry.data.edge_index[0] == i).nonzero()))
    return np.array(n)

class EquiformerWrapper(nn.Module):
    def __init__(self, in_dim, em_dim, gat_model):
        super().__init__()
        self.gat_model = gat_model
        self.em_z = nn.Linear(in_dim, em_dim)
        self.em_x = nn.Linear(in_dim, em_dim)

    def forward(self, data):
        data.z = F.relu(self.em_z(data.z))
        data.x = F.relu(self.em_x(data.x))
        return self.gat_model(data)

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


def ddp_worker(rank, world_size, trial_params):
    setup(rank, world_size)
    device = torch.device(f"cuda:{rank}")

    dataloader_train, dataloader_valid, out_dim, r_max, n_train, idx_train = get_dataloaders_from_cache(rank, world_size, batch_size=trial_params["batch_size"])
    
    INPUT_FEATURE_DIM = 118
    EMBEDDING_DIM = trial_params["embedding_dim"]
    # EMBEDDING_DIM = 64
    irreps_query = f"{trial_params['irreps_0e']}x0e+{trial_params['irreps_1e']}x1e+{trial_params['irreps_2e']}x2e"
    irreps_out = f"{trial_params['irreps_0e'] * 2}x0e+{trial_params['irreps_1e']}x1e+{trial_params['irreps_2e']}x2e"

    model = Network(
        irreps_in=f"{EMBEDDING_DIM}x0e",
        embedding_dim=EMBEDDING_DIM,
        irreps_query=irreps_query,
        irreps_key=irreps_query,
        irreps_out=irreps_out,
        formula="ij=ji",
        mul=trial_params["mul"],
        lmax=2,
        max_radius=trial_params["r_max"],
        number_of_basis=trial_params["number_of_basis"],
        num_nodes=n_train.mean(),
        pool_nodes=True,
    ).to(device)
    model = EquiformerWrapper(INPUT_FEATURE_DIM, EMBEDDING_DIM, model)
    model.to(device)
    model = DDP(model, device_ids=[rank], find_unused_parameters=True)
    loss_balancer = LearnableUncertaintyLoss().to(device)
    opt = torch.optim.AdamW(list(model.parameters()) + list(loss_balancer.parameters()), lr=trial_params["lr"], weight_decay=0.05)

    if trial_params["scheduler"] == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(opt, T_0=10)
    else:
        scheduler = torch.optim.lr_scheduler.ExponentialLR(opt, gamma=0.96)

    run_name = f"eatgnn_optuna_trial_{trial_params['trial_number']}"
    if rank == 0:
        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Total trainable parameters: {total_params}")
        wandb.init(project="EATGNN Sweep with DDP", name=run_name, config=trial_params)

    train(
        model, opt, dataloader_train, dataloader_valid,
        nn.L1Loss(), nn.MSELoss(), nn.L1Loss(), nn.MSELoss(),
        run_name=run_name, max_iter=100, scheduler=scheduler,
        device=device, disable_tqdm=True, loss_balancer=loss_balancer,
        train_sampler=dataloader_train.sampler 
    )

    if rank == 0:
        wandb.finish()
        try:
            hist = torch.load(f"../model/{run_name}.torch", map_location=device)["history"]
            val_losses = [d["valid"]["loss"] for d in hist]
            best_val_loss = min(val_losses)
        except Exception:
            best_val_loss = float("inf")

        with open(f"optuna_trial_{trial_params['trial_number']}_score.txt", "w") as f:
            f.write(str(best_val_loss))

    cleanup()


def objective(trial):
    trial_params = {
        # "batch_size": trial.suggest_categorical("batch_size", [2, 4, 8]),
        "batch_size": 4,
        # "lr": trial.suggest_float("lr", 5e-3, 1e-2, log=True),
        "lr": 1e-2, 
        "scheduler": "cosine",
        "mul": trial.suggest_categorical("mul", [32, 48, 64]),
        "embedding_dim": trial.suggest_categorical("embedding_dim", [32, 64, 96]),
        "irreps_0e": trial.suggest_categorical("irreps_0e", [16, 32, 64]),
        "irreps_1e": trial.suggest_categorical("irreps_1e", [8, 16, 32]),
        "irreps_2e": trial.suggest_categorical("irreps_2e", [4, 8, 16]),
        "number_of_basis": trial.suggest_categorical("number_of_basis", [10 , 20, 40]),
        "r_max": 6,
        "trial_number": trial.number,
    }

    world_size = 4
    mp.spawn(ddp_worker, args=(world_size, trial_params), nprocs=world_size, join=True)

    try:
        with open(f"optuna_trial_{trial.number}_score.txt", "r") as f:
            return float(f.read())
    except:
        return float("inf")


def main():
    study = optuna.create_study(direction="minimize", study_name="tensor_spectrum_ddp")
    study.optimize(objective, n_trials=25)

    print("Best trial:")
    print(study.best_trial)


if __name__ == "__main__":
    main()
