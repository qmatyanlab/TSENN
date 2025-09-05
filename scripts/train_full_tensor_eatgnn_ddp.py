import os
import sys
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(parent_dir)  # Add parent directory to sys.path
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import torch_geometric as tg
import wandb
# from utils.utils_model_full_tensor import train

from utils.utils_data import load_data, train_valid_test_split, save_or_load_onehot, build_data, plot_spherical_harmonics_comparison, plot_cartesian_tensor_comparison
from utils.eatgnn_ori import Network, train
from e3nn.io import CartesianTensor
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from tqdm import tqdm
import matplotlib.pyplot as plt
bar_format = '{l_bar}{bar:10}{r_bar}{bar:-10b}'
default_dtype = torch.float64
torch.set_default_dtype(default_dtype)
torch.manual_seed(3407)


def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def cleanup():
    dist.destroy_process_group()

def interpolate_matrix(matrix, omega, new_x):
    interp = interp1d(omega, matrix, kind='linear', axis=0, fill_value=0, bounds_error=False)
    return interp(new_x)

def get_neighbors(df, idx):
    n = []
    for entry in df.iloc[idx].itertuples():
        N = entry.data.pos.shape[0]
        for i in range(N):
            n.append(len((entry.data.edge_index[0] == i).nonzero()))
    return np.array(n)

def main_worker(rank, world_size):
    setup(rank, world_size)
    device = torch.device(f"cuda:{rank}")
    default_dtype = torch.float64
    torch.set_default_dtype(default_dtype)
    # Load and preprocess data
    data_file = '../dataset/symmetrized_dataset.pkl'
    df, species = load_data(data_file)
    df = df.reset_index(drop=True)

    energy_min, energy_max, nstep = 0, 30, 300
    new_x = np.linspace(energy_min, energy_max, nstep)

    df['imag_Permittivity_Matrices_interp'] = [
        interpolate_matrix(row['imag_symmetrized_permittivity'], row['omega'], new_x) for _, row in df.iterrows()
    ]
    df['energies_interp'] = df.apply(lambda x: new_x, axis=1)

    stack_matrices_tensor = torch.tensor(np.stack(df['imag_Permittivity_Matrices_interp'].values), dtype=default_dtype, device=device)
    x = CartesianTensor("ij=ji")
    sph_coefs_tensor = x.from_cartesian(stack_matrices_tensor)
    df['sph_coefs'] = list(sph_coefs_tensor.cpu().numpy())

    type_onehot, mass_onehot, dipole_onehot, radius_onehot, type_encoding = save_or_load_onehot()
    scale_data = np.median(np.max(np.abs(np.stack(df['sph_coefs'].values)), axis=(1, 2)))

    r_max = 6.
    df['data'] = df.progress_apply(lambda x: build_data(x, 'sph_coefs', scale_data, type_onehot, mass_onehot, dipole_onehot, radius_onehot, type_encoding, r_max), axis=1)
    
    run_time = '250713'
    # idx_train, idx_valid, idx_test = train_valid_test_split(df, valid_size=.1, test_size=.1, plot=False)
    with open('../model/idx_train_'+run_time+'.txt', 'r') as f: idx_train = [int(i.split('\n')[0]) for i in f.readlines()]
    with open('../model/idx_valid_'+run_time+'.txt', 'r') as f: idx_valid = [int(i.split('\n')[0]) for i in f.readlines()]
    with open('../model/idx_test_'+run_time+'.txt', 'r') as f: idx_test = [int(i.split('\n')[0]) for i in f.readlines()]

    batch_size = 4
    train_sampler = DistributedSampler(df.iloc[idx_train]['data'].values, num_replicas=world_size, rank=rank, shuffle=True)
    dataloader_train = tg.loader.DataLoader(df.iloc[idx_train]['data'].values, sampler=train_sampler, batch_size=batch_size)
    dataloader_valid = tg.loader.DataLoader(df.iloc[idx_valid]['data'].values, batch_size=batch_size)

    n_train = get_neighbors(df, idx_train)

    INPUT_FEATURE_DIM = 118
    EMBEDDING_DIM = 64
    number_of_basis = 10  # Number of basis functions for the radial basis function expansion
    trial_params = {'irreps_0e': 32,
                    'irreps_1e': 16,
                    'irreps_2e': 8,
                    'irreps_3e': 4}
    irreps_query = f"{trial_params['irreps_0e']}x0e+{trial_params['irreps_1e']}x1e+{trial_params['irreps_2e']}x2e+{trial_params['irreps_3e']}x3e"
    # irreps_out = f"{trial_params['irreps_0e'] * 2}x0e+{trial_params['irreps_1e'] * 2}x1e+{trial_params['irreps_2e'] * 2}x2e + {trial_params['irreps_3e'] * 2}x3e"
    irreps_out = f"300x0e+300x2e"
    model = Network(
        irreps_in="{}x0e".format(EMBEDDING_DIM),
        embedding_dim=EMBEDDING_DIM,
        irreps_query=irreps_query,
        irreps_key=irreps_query,
        irreps_out = irreps_out,
        formula="ij=ji",
        lmax=0, 
        max_radius=r_max,
        number_of_basis = number_of_basis,
        num_nodes=n_train.mean(),
        pool_nodes=True,
    )
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
    
    class NetWrapper(nn.Module):
        def __init__(self, in_dim, em_dim, gat_model):
            super().__init__()
            self.gat_model = gat_model
            self.em_z = nn.Linear(in_dim, em_dim)
            self.em_x = nn.Linear(in_dim, em_dim)

        def forward(self, data):
            data.z = F.relu(self.em_z(data.z))
            data.x = F.relu(self.em_x(data.x))
            return self.gat_model(data)

    model = NetWrapper(INPUT_FEATURE_DIM, EMBEDDING_DIM, model)
    model.to(device)
    model = DDP(model, device_ids=[rank], find_unused_parameters=True)

    loss_balancer = LearnableUncertaintyLoss().to(device)
    # opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.05)
    opt = torch.optim.AdamW(list(model.parameters()) + list(loss_balancer.parameters()), lr=1e-2, weight_decay=0.05)
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(opt, T_0=10, T_mult=1)
    loss_fn = torch.nn.L1Loss()
    loss_fn_mse = torch.nn.MSELoss()

    loss_fn_eval = torch.nn.L1Loss()
    loss_fn_mse_eval = torch.nn.MSELoss()
    max_iter = 100
    run_name = 'TOSENN-A_Lmax=0_test'
    if rank == 0:
        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Total trainable parameters: {total_params}")
        wandb.init(
            project="Tensor Predict EATGNN",
            name=f'{run_name}',
            config={
                "max_iter": max_iter,
                "lr": opt.param_groups[0]["lr"],
                "batch_size": batch_size,
                "dropout_prob": 0.4,
                "scheduler": type(scheduler).__name__,
                "loss_function": type(loss_fn).__name__
            }
        )
        wandb.watch(model.module, log='all', log_freq=100)  # model.module = unwrap from DDP


    train(model, opt, dataloader_train, dataloader_valid, loss_fn, loss_fn_mse, loss_fn_eval, loss_fn_mse_eval, 
          run_name, max_iter=max_iter, scheduler=scheduler, device=device,
          loss_balancer=loss_balancer)

    if rank == 0:
        # run_name = f"run_gpu_{rank}"  # make sure this matches the saved model
        # model_path = f"../model/{run_name}_best.torch"
        # history_path = f"../model/{run_name}.torch"

        history = torch.load('../model/' + run_name + '_best.torch', map_location=device)['history']
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
        checkpoint = torch.load(f'../model/{run_name}_best.torch', map_location=device)
        state_dict = checkpoint['state']

        if isinstance(model, torch.nn.parallel.DistributedDataParallel):
            model.module.load_state_dict(state_dict)
        else:
            model.load_state_dict(state_dict)
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
                
                # irreps_0e = model.irreps_out.count(o3.Irrep("0e"))
                # irreps_2e = model.irreps_out.count(o3.Irrep("2e")) * 5
                # out_dim = model.irreps_out.count(o3.Irrep("0e")) 
                irreps_0e = 300
                irreps_2e = 300 * 5
                out_dim = 300

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
        cart_tensors = x.to_cartesian(sph_tensors)
        df['y_pred_cart'] = list(cart_tensors.numpy())  # Convert back to list of NumPy arrays

        cart_true = np.stack(df[column].values)  # Shape: (num_samples, 300, 3, 3)
        cart_pred = np.stack(df['y_pred_cart'].values)  # Shape: (num_samples, 300, 3, 3)

        # Convert to PyTorch tensors
        cart_true_tensor = torch.tensor(cart_true, dtype=default_dtype)
        cart_pred_tensor = torch.tensor(cart_pred, dtype=default_dtype)

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

        sph_true_tensor = torch.tensor(sph_true, dtype=default_dtype)
        sph_pred_tensor = torch.tensor(sph_pred, dtype=default_dtype)


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
            "Std MSE in sph": mse_sph_std,
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

    cleanup()

def main():
    world_size = 4
    mp.spawn(main_worker, args=(world_size,), nprocs=world_size, join=True)

if __name__ == "__main__":
    main()