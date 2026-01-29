import os
import sys

# Get the absolute path of the current notebook's directory
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(script_dir, ".."))

# sys.path.append(parent_dir)  # Add parent directory to sys.path
if parent_dir not in sys.path:
    sys.path.append(parent_dir)
# model
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric as tg
import torch_scatter
import e3nn
from e3nn import o3
from typing import Dict, Union

# crystal structure data
from ase import Atom, Atoms
from ase.neighborlist import neighbor_list
from ase.visualize.plot import plot_atoms
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
from e3nn.o3 import wigner_D, so3_generators

# supress error log from font
import logging
logging.getLogger('matplotlib.font_manager').setLevel(level=logging.CRITICAL)
# utilities
import time
from mendeleev import element
from tqdm import tqdm
from utils.utils_data import (load_data, train_valid_test_split, save_or_load_onehot, build_data, plot_spherical_harmonics_comparison, plot_cartesian_tensor_comparison)
from utils.utils_model_full_tensor import Network, train, evaluate
import wandb
from pymatgen.ext.matproj import MPRester
from pymatgen.analysis.structure_analyzer import SpacegroupAnalyzer
from pymatgen.io.ase import AseAtomsAdaptor

plt.rcParams["mathtext.fontset"] = "cm"

bar_format = '{l_bar}{bar:10}{r_bar}{bar:-10b}'
default_dtype = torch.float64
torch.set_default_dtype(default_dtype)

# Create a colormap based on the number of unique symbols
datasets = ['g', 'y', 'r']
colors = dict(zip(datasets, palette))
cmap = mpl.colors.LinearSegmentedColormap.from_list('cmap', [palette[k] for k in [0,2,1]])


from concurrent.futures import ProcessPoolExecutor
from matplotlib.ticker import MaxNLocator

def kramers_kronig_tensor(omega_vals, imag_tensor):
    N = len(omega_vals)
    real_tensor = np.zeros_like(imag_tensor)

    for i in range(3):
        for j in range(i, 3):
            chi_im_vals = imag_tensor[:, i, j]
            chi_re_vals = np.zeros(N)

            for k, omega in enumerate(omega_vals):
                integrand = np.zeros(N)
                for l, omega_p in enumerate(omega_vals):
                    if l != k:
                        integrand[l] = (omega_p * chi_im_vals[l]) / (omega_p**2 - omega**2)
                integral = np.trapz(integrand, omega_vals)
                chi_re_vals[k] = (2 / np.pi) * integral

            real_tensor[:, i, j] = chi_re_vals
            if i != j:
                real_tensor[:, j, i] = chi_re_vals
    return real_tensor

def process_row(row_dict):
    row = row_dict.copy()
    pred_imag_permittivity = row["y_pred_cart"]
    omega = row["energies_interp"]

    omega = np.where(omega == 0, np.finfo(float).eps, omega)
    
    pred_real_permittivity = kramers_kronig_tensor(omega, pred_imag_permittivity)

    pred_permittivity_complex = pred_real_permittivity + 1j * pred_imag_permittivity
    pred_conductivity_complex = -1j * pred_permittivity_complex / omega[:, np.newaxis, np.newaxis]

    row["pred_conductivity_complex"] = pred_conductivity_complex
    row["pred_permittivity_complex"] = pred_permittivity_complex

    return row

# --- Parallelize over rows ---
def parallel_process_df(df, n_workers=None):
    rows = df.to_dict(orient="records")
    with ProcessPoolExecutor(max_workers=36) as executor:
        results = list(executor.map(process_row, rows))
    return pd.DataFrame(results)

# Check device
device = "cuda:3" if torch.cuda.is_available() else "cpu"
print('torch device:' , device)

# Replace with your Materials Project API key
API_KEY = ""  # <----------------- PLEASE INSERT YOUR API KEY!!!!
# MATERIAL_ID = "mp-2534"  # Example Material ID (e.g., GaAs)
MATERIAL_ID = "mp-9538"
# Step 1: Query the material by Material ID and get the structure
with MPRester(API_KEY) as mpr:
    structure = mpr.get_structure_by_material_id(MATERIAL_ID)

# Step 2: Use SpacegroupAnalyzer to refine the structure
sym_prec = 1e-2
spacegroup_analyzer = SpacegroupAnalyzer(structure, symprec=sym_prec)
refined_structure = spacegroup_analyzer.get_refined_structure()

# Step 3: Convert the refined structure to an ASE Atoms object
ase_atoms = AseAtomsAdaptor.get_atoms(refined_structure)
crystal_system = spacegroup_analyzer.get_crystal_system()

# Optional: Print some details to verify
print(f"Material ID: {MATERIAL_ID}")
print(f"Spacegroup (before refinement): {spacegroup_analyzer.get_space_group_symbol()}")
print(f"Spacegroup (after refinement): {SpacegroupAnalyzer(refined_structure).get_space_group_symbol()}")
print(f"Number of atoms in ASE object: {len(ase_atoms)}")
print(f"Chemical symbols: {ase_atoms.get_chemical_symbols()}")
print(f"Cell parameters: {ase_atoms.cell}")
positions = ase_atoms.get_positions()

# Extracting the chemical formula
formula = ase_atoms.get_chemical_formula()

# Extracting atomic symbols
symbols = ase_atoms.get_chemical_symbols()
z = dict(zip(symbols, range(len(symbols))))

# Extracting cell dimensions
cell = ase_atoms.get_cell()

# Parameters
energy_min = 0  # eV
energy_max = 30  # eV
nstep = 300
new_x = np.linspace(energy_min, energy_max, nstep)

# Build structure
atoms = Atoms(symbols=symbols, positions=positions, cell=cell, pbc=True)

# Dummy permittivity tensor (symmetric 3x3 tensor for each energy)
dummy_tensor = np.zeros((nstep, 3, 3))

# Build DataFrame
df = pd.DataFrame({
    "id": [0],
    "formula": [formula],
    "symmetrized_structure": [atoms],
    "energies_interp": [new_x],
    "rel_permittivity_imag_interp": [dummy_tensor],
    "crystal_system":[crystal_system]
})

# Prepare tensor
x = CartesianTensor("ij=ji")  # Symmetric rank-2 tensor

place_holder = torch.from_numpy(df['rel_permittivity_imag_interp'].iloc[0]).float()  #complete dummy don't worry
# Convert from Cartesian to spherical irreps
sph_coefs_tensor = x.from_cartesian(place_holder)  

df['sph_coefs'] = [sph_coefs_tensor.cpu().numpy()]

type_onehot, mass_onehot, dipole_onehot, radius_onehot, type_encoding = save_or_load_onehot()

# Find the scaling value
tmp = np.array([df.iloc[i]['sph_coefs'] for i in range(len(df))])
print(tmp.shape)
n_train = 43.404443289593345
scale_data = 1 # this is unused for now
scale_0e = 12.0045
scale_2e = 2.4356
r_max = 6. # cutoff radius

# --- 1. Time Graph Building ---
start_graph = time.perf_counter()
df['data'] = df.progress_apply(
    lambda x: build_data(
        x, 'sph_coefs', scale_data,
        type_onehot, mass_onehot, dipole_onehot,
        radius_onehot, type_encoding, r_max
    ), axis=1
)
end_graph = time.perf_counter()
print(f"Graph building took {end_graph - start_graph:.2f} seconds")

run_time = '250909'
with open('../model/idx_train_'+run_time+'.txt', 'r') as f: idx_train = [int(i.split('\n')[0]) for i in f.readlines()]
with open('../model/idx_valid_'+run_time+'.txt', 'r') as f: idx_valid = [int(i.split('\n')[0]) for i in f.readlines()]
with open('../model/idx_test_'+run_time+'.txt', 'r') as f: idx_test = [int(i.split('\n')[0]) for i in f.readlines()]


class Netwrapper(Network):
    def __init__(self, in_dim, em_dim , **kwargs):            
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
        # RELU issue, from e3nn discussion, removing because it might break the symmetry
        #output = torch.relu(output)
        
        # if pool_nodes was set to True, use scatter_mean to aggregate
        if self.pool == True:
            output = torch_scatter.scatter_mean(output, data.batch, dim=0)  # take mean over atoms per example
            # output = torch_scatter.scatter_add(output, data.batch, dim=0)  # take mean over atoms per example
            # output, _ = torch_scatter.scatter_max(output, data.batch, dim=0)  # max over atoms per examples
        return output

out_dim = len(df.iloc[0]['energies_interp'])      # about 200 points
em_dim = 128
lmax = 2

use_batch_norm = False
dropout_prob=0
batch_size = 8
layers = 4
lr = 1e-2

model = Netwrapper(
    in_dim=118,
    em_dim=em_dim,
    irreps_in=str(em_dim)+"x0e",
    irreps_out=str(out_dim)+"x0e +" + str(out_dim) + "x2e",
    irreps_node_attr=str(em_dim)+"x0e",
    layers=layers,
    mul=32,
    lmax=lmax,
    max_radius=r_max,
    num_neighbors=n_train,
    reduce_output=True,
    dropout_prob=dropout_prob,
    use_batch_norm = use_batch_norm
)
run_name = f'revision_{run_time}_Lmax_{lmax}_Lr_{lr}_bs_{batch_size}_em_{em_dim}_layers_{layers}'


wandb.init(
    project="Prediction",  # Change this to your project name
    name=run_name,  # Unique identifier for this run
    config={
    }
)

# predict on all data
model.load_state_dict(torch.load('../model/'+run_name + '_best.torch', map_location=device)['state'])
model.pool = True

dataloader = tg.loader.DataLoader(df['data'].values, batch_size=64)
df['mse_sph'] = 0.
df['y_pred_sph'] = np.empty((len(df), 0)).tolist()

model.to(device)
model.eval()
# =========================
# Benchmark full pipeline for fixed N
# =========================
N = 1  # fixed number of materials
print(f"\n=== Benchmark full pipeline with N={N} ===")

# Duplicate df to simulate N materials
df_test = pd.concat([df] * N, ignore_index=True)

# --- 1. Graph building ---
t0 = time.perf_counter()
df_test['data'] = df_test.progress_apply(
    lambda x: build_data(
        x, 'sph_coefs', scale_data,
        type_onehot, mass_onehot, dipole_onehot,
        radius_onehot, type_encoding, r_max
    ), axis=1
)
t1 = time.perf_counter()
graph_time = t1 - t0
per_mat_graph_time = (graph_time / N) * 1000  # ms per material
print(f"Graph building took  {per_mat_graph_time:.2f} ms per material")

# --- 2. NN inference and y_pred_cart ---
dataloader = tg.loader.DataLoader(df_test['data'].values, batch_size=64)
df_test['y_pred_sph'] = np.empty((len(df_test), 0)).tolist()
i0 = 0

if device.startswith("cuda"):
    torch.cuda.synchronize()
t0 = time.perf_counter()

with torch.no_grad():
    for d in dataloader:
        d = d.to(device)
        output = model(d)

        irreps_0e = model.irreps_out.count(o3.Irrep("0e"))
        irreps_2e = model.irreps_out.count(o3.Irrep("2e")) * 5
        out_dim = irreps_0e

        output_0e = output[:, :irreps_0e] * scale_0e
        output_2e = output[:, irreps_0e:irreps_0e + irreps_2e].contiguous().view(
            output.shape[0], out_dim, 5
        ) * scale_2e

        combined_output = torch.cat([output_0e.unsqueeze(2), output_2e], dim=2)

        for batch_idx in range(d.y.shape[0]):
            df_test.loc[i0 + batch_idx, 'y_pred_sph'] = [combined_output[batch_idx].cpu().numpy()]
        i0 += d.y.shape[0]

if device.startswith("cuda"):
    torch.cuda.synchronize()
t1 = time.perf_counter()
inference_time = t1 - t0
# print(f"NN inference took {inference_time:.2f} seconds")
per_mat_inference_time = (inference_time / N) * 1000  # ms per material
print(f"NN inference: {per_mat_inference_time:.2f} ms per material")

# Convert spherical predictions to Cartesian
df_test['y_pred_sph'] = df_test['y_pred_sph'].map(lambda x: x[0]) * scale_data
sph_tensors = torch.tensor(np.stack(df_test['y_pred_sph'].values))
cart_tensors = x.to_cartesian(sph_tensors)
df_test['y_pred_cart'] = list(cart_tensors.numpy())

# --- 3. KK calculation (using y_pred_cart) ---
t0 = time.perf_counter()
df_out = parallel_process_df(df_test, n_workers=36)
t1 = time.perf_counter()
kk_time = t1 - t0
per_mat_kk_time = (kk_time / N) * 1000
print(f"KK (permittivity) calculation took {per_mat_kk_time:.2f} ms per material")

# --- 4. Total ---
total_time = graph_time + inference_time + kk_time
# print(f"Total pipeline runtime for N={N}: {total_time:.2f} seconds")
per_mat_total_time = (total_time / N) * 1000
print(f"Total pipeline: {per_mat_total_time:.2f} ms per material")
wandb.finish()
