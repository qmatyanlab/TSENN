from typing import Dict, Union

import torch
from torch.nn.modules.loss import _Loss
from torch_geometric.data import Data
from torch_cluster import radius_graph
import torch.nn as nn

from e3nn import o3
from e3nn.math import soft_one_hot_linspace
from e3nn.nn import Gate, Dropout, BatchNorm
from e3nn.nn.models.gate_points_2101 import Convolution, smooth_cutoff, tp_path_exists

import matplotlib.pyplot as plt
import math
import time
from tqdm import tqdm
import wandb

# format progress bar
bar_format = '{l_bar}{bar:10}{r_bar}{bar:-10b}'


# standard formatting for plots
fontsize = 16
textsize = 14
sub = str.maketrans("0123456789", "₀₁₂₃₄₅₆₇₈₉")
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['axes.linewidth'] = 1
plt.rcParams['mathtext.default'] = 'regular'
plt.rcParams['xtick.bottom'] = True
plt.rcParams['ytick.left'] = True
plt.rcParams['font.size'] = fontsize
plt.rcParams['axes.labelsize'] = fontsize
plt.rcParams['xtick.labelsize'] = fontsize
plt.rcParams['ytick.labelsize'] = fontsize
plt.rcParams['legend.fontsize'] = textsize

default_dtype = torch.float64
torch.set_default_dtype(default_dtype)

class CustomCompose(torch.nn.Module):
    def __init__(self, first, second):
        super().__init__()
        self.first = first
        self.second = second
        self.irreps_in = self.first.irreps_in
        self.irreps_out = self.second.irreps_out

    def forward(self, *input):
        x = self.first(*input)
        self.first_out = x.clone()
        x = self.second(x)
        self.second_out = x.clone()
        return x

class CustomComposeWithDropoutAndBN(torch.nn.Module):
    def __init__(self, first, second, dropout, batch_norm, use_batch_norm=True):
        super().__init__()
        self.first = first
        self.second = second
        self.dropout = dropout
        self.batch_norm = batch_norm if use_batch_norm else None  # Store batch_norm only if enabled
        self.use_batch_norm = use_batch_norm
        self.irreps_in = self.first.irreps_in
        self.irreps_out = self.second.irreps_out

    def forward(self, *input):
        x = input[0]  # Extract the node features (first argument)
        x_in = x.clone()  # Save input for residual connection
        x = self.first(*input)
        self.first_out = x.clone()
        x = self.second(x)
        self.second_out = x.clone()
        if self.use_batch_norm and self.batch_norm is not None:
            x = self.batch_norm(x)  # Apply batch normalization only if enabled
        x = self.dropout(x)  # Apply dropout
        # Add residual connection if irreps match
        if x.shape[-1] == x_in.shape[-1]:  # Check if dimensions match (implies same irreps)
            x = x + x_in
        return x
class Network(torch.nn.Module):
    r"""equivariant neural network
    Parameters
    ----------
    irreps_in : `e3nn.o3.Irreps` or None
        representation of the input features
        can be set to ``None`` if nodes don't have input features
    irreps_out : `e3nn.o3.Irreps`
        representation of the output features
    irreps_node_attr : `e3nn.o3.Irreps` or None
        representation of the nodes attributes
        can be set to ``None`` if nodes don't have attributes
    irreps_edge_attr : `e3nn.o3.Irreps`
        representation of the edge attributes
        the edge attributes are :math:`h(r) Y(\vec r / r)`
        where :math:`h` is a smooth function that goes to zero at ``max_radius``
        and :math:`Y` are the spherical harmonics polynomials
    layers : int
        number of gates (non linearities)
    max_radius : float
        maximum radius for the convolution
    number_of_basis : int
        number of basis on which the edge length are projected
    radial_layers : int
        number of hidden layers in the radial fully connected network
    radial_neurons : int
        number of neurons in the hidden layers of the radial fully connected network
    num_neighbors : float
        typical number of nodes at a distance ``max_radius``
    num_nodes : float
        typical number of nodes in a graph
    dropout_prob : float
        probability of dropping an irrep during training (default: 0.5)
    use_batch_norm : bool
        whether to apply batch normalization in the layers (default: True)
    """
    def __init__(
        self,
        irreps_in,
        irreps_out,
        irreps_node_attr,
        layers,
        mul,
        lmax,
        max_radius,
        number_of_basis=10,
        radial_layers=1,
        radial_neurons=100,
        num_neighbors=1.,
        num_nodes=1.,
        reduce_output=True,
        dropout_prob=0.5,
        use_batch_norm=True,  # New parameter
    ) -> None:
        super().__init__()
        self.mul = mul
        self.lmax = lmax
        self.max_radius = max_radius
        self.number_of_basis = number_of_basis
        self.num_neighbors = num_neighbors
        self.num_nodes = num_nodes
        self.reduce_output = reduce_output
        self.dropout_prob = dropout_prob
        self.use_batch_norm = use_batch_norm  # Store the flag

        self.irreps_in = o3.Irreps(irreps_in) if irreps_in is not None else None
        self.irreps_hidden = o3.Irreps([(self.mul, (l, p)) for l in range(lmax + 1) for p in [-1, 1]])
        self.irreps_out = o3.Irreps(irreps_out)
        self.irreps_node_attr = o3.Irreps(irreps_node_attr) if irreps_node_attr is not None else o3.Irreps("0e")
        self.irreps_edge_attr = o3.Irreps.spherical_harmonics(lmax)

        self.input_has_node_in = (irreps_in is not None)
        self.input_has_node_attr = (irreps_node_attr is not None)

        irreps = self.irreps_in if self.irreps_in is not None else o3.Irreps("0e")

        act = {
            1: torch.nn.functional.silu,
            -1: torch.tanh,
        }
        act_gates = {
            1: torch.sigmoid,
            -1: torch.tanh,
        }

        self.layers = torch.nn.ModuleList()

        for _ in range(layers):
            irreps_scalars = o3.Irreps([(mul, ir) for mul, ir in self.irreps_hidden if ir.l == 0 and tp_path_exists(irreps, self.irreps_edge_attr, ir)])
            irreps_gated = o3.Irreps([(mul, ir) for mul, ir in self.irreps_hidden if ir.l > 0 and tp_path_exists(irreps, self.irreps_edge_attr, ir)])
            ir = "0e" if tp_path_exists(irreps, self.irreps_edge_attr, "0e") else "0o"
            irreps_gates = o3.Irreps([(mul, ir) for mul, _ in irreps_gated])

            gate = Gate(
                irreps_scalars, [act[ir.p] for _, ir in irreps_scalars],  # scalar
                irreps_gates, [act_gates[ir.p] for _, ir in irreps_gates],  # gates (scalars)
                irreps_gated  # gated tensors
            )
            conv = Convolution(
                irreps,
                self.irreps_node_attr,
                self.irreps_edge_attr,
                gate.irreps_in,
                number_of_basis,
                radial_layers,
                radial_neurons,
                num_neighbors
            )
            # Add dropout and batch normalization after the gate
            dropout = Dropout(gate.irreps_out, p=dropout_prob)
            batch_norm = BatchNorm(gate.irreps_out, normalization='component') if self.use_batch_norm else None  # Instantiate only if enabled
            # batch_norm = BatchNorm(gate.irreps_out) if self.use_batch_norm else None  # Instantiate only if enabled

            # Use the new CustomCompose with dropout, batch norm, and residual connection
            self.layers.append(CustomComposeWithDropoutAndBN(conv, gate, dropout, batch_norm, use_batch_norm=self.use_batch_norm))
            irreps = gate.irreps_out

        # Final convolution layer (no gate, dropout, batch norm, or residual connection)
        self.layers.append(
            Convolution(
                irreps,
                self.irreps_node_attr,
                self.irreps_edge_attr,
                self.irreps_out,
                number_of_basis,
                radial_layers,
                radial_neurons,
                num_neighbors
            )
        )

    def preprocess(self, data: Union[Data, Dict[str, torch.Tensor]]) -> torch.Tensor:
        if 'batch' in data:
            batch = data['batch']
        else:
            batch = data['pos'].new_zeros(data['pos'].shape[0], dtype=torch.long)

        if 'edge_index' in data:
            edge_src = data['edge_index'][0]  # edge source
            edge_dst = data['edge_index'][1]  # edge destination
            edge_vec = data['edge_vec']
        else:
            edge_index = radius_graph(data['pos'], self.max_radius, batch)
            edge_src = edge_index[0]
            edge_dst = edge_index[1]
            edge_vec = data['pos'][edge_src] - data['pos'][edge_dst]

        return batch, edge_src, edge_dst, edge_vec

    def forward(self, data: Union[Data, Dict[str, torch.Tensor]]) -> torch.Tensor:
        """evaluate the network
        Parameters
        ----------
        data : `torch_geometric.data.Data` or dict
            data object containing
            - ``pos`` the position of the nodes (atoms)
            - ``x`` the input features of the nodes, optional
            - ``z`` the attributes of the nodes, for instance the atom type, optional
            - ``batch`` the graph to which the node belong, optional
        """
        batch, edge_src, edge_dst, edge_vec = self.preprocess(data)
        edge_sh = o3.spherical_harmonics(self.irreps_edge_attr, edge_vec, True, normalization='component')
        edge_length = edge_vec.norm(dim=1)
        edge_length_embedded = soft_one_hot_linspace(
            x=edge_length,
            start=0.0,
            end=self.max_radius,
            number=self.number_of_basis,
            basis='gaussian',
            cutoff=False
        ).mul(self.number_of_basis**0.5)
        edge_attr = smooth_cutoff(edge_length / self.max_radius)[:, None] * edge_sh

        if self.input_has_node_in and 'x' in data:
            assert self.irreps_in is not None
            x = data['x']
        else:
            assert self.irreps_in is None
            x = data['pos'].new_ones((data['pos'].shape[0], 1))

        if self.input_has_node_attr and 'z' in data:
            z = data['z']
        else:
            assert self.irreps_node_attr == o3.Irreps("0e")
            z = data['pos'].new_ones((data['pos'].shape[0], 1))

        for lay in self.layers:
            x = lay(x, z, edge_src, edge_dst, edge_attr, edge_length_embedded)

        if self.reduce_output:
            return scatter(x, batch, dim=0).div(self.num_nodes**0.5)
        else:
            return x


def visualize_layers(model):
    layer_dst = dict(zip(['sc', 'lin1', 'tp', 'lin2'], ['gate', 'tp', 'lin2', 'gate']))
    try: layers = model.mp.layers
    except: layers = model.layers

    num_layers = len(layers)
    num_ops = max([len([k for k in list(layers[i].first._modules.keys()) if k not in ['fc', 'alpha']])
                   for i in range(num_layers-1)])

    fig, ax = plt.subplots(num_layers, num_ops, figsize=(14,3.5*num_layers))
    for i in range(num_layers - 1):
        ops = layers[i].first._modules.copy()
        ops.pop('fc', None); ops.pop('alpha', None)
        for j, (k, v) in enumerate(ops.items()):
            ax[i,j].set_title(k, fontsize=textsize)
            v.cpu().visualize(ax=ax[i,j])
            ax[i,j].text(0.7,-0.15,'--> to ' + layer_dst[k], fontsize=textsize-2, transform=ax[i,j].transAxes)

    layer_dst = dict(zip(['sc', 'lin1', 'tp', 'lin2'], ['output', 'tp', 'lin2', 'output']))
    ops = layers[-1]._modules.copy()
    ops.pop('fc', None); ops.pop('alpha', None)
    for j, (k, v) in enumerate(ops.items()):
        ax[-1,j].set_title(k, fontsize=textsize)
        v.cpu().visualize(ax=ax[-1,j])
        ax[-1,j].text(0.7,-0.15,'--> to ' + layer_dst[k], fontsize=textsize-2, transform=ax[-1,j].transAxes)

    fig.subplots_adjust(wspace=0.3, hspace=0.5)


def loglinspace(rate, step, end=None):
    t = 0
    while end is None or t <= end:
        yield t
        t = int(t + 1 + step*(1 - math.exp(-t*rate/step)))


def evaluate(model, dataloader, loss_fn_eval, loss_fn_mae_eval, device, weighting=None):
    model.eval()
    loss_cumulative = 0.
    loss_cumulative_mae = 0.
    start_time = time.time()
    
    # Constants
    omega_points = 201
    out_dim = omega_points

    with torch.no_grad():
        for j, batch in enumerate(dataloader):
            batched_graph, wigner_Ds, labels = batch  # Unpack the batch
            batched_graph = batched_graph.to(device=device)
            labels = labels.to(device=device)
            wigner_Ds = [[w.to(device=device) for w in wigner_D] for wigner_D in wigner_Ds]

            # Forward pass
            output = model(batched_graph)  # Shape: (batch_size, 1206), e.g., (16, 1206)
            batch_size = batched_graph.batch.max() + 1
            
            # Define irreps dimensions (total across all omega points)
            irreps_0e = model.irreps_out.count(o3.Irrep("0e"))  # e.g., 201
            irreps_2e = model.irreps_out.count(o3.Irrep("2e")) * 5  # e.g., 201 * 5 = 1005
            total_dim = irreps_0e + irreps_2e  # e.g., 201 + 1005 = 1206

            # Verify output shape
            expected_shape = (batch_size, total_dim)
            assert output.shape == expected_shape, f"Expected {expected_shape}, got {output.shape}"

            # Split into 0e and 2e components
            output_0e = output[:, :irreps_0e].view(batch_size, omega_points, 1)  # Shape: (batch_size, 201, 1)
            output_2e = output[:, irreps_0e:irreps_0e + irreps_2e].contiguous().view(batch_size, out_dim, 5)  # Shape: (batch_size, 201, 5)

            # Apply Wigner-D transformations per batch element
            invariant_output_0e = []
            invariant_output_2e = []
            assert len(wigner_Ds) == batch_size, f"wigner_Ds length {len(wigner_Ds)} != batch_size {batch_size}"
            
            for i, wigner_D in enumerate(wigner_Ds):
                # Process 0e component (scalar per omega point, shape: (201, 1))
                out_0e_i = output_0e[i]  # Shape: (201, 1)
                invariant_out_0e_i = out_0e_i  # 0e is invariant, no Wigner-D needed
                invariant_output_0e.append(invariant_out_0e_i)

                # Process 2e component (tensor, shape: (201, 5))
                out_2e_i = output_2e[i]  # Shape: (201, 5)
                wigner_D_tensor = torch.stack(wigner_D)  # Shape: (N_i, 6, 6)
                wigner_D_2e = wigner_D_tensor[:, 1:, 1:]  # Shape: (N_i, 5, 5), assuming 0e is first
                transformed_2e = torch.einsum("nij,wj->wni", wigner_D_2e, out_2e_i)  # Shape: (201, N_i, 5)
                invariant_out_2e_i = transformed_2e.mean(dim=1)  # Shape: (201, 5)
                invariant_output_2e.append(invariant_out_2e_i)

            # Stack invariant outputs
            invariant_output_0e = torch.stack(invariant_output_0e)  # Shape: (batch_size, 201, 1)
            invariant_output_2e = torch.stack(invariant_output_2e)  # Shape: (batch_size, 201, 5)

            # Combine for loss computation
            invariant_output = torch.cat([invariant_output_0e, invariant_output_2e], dim=2)  # Shape: (batch_size, 201, 6)

            # Split ground truth (assuming labels match this structure)
            y = labels.view(batch_size, omega_points, 6)  # Shape: (batch_size, 201, 6)
            y_0e = y[:, :, :1]  # Shape: (batch_size, 201, 1)
            y_2e = y[:, :, 1:]  # Shape: (batch_size, 201, 5)

            # Compute weighted loss
            if weighting is not None:
                weight_broadcast = weighting.view(1, -1, 1).to(device)  # Shape: (1, 201, 1)
                loss_0e = (loss_fn_eval(invariant_output_0e, y_0e) * weight_broadcast).mean()
                loss_2e = (loss_fn_eval(invariant_output_2e, y_2e) * weight_broadcast).mean()
                loss_0e_mae = (loss_fn_mae_eval(invariant_output_0e, y_0e) * weight_broadcast).mean()
                loss_2e_mae = (loss_fn_mae_eval(invariant_output_2e, y_2e) * weight_broadcast).mean()
            else:
                loss_0e = loss_fn_eval(invariant_output_0e, y_0e).mean()
                loss_2e = loss_fn_eval(invariant_output_2e, y_2e).mean()
                loss_0e_mae = loss_fn_mae_eval(invariant_output_0e, y_0e).mean()
                loss_2e_mae = loss_fn_mae_eval(invariant_output_2e, y_2e).mean()

            # Compute total loss
            loss = loss_0e + loss_2e
            loss_mae = loss_0e_mae + loss_2e_mae

            loss_cumulative += loss.item()
            loss_cumulative_mae += loss_mae.item()

    return loss_cumulative / len(dataloader), loss_cumulative_mae / len(dataloader)



def train(model, optimizer, dataloader_train, dataloader_valid, loss_fn, loss_fn_mae, loss_fn_eval, loss_fn_mae_eval, run_name, energy_grid,
          max_iter=101, scheduler=None, device="cpu", use_weighting=True, weighting_mode="exponential_decay", lambda_decay=5.0):
    # Set default dtype to float32 to avoid dtype mismatches
    
    # Ensure model is in float32 and on device
    # Ensure energy_grid is float32
    energy_grid = energy_grid.to(device=device)

    model.to(device)
    
    if use_weighting:
        if weighting_mode == "exponential_decay":
            weighting = torch.exp(-energy_grid / lambda_decay)  # Shape: (201,)
        elif weighting_mode == "inverse_decay":
            weighting = 1 / (1 + energy_grid / lambda_decay)
        else:
            raise ValueError(f"Unknown weighting_mode: {weighting_mode}")
    else:
        weighting = torch.ones_like(energy_grid)

    # Extract irreps information
    irreps_0e = 1  # Scalar component
    irreps_2e = 5  # Rank-2 tensor component
    omega_points = 201  # Number of omega points
    irreps_dim = irreps_0e + irreps_2e  # 6
    out_dim = 201
    checkpoint_generator = loglinspace(0, 5)
    checkpoint = next(checkpoint_generator)
    start_time = time.time()

    # Load checkpoint if available
    try:
        model.load_state_dict(torch.load(run_name + '.torch')['state'])
        results = torch.load(run_name + '.torch')
        history = results['history']
        s0 = history[-1]['step'] + 1
        best_valid_loss = min(h['valid']['loss'] for h in history)
    except FileNotFoundError:
        results = {}
        history = []
        s0 = 0
        best_valid_loss = float('inf')

    for step in range(s0, max_iter):
        model.train()
        loss_cumulative = 0.
        loss_cumulative_mae = 0.

        for j, batch in tqdm(enumerate(dataloader_train), total=len(dataloader_train), bar_format="{l_bar}{bar:30}{r_bar}"):
            batched_graph, wigner_Ds, labels = batch  # Unpack the batch
            batched_graph = batched_graph.to(device=device)
            labels = labels.to(device=device)
            wigner_Ds = [[w.to(device=device) for w in wigner_D] for wigner_D in wigner_Ds]

            # Forward pass
            output = model(batched_graph)  # Shape: (batch_size, 1206), e.g., (16, 1206)
            batch_size = batched_graph.batch.max() + 1
            
            # Define irreps dimensions per omega point
            irreps_0e = model.irreps_out.count(o3.Irrep("0e"))  # e.g., 201
            irreps_2e = model.irreps_out.count(o3.Irrep("2e")) * 5  # e.g., 201 * 5 = 1005
            total_dim = irreps_0e + irreps_2e  # e.g., 201 + 1005 = 1206

            # Verify output shape
            expected_shape = (batch_size, total_dim)
            assert output.shape == expected_shape, f"Expected {expected_shape}, got {output.shape}"

            # Split into 0e and 2e components
            output_0e = output[:, :irreps_0e].view(batch_size, omega_points, 1)  # Shape: (batch_size, 201, 1), e.g., (16, 201, 1)
            output_2e = output[:, irreps_0e:irreps_0e + irreps_2e].contiguous().view(batch_size, out_dim, 5)  # Shape: (batch_size, 201, 5), e.g., (16, 201, 5)

            # Apply Wigner-D transformations per batch element
            invariant_output_0e = []
            invariant_output_2e = []
            assert len(wigner_Ds) == batch_size, f"wigner_Ds length {len(wigner_Ds)} != batch_size {batch_size}"
            
            for i, wigner_D in enumerate(wigner_Ds):
                # Process 0e component (scalar per omega point, shape: (201, 1))
                out_0e_i = output_0e[i]  # Shape: (201, 1)
                invariant_out_0e_i = out_0e_i  # 0e is invariant, no Wigner-D needed
                invariant_output_0e.append(invariant_out_0e_i)

                # Process 2e component (tensor, shape: (201, 5))
                out_2e_i = output_2e[i]  # Shape: (201, 5)
                wigner_D_tensor = torch.stack(wigner_D)  # Shape: (N_i, 6, 6)
                wigner_D_2e = wigner_D_tensor[:, 1:, 1:]  # Shape: (N_i, 5, 5), assuming 0e is first
                transformed_2e = torch.einsum("nij,wj->wni", wigner_D_2e, out_2e_i)  # Shape: (201, N_i, 5)
                invariant_out_2e_i = transformed_2e.mean(dim=1)  # Shape: (201, 5)
                invariant_output_2e.append(invariant_out_2e_i)

            # Stack invariant outputs
            invariant_output_0e = torch.stack(invariant_output_0e)  # Shape: (batch_size, 201, 1)
            invariant_output_2e = torch.stack(invariant_output_2e)  # Shape: (batch_size, 201, 5)

            # Combine for downstream processing
            invariant_output = torch.cat([invariant_output_0e, invariant_output_2e], dim=2)  # Shape: (batch_size, 201, 6)

            # Split predictions into 0e and 2e components
            output_0e = invariant_output[:, :, :irreps_0e]  # Shape: (batch_size, 201, 1)
            output_2e = invariant_output[:, :, irreps_0e:]  # Shape: (batch_size, 201, 5)

            # Split ground truth
            y = labels.view(batch_size, omega_points, irreps_dim)  # Shape: (batch_size, 201, 6)
            y_0e = y[:, :, :irreps_0e]  # Shape: (batch_size, 201, 1)
            y_2e = y[:, :, irreps_0e:]  # Shape: (batch_size, 201, 5)

            # Compute weighted loss
            weight_broadcast = weighting.view(1, -1, 1).to(device)  # Shape: (1, 201, 1)
            loss_0e = loss_fn(output_0e, y_0e) * weight_broadcast  # Shape: (batch_size, 201, 1)
            loss_2e = loss_fn(output_2e, y_2e) * weight_broadcast  # Shape: (batch_size, 201, 5)

            # Average losses
            loss = loss_0e.mean() + loss_2e.mean()

            # Compute MAE loss
            loss_0e_mae = loss_fn_mae(output_0e, y_0e) * weight_broadcast
            loss_2e_mae = loss_fn_mae(output_2e, y_2e) * weight_broadcast
            loss_mae = loss_0e_mae.mean() + loss_2e_mae.mean()

            # Accumulate losses
            loss_cumulative += loss.detach().item()
            loss_cumulative_mae += loss_mae.detach().item()

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        end_time = time.time()
        wall = end_time - start_time
        # Validation step
        if step == checkpoint:
            checkpoint = next(checkpoint_generator)
            assert checkpoint > step
            valid_avg_loss = evaluate(model, dataloader_valid, loss_fn_eval, loss_fn_mae_eval, device, weighting=weighting)
            train_avg_loss = evaluate(model, dataloader_train, loss_fn_eval, loss_fn_mae_eval, device, weighting=weighting)

            history.append({
                'step': s0 + step,
                'wall': wall,
                'batch': {
                    'loss': loss.item(),
                    'mean_abs': loss_mae.item(),
                },
                'valid': {
                    'loss': valid_avg_loss[0],
                    'mean_abs': valid_avg_loss[1],
                },
                'train': {
                    'loss': train_avg_loss[0],
                    'mean_abs': train_avg_loss[1],
                },
            })

            results = {
                'history': history,
                'state': model.state_dict()
            }

            wandb.log({
                "train/loss": train_avg_loss[0],
                "train/mae": train_avg_loss[1],
                "valid/loss": valid_avg_loss[0],
                "valid/mae": valid_avg_loss[1],
                "step": step,
                "wall_time": wall
            })

            if valid_avg_loss[0] < best_valid_loss:
                best_valid_loss = valid_avg_loss[0]
                print(f"New best validation loss: {best_valid_loss:.4f}. Saving model...")
                best_model_path = f'../model/{run_name}_best.torch'
                torch.save(results, best_model_path)

            print(f"Iteration {step+1:4d}   " +
                  f"train loss = {train_avg_loss[0]:8.4f}   " +
                  f"valid loss = {valid_avg_loss[0]:8.4f}   " +
                  f"elapsed time = {time.strftime('%H:%M:%S', time.gmtime(wall))}")

            latest_model_path = f'../model/{run_name}.torch'
            torch.save(results, latest_model_path)

        if scheduler is not None:
            scheduler.step(valid_avg_loss[0])

    print("Training complete!")