from typing import Dict, Union

import torch
from torch.nn.modules.loss import _Loss
from torch_geometric.data import Data
from torch_cluster import radius_graph
import torch.nn as nn
import os
from e3nn import o3
from e3nn.math import soft_one_hot_linspace
from e3nn.nn import Gate, Dropout, BatchNorm, FullyConnectedNet
# from e3nn.nn.models.gate_points_2101 import Convolution, smooth_cutoff, tp_path_exists
from e3nn.o3 import FullyConnectedTensorProduct, TensorProduct, Linear

import matplotlib.pyplot as plt
import math
import time
from tqdm import tqdm
import wandb
from e3nn.util.jit import compile_mode
import numpy as np 

bar_format = '{l_bar}{bar:10}{r_bar}{bar:-10b}'
# standard formatting for plots
fontsize = 24
textsize = 16
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
    
from e3nn.o3 import Irreps

def to_irreps(x):
    if x is None:
        return None
    if isinstance(x, Irreps):
        return x
    if isinstance(x, int):
        return Irreps(f"{x}x0e")
    return Irreps(x)  # assume it's a valid string

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
        fc_neurons = [self.number_of_basis, 100]


        # self.irreps_in = o3.Irreps(irreps_in) if irreps_in is not None else None
        self.irreps_in = irreps_in
        self.irreps_hidden = o3.Irreps([(self.mul, (l, p)) for l in range(lmax + 1) for p in [-1, 1]])
        self.irreps_out = o3.Irreps(irreps_out)
        # self.irreps_node_attr = o3.Irreps(irreps_node_attr) if irreps_node_attr is not None else o3.Irreps("0e")
        self.irreps_node_attr = irreps_node_attr

        self.irreps_edge_attr = o3.Irreps.spherical_harmonics(lmax)

        self.input_has_node_in = (irreps_in is not None)
        self.input_has_node_attr = (irreps_node_attr is not None)

        irreps = self.irreps_in if self.irreps_in is not None else o3.Irreps("0e")

        act = {
            1: torch.nn.functional.silu,
            # 1: torch.nn.functional.relu,
            # 1: torch.nn.functional.softplus,
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
            # conv = Convolution(
            #     irreps,
            #     self.irreps_node_attr,
            #     self.irreps_edge_attr,
            #     gate.irreps_in,
            #     number_of_basis,
            #     radial_layers,
            #     radial_neurons,
            #     num_neighbors
            # )
            conv = Convolution(
                irreps,
                self.irreps_node_attr,
                self.irreps_edge_attr,
                gate.irreps_in,
                fc_neurons,
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
                # number_of_basis,
                # radial_layers,
                # radial_neurons,
                fc_neurons,
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

        # === Apply softplus to 0e outputs only ===
        x = self._apply_softplus_to_0e(x) # only for new model for the test

        if self.reduce_output:
            return scatter(x, batch, dim=0).div(self.num_nodes**0.5)
        else:
            return x

    def _apply_softplus_to_0e(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply softplus to all '0e' components of the output tensor.
        """
        slices = self.irreps_out.slices()
        irreps = self.irreps_out
        out_chunks = []

        for i, (_, ir) in enumerate(irreps):
            chunk = x[..., slices[i]]
            if ir == o3.Irrep("0e"):
                out_chunks.append(torch.nn.functional.softplus(chunk))
            else:
                out_chunks.append(chunk)

        return torch.cat(out_chunks, dim=-1)


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

# def evaluate(model, dataloader, loss_fn_eval, loss_fn_mse_eval, device, step=None, save_dir=None, save_vis=False, max_vis_samples=None):
def evaluate(model, dataloader, loss_fn_eval, loss_fn_mse_eval, device):

    model.eval()
    loss_cumulative = 0.
    loss_cumulative_mse = 0.
    dataloader = [d.to(device) for d in dataloader]

    # Set constants for slicing
    # irreps_0e = 300
    # irreps_2e = 1500
    # out_dim = 300
    irreps_0e = model.irreps_out.count(o3.Irrep("0e"))
    irreps_2e = model.irreps_out.count(o3.Irrep("2e")) * 5
    out_dim = model.irreps_out.count(o3.Irrep("0e"))
    
    # os.makedirs(save_dir, exist_ok=True) if save_vis and save_dir is not None else None

    with torch.no_grad():
        for batch_idx, d in enumerate(dataloader):
            output = model(d)

            output_0e = output[:, :irreps_0e]
            output_2e = output[:, irreps_0e:irreps_0e + irreps_2e].contiguous().view(output.shape[0], out_dim, 5)

            y_0e = d.y[:, :, 0].view(d.y.shape[0], out_dim)
            y_2e = d.y[:, :, 1:].view(d.y.shape[0], out_dim, 5)

            loss_0e = loss_fn_eval(output_0e, y_0e)
            loss_2e = loss_fn_eval(output_2e, y_2e)
            loss_0e_mse = loss_fn_mse_eval(output_0e, y_0e)
            loss_2e_mse = loss_fn_mse_eval(output_2e, y_2e)

            loss = loss_0e + loss_2e
            loss_mse = loss_0e_mse + loss_2e_mse

            loss_cumulative += loss.item()
            loss_cumulative_mse += loss_mse.item()

            # # === Plot all samples in first batch only ===
            # if save_vis and batch_idx == 0:
            #     num_samples = output_0e.shape[0]
            #     if max_vis_samples is not None:
            #         num_samples = min(num_samples, max_vis_samples)

            #     for n in range(num_samples):
            #         pred_0e = output_0e[n].cpu().numpy()
            #         pred_2e = output_2e[n].cpu().numpy().T.reshape(-1)
            #         pred = np.concatenate([pred_0e, pred_2e])

            #         target_0e = y_0e[n].cpu().numpy()
            #         target_2e = y_2e[n].cpu().numpy().T.reshape(-1)
            #         target = np.concatenate([target_0e, target_2e])

            #         fig, ax = plt.subplots(figsize=(10, 8))
            #         ax.plot(pred, label='Prediction', alpha=0.9)
            #         ax.plot(target, label='Ground Truth', alpha=0.9)
            #         ax.set_title(f'[VALID] Sample {n} | Step {step}')
            #         ax.set_ylabel('Value')
            #         ax.legend()

            #         # vertical tick lines
            #         block_size = out_dim
            #         tick_positions = list(range(0, 6 * block_size, block_size))
            #         ax.set_xticks([])
            #         for x in tick_positions[1:]:
            #             ax.axvline(x=x, color='gray', linestyle='--', linewidth=0.5, alpha=0.4)

            #         # spherical harmonic labels
            #         block_labels = [r'$Y^0_0$', r'$Y^2_{-2}$', r'$Y^2_{-1}$', r'$Y^2_{0}$', r'$Y^2_{1}$', r'$Y^2_{2}$']
            #         for i, label in enumerate(block_labels):
            #             center = tick_positions[i] + block_size / 2
            #             ax.text(center, ax.get_ylim()[0] - 0.05 * (ax.get_ylim()[1] - ax.get_ylim()[0]),
            #                     label, fontsize=24, ha='center', va='top', transform=ax.transData)

            #         fig.subplots_adjust(bottom=0.15)
            #         fig.tight_layout()
            #         fig.savefig(os.path.join(save_dir, f"valid_sample{n}_step{step}.png"))
            #         plt.close(fig)

    return loss_cumulative / len(dataloader), loss_cumulative_mse / len(dataloader)


def train(model, optimizer, dataloader_train, dataloader_valid, loss_fn, loss_fn_mse, loss_fn_eval, loss_fn_mse_eval, run_name, 
          max_iter=101, scheduler=None, device="cpu", disable_tqdm=False, alpha = 1., beta = 50., loss_balancer=None):
    model.to(device)

    irreps_0e = model.irreps_out.count(o3.Irrep("0e"))
    irreps_2e = model.irreps_out.count(o3.Irrep("2e")) * 5
    out_dim = model.irreps_out.count(o3.Irrep("0e"))

    # irreps_0e = 150
    # irreps_2e = 150 * 5
    # out_dim = 150

    start_time = time.time()

    try:
        model.load_state_dict(torch.load(run_name + '.torch')['state'])
    except:
        results = {}
        history = []
        s0 = 0
        best_valid_loss = float('inf')
    else:
        results = torch.load(run_name + '.torch')
        history = results['history']
        s0 = history[-1]['step'] + 1
        best_valid_loss = min(h['valid']['loss'] for h in history)

    for step in range(max_iter):
        model.train()
        loss_cumulative = 0.
        loss_cumulative_mse = 0.

        # for d in tqdm(dataloader_train, total=len(dataloader_train), bar_format="{l_bar}{bar:30}{r_bar}", disable=disable_tqdm):
        for batch_idx, d in enumerate(tqdm(dataloader_train, total=len(dataloader_train), 
                            bar_format="{l_bar}{bar:30}{r_bar}", disable=disable_tqdm)):
            d = d.to(device)
            output = model(d)

            output_0e = output[:, :irreps_0e]
            output_2e = output[:, irreps_0e:irreps_0e + irreps_2e].contiguous().view(output.shape[0], out_dim, 5)

            y_0e = d.y[:, :, 0].view(d.y.shape[0], out_dim)
            y_2e = d.y[:, :, 1:].view(d.y.shape[0], out_dim, 5)

            loss_0e = loss_fn(output_0e, y_0e)
            loss_2e = loss_fn(output_2e, y_2e)
            # Compute MSE per subchannel: shape → (5,)
            # loss_2e_per_channel = torch.mean((output_2e - y_2e) ** 2, dim=(0, 1))  # over batch and dim
            # loss_0e_mean = loss_fn(output_0e, y_0e).mean()
            # # if loss_balancer is not None:
            # #     loss = loss_balancer(loss_0e.mean(), loss_2e.mean())
            # # else:
            # #     loss = alpha * loss_0e.mean() + beta * loss_2e.mean()
            # if loss_balancer is not None:
            #     loss = loss_balancer(loss_0e_mean, loss_2e_per_channel)
            # else:
            #     loss = alpha * loss_0e_mean + beta * loss_2e_per_channel.sum()
            if loss_balancer is not None:
                loss = loss_balancer(loss_0e.mean(), loss_2e.mean())
            else:
                loss = alpha * loss_0e.mean() + beta * loss_2e.mean()

            loss_0e_mse = loss_fn_mse(output_0e, y_0e)
            loss_2e_mse = loss_fn_mse(output_2e, y_2e)
            loss_mse = loss_0e_mse.mean() + loss_2e_mse.mean()
             
            loss_cumulative += loss.item()
            loss_cumulative_mse += loss_mse.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if step % 4 == 0 and batch_idx == 1:
                batch_size = output_0e.shape[0]
                for sample_idx in range(1):
                    pred_0e = output_0e[sample_idx].detach().cpu().numpy()               # (out_dim,)
                    pred_2e = output_2e[sample_idx].detach().cpu().numpy().T.reshape(-1)  # (5*out_dim,)
                    pred = np.concatenate([pred_0e, pred_2e])  # shape: (6*out_dim,)

                    target_0e = y_0e[sample_idx].detach().cpu().numpy()
                    target_2e = y_2e[sample_idx].detach().cpu().numpy().T.reshape(-1)
                    target = np.concatenate([target_0e, target_2e])

                    fig, ax = plt.subplots(figsize=(10, 8))
                    ax.plot(target, label='Ground Truth', alpha=0.9, color = 'black')
                    ax.plot(pred, label='Prediction', alpha=0.9, color = 'red')
                    ax.set_title(f'Step {step} | Batch {batch_idx} | Sample {sample_idx}')
                    ax.set_ylabel('Value')
                    ax.legend()

                    # Block boundaries and labels
                    block_size = out_dim
                    tick_positions = list(range(0, 6 * block_size, block_size))
                    ax.set_xticks([])

                    for x in tick_positions[1:]:
                        ax.axvline(x=x, color='gray', linestyle='--', linewidth=0.5, alpha=0.4)

                    block_labels = [r'$Y^0_0$', r'$Y^2_{-2}$', r'$Y^2_{-1}$', r'$Y^2_{0}$', r'$Y^2_{1}$', r'$Y^2_{2}$']
                    for i, label in enumerate(block_labels):
                        center = tick_positions[i] + block_size / 2
                        ax.text(center, ax.get_ylim()[0] - 0.05 * (ax.get_ylim()[1] - ax.get_ylim()[0]),
                                label, fontsize=24, ha='center', va='top', transform=ax.transData)

                    fig.subplots_adjust(bottom=0.15)
                    fig.tight_layout()
                    fig.savefig(f'../pngs/pred_vs_gt_step{step}_batch{batch_idx}_sample{sample_idx}.png')
                    plt.close(fig)



        end_time = time.time()
        wall = end_time - start_time

        valid_avg_loss = evaluate(model, dataloader_valid, loss_fn_eval, loss_fn_mse_eval, device)
        # valid_avg_loss = evaluate(
        #     model, 
        #     dataloader_valid, 
        #     loss_fn_eval, 
        #     loss_fn_mse_eval, 
        #     device, 
        #     step=step, 
        #     save_dir="../pngs_eval", 
        #     save_vis=(step % 5 == 0), 
        #     max_vis_samples=16  # optional
        # )
        train_avg_loss = evaluate(model, dataloader_train, loss_fn_eval, loss_fn_mse_eval, device)

        history.append({
            'step': s0 + step,
            'wall': wall,
            'batch': {
                'loss': loss.item(),
                'mean_abs': loss_mse.item(),
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
        current_lr = optimizer.param_groups[0]['lr']
        log_dict = {
            "train/loss": train_avg_loss[0],
            "train/mse": train_avg_loss[1],
            "valid/loss": valid_avg_loss[0],
            "valid/mse": valid_avg_loss[1],
            "learning_rate": current_lr,
            "step": s0 + step,
            "wall_time": wall,
        }

        if loss_balancer is not None:
            log_dict["loss_weight/0e"] = torch.exp(-2 * loss_balancer.log_sigma_0e).item()
            log_dict["loss_weight/2e"] = torch.exp(-2 * loss_balancer.log_sigma_2e).item()
            # # For 2e: log each subchannel
            # for i in range(loss_balancer.log_sigma_2e.shape[0]):
            #     weight_2e_i = torch.exp(-2 * loss_balancer.log_sigma_2e[i]).item()
            #     log_dict[f"loss_weight/2e_m{i - 2}"] = weight_2e_i  # m = -2, -1, 0, 1, 2

        wandb.log(log_dict)

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
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(valid_avg_loss[0])  # Requires validation loss as input
            else:
                scheduler.step()  # Step without metric

    print("Training complete!")



def smooth_cutoff(x):
    u = 2 * (x - 1)
    y = (math.pi * u).cos().neg().add(1).div(2)
    y[u > 0] = 0
    y[u < -1] = 1
    return y


def tp_path_exists(irreps_in1, irreps_in2, ir_out):
    irreps_in1 = o3.Irreps(irreps_in1).simplify()
    irreps_in2 = o3.Irreps(irreps_in2).simplify()
    ir_out = o3.Irrep(ir_out)

    for _, ir1 in irreps_in1:
        for _, ir2 in irreps_in2:
            if ir_out in ir1 * ir2:
                return True
    return False


def scatter(src: torch.Tensor, index: torch.Tensor, dim_size: int) -> torch.Tensor:
    # special case of torch_scatter.scatter with dim=0
    out = src.new_zeros(dim_size, src.shape[1])
    index = index.reshape(-1, 1).expand_as(src)
    return out.scatter_add_(0, index, src)

# Euclidean neural networks (e3nn) Copyright (c) 2020, The Regents of the
# University of California, through Lawrence Berkeley National Laboratory
# (subject to receipt of any required approvals from the U.S. Dept. of Energy), 
# Ecole Polytechnique Federale de Lausanne (EPFL), Free University of Berlin 
# and Kostiantyn Lapchevskyi. All rights reserved.
# Modified from https://github.com/e3nn/e3nn/blob/05b386177ed039156526f9c67d0d87b6c21ff5d3/e3nn/nn/models/v2103/points_convolution.py
#  - Remove torch_scatter dependency
#  - Add support for differently indexed sending/receiver nodes.
#  - Sender and receiver nodes can have different irreps.
@compile_mode("script")
class Convolution(torch.nn.Module):
    """
    Equivariant Convolution
    Args:
        irreps_node_input (e3nn.o3.Irreps): representation of the input node features
        irreps_node_attr (e3nn.o3.Irreps): representation of the node attributes
        irreps_edge_attr (e3nn.o3.Irreps): representation of the edge attributes
        irreps_node_output (e3nn.o3.Irreps or None): representation of the output node features
        fc_neurons (list[int]): number of neurons per layers in the fully connected network
            first layer and hidden layers but not the output layer
        num_neighbors (float): typical number of nodes convolved over
    """

    def __init__(
        self,
        irreps_in,
        irreps_node_attr,
        irreps_edge_attr,
        irreps_out,
        fc_neurons,
        num_neighbors,
    ) -> None:
        super().__init__()
        self.irreps_in = o3.Irreps(irreps_in)
        self.irreps_node_attr = o3.Irreps(irreps_node_attr)
        self.irreps_edge_attr = o3.Irreps(irreps_edge_attr)
        self.irreps_out = o3.Irreps(irreps_out)
        self.num_neighbors = num_neighbors

        self.sc = FullyConnectedTensorProduct(
            self.irreps_in, self.irreps_node_attr, self.irreps_out
        )

        self.lin1 = FullyConnectedTensorProduct(
            self.irreps_in, self.irreps_node_attr, self.irreps_in
        )

        irreps_mid = []
        instructions = []
        for i, (mul, ir_in) in enumerate(self.irreps_in):
            for j, (_, ir_edge) in enumerate(self.irreps_edge_attr):
                for ir_out in ir_in * ir_edge:
                    if ir_out in self.irreps_out or ir_out == o3.Irrep(0, 1):
                        k = len(irreps_mid)
                        irreps_mid.append((mul, ir_out))
                        instructions.append((i, j, k, "uvu", True))
        irreps_mid = o3.Irreps(irreps_mid)
        irreps_mid, p, _ = irreps_mid.sort()

        instructions = [
            (i_1, i_2, p[i_out], mode, train)
            for i_1, i_2, i_out, mode, train in instructions
        ]

        tp = TensorProduct(
            self.irreps_in,
            self.irreps_edge_attr,
            irreps_mid,
            instructions,
            internal_weights=False,
            shared_weights=False,
        )
        self.fc = FullyConnectedNet(
            fc_neurons + [tp.weight_numel], torch.nn.functional.silu
        )
        self.tp = tp

        self.lin2 = FullyConnectedTensorProduct(
            irreps_mid, self.irreps_node_attr, self.irreps_out
        )
        # self.lin3 = FullyConnectedTensorProduct(irreps_mid, self.irreps_node_attr, "0e")

        # inspired by https://arxiv.org/pdf/2002.10444.pdf
        # self.alpha = FullyConnectedTensorProduct(irreps_mid, self.irreps_node_attr, "0e")
        # with torch.no_grad():
        #     self.alpha.weight.zero_()
        # assert (
        #     self.alpha.output_mask[0] == 1.0
        # ), f"irreps_mid={irreps_mid} and irreps_node_attr={self.irreps_node_attr} are not able to generate scalars"

#------------------------- THREE DIFFERENT VERSION OF FORWARD FUNCTION PROVIDED BY E3NN -------------------------#
    # def forward(self, node_input, node_attr, edge_src, edge_dst, edge_attr, edge_scalars) -> torch.Tensor:
    #     weight = self.fc(edge_scalars)

    #     node_self_connection = self.sc(node_input, node_attr)
    #     node_features = self.lin1(node_input, node_attr)

    #     edge_features = self.tp(node_features[edge_src], edge_attr, weight)
    #     node_features = scatter(edge_features, edge_dst, dim_size=node_input.shape[0]).div(self.num_neighbors**0.5)

    #     node_conv_out = self.lin2(node_features, node_attr)
    #     node_angle = 0.1 * self.lin3(node_features, node_attr)
    #     #            ^^^------ start small, favor self-connection

    #     cos, sin = node_angle.cos(), node_angle.sin()
    #     m = self.sc.output_mask
    #     sin = (1 - m) + sin * m
    #     return cos * node_self_connection + sin * node_conv_out
    

    # def forward(self, node_input, node_attr, edge_src, edge_dst, edge_attr, edge_scalars) -> torch.Tensor:
    #     weight = self.fc(edge_scalars)

    #     node_self_connection = self.sc(node_input, node_attr)
    #     node_features = self.lin1(node_input, node_attr)

    #     edge_features = self.tp(node_features[edge_src], edge_attr, weight)
    #     node_features = scatter(edge_features, edge_dst, dim_size=node_input.shape[0]).div(self.num_neighbors**0.5)

    #     node_conv_out = self.lin2(node_features, node_attr)
    #     alpha = self.alpha(node_features, node_attr)

    #     m = self.sc.output_mask
    #     alpha = (1 - m) + alpha * m
    #     return node_self_connection + alpha * node_conv_out
    
    def forward(self, node_input, node_attr, edge_src, edge_dst, edge_attr, edge_length_embedded) -> torch.Tensor:
        weight = self.fc(edge_length_embedded)

        x = node_input

        s = self.sc(x, node_attr)
        x = self.lin1(x, node_attr)

        edge_features = self.tp(x[edge_src], edge_attr, weight)
        x = scatter(edge_features, edge_dst, dim_size=x.shape[0]).div(self.num_neighbors**0.5)

        x = self.lin2(x, node_attr)

        c_s, c_x = math.sin(math.pi / 8), math.cos(math.pi / 8)
        m = self.sc.output_mask
        c_x = (1 - m) + c_x * m
        return c_s * s + c_x * x
