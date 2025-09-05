# model
import torch
import torch_geometric as tg
import os 

# data pre-processing
import numpy as np
import pandas as pd
import pickle as pkl
from sklearn.model_selection import train_test_split
from scipy.stats import gaussian_kde
import math
import seaborn as sns

# data visualization
import matplotlib as mpl
import matplotlib.pyplot as plt
import networkx as nx
from ase import Atoms, Atom
from ase.visualize.plot import plot_atoms
from ase.neighborlist import neighbor_list
from matplotlib.ticker import MaxNLocator

# utilities
from tqdm import tqdm
from mendeleev import element

# format progress bar
bar_format = '{l_bar}{bar:10}{r_bar}{bar:-10b}'
tqdm.pandas(bar_format=bar_format)
default_dtype = torch.float64

# standard formatting for plots
fontsize = 16
textsize = 14
sub = str.maketrans("0123456789", "₀₁₂₃₄₅₆₇₈₉")
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.sans-serif'] = ['Helvetica', 'Arial', 'Liberation Sans', 'sans-serif']
plt.rcParams['axes.linewidth'] = 1
plt.rcParams['mathtext.default'] = 'regular'
plt.rcParams['xtick.bottom'] = True
plt.rcParams['ytick.left'] = True
plt.rcParams['font.size'] = fontsize
plt.rcParams['axes.labelsize'] = fontsize
plt.rcParams['xtick.labelsize'] = fontsize
plt.rcParams['ytick.labelsize'] = fontsize
plt.rcParams['legend.fontsize'] = textsize
plt.rcParams['text.usetex'] = False


# colors for datasets
palette = ['#2876B2', '#F39957', '#67C7C2', '#C86646']
datasets = ['train', 'valid', 'test']
colors = dict(zip(datasets, palette[:-1]))
cmap = mpl.colors.LinearSegmentedColormap.from_list('cmap', [palette[k] for k in [0,2,1]])


def load_data(filename):
    # load data from a pkl file and derive formula and species columns from structure
    df = pd.read_pickle(filename)
    df['species'] = df['symmetrized_structure'].map(lambda x: list(set(x.get_chemical_symbols())))
    species = sorted(list(set(df['species'].sum())))

    #df['energies'] = df['energies'].apply(eval).apply(np.array)
    #df['absorption_coefficient'] = df['absorption_coefficient'].apply(eval).apply(np.array)
    #df['pdos'] = df['pdos'].apply(eval)

    return df, species

def train_valid_test_split(df, valid_size, test_size, seed=12, plot=False):
    """
    Perform a crystal system-balanced train/valid/test split.

    Parameters:
    - df: DataFrame containing the dataset with a 'crystal_system' column.
    - crystal_systems: List of unique crystal systems in the dataset.
    - valid_size: Fraction of data to use for validation.
    - test_size: Fraction of data to use for testing.
    - seed: Random seed for reproducibility.
    - plot: Boolean to plot the distribution of crystal systems in each split.

    Returns:
    - idx_train, idx_valid, idx_test: Lists of indices for training, validation, and test sets.
    """
    crystal_systems = df['crystal_system'].unique().tolist()
    print(f"Unique crystal systems found: {crystal_systems}")
    # Perform a crystal system-balanced train/valid/test split
    print('Splitting train/dev ...')
    dev_size = valid_size + test_size
    stats = get_crystal_system_statistics(df, crystal_systems)
    idx_train, idx_dev = split_data(stats, dev_size, seed)
    
    print('Splitting valid/test ...')
    stats_dev = get_crystal_system_statistics(df.iloc[idx_dev], crystal_systems)
    idx_valid, idx_test = split_data(stats_dev, test_size/dev_size, seed)
    
    # Ensure all indices are included
    idx_train += df[~df.index.isin(idx_train + idx_valid + idx_test)].index.tolist()

    print('Number of training examples:', len(idx_train))
    print('Number of validation examples:', len(idx_valid))
    print('Number of testing examples:', len(idx_test))
    print('Total number of examples:', len(idx_train + idx_valid + idx_test))
    assert len(set.intersection(*map(set, [idx_train, idx_valid, idx_test]))) == 0, "Overlapping indices found!"

    if plot:
            # Calculate counts for each dataset
            stats['train'] = stats['data'].map(lambda x: crystal_system_representation(x, np.sort(idx_train)))
            stats['valid'] = stats['data'].map(lambda x: crystal_system_representation(x, np.sort(idx_valid)))
            stats['test'] = stats['data'].map(lambda x: crystal_system_representation(x, np.sort(idx_test)))
            stats = stats.sort_values('crystal_system')

            # Create a single plot with side-by-side bars
            fig, ax = plt.subplots(figsize=(10, 6), dpi=1600)
            bar_width = 0.25  # Width of each bar
            index = np.arange(len(crystal_systems))  # Positions for the bars

            # Plot bars for each dataset
            ax.bar(index, stats['train'], bar_width, label='Train', color='skyblue', edgecolor='black')
            ax.bar(index + bar_width, stats['valid'], bar_width, label='Valid', color='lightcoral', edgecolor='black')
            ax.bar(index + 2 * bar_width, stats['test'], bar_width, label='Test', color='lightgreen', edgecolor='black')

            # Customize the plot
            ax.set_xlabel('Crystal System', fontsize=fontsize)
            ax.set_ylabel('Number of Samples', fontsize=fontsize)
            ax.set_title('Distribution of Crystal Systems in Train/Valid/Test Sets', fontsize=fontsize, pad=15)
            ax.set_xticks(index + bar_width)
            ax.set_xticklabels(stats['crystal_system'], rotation=45, ha='right', fontsize=fontsize)
            ax.legend(ncol = 3, fontsize=fontsize)

            # Add grid for better readability
            ax.set_axisbelow(True)

            # Dynamically adjust y-axis limit
            max_count = max(stats['train'].max(), stats['valid'].max(), stats['test'].max())
            ax.set_ylim(0, max_count * 1.2)  # Add 20% padding above the max count

            # Add value labels on top of each bar
            for i in range(len(crystal_systems)):
                # Train
                height = stats['train'].iloc[i]
                ax.text(i, height + 0.02 * max_count, f'{int(height)}', ha='center', va='bottom', fontsize=textsize)
                # Valid
                height = stats['valid'].iloc[i]
                ax.text(i + bar_width, height + 0.02 * max_count, f'{int(height)}', ha='center', va='bottom', fontsize=textsize)
                # Test
                height = stats['test'].iloc[i]
                ax.text(i + 2 * bar_width, height + 0.02 * max_count, f'{int(height)}', ha='center', va='bottom', fontsize=textsize)

            plt.tight_layout()
            os.makedirs("../pngs", exist_ok=True)
            save_path = f"../pngs/dataset_split.png"
            fig.savefig(save_path, dpi=300)

            plt.show()

    return idx_train, idx_valid, idx_test

# def train_valid_test_split(df, species, valid_size, test_size, seed=12, plot=False):
#     # perform an element-balanced train/valid/test split
#     print('split train/dev ...')
#     dev_size = valid_size + test_size
#     stats = get_element_statistics(df, species)
#     idx_train, idx_dev = split_data(stats, dev_size, seed)
    
#     print('split valid/test ...')
#     stats_dev = get_element_statistics(df.iloc[idx_dev], species)
#     idx_valid, idx_test = split_data(stats_dev, test_size/dev_size, seed)
#     idx_train += df[~df.index.isin(idx_train + idx_valid + idx_test)].index.tolist()

#     print('number of training examples:', len(idx_train))
#     print('number of validation examples:', len(idx_valid))
#     print('number of testing examples:', len(idx_test))
#     print('total number of examples:', len(idx_train + idx_valid + idx_test))
#     assert len(set.intersection(*map(set, [idx_train, idx_valid, idx_test]))) == 0

#     if plot:
#         # plot element representation in each dataset
#         stats['train'] = stats['data'].map(lambda x: element_representation(x, np.sort(idx_train)))
#         stats['valid'] = stats['data'].map(lambda x: element_representation(x, np.sort(idx_valid)))
#         stats['test'] = stats['data'].map(lambda x: element_representation(x, np.sort(idx_test)))
#         stats = stats.sort_values('symbol')

#         fig, ax = plt.subplots(2,1, figsize=(14,7))
#         b0, b1 = 0., 0.
#         for i, dataset in enumerate(datasets):
#             split_subplot(ax[0], stats[:len(stats)//2], species[:len(stats)//2], dataset, bottom=b0, legend=True)
#             split_subplot(ax[1], stats[len(stats)//2:], species[len(stats)//2:], dataset, bottom=b1)

#             b0 += stats.iloc[:len(stats)//2][dataset].values
#             b1 += stats.iloc[len(stats)//2:][dataset].values

#         fig.tight_layout()
#         fig.subplots_adjust(hspace=0.1)

#     return idx_train, idx_valid, idx_test


def get_element_statistics(df, species):    
    # create dictionary indexed by element names storing index of samples containing given element
    species_dict = {k: [] for k in species}
    for entry in df.itertuples():
        for specie in entry.species:
            species_dict[specie].append(entry.Index)

    # create dataframe of element statistics
    stats = pd.DataFrame({'symbol': species})
    stats['data'] = stats['symbol'].astype('object')
    for specie in species:
        stats.at[stats.index[stats['symbol'] == specie].values[0], 'data'] = species_dict[specie]
    stats['count'] = stats['data'].apply(len)

    return stats
def get_crystal_system_statistics(df, crystal_systems):
    """
    Create a DataFrame with statistics of crystal systems in the dataset.

    Parameters:
    - df: DataFrame containing the dataset with a 'crystal_system' column.
    - crystal_systems: List of unique crystal systems.

    Returns:
    - stats: DataFrame with crystal system statistics.
    """
    # Create dictionary indexed by crystal system names storing indices of samples
    crystal_system_dict = {k: [] for k in crystal_systems}
    for entry in df.itertuples():
        crystal_system = entry.crystal_system  # Assumes 'crystal_system' column exists
        crystal_system_dict[crystal_system].append(entry.Index)

    # Create DataFrame of crystal system statistics
    stats = pd.DataFrame({'crystal_system': crystal_systems})
    stats['data'] = stats['crystal_system'].astype('object')
    for system in crystal_systems:
        stats.at[stats.index[stats['crystal_system'] == system].values[0], 'data'] = crystal_system_dict[system]
    stats['count'] = stats['data'].apply(len)

    return stats

# def split_data(df, test_size, seed):
#     # initialize output arrays
#     idx_train, idx_test = [], []
    
#     # remove empty examples
#     df = df[df['data'].str.len()>0]
    
#     # sort df in order of fewest to most examples
#     df = df.sort_values('count')
    
#     for _, entry in tqdm(df.iterrows(), total=len(df), bar_format=bar_format):
#         df_specie = entry.to_frame().T.explode('data')

#         try:
#             idx_train_s, idx_test_s = train_test_split(df_specie['data'].values, test_size=test_size,
#                                                        random_state=seed)
#         except:
#             # too few examples to perform split - these examples will be assigned based on other constituent elements
#             # (assuming not elemental examples)
#             pass

#         else:
#             # add new examples that do not exist in previous lists
#             idx_train += [k for k in idx_train_s if k not in idx_train + idx_test]
#             idx_test += [k for k in idx_test_s if k not in idx_train + idx_test]
    
#     return idx_train, idx_test
def split_data(stats, split_size, seed):
    """
    Split data indices while balancing across categories (crystal systems).

    Parameters:
    - stats: DataFrame with 'data' column containing lists of indices.
    - split_size: Fraction of data to split out.
    - seed: Random seed for reproducibility.

    Returns:
    - idx_train, idx_dev: Lists of indices for training and development sets.
    """
    idx_train, idx_dev = [], []
    for _, row in stats.iterrows():
        indices = row['data']
        if len(indices) > 0:
            train_idx, dev_idx = train_test_split(indices, test_size=split_size, random_state=seed)
            idx_train.extend(train_idx)
            idx_dev.extend(dev_idx)
    return idx_train, idx_dev


def element_representation(x, idx):
    # get fraction of samples containing given element in dataset
    return len([k for k in x if k in idx])/len(x)

def crystal_system_representation(indices, subset_indices):
    """
    Calculate the number of samples in subset_indices that belong to indices.

    Parameters:
    - indices: List of all indices for a crystal system.
    - subset_indices: List of indices in a subset (e.g., train, valid, test).

    Returns:
    - count: Number of overlapping indices.
    """
    return len(set(indices).intersection(set(subset_indices)))

# def split_subplot(ax, df, species, dataset, bottom=0., legend=False):    
#     # plot element representation
#     width = 0.4
#     color = [int(colors[dataset].lstrip('#')[i:i+2], 16)/255. for i in (0,2,4)]
#     bx = np.arange(len(species))
        
#     ax.bar(bx, df[dataset], width, fc=color+[0.7], ec=color, lw=1.5, bottom=bottom, label=dataset)
        
#     ax.set_xticks(bx)
#     ax.set_xticklabels(species)
#     ax.tick_params(direction='in', length=0, width=1)
#     ax.set_ylim(top=1.18)
#     if legend: ax.legend(frameon=False, ncol=3, loc='upper left')
        
def split_subplot(ax, stats, crystal_systems, dataset, bottom, legend=False):
    """
    Plot a subplot for the distribution of crystal systems in a dataset.

    Parameters:
    - ax: Matplotlib axis to plot on.
    - stats: DataFrame with crystal system statistics.
    - crystal_systems: List of crystal systems to plot.
    - dataset: Name of the dataset ('train', 'valid', 'test').
    - bottom: Starting position for stacking bars.
    - legend: Boolean to include a legend.
    """
    ax.bar(stats['crystal_system'], stats[dataset], bottom=bottom, label=dataset)
    ax.set_xticks(range(len(crystal_systems)))
    ax.set_xticklabels(crystal_systems, rotation=45)
    if legend:
        ax.legend()

def plot_example(df, i=12, label_edges=False):
    # plot an example crystal structure and graph
    entry = df.iloc[i]['data']

    # get graph with node and edge attributes
    g = tg.utils.to_networkx(entry, node_attrs=['symbol'], edge_attrs=['edge_len'], to_undirected=True)

    # remove self-loop edges for plotting
    g.remove_edges_from(list(nx.selfloop_edges(g)))
    node_labels = dict(zip([k[0] for k in g.nodes.data()], [k[1]['symbol'] for k in g.nodes.data()]))
    edge_labels = dict(zip([(k[0], k[1]) for k in g.edges.data()], [k[2]['edge_len'] for k in g.edges.data()]))

    # project positions of nodes to 2D for plotting
    pos = dict(zip(list(g.nodes), [np.roll(k,2)[:-1][::-1] for k in entry.pos.numpy()]))

    # plot unit cell
    fig, ax = plt.subplots(1,2, figsize=(14,10), gridspec_kw={'width_ratios': [2,3]})
    atoms = Atoms(symbols=entry.symbol, positions=entry.pos.numpy(), cell=entry.lattice.squeeze().numpy(), pbc=True)
    symbols = np.unique(entry.symbol)
    z = dict(zip(symbols, range(len(symbols))))
    norm = plt.Normalize(vmin=0, vmax=len(symbols)-1)
    color = [mpl.colors.to_hex(k) for k in cmap(norm([z[j] for j in entry.symbol]))]
    plot_atoms(atoms, ax[0], radii=0.25, colors=color, rotation=('0x,90y,0z'))

    # plot graph
    nx.draw_networkx(g, ax=ax[1], labels=node_labels, pos=pos, font_family='Arial', node_size=500, node_color=color,
                     edge_color='gray')
    
    if label_edges:
        nx.draw_networkx_edge_labels(g, ax=ax[1], edge_labels=edge_labels, pos=pos, label_pos=0.5, font_family='Arial')
    
    # format axes
    ax[0].set_xlabel(r'$x_1\ (\AA)$')
    ax[0].set_ylabel(r'$x_2\ (\AA)$')
    ax[0].set_title('Crystal structure', fontsize=fontsize)
    ax[1].set_aspect('equal')
    ax[1].axis('off')
    ax[1].set_title('Crystal graph', fontsize=fontsize)
    pad = np.array([-0.5, 0.5])
    ax[1].set_xlim(np.array(ax[1].get_xlim()) + pad)
    ax[1].set_ylim(np.array(ax[1].get_ylim()) + pad)
    fig.subplots_adjust(wspace=0.4)



def plot_predictions(df, idx, column, header, title=None, plot_gt=True):    
    # Get sorted indices based on mse
    i_mse = np.argsort(df.iloc[idx]['mse'])
    ds = df.iloc[idx].iloc[i_mse][['formula', 'energies_interp', column, 'y_pred', 'mse']].reset_index(drop=True)

    # Compute adaptive quartiles
    quartiles = np.percentile(ds['mse'].values, [25, 50, 75, 100])
    iq = sorted(set([0] + [np.argmin(np.abs(ds['mse'].values - q)) for q in quartiles]))

    # Select representative samples within each quartile range
    n = 6  # Number of samples per quartile
    s = np.concatenate([np.sort(np.random.choice(np.arange(iq[k-1], iq[k], 1), size=n, replace=False)) 
                        for k in range(1, len(iq))])

    x = df.iloc[0]['energies_interp']
    fig, axs = plt.subplots(4, n, figsize=(12,8))  # Removed KDE column

    fontsize = 14
    cols = [palette[k] for k in [2, 0, 1, 3]][::-1]
    cols = np.repeat(cols[::-1], n)
    axs = axs.ravel()

    for k in range(4*n):
        ax = axs[k]
        i = s[k]
        if plot_gt:
            ax.plot(x, ds.iloc[i][column], color='black')
        ax.plot(x, ds.iloc[i]['y_pred'], color=cols[k])
        ax.set_xticks([]); ax.set_yticks([])

        # Add formula and MSE in title
        formula = ds.iloc[i]['formula'].translate(sub)
        mse_value = ds.iloc[i]['mse']
        ax.set_title(f"{formula}\nMSE: {mse_value:.2e}", fontname='DejaVu Sans', fontsize=fontsize-4, y=1.0)
        for ax in axs[-n:]:  
            ax.set_xticks(np.linspace(x.min(), x.max(), 5))  # Example tick positions

    # fig.text(0.5, 0.02, "Photon energy (eV)", ha='center', va='center', fontsize=fontsize )

    # fig.subplots_adjust(hspace=0.75)
    # if title:
    #     fig.suptitle(title, ha='center', y=1.05, fontsize=fontsize + 4)
    # fig.savefig(f"{header}_{title}_spectra.pdf")
    fig.supxlabel("Photon energy (eV)", fontsize=fontsize + 2)

    # Adjust layout and ensure suptitle is visible
    fig.tight_layout(rect=[0, 0, 1, 0.94])  # Make space for the title
    if title:
        fig.suptitle(title, ha='center', fontsize=fontsize + 4, y=0.98)

    fig.savefig(f"../pdfs/{header}_{title}_spectra.pdf")
    plt.show()


def plot_partials(model, df, idx, device='cpu'):
    # randomly sample r compounds from the dataset
    r = 6
    ids = np.random.choice(df.iloc[idx][df.iloc[idx]['pdos'].str.len()>0].index.tolist(), size=r, replace=False)
    
    # initialize figure axes
    N = df.iloc[ids]['species'].str.len().max()
    fig, ax = plt.subplots(r, N+1, figsize=(2.4*(N+1),1.2*r), sharex=True, sharey=True)

    # predict output of each site for each sample
    for row, i in enumerate(ids):
        entry = df.iloc[i]
        d = tg.data.Batch.from_data_list([entry.data])

        model.eval()
        with torch.no_grad():
            d.to(device)
            output = model(d).cpu().numpy()

        # average contributions from the same specie over all sites
        n = len(entry.species)
        pdos = dict(zip(entry.species, [np.zeros((output.shape[1])) for k in range(n)]))
        for j in range(output.shape[0]):
            pdos[entry.data.symbol[j]] += output[j,:]

        for j, s in enumerate(entry.species):
            pdos[s] /= entry.data.symbol.count(s)

        # plot total DoS
        ax[row,0].plot(entry.phfreq, entry.phdos, color='black')
        ax[row,0].plot(entry.phfreq, entry.phdos_pred, color=palette[0])
        ax[row,0].set_title(entry.formula.translate(sub), fontsize=fontsize - 2, y=0.99)
        ax[row,0].set_xticks([]); ax[row,0].set_yticks([])

        # plot partial DoS
        for j, s in enumerate(entry.species):
            ax[row,j+1].plot(entry.phfreq, entry.pdos[s], color='black')
            ax[row,j+1].plot(entry.phfreq, pdos[s]/pdos[s].max(), color=palette[1], lw=2)
            ax[row,j+1].set_title(s, fontsize=fontsize - 2, y=0.99)
            ax[row,j+1].set_xticks([]); ax[row,j+1].set_yticks([])

        for j in range(len(entry.species) + 1, N+1):
            ax[row,j].remove()

    try: fig.supylabel('Intensity', fontsize=fontsize, x=0.08)
    except: pass
    else: fig.supxlabel('Frequency', fontsize=fontsize, y=0.06)
    fig.subplots_adjust(hspace=0.8)


def weighted_mean(energies, values):
    """
    Compute the weighted average of 3×3 matrix-valued functions over energy, for multiple samples.

    Parameters
    ----------
    energies : array-like of shape (N, E)
        Energy grids for N samples, each with E energy points.

    values : array-like of shape (N, E, 3, 3)
        Matrix values corresponding to each energy point for N samples.

    Returns
    -------
    weighted_avgs : ndarray of shape (N, 3, 3)
        The weighted average 3x3 matrix per sample.
    """
    energies = np.asarray(energies)
    values = np.asarray(values)

    if energies.shape[0] != values.shape[0] or energies.shape[1] != values.shape[1]:
        raise ValueError("Mismatched shapes between energies and values")

    weighted_avgs = []
    for e, v in zip(energies, values):
        weights = np.gradient(e)[:, np.newaxis, np.newaxis]  # shape (E, 1, 1)
        weighted_avg = np.sum(v * weights, axis=0) / np.sum(weights)
        weighted_avgs.append(weighted_avg)

    return np.array(weighted_avgs)  # shape (N, 3, 3)

# Load data per split
# def extract_and_average(df, idx, column):
#     subset = df.iloc[idx][['formula', 'energies_interp', column, 'y_pred_cart', 'mse_cart']]
#     dx = np.array([sample['energies_interp'] for _, sample in subset.iterrows()])
#     gt = np.array([sample[column] for _, sample in subset.iterrows()])
#     pr = np.array([sample['y_pred_cart'] for _, sample in subset.iterrows()])
    
#     wgt = weighted_mean(dx, gt)
#     wpr = weighted_mean(dx, pr)
    
#     return wgt, wpr
def extract_and_average(df, idx, column):
    subset = df.iloc[idx][['formula', 'energies_interp', column, 'y_pred_cart', 'mse_cart']]
    dx = np.array([sample['energies_interp'] for _, sample in subset.iterrows()])
    gt = np.array([sample[column] for _, sample in subset.iterrows()])
    pr = np.array([sample['y_pred_cart'] for _, sample in subset.iterrows()])
    
    wgt = weighted_mean(dx, gt)
    wpr = weighted_mean(dx, pr)
    
    return wgt, wpr, gt, pr  # Return weighted means and full spectrum

def r2_score(y_true, y_pred):
    """
    Calculate the R^2 (coefficient of determination) value between true and predicted values.
    
    Parameters:
        y_true (array-like): Array of true Y values.
        y_pred (array-like): Array of predicted Y values.
        
    Returns:
        float: R^2 value.
    """
    # Calculate the mean of true values
    y_true_mean = np.mean(y_true)
    # Calculate the total sum of squares
    ss_total = np.sum((y_true - y_true_mean) ** 2)
    # Calculate the residual sum of squares
    ss_residual = np.sum((y_true - y_pred) ** 2)
    # Calculate R^2
    r_squared = 1 - (ss_residual / ss_total)
    return r_squared

def process_atom(Z):
    """Process atomic properties for an element."""
    specie = Atom(Z)
    Z_mass = specie.mass
    Z_dipole = element(specie.symbol).dipole_polarizability or 67.0
    Z_radius = element(specie.symbol).covalent_radius_pyykko
    return specie.symbol, Z - 1, Z_mass, Z_dipole, Z_radius


def save_or_load_onehot(save_path="../onehot_data"):
    # Define file paths
    type_onehot_path = os.path.join(save_path, "type_onehot.torch")
    mass_onehot_path = os.path.join(save_path, "mass_onehot.torch")
    dipole_onehot_path = os.path.join(save_path, "dipole_onehot.torch")
    radius_onehot_path = os.path.join(save_path, "radius_onehot.torch")
    encoding_path = os.path.join(save_path, "type_encoding.torch")

    # Check if data already exists
    if all(os.path.exists(path) for path in [type_onehot_path, mass_onehot_path, dipole_onehot_path, radius_onehot_path, encoding_path]):
        print("Loading existing data...")
        type_onehot = torch.load(type_onehot_path)
        mass_onehot = torch.load(mass_onehot_path)
        dipole_onehot = torch.load(dipole_onehot_path)
        radius_onehot = torch.load(radius_onehot_path)
        type_encoding = torch.load(encoding_path)
    else:
        print("Processing data...")
        type_encoding = {}
        specie_mass = []
        specie_dipole = []
        specie_radius = []

        for Z in tqdm(range(1, 119), desc="Processing Elements"):
            symbol, encoding, mass, dipole, radius = process_atom(Z)
            type_encoding[symbol] = encoding
            specie_mass.append(mass)
            specie_dipole.append(dipole)
            specie_radius.append(radius)

        # Convert to one-hot encodings
        type_onehot = torch.eye(len(type_encoding))
        mass_onehot = torch.diag(torch.tensor(specie_mass))
        dipole_onehot = torch.diag(torch.tensor(specie_dipole))
        radius_onehot = torch.diag(torch.tensor(specie_radius))

        # Save the data
        os.makedirs(save_path, exist_ok=True)
        torch.save(type_onehot, type_onehot_path)
        torch.save(mass_onehot, mass_onehot_path)
        torch.save(dipole_onehot, dipole_onehot_path)
        torch.save(radius_onehot, radius_onehot_path)
        torch.save(type_encoding, encoding_path)
        print("Data saved!")

    return type_onehot, mass_onehot, dipole_onehot, radius_onehot, type_encoding

# build data
def build_data(entry, column, scale_data, type_onehot, mass_onehot, dipole_onehot, radius_onehot, type_encoding, r_max=5., ):
    symbols = list(entry.symmetrized_structure.symbols).copy()
    positions = torch.from_numpy(entry.symmetrized_structure.positions.copy())
    lattice = torch.from_numpy(entry.symmetrized_structure.cell.array.copy()).unsqueeze(0)

    # edge_src and edge_dst are the indices of the central and neighboring atom, respectively
    # edge_shift indicates whether the neighbors are in different images or copies of the unit cell
    edge_src, edge_dst, edge_shift = neighbor_list("ijS", a=entry.symmetrized_structure, cutoff=r_max, self_interaction=True)
    
    # compute the relative distances and unit cell shifts from periodic boundaries
    edge_batch = positions.new_zeros(positions.shape[0], dtype=torch.long)[torch.from_numpy(edge_src)]
    edge_vec = (positions[torch.from_numpy(edge_dst)]
                - positions[torch.from_numpy(edge_src)]
                + torch.einsum('ni,nij->nj', torch.tensor(edge_shift, dtype=default_dtype), lattice[edge_batch]))

    # compute edge lengths (rounded only for plotting purposes)
    edge_len = np.around(edge_vec.norm(dim=1).numpy(), decimals=2)

    # Define the crystal systems and their one-hot encoding
    crystal_systems = ["triclinic", "monoclinic", "orthorhombic", "tetragonal", "trigonal", "hexagonal", "cubic"]
    num_crystal_systems = len(crystal_systems)
    
    # Get the crystal system from the DataFrame entry
    crystal_system = entry["crystal_system"].lower()
    # Create the one-hot encoding for the crystal system
    crystal_system_idx = crystal_systems.index(crystal_system)
    crystal_system_onehot = torch.zeros(num_crystal_systems, dtype=default_dtype)
    crystal_system_onehot[crystal_system_idx] = 1.0


    data = tg.data.Data(
        pos=positions, lattice=lattice, symbol=symbols,
        x_mass=mass_onehot[[type_encoding[specie] for specie in symbols]],       # atomic mass (node feature)
        x_dipole=dipole_onehot[[type_encoding[specie] for specie in symbols]],   # atomic dipole polarizability (node feature)
        x_radius=radius_onehot[[type_encoding[specie] for specie in symbols]],   # atomic covalent radius (node feature)
        z=type_onehot[[type_encoding[specie] for specie in symbols]],            # atom type (node attribute)
        x=type_onehot[[type_encoding[specie] for specie in symbols]],            # atom type (node attribute)
        edge_index=torch.stack([torch.LongTensor(edge_src), torch.LongTensor(edge_dst)], dim=0),
        edge_shift=torch.tensor(edge_shift, dtype=default_dtype),
        edge_vec=edge_vec, edge_len=edge_len,
        y=torch.from_numpy((entry[column])/scale_data).unsqueeze(0),
        # y = torch.from_numpy(entry[column] / scale_data).T.unsqueeze(0).reshape(1, -1),
        crystal_system_onehot=crystal_system_onehot.unsqueeze(0)  # New global feature
    )
    
    return data


def plot_spherical_harmonics_comparison(df, idx, column_name, title_prefix="", n=4):
    """
    Plots heatmaps comparing target spherical harmonics with predicted spherical harmonics for multiple samples
    in a more compact layout (e.g., 4 rows × multiple columns instead of a single wide row).

    Parameters:
    - df: Pandas DataFrame containing 'sph_coefs' (targets), 'y_pred_sph' (predictions), 'mse'.
    - idx: List or array of indices to consider (e.g., idx_train, idx_test).
    - title_prefix: Prefix to be added to the plot title (e.g., Material ID).
    - n: Number of samples to plot.
    """
    # Filter DataFrame based on provided indices
    from e3nn.io import CartesianTensor
    
    with sns.plotting_context("notebook", font_scale=2):
        ct = CartesianTensor("ij=ji")
        ds = df.iloc[idx].reset_index(drop=True)

        # Extract energy grid (assume shared for all)
        omega = df["energies_interp"].iloc[0]

        # Define subplot grid layout
        n_rows = 4  # Fixed number of rows
        n_cols = math.ceil(n / n_rows) * 2  # Ensures space for real & predicted pairs
        # Setup figure with multiple rows & columns for compact layout
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(3.5 * n_cols, 3 * n_rows), sharex=True, sharey=True)
        axes = axes.ravel()  # Flatten axes array for easy iteration

        # Spherical harmonic component labels
        component_ticks = [r"$Y^2_2$", r"$Y^2_{1}$", r"$Y^2_0$", r"$Y^2_{-1}$", r"$Y^2_{-2}$", r"$Y^0_0$"]
        
        
        for i, sample_idx in enumerate(idx[:n]):  # Use provided indices directly
            material_id = ds.iloc[i]['mp_id']
            crystal_system = ds.iloc[i]['crystal_system']
            real_permittivity = ds[column_name].iloc[i]  # (num_energies, 3, 3)
            pred_permittivity = ds["y_pred_cart"].iloc[i]  # (num_energies, 3, 3)
            perm = torch.tensor([1, 2, 0])  # Permutation: x->2, y->0, z->1

            # Convert to torch tensors if not already
            real_permittivity = torch.tensor(real_permittivity) if not torch.is_tensor(real_permittivity) else real_permittivity
            pred_permittivity = torch.tensor(pred_permittivity) if not torch.is_tensor(pred_permittivity) else pred_permittivity

            # Permute the 3x3 dimensions (axes 1 and 2)
            real_permittivity_permuted = real_permittivity[:, perm, :][:, :, perm]  # Shape: [201, 3, 3]
            pred_permittivity_permuted = pred_permittivity[:, perm, :][:, :, perm]  # Shape: [201, 3, 3]

            real_sph = ct.from_cartesian(real_permittivity_permuted)
            pred_sph = ct.from_cartesian(pred_permittivity_permuted)

            # Extract target and prediction, then transpose & reverse for correct visualization
            X_irrep_target = real_sph.numpy().T[::-1, :]
            X_irrep_pred = pred_sph.numpy().T[::-1, :]
            vmin = X_irrep_target.min()
            vmax = X_irrep_target.max()
            # Left: Target heatmap
            ax_target = axes[2 * i]
            sns.heatmap(X_irrep_target, cmap='bwr', center=0, ax=ax_target,
                        yticklabels=component_ticks, xticklabels=50,
                        vmin=vmin, vmax=vmax)

            # ax_target.set_title(f"{material_id} {crystal_system}\nTarget", fontsize=10)
            ax_target.tick_params(axis='x', rotation=0, labelsize=16)  # Ensures x-axis labels are horizontal
            ax_target.tick_params(axis='y', rotation=0, labelsize=16)  # Ensures y-axis labels are horizontal

            # Right: Predicted heatmap
            ax_pred = axes[2 * i + 1]
            sns.heatmap(X_irrep_pred, cmap='bwr', center=0, ax=ax_pred,
                        yticklabels=component_ticks, xticklabels=50,
                        vmin=vmin, vmax=vmax)

            # ax_pred.set_title(f"{material_id} {crystal_system}\nPredicted MSE={ds['mse_sph'].iloc[i]:.2e}", fontsize=12)
            ax_pred.tick_params(axis='x', rotation=0, labelsize=16)  # Ensures x-axis labels are horizontal
            ax_pred.tick_params(axis='y', rotation=0, labelsize=16)  # Ensures x-axis labels are horizontal

        # Set x-axis ticks at reasonable intervals
        num_omega = len(omega)
        tick_indices = np.linspace(0, num_omega - 1, 7, dtype=int)  # 5 integer ticks across the range
        tick_labels = [int(round(omega[idx])) for idx in tick_indices]  # Round omega values to integers

        for ax in axes[:2 * n]:  # Apply to only used axes
            ax.set_xticks(tick_indices)
            ax.set_xticklabels(tick_labels)

        plt.tight_layout(rect=[0, 0.05, 1, 1])  # Adjust layout
        plt.subplots_adjust(wspace=0.2, hspace=0.1)  
        os.makedirs("../pngs", exist_ok=True)
        save_path = f"../pngs/{title_prefix}_spectra.png"
        fig.savefig(save_path, dpi=300)

        plt.show()

def format_chemical_formula(formula):
    if not formula:
        return formula
    result = ''
    i = 0
    while i < len(formula):
        char = formula[i]
        if char.isalpha():
            element = char
            i += 1
            if i < len(formula) and formula[i].islower():
                element += formula[i]
                i += 1
            result += element
            number = ''
            while i < len(formula) and formula[i].isdigit():
                number += formula[i]
                i += 1
            if number:
                result += f'_{{{number}}}'
            if i < len(formula) and formula[i] == '^':
                i += 1
                superscript = ''
                while i < len(formula) and (formula[i].isdigit() or formula[i] in ['+', '-']):
                    superscript += formula[i]
                    i += 1
                if superscript:
                    result += f'^{{{superscript}}}'
        else:
            result += char
            i += 1
    return result

def plot_cartesian_tensor_comparison(df, idx, column_name, title_prefix="", n=3):
    """
    Plots multiple side-by-side line plots comparing predicted vs. real Cartesian tensors 
    in a more compact horizontal layout with minimal white space.

    Parameters:
    - df: Pandas DataFrame containing 'y_pred_cart' (predictions), 
          'real_Permittivity_Matrices_interp' (targets), and 'energies_interp'.
    - idx: List or array of indices to consider (e.g., idx_train, idx_test).
    - title_prefix: Prefix to be added to the plot title.
    - n: Number of samples to plot.
    """
    # Filter DataFrame based on provided indices
    ds = df.iloc[idx].reset_index(drop=True)

    # Define component labels
    xyz_list = ['x', 'y', 'z']
    tensor_components = [f"$\chi_{{{xyz_list[a]}{xyz_list[b]}}}$" for a in range(3) for b in range(a, 3)]

    # Define subplot grid layout
    n_rows = min(n, 4)  # Limit number of rows to at most 4
    n_cols = math.ceil(n / n_rows) * 2  # Ensures space for real & predicted pairs

    # Adjust figure size to reduce white space
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(2.8 * n_cols, 2.5 * n_rows), dpi=300, sharex=True)
    axes = axes.ravel()  # Flatten axes array for easy iteration

    handles, labels = [], []

    for i, sample_idx in enumerate(idx[:n]):  # Directly loop over provided idx
        material_id = ds.iloc[i]['mp_id']
        # mse_cart = ds.iloc[i]['mse_cart']
        mae_cart = ds.iloc[i]['mae_cart']
        # abs_diff_RMS = ds.iloc[i]['abs_diff_RMS']
        
        crystal_system = ds.iloc[i]['crystal_system']
        formula = ds.iloc[i]['formula']
        omega = ds["energies_interp"].iloc[i]
        formatted_formula = format_chemical_formula(formula)
        real_permittivity = ds[column_name].iloc[i]  # (num_energies, 3, 3)
        pred_permittivity = ds["y_pred_cart"].iloc[i]  # (num_energies, 3, 3)
        
        y_min = min(real_permittivity.min(), pred_permittivity.min())
        y_max = max(real_permittivity.max(), pred_permittivity.max())
        
        # Left subplot: Real permittivity
        ax_real = axes[2 * i]  # Every even index is real data
        for idx, (a, b) in enumerate([(x, y) for x in range(3) for y in range(x, 3)]):
            h, = ax_real.plot(omega, real_permittivity[:, a, b], label=tensor_components[idx], linestyle='-')
            if i == 0:  # Collect legend items only from the first plot
                handles.append(h)
                labels.append(tensor_components[idx])
        if column_name =="real_Permittivity_Matrices_interp":
            ax_real.set_ylim((y_min-0.1) * 1.4, y_max * 1.2)
        else:
            ax_real.set_ylim((y_min-0.5) * 1.4, y_max * 1.2)
        ax_real.text(0.95, 0.95, f"${formatted_formula}$ \n {crystal_system}", 
             transform=ax_real.transAxes, fontsize=16, verticalalignment='top', 
             horizontalalignment='right', bbox=dict(facecolor='white', alpha=0.6, edgecolor='none'))
        # ax_real.text(0.95, 0.95, r"${formatted_formula}$ \n {crystal_system}", transform=ax_real.transAxes, 
        #             fontsize=19, verticalalignment='top', horizontalalignment='right', bbox=dict(facecolor='white', alpha=0.6, edgecolor='none'))
        ax_real.yaxis.set_major_locator(MaxNLocator(integer=True, nbins=5))

        # Right subplot: Predicted permittivity
        ax_pred = axes[2 * i + 1]  # Every odd index is predicted data
        for idx, (a, b) in enumerate([(x, y) for x in range(3) for y in range(x, 3)]):
            ax_pred.plot(omega, pred_permittivity[:, a, b], linestyle='-')
        if column_name =="real_Permittivity_Matrices_interp":
            ax_pred.set_ylim((y_min-0.1) * 1.4, y_max * 1.2)
        else:
            ax_pred.set_ylim((y_min-0.5) * 1.4, y_max * 1.2)
        # Format the text with fixed-width numbers and aligned labels
        # Use a monospace font and ensure consistent spacing
        text_str = (
            f"{'MAE:':<1} {mae_cart:>4.4f}\n"
            # f"{'FME:':<1} {abs_diff_RMS:>4.4f}"
        )

        # Add the text to the subplot
        ax_pred.text(
            0.95, 0.95,  # Position in axes coordinates (top-right)
            text_str,
            transform=ax_pred.transAxes,
            fontsize=13,
            verticalalignment='top',
            horizontalalignment='right',
            bbox=dict(facecolor='white', alpha=0.6, edgecolor='none')
        )
        ax_pred.yaxis.set_major_locator(MaxNLocator(integer=True, nbins=5))
        
    # Remove x-labels from individual subplots
    for ax in axes:
        ax.set_xticks([])
    
    for ax in axes[-n_cols:]:  
        ax.set_xticks(np.linspace(omega.min(), omega.max(), 7))

    # Add a single, global x-axis label
    # fig.supxlabel("Photon energy (eV)", fontsize=24)
    # fig.supylabel(r"Im $\chi_{ij}(\omega) (F/m)$", fontsize=24)

    # Add a single legend outside the figure
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, 0.01), ncol=6, fontsize=22, frameon=False)
    
    plt.tight_layout(rect=[0, 0.05, 1, 1])  # Adjust layout
    plt.subplots_adjust(wspace=0.2, hspace=0.1)  

    os.makedirs("../pngs", exist_ok=True)
    save_path = f"../pngs/{title_prefix}_cart_spectra.png"
    fig.savefig(save_path, dpi=300)

    plt.show()




def format_chemical_formula(formula):
    if not formula:
        return formula
    result = ''
    i = 0
    while i < len(formula):
        char = formula[i]
        if char.isalpha():
            element = char
            i += 1
            if i < len(formula) and formula[i].islower():
                element += formula[i]
                i += 1
            result += element
            number = ''
            while i < len(formula) and formula[i].isdigit():
                number += formula[i]
                i += 1
            if number:
                result += f'_{{{number}}}'
            if i < len(formula) and formula[i] == '^':
                i += 1
                superscript = ''
                while i < len(formula) and (formula[i].isdigit() or formula[i] in ['+', '-']):
                    superscript += formula[i]
                    i += 1
                if superscript:
                    result += f'^{{{superscript}}}'
        else:
            result += char
            i += 1
    return result