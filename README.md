# TSENN

This repository contains the source code and dataset for the **T**ensorial-**S**pectra **E**quivariant **N**eural **N**etwork (**TSENN**).

## Table of Contents

- [TSENN](#tsenn)
  - [Table of Contents](#table-of-contents)
  - [Install Dependencies](#install-dependencies)
    - [Version-Specific Packages](#version-specific-packages)
    - [Other Dependencies](#other-dependencies)
  - [Dataset](#dataset)
  - [Example Usage](#example-usage)
  - [Pretrained Model \& Visualization](#pretrained-model--visualization)

## Install Dependencies

### Version-Specific Packages

The following key dependencies were used in this project:
- `python==3.11.7`
- `torch==2.0.1+cu117`
- `torch_geometric==2.4.0`
- `torch_scatter==2.1.2+pt20cu117` (plus other torch_geometric dependencies)
- `pymatgen==2025.2.18`
- `e3nn==0.5.5`

It is recommended to install `torch` first, followed by `torch_geometric` and its corresponding dependencies for your CUDA version.

### Other Dependencies

You may also need the following common packages:
- `pandas`
- `wandb`
- `seaborn`
- `mendeleev`
- `ase`
- ...

You can install them using pip:

```bash
pip install pandas wandb seaborn mendeleev ase
```

To ensure all packages are installed, run the notebook at `notebook/full_tensor_train_data_prep.ipynb`. Missing packages can be added via pip as needed.

## Dataset

The dataset used in our work is located in the `dataset/` directory.
An example of how to load and use it is provided in:
`notebook/full_tensor_train_data_prep.ipynb`

Each entry in the dataset corresponds to a material and includes:

* `symmetrized_structure`
* `mp_id` (Materials Project ID)
* `chemical formula`
* `band gap`
* `crystal system`
* `photon energy (omega)`
* `real part of permittivity`
* `imaginary part of permittivity`

The permittivity data is stored as a NumPy array of shape `(3001, 3, 3)`,
where `3001` corresponds to the number of photon energy points.
Each tensor is a $3 \times 3$ matrix in Cartesian coordinates ($xx, xy, ..., zz$).

## Example Usage

To train the model, navigate to the scripts folder and run:

```bash
cd scripts/
python train_full_im_tensor.py
```

This will begin training the full-tensor model using the default parameters.

## Pretrained Model & Visualization

You can explore the pretrained model and visualize its results in the `notebook/` directory:

* `imaginary_part_prediction.ipynb`: demonstrates predictions using the pretrained model on curated datasets.
* `multiple_material_prediction.ipynb` and `material_prediction.ipynb`: shows how to query new materials from the [Materials Project](https://next-gen.materialsproject.org/) and make predictions.

---

Feel free to cite or fork this repository for your own research.
```
@misc{hsu2025accuratepredictionsequentialtensor,
      title={Accurate Prediction of Tensorial Spectra Using Equivariant Graph Neural Network}, 
      author={Ting-Wei Hsu and Zhenyao Fang and Arun Bansil and Qimin Yan},
      year={2025},
      eprint={2505.04862},
      archivePrefix={arXiv},
      primaryClass={cond-mat.mtrl-sci},
      url={https://arxiv.org/abs/2505.04862}, 
}
```

