### Training Scripts

This directory includes several different training scripts for various tasks:

- **`train_full_im_tensor.py`** — Trains the full *imaginary part* of the dielectric tensor.  
- **`optuna_worker.py`** — Handles Optuna-based *hyperparameter optimization*. It is launched via `main_launcher.py` to distribute the search across multiple GPUs.  
- **`train_full_tensor_eatgnn_ddp.py`** — Trains the **TSENN–Attention** model for the same task.  
  > ⚠️ It is recommended to train this script using **Distributed Data Parallel (DDP)**, as tensor products are computationally expensive in single-GPU mode.  
- **`train_im_scalar.py`** — Trains the *scalar baseline* model.  
- **`train_im_scalar_cart_directly.py`** — Trains the scalar model *directly in Cartesian coordinates* (`x`, `y`, `z`) without performing the basis transformation.  
- **`train_sub_im_tensor.py`** — Trains *subcomponents* of the tensor (e.g., only the ℓ = 0 or ℓ = 2 part).  
  > Used for the **ablation study** described in our manuscript; see the *Supplementary Information* for details.
