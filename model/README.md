# Pre-trained models (Figshare)

All model checkpoints are stored on Figshare because Git LFS is disabled in this repository.
Download them from Figshare and place them in the `model/` directory.

Figshare DOI: https://doi.org/10.6084/m9.figshare.31180054

Download script:
`python3 scripts/download_figshare.py --extract`

Models for main results:
- `revision_250909_Lmax_2_Lr_0.01_bs_8_em_128_layers_4_best.torch`
- `optuna_trial_27_best.torch`

Model for auxiliary dataset:
- `auxiliary_optuna_trial_5_em_dim_128_layers_3_mul_64_lmax_2_best.torch`

Models for ablation study:
- TSENN_250909_Lmax_*_Lr_0.01_bs_12_em_128_layers_2_mul_32_best.torch
- TSENN-A_250909_Lmax_*_best.torch
- TSENN-B_250909_Lmax_*_Lr_0.01_bs_12_em_128_layers_2_mul_32_best.torch
- TSENN-S_250909_Lmax_*_Lr_0.01_bs_4_em_128_layers_2_mul_32_best.torch
- TSENN_ablation_250929_Lmax_2_Lr_0.01_layers_2_mul_32_0e_best.torch
- TSENN_ablation_250929_Lmax_2_Lr_0.01_layers_2_mul_32_2e_best.torch
- TSENN_ablation_250929_Lmax_2_Lr_0.01_layers_2_mul_32_both_best.torch
- TSENN_cart_model_250909_xx_yy_zz_best.torch
- TSENN_cart_model_250909_xx_yy_zz_xy_xz_yz_best.torch
- TSENN_cart_model_250909_xx_yy_zz_xz_best.torch
- unconverged_TSENN-B_250922_Lmax_2_Lr_0.01_bs_12_em_128_layers_2_mul_32_best.torch
