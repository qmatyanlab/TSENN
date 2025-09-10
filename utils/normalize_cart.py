import torch
from e3nn.io import CartesianTensor

EPS = 1e-12

def compute_norm_params(eps: torch.Tensor) -> torch.Tensor:
    """
    eps: (N, F, 3, 3) full dataset (imag part) on the *target* device/dtype
    Returns norm_params: (3, 3, 2)
      - diagonals: [min, max]
      - off-diagonals: [0, max_abs]
    Mirrors (i,j) -> (j,i) so it's symmetric by construction.
    """
    assert eps.ndim == 4 and eps.shape[-2:] == (3, 3)
    device, dtype = eps.device, eps.dtype

    # Flatten across samples+freqs per component
    comp_flat = eps.view(-1, 3, 3)

    # Initialize
    norm_params = torch.zeros((3, 3, 2), device=device, dtype=dtype)

    # Diagonals: min/max
    for i in range(3):
        comp = comp_flat[:, i, i]
        mn = comp.min()
        mx = comp.max()
        # Guard (constant channel)
        if torch.isclose(mx, mn):
            mx = mn + EPS
        norm_params[i, i, 0] = mn
        norm_params[i, i, 1] = mx

    # Off-diagonals: max_abs (center stays 0)
    for (i, j) in [(0,1), (0,2), (1,2)]:
        comp = comp_flat[:, i, j]
        mabs = comp.abs().max()
        if torch.isclose(mabs, torch.tensor(0.0, device=device, dtype=dtype)):
            mabs = torch.tensor(EPS, device=device, dtype=dtype)
        # store for (i,j) and mirror to (j,i)
        norm_params[i, j, 0] = 0.0
        norm_params[i, j, 1] = mabs
        norm_params[j, i, 0] = 0.0
        norm_params[j, i, 1] = mabs

    return norm_params


def normalize_with_params(eps: torch.Tensor, norm_params: torch.Tensor) -> torch.Tensor:
    """
    Apply the scheme:
      diag: (x - min) / (max - min)
      off:  x / max_abs
    """
    device, dtype = eps.device, eps.dtype
    eps_norm = torch.zeros_like(eps, device=device, dtype=dtype)

    for i in range(3):
        # diagonal
        mn, mx = norm_params[i, i]
        scale = (mx - mn).clamp_min(EPS)
        eps_norm[:, :, i, i] = (eps[:, :, i, i] - mn) / scale

        # off-diagonals
        for j in range(3):
            if i == j:
                continue
            _, mabs = norm_params[i, j]
            scale_off = mabs.clamp_min(EPS)
            eps_norm[:, :, i, j] = eps[:, :, i, j] / scale_off

    return eps_norm


def denormalize(eps_norm: torch.Tensor, norm_params: torch.Tensor) -> torch.Tensor:
    """
    Inverse of normalize_with_params
    """
    device, dtype = eps_norm.device, eps_norm.dtype
    eps_rec = torch.zeros_like(eps_norm, device=device, dtype=dtype)

    for i in range(3):
        # diagonal
        mn, mx = norm_params[i, i]
        scale = (mx - mn).clamp_min(EPS)
        eps_rec[:, :, i, i] = eps_norm[:, :, i, i] * scale + mn

        # off-diagonals
        for j in range(3):
            if i == j:
                continue
            _, mabs = norm_params[i, j]
            scale_off = mabs.clamp_min(EPS)
            eps_rec[:, :, i, j] = eps_norm[:, :, i, j] * scale_off

    return eps_rec


def cart_to_sph(eps_cart_norm: torch.Tensor) -> torch.Tensor:
    """
    (N, F, 3, 3) -> (N, F, 6) with ij=ji convention
    """
    x = CartesianTensor("ij=ji")
    return x.from_cartesian(eps_cart_norm)


def sph_to_cart(sph: torch.Tensor) -> torch.Tensor:
    """
    (N, F, 6) -> (N, F, 3, 3)
    """
    x = CartesianTensor("ij=ji")
    return x.to_cartesian(sph)

def save_norm_params(path: str, norm_params: torch.Tensor):
    torch.save(norm_params.detach().cpu(), path)

def load_norm_params(path: str, device=None, dtype=None) -> torch.Tensor:
    t = torch.load(path, map_location="cpu")
    if device is not None: t = t.to(device)
    if dtype is not None:  t = t.to(dtype)
    return t
