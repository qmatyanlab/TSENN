from pymatgen.core import Structure
import numpy as np
import sys

def get_kgrid_from_spacing(poscar_file, k_spacing=0.05):
    """
    Compute Monkhorst-Pack k-point grid dimensions for a given structure
    and target reciprocal spacing.

    Args:
        poscar_file (str): Path to POSCAR/CONTCAR.
        k_spacing (float): Target spacing in reciprocal space [1/Å].

    Returns:
        tuple: (kx, ky, kz) integers for KPOINTS grid.
    """
    # Load structure
    structure = Structure.from_file(poscar_file)

    # Reciprocal lattice vectors (in 1/Å)
    rec_lattice = structure.lattice.reciprocal_lattice
    b1, b2, b3 = rec_lattice.matrix

    # Norms of reciprocal vectors
    norms = [np.linalg.norm(b) for b in [b1, b2, b3]]

    # k-points = |b_i| / k_spacing (rounded up)
    kgrid = [max(1, int(np.ceil(norm / k_spacing))) for norm in norms]

    return tuple(kgrid)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python kgrid_from_poscar.py POSCAR [k_spacing]")
        sys.exit(1)

    poscar = sys.argv[1]
    spacing = float(sys.argv[2]) if len(sys.argv) > 2 else 0.05

    kgrid = get_kgrid_from_spacing(poscar, spacing)
    print(f"Suggested k-grid for spacing {spacing:.3f} Å⁻¹: {kgrid}")