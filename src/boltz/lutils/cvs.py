"""All code for CV calculation for Boltz umbrella sampling is defined 
within this module. To add a new CV functor, define a new class with a 
'@register_class' decorator. The class name can then be passed through 
the --umbrella_functor argument in main.py.

All functors' __call__ functions' only parameter should be the system 
coordinates in units of nanometers.
"""

import torch
import mdtraj as md
import numpy as np
import os

CLASS_REGISTRY = {}

def register_class(cls):
    """Decorator to register a class in the registry by its name."""
    CLASS_REGISTRY[cls.__name__] = cls
    return cls

@register_class
class Chignolin:
    """Computes the first two tICA coordinates for Chignolin based on 
    CA atom distances.

    Parameters
    ----------
    top : md.Topology
        mdtraj topology of the PDB.
    """
    def __init__(self, top, device="cpu"):
        base_dir = os.path.dirname(os.path.dirname(__file__)) # .../src/boltz/
        means_path = os.path.join(
            base_dir, 
            "assets", "cvs",
            "chignolin_tica_mean.npy"
        )
        self.means = torch.tensor(
            np.load(means_path), 
            dtype=torch.float32, 
            device=device
        )
        eigens_path = os.path.join(
            base_dir, 
            "assets", "cvs",
            "chignolin_tica_eigenvectors.npy"
        )
        self.eigenvectors = torch.tensor(
            np.load(eigens_path), 
            dtype=torch.float32, 
            device=device
        )

        self.device = device

        # Select CA atoms and precompute distance indices
        self.ca_indices = top.select("name CA")
        traj = md.Trajectory(np.zeros((1, top.n_atoms, 3)), top)
        traj = traj.atom_slice(self.ca_indices)
        self.distance_indices = np.array(
            [
                [i, j] for i in range(traj.n_atoms)
                for j in range(i + 1, traj.n_atoms)
            ],
            dtype=np.int32
        )

    def __call__(self, coords_nm):
        """
        Compute the tICA coordinates based on CA atom distances.

        Parameters
        ----------
        coords_nm: torch.Tensor of shape (n_atoms, 3) 
            System coordinates in nm.

        Returns
        -------
        torch.Tensor of shape (2,) 
            The first two tICA coordinates.
        """
        # Select CA coords
        ca_coords = coords_nm[self.ca_indices]  # (n_CA, 3)

        # Compute pairwise distances
        # Broadcasting to get (n_pairs,)
        diffs = ca_coords[self.distance_indices[:, 0]] - ca_coords[self.distance_indices[:, 1]]
        distances = torch.linalg.norm(diffs, dim=1)

        # tICA projection
        proj = (distances - self.means) @ self.eigenvectors
        return proj[:2]  # first 2 tICs