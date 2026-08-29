"""Script to plot unbiased trajectories or umbrella sampling 
trajectories + free energy landscape in CV space.
"""

import argparse
import os
import h5py
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from openmm import unit
from glob import glob
from tqdm import tqdm
from pymbar import FES

# TODO: change.
# Load in bias force.
bba_pdb = md.load("/home/ethanz/data_boltz_likelihood/bba/bba_1fme.pdb") # NMR ensemble, 34 conformers.
bba_ca_pdb = bba_pdb.atom_slice(bba_pdb.topology.select("name CA")) # Trajectory w/ 34 frames, 28 CA atoms each.
ref_ca_p = bba_ca_pdb.xyz * 10 # Converting MDTraj nm to Angstroms.

helix_pdb = md.load("/home/ethanz/data_boltz_likelihood/bba/helix.pdb")
helix_ca_pdb = helix_pdb.atom_slice(helix_pdb.topology.select("name CA"))
helix_ca_p = helix_ca_pdb.xyz * 10 # Converting MDTraj nm to Angstroms.

import torch.nn as nn
class BiasForceEnergy(nn.Module):
    def __init__(self, centers: torch.Tensor, simulation_topology: md.Topology, bias_force:int =100.0):
        # simulation topology has the number of atoms you actually expect to simulate
        # for Boltz this means everything but the hydrogens
        # all positions should be in Angstroms
        super().__init__()
        self.centers = nn.Parameter(torch.tensor(centers, dtype=torch.float32))
        self.bias_force = bias_force
        self.batch_size = centers.shape[0]
        self.reference_ca = torch.tensor(ref_ca_p, dtype=torch.float32)
        self.reference_helix_ca = torch.tensor(helix_ca_p, dtype=torch.float32)

        ca_indices = simulation_topology.select("name CA")
        alpha_indices = ca_indices[15:26]
        beta_indices = ca_indices[3:15]


        self.alpha_indices = torch.tensor(alpha_indices, dtype=torch.long)
        self.beta_indices = torch.tensor(beta_indices, dtype=torch.long)

        self.ref_beta = self.reference_ca[:, 3:15, :]
        self.ref_beta = nn.Parameter(torch.cat([self.ref_beta for _ in range(self.batch_size)], dim=0))
        self.ref_helix = self.reference_helix_ca[:, 15:26, :]
        self.ref_helix = nn.Parameter(torch.cat([self.ref_helix for _ in range(self.batch_size)], dim=0))
        self.n_atoms = simulation_topology.n_atoms


    def rmsd(self, P: torch.Tensor, Q: torch.Tensor) -> torch.Tensor:
        """
        Compute RMSD per frame using Theobald's method (no explicit alignment).
        
        P: (n_frames, n_atoms, 3)
        Q: (n_frames, n_atoms, 3)

        Returns:
            rmsd_per_frame: (n_frames,)
        """
        # Center both structures
        P_centered = P - P.mean(dim=1, keepdim=True)
        Q_centered = Q - Q.mean(dim=1, keepdim=True)

        # Compute covariance matrix
        C = torch.matmul(P_centered.transpose(1, 2), Q_centered)  # (n_frames, 3, 3)

        # SVD on covariance matrix
        V, S, W = torch.linalg.svd(C)
        det = torch.det(torch.matmul(V, W.transpose(-1, -2)))
        sign_correction = det.sign()
        S_corrected = S.clone()
        S_corrected[:, -1] *= sign_correction  # fix last singular value

        dot = 2 * torch.sum(S_corrected, dim=-1)

        # Compute squared norms
        P_norm_sq = torch.sum(P_centered ** 2, dim=(1, 2))
        Q_norm_sq = torch.sum(Q_centered ** 2, dim=(1, 2))

        # RMSD² = (||P||² + ||Q||² - 2 * sum(S)) / N
        # dot = 2 * torch.sum(S, dim=-1)
        rmsd_sq = (P_norm_sq + Q_norm_sq - dot) / P.shape[1]
        # rmsd_sq = torch.clamp(rmsd_sq, min=0.0)

        return torch.sqrt(rmsd_sq)

    def transform_a(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.log(x + 2)
        return x

    def transform_b(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.log(x + 1)
        return x

    def get_cv(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        a = self.transform_a(self.rmsd(x[:, self.alpha_indices, :], self.ref_helix))
        b = self.transform_b(self.rmsd(x[:, self.beta_indices, :], self.ref_beta)) 
        return a, b

    def energy(self, x: torch.Tensor) -> torch.Tensor:
        # x is shape (batch, n_atoms, 3)
        alpha_cv, beta_cv = self.get_cv(x)

        alpha_cv = (self.centers[:, 0] - alpha_cv) ** 2
        beta_cv = (self.centers[:, 1] - beta_cv) ** 2
        energy = self.bias_force * (alpha_cv + beta_cv)
        return energy
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x is shape (batch, n_atoms, 3)
        # Ensure input tensor requires gradients
        if not x.requires_grad:
            x = x.clone().detach().requires_grad_(True)
        
        energy: torch.Tensor = self.energy(x)
        return energy.detach()

def compute_bias_energies(positions, bias_force):
    """Computes bias energies from position tensors.

    Parameters
    ----------
    bias_force : BiasForceBBA object from TODO

    Returns
    -------
    bias_energies : torch.Tensor of shape (frames, windows).
    """
    n_frames, n_windows, n_atoms, _ = positions.shape
    positions_flattened = (
        positions.permute(1, 0, 2, 3)
    ).reshape(n_windows * n_frames, n_atoms, 3)
    positions_repeated = (
        positions_flattened.unsqueeze(1)
    ).expand(-1, n_windows, -1, -1) # (windows * frames, windows, atoms, 3)

    bias_energies = []
    for repeated_single_window in tqdm(positions_repeated):
        '''repeated_single_window is (windows, atoms, 3) and represents a 
        single frame of a trajectory repeated windows times.'''
        bias_energies.append(bias_force(repeated_single_window))

    bias_energies = torch.stack(bias_energies).numpy().transpose(1, 0) # (windows, windows * frames)

    # TODO: change as appropriate based on chosen simulation energy units and such. 
    sim_energy_unit = unit.kilocalorie_per_mole 
    sim_temp = 350
    sim_temp_unit = unit.kelvin

    bias_energies *= sim_energy_unit / (sim_temp_unit * sim_temp * unit.constants.BOLTZMANN_CONSTANT_kB * unit.AVOGADRO_CONSTANT_NA).in_units_of(sim_energy_unit)
    # bias_energies are now in units of kT
    N_k = np.array([n_frames for _ in range(n_windows)])
    
    return bias_energies, N_k

def process_checkpoints(input_dir):
    stride = 1 # TODO: make it a param?
    """Loads position tensors from checkpoints, subsamples them, and 
    removes invalid frames.

    Returns
    -------
    positions : torch.Tensor of shape (frames, windows, atoms, 3).
        In units of Angstroms.
    """
    checkpoints = glob(os.path.join(input_dir, "cg_dataset_*.hdf5"))
    checkpoints = [ckpt for ckpt in checkpoints if os.path.exists(ckpt.replace(".hdf5", "_chk.pt"))]
    print(len(checkpoints), "valid checkpoints.")

    checkpoints = sorted(checkpoints, key=lambda x: int(x.split("_")[-1].split(".")[0]))

    sampled_positions = []
    for file in checkpoints:
        with h5py.File(file, "r") as file:
            pos_tensor = torch.tensor(file["positions"][::stride])
            sampled_positions.append(pos_tensor)
    sampled_positions = torch.cat(sampled_positions, dim=0) # (frames, windows, atoms, 3)
    print(
        "Position tensor shape before subsampling:", 
        sampled_positions.shape
    )

    # Removing invalid frames (if any).
    print(
        f"""{(sampled_positions != 0.0).all((1,2,3)).sum()} nonzero frames out 
        of {sampled_positions.shape[0]} total frames."""
    )
    print(
        f"""{(sampled_positions.isinf()).all((1,2,3)).sum()} inf out of 
        {sampled_positions.shape[0]}"""
    )
    invalid = ((sampled_positions == 0.0).all((1,2,3)) | (sampled_positions.isinf()).all((1,2,3)))
    sampled_positions = sampled_positions[~invalid]
    print(
        "Final position tensor shape after removing invalid frames:", 
        sampled_positions.shape
    )
    return sampled_positions

def main():
    # TODO: change.  
    centers = torch.from_numpy(np.load('/home/ethanz/data_boltz_likelihood/bba/all_bba_start_positions_cvs.npy'))
    simulation_topology = boltz_pdb = md.load("/home/ethanz/data_boltz_likelihood/bba/bba_model_0.pdb").topology
    bias_force = BiasForceEnergy(centers=centers, simulation_topology=simulation_topology, bias_force=500.0) 

    
    parser = argparse.ArgumentParser(
        description=(
            "Plot unbiased trajectories or umbrella sampling "
            "trajectories + free energy landscape in CV space."
        )
    )
    # Directory containing simulation trajectories.
    parser.add_argument(
        "--input",
        type=str,
        help=(
            "Path to umbrella_x directory."
        ),
    )
    parser.add_argument(
        "--output",
        type=str,
        help=(
            "Path to output file."
        ),
    )
    args = parser.parse_args()
    input_dir = args.input
    outpath = args.output

    # Make output directory if it doesn't exist.
    os.makedirs(os.path.dirname(outpath), exist_ok=True)
    
    positions = process_checkpoints(input_dir)

    # Computing all CV values (n_total_frames, num_cvs).
    tups = [bias_force.get_cv(all_window_frame) for all_window_frame in positions]
    cv_values_all_frames = torch.concat(
        [torch.stack(tup_elem) for tup_elem in tups], dim=1
    ).detach().permute(1, 0).numpy() # (n_total_frames, dim)

    bias_energies, N_k = compute_bias_energies(positions, bias_force)

    # Plotting
    # ----------------------------------------------------------------------- #
    _, x_edges, y_edges = np.histogram2d(cv_values_all_frames[:, 0], cv_values_all_frames[:, 1], bins=20)
    bin_centers_0 = 0.5 * (x_edges[:-1] + x_edges[1:])
    bin_centers_1 = 0.5 * (y_edges[:-1] + y_edges[1:])
    bin_centers = np.array(np.meshgrid(bin_centers_0, bin_centers_1)).T.reshape(-1, 2)
    free_energy = np.inf * np.ones(bin_centers.shape[0])

    center_indices_to_use = []
    centers_to_use = []

    width_0 = bin_centers_0[1] - bin_centers_0[0]
    width_1 = bin_centers_1[1] - bin_centers_1[0]

    for i in range(bin_centers.shape[0]):
        bin_center = bin_centers[i]
        x0, x1 = bin_center[0] - width_0 / 2, bin_center[0] + width_0 / 2
        y0, y1 = bin_center[1] - width_1 / 2, bin_center[1] + width_1 / 2
        
        mask = (cv_values_all_frames[:, 0] >= x0) & (cv_values_all_frames[:, 0] < x1) & (cv_values_all_frames[:, 1] >= y0) & (cv_values_all_frames[:, 1] < y1)
        num_points = np.sum(mask)
        if num_points > 0:
            centers_to_use.append(bin_center)
            center_indices_to_use.append(i)
    centers_to_use = np.array(centers_to_use)
    center_indices_to_use = np.array(center_indices_to_use)

    fes = FES(bias_energies, N_k)
    fes.generate_fes(np.zeros(cv_values_all_frames.shape[0]), cv_values_all_frames, fes_type="kde")
    results = fes.get_fes(centers_to_use, uncertainty_method=None)
    f = results["f_i"] # (36, )
    f = f - f.min()
    free_energy[center_indices_to_use] = f

    plt.figure(figsize=(5, 4))
    vmax = 50
    plt.contourf(bin_centers_0, bin_centers_1, free_energy.reshape(len(bin_centers_0), len(bin_centers_1)).T,
                    levels=np.linspace(0, vmax, 10), cmap='coolwarm')
    cbar = plt.colorbar(boundaries=np.linspace(0, vmax, 10), pad=0.01)
    cbar.set_label('Free Energies', rotation=270, labelpad=20)



    # only label integer values
    cbar.set_ticks(np.arange(0, vmax + 1, 1))
    # # plt.scatter(centers[:, 0], centers[:, 1], color='black', s=10, marker="x", label="Centers")
    # plt.scatter(starting_tica[:, 0], starting_tica[:, 1], color='red', s=10)
    # cbar.ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))

    # make X and Y axis ticks every 0.5
    plt.gca().xaxis.set_major_locator(ticker.MultipleLocator(1.0))
    plt.gca().yaxis.set_major_locator(ticker.MultipleLocator(1.0))


    # plt.xlabel(r"$\textrm{Native Contacts}$")
    # plt.ylabel(r"$\textrm{Dihedral}$")
    plt.xlim(-2, 2)
    plt.ylim(-2, 4)
    plt.tight_layout()
    plt.savefig(outpath, dpi=300)


    

