import os
import numpy as np
import mdtraj as md
# import hydra
import torch
from sim import OVRVO, Brownian, generate_trajectory, TrajWriter
from omegaconf import OmegaConf
import random
from einops import rearrange, repeat, reduce

# @hydra.main(version_base="1.3", config_path="../cfgs", config_name="cg_sim")
# def main(cfg):
def run_cg_sim(
    u_model, 
    start_positions, 
    masses, 
    cfg="cg_sim.yaml",
    integrator="OVRVO"
):
    """
    u_model: nn.Module subclass that has a get_forces(positions) method. The bias force should be included in this model if desired.
    start_positions: np.ndarray of shape (batch_size, n_atoms, 3) defining the initial positions of the system.
    cfg: str (path to YAML file) or dict (configuration dictionary)
    """
    batch_size = u_model.batch_size_force
    
    # Handle both YAML file path and configuration dictionary
    if isinstance(cfg, str):
        cfg = OmegaConf.load(cfg)
    else:
        # Convert dict to OmegaConf structure
        cfg = OmegaConf.create(cfg)
    print(cfg)
    global_args = cfg.global_args

    if torch.cuda.is_available():
        u_model = u_model.cuda()

    u_model.eval()

    num_atoms = len(masses)

    masses = np.array(masses).astype(np.float32)

    chk_files = os.listdir(global_args["save_folder_name"])
    chk_files = [f for f in chk_files if f.endswith("_chk.pt")]
    
    if len(chk_files) > 0:
        chk_files = sorted(chk_files, key=lambda x: int(x.split("_")[-2]))
        print(f"Found {len(chk_files)} checkpoint files. Resuming from {chk_files[-1]}", flush=True)
        chk = torch.load(f"{global_args['save_folder_name']}/{chk_files[-1]}", weights_only=False)
        start_chk = int(chk_files[-1].split("_")[-2]) + 1
        init_x = chk['positions']
        init_v = chk['velocities']
        torch.set_rng_state(chk['torch'])
        if torch.cuda.is_available() and chk['cuda'] is not None:
            torch.cuda.set_rng_state_all(chk['cuda'])
        np.random.set_state(chk['numpy'])
        random.setstate(chk['python'])
    elif start_positions is not None:
        print("Starting from provided starting positions.", flush=True)
        init_x = torch.tensor(start_positions, dtype=torch.float32)
        # init_x = init_x.reshape(-1, 3)
        init_x = rearrange(init_x, "batch atoms dim -> (batch atoms) dim")
        init_v = None
        start_chk = 0
    else:
        print("No checkpoint or starting positions found. Starting from random positions.", flush=True)
        init_x = torch.randn(batch_size * num_atoms, 3, requires_grad=True)
        init_v = None
        start_chk = 0

    print(f"{init_x.shape = }", flush=True)

    if integrator == 'OVRVO':
        integrator = OVRVO(
            u_model, 
            masses, 
            batch_size=batch_size, 
            **cfg["integrator_args"]
        )
    elif integrator == 'Brownian':
        integrator = Brownian(
            u_model, 
            masses,
            batch_size=batch_size,
            **cfg["integrator_args"]
        )
    else:
        raise ValueError('Invalid integrator. Options: "OVRVO", "Brownian".')

    generate_trajectory(integrator=integrator,
                        number_atoms=num_atoms, 
                        batch_size=batch_size,
                        num_data_points=int(global_args["num_data_points"]),
                        save_freq=global_args["save_freq"],
                        chk_freq=global_args["chk_freq"],
                        start_chk=start_chk,
                        save_filename=f"{global_args['save_folder_name']}/cg_dataset",
                        init_x=init_x,
                        init_v=init_v)





import h5py
from tqdm import trange
from openmm import unit
class MALAWriter:
    def __init__(self, filename, batch_size,
                 num_atoms):
        self.batch_size = batch_size
        self.num_atoms = num_atoms
        self.filename = filename

        if os.path.exists(self.filename):
            with h5py.File(self.filename, "r") as file:
                self.current_frame = file["positions"].shape[0]
                print(f"Appending to existing Writer file: {self.filename}, starting at frame {self.current_frame}.")
        else:
            self.current_frame = 0
            print(f"Creating Writer with file: {self.filename}.")
            with h5py.File(self.filename, "w", libver='latest') as file:
                file.create_dataset(f"positions", (0, batch_size, num_atoms, 3),
                                    maxshape=(None, batch_size, num_atoms, 3),
                                    dtype='f4')
                file.create_dataset(f"energies", (0, batch_size),
                                    maxshape=(None, batch_size),
                                    dtype='f4')
                file.create_dataset(f"forces", (0, batch_size, num_atoms, 3),
                                    maxshape=(None, batch_size, num_atoms, 3),
                                    dtype='f4')
                file.create_dataset("accepted", (0, batch_size),
                                    maxshape=(None, batch_size),
                                    dtype='bool')

    def write(self, positions, energies, forces, accepted):
        with h5py.File(self.filename, "a", libver='latest') as file:

            if positions is not None:
                file["positions"].resize((self.current_frame + 1, self.batch_size, self.num_atoms, 3))
                file["positions"][self.current_frame] = positions

            if energies is not None:
                file["energies"].resize((self.current_frame + 1, self.batch_size))
                file["energies"][self.current_frame] = energies
            
            if forces is not None:
                file["forces"].resize((self.current_frame + 1, self.batch_size, self.num_atoms, 3))
                file["forces"][self.current_frame] = forces

            if accepted is not None:
                file["accepted"].resize((self.current_frame + 1, self.batch_size))
                file["accepted"][self.current_frame] = accepted

        self.current_frame += 1
        
    def add_attributes(self, attrs: dict):
        with h5py.File(self.filename, "a", libver='latest') as file:
            for key, value in attrs.items():
                file.attrs[key] = value

def mala_step(x, cg_model, step_size=0.1, beta=1.0, masses=None):
    """
    Perform a single MALA step for all-atom coordinates x under potential U(x)
    x is shape (batch_size, num_atoms, 3)
    masses is shape (1, num_atoms, 1)
    """
    current_state = x.clone()
    
    if masses is None:
        masses = torch.ones_like(current_state)

    # get_score_and_energy_kt method needs to return in kT units
    current_score, current_energy = cg_model.get_score_and_energy_kt(current_state, beta=beta)
    current_score = current_score / masses  # pre-conditioned score using mass matrix (diagonal here)

    noise = torch.randn_like(current_state) * (2.0 * step_size / masses).sqrt()
    proposed_state = current_state + (step_size * current_score) + noise
    
    # Reverse proposal (for proposed -> current)
    proposed_state_grad = proposed_state.detach().clone()
    proposed_score, proposed_energy = cg_model.get_score_and_energy_kt(proposed_state_grad, beta=beta)
    proposed_score = proposed_score / masses  # pre-conditioned score for reverse proposal
    
    # Calculate forward and reverse log probabilities (with mass preconditioning)
    # Proposal covariance is 2*step_size*M^(-1), so precision matrix is M/(2*step_size)
    # note we can get rid of the multivariate normalization constant since it cancels in the acceptance ratio 
    # but only because the preconditioning matrix is constant
    forward_logp = -((proposed_state - current_state - (step_size * current_score)) ** 2 * masses).sum((1,2)) / (4 * step_size)
    reverse_logp = -((current_state - proposed_state - (step_size * proposed_score)) ** 2 * masses).sum((1,2)) / (4 * step_size)
    # Get target logp
    proposed_tlogp = -proposed_energy
    current_tlogp = -current_energy
    return_dict = {
        "proposed_state": proposed_state,
        "proposed_tlogp": proposed_tlogp,
        "current_tlogp": current_tlogp,
        "forward_logp": forward_logp,
        "reverse_logp": reverse_logp,
        "score": current_score
    }

    return return_dict

def langevin_step(x, cg_model, step_size=0.1, beta=1.0, masses=None):
    """
    Perform a single Langevin step for all-atom coordinates x under potential U(x)
    No Metropolis acceptance/rejection, just a pure Langevin step
    Often this will be for methods where we don't have access to the energy so in this case get_force_and_energy can just return None for energy and we will ignore it here
    x is shape (batch_size, num_atoms, 3)
    masses is shape (1, num_atoms, 1)
    """
    current_state = x.clone()
    
    if masses is None:
        masses = torch.ones_like(current_state)

    # get_score_and_energy_kt method needs to return in kT units
    current_score, current_energy = cg_model.get_score_and_energy_kt(current_state, beta=beta)
    current_score = current_score / masses  # pre-conditioned score using mass matrix (diagonal here)

    noise = torch.randn_like(current_state) * (2.0 * step_size / masses).sqrt()
    proposed_state = current_state + (step_size * current_score) + noise
    proposed_tlogp = None if current_energy is None else -current_energy
    return_dict = {
        "proposed_state": proposed_state,
        "proposed_tlogp": proposed_tlogp,
        "current_tlogp": None,
        "forward_logp": None,
        "reverse_logp": None,
        "score": current_score
    }

    return return_dict

def mala_sampling(initial_x, cg_model, metropolize=True, num_steps=1000, step_size=0.1, beta=1.0, masses=None,save_freq=100, writer=None):
    """
    Run MALA sampling for a number of steps
    """
    device = cg_model.device
    x = initial_x.to(device)
    if masses is None:
        masses = torch.ones_like(x)
    masses = masses.to(device)
    
    for i in trange(num_steps, mininterval=30, desc="MALA Sampling"):
        if metropolize:
            step_results = mala_step(x, cg_model, step_size, beta=beta, masses=masses,)
            proposed_state = step_results["proposed_state"]
            proposed_tlogp = step_results["proposed_tlogp"]
            current_tlogp = step_results["current_tlogp"]
            forward_logp = step_results["forward_logp"]
            reverse_logp = step_results["reverse_logp"]
            log_accept_ratio = proposed_tlogp + reverse_logp - current_tlogp - forward_logp
            accept = torch.log(torch.rand(log_accept_ratio.shape, device=device)) < log_accept_ratio
            x_new = x.clone()
            x_new[accept] = proposed_state[accept]
            x = x_new
        else: 
            step_results = langevin_step(x, cg_model, step_size, beta=beta, masses=masses)
            proposed_state = step_results["proposed_state"]
            proposed_tlogp = step_results["proposed_tlogp"]
            current_tlogp = step_results["current_tlogp"]
            accept = None
            x = proposed_state.clone()
        if writer is not None and (i + 1) % save_freq == 0:
            writer.write(x.detach().cpu().numpy(), None if current_tlogp is None else -current_tlogp.detach().cpu().numpy(),
                         step_results["score"].detach().cpu().numpy(), None if accept is None else accept.detach().cpu().numpy())


def run_langevin(initial_x, cg_model, num_steps=1000, step_size=0.1, temperature=300, temperature_units="kelvin", energy_units="kilojoule_per_mole", save_freq=100, save_folder="langevin_output"):
    """
    Wrapper function to run Langevin sampling with the given model and parameters
    """
    if getattr(unit, energy_units).is_compatible(unit.kilojoule_per_mole):
        beta = 1/(temperature * getattr(unit, temperature_units) * unit.AVOGADRO_CONSTANT_NA * unit.BOLTZMANN_CONSTANT_kB).value_in_unit(getattr(unit, energy_units))
    else:
        beta = 1/(temperature * getattr(unit, temperature_units) * unit.BOLTZMANN_CONSTANT_kB).value_in_unit(getattr(unit, energy_units))

    batch_size, num_atoms, _ = initial_x.shape
    writer = MALAWriter(filename=f"{save_folder}/langevin_samples.hdf5", batch_size=batch_size, num_atoms=num_atoms)
    writer.write(initial_x.detach().cpu().numpy(), None, None, None)  # save initial positions
    mala_sampling(initial_x, cg_model, metropolize=False, num_steps=num_steps, step_size=step_size, beta=beta, masses=None, save_freq=save_freq, writer=writer)