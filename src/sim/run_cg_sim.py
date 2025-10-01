import os
import numpy as np
import mdtraj as md
# import hydra
import torch
from sim import OVRVO, generate_trajectory, TrajWriter
from omegaconf import OmegaConf
import random
from einops import rearrange, repeat, reduce

# @hydra.main(version_base="1.3", config_path="../cfgs", config_name="cg_sim")
# def main(cfg):
def run_cg_sim(u_model, start_positions, masses, cfg="cg_sim.yaml"):
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
    integrator = OVRVO(u_model, masses, batch_size = batch_size, **cfg["integrator_args"])
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
