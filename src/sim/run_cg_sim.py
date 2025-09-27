import os
import numpy as np
import mdtraj as md
# import hydra
import torch
from sim import OVRVO, generate_trajectory, TrajWriter
from omegaconf import OmegaConf
import random

# @hydra.main(version_base="1.3", config_path="../cfgs", config_name="cg_sim")
# def main(cfg):
def run_cg_sim(u_model, start_positions, masses=None, cfg="cg_sim.yaml"):
    """
    u_model: nn.Module subclass that has a get_forces(positions) method. The bias force should be included in this model if desired.
    topology: md.Trajectory.topology object that defines the system to be simulated.
    start_positions: np.ndarray of shape (batch_size, n_atoms, 3) defining the initial positions of the system.
    """
    batch_size = u_model.batch_size_force
    cfg = OmegaConf.load(cfg)
    global_args = cfg.global_args
    topology = md.load(global_args["pdb_file"]).topology
    # model_folder = global_args["model_folder"]
    # if model_folder is None:
    #     model_cfg = cfg
    # else:
    #     model_cfg = OmegaConf.load(f"{model_folder}/config.yaml")
    
    # nn_config = model_cfg["nn"]
    # prior_config = model_cfg["prior"]
    # moe_config = model_cfg["moe"]
    # train_config = model_cfg["train"]
    # bias_force = global_args["bias_force"]

    # if not model_cfg.global_args.use_nn:
    #     nn_model_name = None
    #     nn_config = None
    # else:
    #     nn_model_name = nn_config["model"]
    #     nn_model_args = nn_config["model_args"]
    # if not model_cfg.global_args.use_prior:
    #     prior_model_name = None
    #     prior_config = None
    # else:
    #     prior_model_name = prior_config["model"]
    #     prior_model_args = prior_config["model_args"]
    # if not model_cfg.global_args.use_moe:
    #     moe_model_name = None
    #     moe_model_args = None
    #     moe_config = None
    # else:
    #     moe_model_name = moe_config["model"]
    #     moe_model_args = moe_config["model_args"]
    # edge_args = model_cfg["dataset"]["edge_args"]

    # u_model = CGSimModel(nn_model_name=nn_model_name, 
    #                     nn_model_args=nn_model_args,
    #                     prior_model_name=prior_model_name,
    #                     prior_model_args=prior_model_args,
    #                     moe_model_name=moe_model_name,
    #                     moe_model_args=moe_model_args,
    #                     bias_force=bias_force,
    #                     **train_config["lightning_model_args"])
    # if global_args["model_folder"] is not None:
    #     u_model.load_model_from_ckpt(f"{model_folder}/checkpoints/best_model.ckpt")
    if torch.cuda.is_available():
        u_model = u_model.cuda()

    u_model.eval()
    # u_model.store_features(global_args["batch_size"], global_args["pdb_file"], 
    #                        nn_config, prior_config, moe_config, edge_args)
    # cfg.nn = nn_config
    # cfg.prior = prior_config
    # cfg.train = train_config
    # OmegaConf.save(cfg, f"{global_args['save_folder_name']}/config.yaml")

    # backbone = topology.select("name CA or name N or name C")
    num_atoms = topology.n_atoms
    print(f"Number of atoms to simulate: {num_atoms}", flush=True)
    if masses is None:
        masses = []
        for atom in topology.atoms():
            masses.append(atom.element.mass) # units of amu aka Dalton

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
    elif global_args["start_positions"] is None and start_positions is None:
        print("No checkpoint or starting positions found. Starting from random positions.", flush=True)
        init_x = torch.randn(batch_size, num_atoms, 3, requires_grad=True)
        init_v = None
        start_chk = 0
    elif start_positions is not None:
        print("Starting from provided starting positions.", flush=True)
        init_x = torch.tensor(start_positions, dtype=torch.float32)
        init_x = init_x.reshape(-1, 3)
        init_v = None
        start_chk = 0
    else:
        start_positions = torch.tensor(np.load(global_args["start_positions"]))
        start_indices = np.load(global_args["start_indices"])
        init_x = start_positions[start_indices, :, :]
        init_x = init_x.reshape(-1, 3)
        init_v = None
        start_chk = 0

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
