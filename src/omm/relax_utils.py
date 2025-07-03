import torch
import numpy as np
import openmm
import openmm.unit


class VelocitySampler:
    def __init__(self, p, gen_indices, temperature, temperature_units):
        """Initializes a VelocitySampler object
        Args:
            p (cgp.omm.Protein): protein object
            gen_indices (torch.tensor): indices to generate
            temperature (float): temperature
            temperature_units (str): temperature units
        """
        self.p = p
        self.temperature = temperature
        self.temperature_units = getattr(openmm.unit, temperature_units)
        self.gen_indices = gen_indices
        self.num_particles = self.p.simulation.system.getNumParticles()
        self.velocity_dist = self._init_velocity_dist()

    def _init_velocity_dist(self):
        """Gets a velocity sampler
        Returns:

        """
        temperature = self.temperature * self.temperature_units
        all_scales = np.array([((openmm.unit.AVOGADRO_CONSTANT_NA * openmm.unit.BOLTZMANN_CONSTANT_kB * temperature)
                                / (self.p.simulation.system.getParticleMass(index).in_units_of(openmm.unit.kilograms/openmm.unit.mole))).sqrt().in_units_of(openmm.unit.nanometers/openmm.unit.picosecond)._value
                               for index in self.p.gen_indices])
        all_scales = torch.tensor(all_scales,
                                  device=torch.device("cpu")).float()

        velocity_dist = torch.distributions.normal.Normal(loc=torch.zeros_like(all_scales),
                                                          scale=all_scales)
        return velocity_dist

    def sample(self, num_samples):
        """Samples velocities
        Args:
            num_samples (int): number of samples
        Returns:
            all_velocities (np.ndarray): all velocities
            all_velocities_log_prob (np.ndarray): all velocities log probability
        """
        all_velocities_gen = self.velocity_dist.sample((num_samples,))
        all_velocities_log_prob = self.velocity_dist.log_prob(
            all_velocities_gen).sum(-1)
        all_velocities_gen = all_velocities_gen.reshape(-1,
                                                        self.gen_indices.shape[0]//3,
                                                        3)

        all_velocities = torch.zeros((num_samples, self.num_particles, 3),
                                     device=torch.device("cpu"))
        all_velocities[:, self.gen_indices, :] = all_velocities_gen
        all_velocities = all_velocities.detach().cpu().numpy()
        all_velocities_log_prob = all_velocities_log_prob.detach().cpu().numpy()
        return all_velocities, all_velocities_log_prob
    

def relax_structures(p, file, vs, num_steps):
    """Relaxes structures
    Args:
        p (cgp.omm.Protein): protein object
        file (h5py.File): file object
        vs (cgp.omm.VelocitySampler): velocity sampler
        num_steps (int): number of steps
    """
    length_units = getattr(openmm.unit, p.simulation_args["length_units"])
    energy_units = getattr(openmm.unit, p.simulation_args["energy_units"])

    num_samples = file["all_atom_positions"].shape[0]
    all_velocities, all_velocities_log_prob = vs.sample(num_samples)
    all_init_pe = []
    all_final_pe = []
    
    all_init_ke = []
    all_final_ke = []
    for i in range(len(file["all_atom_positions"])):
        position = file["all_atom_positions"][i]
        position = position * length_units
        p.simulation.context.setPositions(position)
        #This will always be in nanometers/picosecond because it is hardcoded in the velocity sampler
        p.simulation.context.setVelocities((all_velocities[i] * openmm.unit.nanometers/openmm.unit.picosecond))
        try:
            init_pe, init_ke = p.get_info()
            p.simulation.step(num_steps)
            final_pe, final_ke = p.get_info()
        except (openmm.OpenMMException, ValueError):
            init_pe = np.inf
            init_ke = np.inf
            final_pe = np.inf
            final_ke = np.inf
        all_init_pe.append(init_pe)
        all_final_pe.append(final_pe)
        all_init_ke.append(init_ke)
        all_final_ke.append(final_ke)

    # TODO: SAVE ALL THE ELEMENTS USING THE KEYS and ADD UNITS
    file.create_dataset("all_init_pe", data=all_init_pe)
    file.create_dataset("all_final_pe", data=all_final_pe)
    file.create_dataset("all_init_ke", data=all_init_ke)
    file.create_dataset("all_final_ke", data=all_final_ke)
    file.create_dataset("all_velocities", data=all_velocities)
    file.create_dataset("all_velocities_log_prob", data=all_velocities_log_prob)


