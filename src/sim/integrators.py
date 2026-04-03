import torch
import math
import openmm.unit as unit
from einops import rearrange, reduce, repeat

class OVRVO:
    """
    https://pubs.acs.org/doi/pdf/10.1021/jp411770f
    pos in units of A
    forces in units of kcal/A
    """
    def __init__(self, u_model, 
                 masses,
                 batch_size=1,
                 dt=0.001, 
                 friction=10., 
                 temperature=300.0, 
                 length_units="angstroms",
                 energy_units="kilocalories_per_mole", 
                 time_units="picoseconds",
                 temperature_units="kelvin"):
        self.u_model = u_model

        self.energy_units = energy_units
        self.length_units = length_units
        self.time_units = time_units
        self.temperature_units = temperature_units

        '''TODO: self.masses must be a torch tensor in internal units of 
        energy_units * time_units^2 / length_units^2, and kT needs to 
        be in internal energy_units. This is because utils.py will do
        scale = (integrator.kT/ integrator.masses) ** 0.5 to set
        initial velocities. 
        '''
        masses = 0.001 * torch.tensor(masses) # Converting masses from amu (g/mol) to kg/mol.
        unit_length_omm = 1.0 * getattr(unit, length_units)
        unit_energy_omm = 1.0 * getattr(unit, energy_units)
        unit_time_omm = 1.0 * getattr(unit, time_units)
        unit_temperature_omm = 1.0 * getattr(unit, temperature_units)
        # 1 kg/mol = 1 (s^2 * J) / (m^2 * mol).
        unit_mass_omm = 1.0 * ((getattr(unit, "second") ** 2) 
                               * getattr(unit, "joules") 
                               / (getattr(unit, "meters") ** 2)
                               / getattr(unit, "mole"))

        friction_omm = friction / unit_time_omm
        dt_omm = dt * unit_time_omm
        temperature_omm = temperature * unit_temperature_omm
        beta_omm = 1/(temperature_omm
                      * unit.BOLTZMANN_CONSTANT_kB
                      * unit.AVOGADRO_CONSTANT_NA)
        kT_omm = 1.0 / beta_omm

        # TODO: brittle, breaks if unit_energy_omm is not a per mol energy unit.
        masses_conversion_factor = unit_mass_omm.value_in_unit((unit_energy_omm * unit_time_omm**2/unit_length_omm**2).unit)
        self.masses = torch.tensor(masses) * masses_conversion_factor # energy_units * time_units^2 / length_units^2
        num_atoms = len(masses)
        # self.masses = self.masses.repeat_interleave(3, 0).reshape(num_atoms, 3) 
        # einops equivalent: self.masses = einops.repeat(self.masses, 'n -> n 3')
        # self.masses = self.masses.repeat(batch_size, 1, 1).reshape(batch_size * num_atoms, 3).to(u_model.device)
        # einops equivalent: self.masses = einops.repeat(self.masses, 'n d -> b n d', b=batch_size); self.masses = einops.rearrange(self.masses, 'b n d -> (b n) d')

        self.masses = repeat(self.masses, 'atoms -> (batch atoms) 3', batch=batch_size).to(u_model.device)
        self.temperature = temperature
        self.dt = dt_omm.value_in_unit(getattr(unit, time_units))
        self.kT = kT_omm.value_in_unit(getattr(unit, energy_units))
        self.friction_omm = friction_omm
        self.dt_omm = dt_omm
        self.t_rescale = torch.sqrt((2/(friction_omm * dt_omm)) 
                                    * torch.tanh(torch.tensor(friction_omm * dt_omm/2))) #unitless
        self.a = math.exp(-friction_omm * dt_omm)  # Unitless
        self.b = torch.sqrt(((1 - self.a) * self.kT)/self.masses) # length_units/time_units
        

    def ovrvo_step(self, x, v, f_prev):
        # time-step rescaling (b in paper) is set to be 1
        v = math.sqrt(self.a) * v + self.b * torch.randn_like(x) # length_units/time_units
        v = v + 0.5 * self.dt * (f_prev / self.masses) * self.t_rescale  # length_units/time_units
        x = x + self.dt * v * self.t_rescale # length_units
        f = self.u_model.get_force(x) # energy_units/length_units
        f = f.detach()  # energy_units/length_units
        x = x.detach() # length_units
        v = v + 0.5 * self.dt * (f / self.masses) * self.t_rescale # length_units/time_units
        # import pdb; pdb.set_trace()
        v = math.sqrt(self.a) * v + self.b * torch.randn_like(x) # length_units/time_units
        return x, v, f

    def integrate(self, init_x, init_v, steps, writer, save_freq=100):
        x = init_x.to(self.u_model.device) # length_units
        v = init_v.to(self.u_model.device) # length_units/time_units
        f_prev = self.u_model.get_force(x) #energy_units/length_units
        f_prev = f_prev.detach()  # energy_units/length_units
        x = x.detach() # length_units
        
        # Get the current time offset if it exists, otherwise start at 0
        time_offset = getattr(self, 'current_time_offset', 0.0)
        
        for i in range(steps):
            if (i % save_freq) == 0:
                # Calculate simulation time in the integrator's time units
                simulation_time = time_offset + i * self.dt
                writer.write(x.cpu(), v.cpu(), f_prev.cpu(), i // save_freq, simulation_time)
            x, v, f_prev = self.ovrvo_step(x, v, f_prev)
        writer.close()
        return x, v
    
    def update_temperature(self, temperature, 
                           temperature_units="kelvin"):
        """
        Update the temperature of the system.
        """
        assert temperature_units == self.temperature_units
        unit_temperature_omm = 1.0 * getattr(unit, temperature_units)
        temperature_omm = temperature * unit_temperature_omm
        beta_omm = 1/(temperature_omm
                      * unit.BOLTZMANN_CONSTANT_kB
                      * unit.AVOGADRO_CONSTANT_NA)
        kT_omm = 1.0 / beta_omm
        self.kT = kT_omm.value_in_unit(getattr(unit, self.energy_units))
        self.b = torch.sqrt(((1 - self.a) * self.kT)/self.masses)
        self.t_rescale = torch.sqrt((2/(self.friction_omm * self.dt_omm)) 
                                    * torch.tanh(torch.tensor(self.friction_omm * self.dt_omm/2)))
        self.temperature = temperature

    def heat_system(self, init_x, init_v, writer, save_freq,
                    end_temperature, heating_steps,
                    temperature_units="kelvin",
                    time_units="picoseconds"):
        assert temperature_units == self.temperature_units
        assert time_units == self.time_units
        temp_increase_per_step = (end_temperature - self.temperature) / heating_steps
        x = init_x.to(self.u_model.device) # length_units
        v = init_v.to(self.u_model.device) # length_units/time_units
        f_prev = self.u_model.get_force(x) #energy_units/length_units
        f_prev = f_prev.detach()  # energy_units/length_units
        x = x.detach() # length_units

        for i in range(heating_steps):
            if (i % save_freq) == 0:
                writer.write(x.cpu(), f_prev.cpu(), i // save_freq)
            x, v, f_prev = self.ovrvo_step(x, v, f_prev)
            self.update_temperature(self.temperature + temp_increase_per_step,
                                    temperature_units=temperature_units)
        writer.close()
        return x, v
    
    def cool_system(self, init_x, init_v, writer, save_freq,
                    end_temperature, cooling_steps,
                    temperature_units="kelvin",
                    time_units="picoseconds"):
        assert temperature_units == self.temperature_units
        assert time_units == self.time_units
        temp_decrease_per_step = (self.temperature - end_temperature) / cooling_steps
        x = init_x.to(self.u_model.device) 
        v = init_v.to(self.u_model.device)
        f_prev = self.u_model.get_force(x)
        f_prev = f_prev.detach()
        x = x.detach()

        for i in range(cooling_steps):
            if (i % save_freq) == 0:
                writer.write(x.cpu(), f_prev.cpu(), i // save_freq)
            x, v, f_prev = self.ovrvo_step(x, v, f_prev)
            self.update_temperature(self.temperature - temp_decrease_per_step,
                                    temperature_units=temperature_units)
        writer.close()
        return x, v

class Brownian:
    """
    Positions in units of Angstroms.
    """
    def __init__(
        self, 
        u_model, 
        masses,
        batch_size=1,
        dt=0.001, 
        friction=10., 
        temperature=300.0, 
        length_units="angstroms",
        energy_units="kilocalories_per_mole", 
        time_units="picoseconds",
        temperature_units="kelvin"
    ):
        self.u_model = u_model
        self.temperature = temperature
        self.friction = friction
        self.dt = dt

        masses = torch.tensor(masses)

        '''Masses are computed in boltz1.py and are initially in amu 
        (g/mol), which is equivalent to 
        1e-6 (kJ * s^2) / (m^2 * mol) (unit_mass_omm). 

        If self.energy_units is per-mole, then mass will be in units of
        self.energy_units * self.time_units ** 2 
        / self.length_units ** 2. This is currently not the case if 
        self.energy_units are per-particle. Thus, in this case, we 
        enforce consistency by performing an initial conversion of the 
        mass units. We scale masses so that they are in units of g, 
        which is equivalent to 1e-6 (kJ * s^2) / m^2, which is in units 
        of self.energy_units * self.time_units ** 2 
        / self.length_units ** 2, as desired.
        '''
        MASSES_CONVERSION_FACTOR = 1e-6
        if getattr(unit, energy_units).is_compatible(unit.kilojoule_per_mole):
            mass_unit = unit.kilojoule * unit.second ** 2 / (unit.meter ** 2 * unit.mole)
        else:
            masses /= unit.AVOGADRO_CONSTANT_NA * unit.mole # Scaling masses from g/mol to g.
            mass_unit = unit.kilojoule * unit.second ** 2 / (unit.meter ** 2)

        target_unit = (
            getattr(unit, energy_units) 
            * getattr(unit, time_units) ** 2 
            / getattr(unit, length_units) ** 2
        )
        conversion_factor = (1.0 * mass_unit).in_units_of(target_unit)._value
        masses_converted = masses * conversion_factor * MASSES_CONVERSION_FACTOR

        self.energy_units = energy_units
        self.length_units = length_units
        self.time_units = time_units
        self.temperature_units = temperature_units
        self.batch_size = batch_size # For reshaping init_x in integrate().

        # Beta (and kT) unit depends on if energy_units is per-mole or per-particle.
        # Beta will always have the same dimensions as inverse energy_units.
        if getattr(unit, energy_units).is_compatible(unit.kilojoule_per_mole):
            beta = (
                1 / (
                    temperature * getattr(unit, temperature_units) 
                    * unit.AVOGADRO_CONSTANT_NA 
                    * unit.BOLTZMANN_CONSTANT_kB
                ).value_in_unit(getattr(unit, self.energy_units))
            ) # 1 / (T * mol^-1 * J/T) = mol / J.
        else:
            beta = (
                1 / (
                    temperature * getattr(unit, temperature_units) 
                    * unit.BOLTZMANN_CONSTANT_kB
                ).value_in_unit(getattr(unit, self.energy_units))
            ) # 1 / (T * J/T) = 1 / J.
        self.beta = beta

        # Required by utils.py. 
        self.kT = 1 / beta
        self.masses = repeat(masses_converted, 'atoms -> (batch atoms) 3', batch=batch_size).to(u_model.device)

    def integrate(self, init_x, _, steps, writer, save_freq=100):
        """Calculate and return the final coordinates and velocities 
        after one checkpoint's worth of simulation steps using 
        EM-discretized Brownian equation. Saves every save_freq-th 
        step's simulation information using TrajWriter.

        Parameters
        ----------
        init_x : torch.tensor (batch * atoms, 3)
        steps : int
            The number of simulation timesteps (dt) to integrate for
            this current checkpoint.

        Returns
        -------
        x : torch.tensor (batch * atoms, 3)
        v : torch.tensor (batch * atoms, 3)
        """
        x = rearrange(init_x, "(batch atom) dim -> batch atom dim", batch=writer.batch_size).detach()
        masses = rearrange(self.masses, "(batch atom) dim -> batch atom dim", batch=writer.batch_size)
        per_atom_gm = masses * self.friction # Friction is in inverse time.
        diffusion_constants = self.kT / per_atom_gm # length^2 / time

        # Includes bias forces. If no bias is present, this is just the score.
        score_prev, _ = self.u_model.get_score_and_energy_kt(x, beta=self.beta)

        # Get the current time offset if it exists, otherwise start at 0
        time_offset = getattr(self, 'current_time_offset', 0.0)
        v = torch.zeros_like(init_x).cpu() # Velocities are zero in Brownian motion.

        for i in range(steps):
            if (i % save_freq) == 0:
                # Calculate simulation time in the integrator's time units
                simulation_time = time_offset + i * self.dt
                writer.write(x.cpu(), v, (score_prev / self.beta).cpu(), i // save_freq, simulation_time)
            
            # Updating coordinates.
            x += (self.dt * diffusion_constants * score_prev
                + torch.randn_like(x) * (2.0 * self.dt * diffusion_constants))
            score_prev, _ = self.u_model.get_score_and_energy_kt(x, beta=self.beta)
        writer.close()
        x = rearrange(x, "batch atom dim -> (batch atom) dim", batch=writer.batch_size)
        return x, v