from openmm import System, HarmonicBondForce, HarmonicAngleForce, PeriodicTorsionForce, NonbondedForce, Platform, CustomIntegrator
from openmm.app import AmberInpcrdFile, AmberPrmtopFile, Modeller, Simulation
from openmm.unit import femtoseconds
from .subset_utils import get_harmonic_bonds_subset, get_harmonic_angles_subset, get_periodic_torsion_subset, get_nonbonded_force_subset, get_nonbonded_force_exception_subset


class ProteinSubset:
    def __init__(self, filename, indices_to_keep,
                 prior_force_args):
        """Creates a Protein object that contains a subset of atoms from an initial Protein object
        Arguments:
            p: A Protein object from which to extract a subset system
            indices_to_keep: A list of sorted indices to keep
        """
        fg_prmtop = AmberPrmtopFile(f'{filename}.prmtop')
        fg_topology = fg_prmtop.topology

        inpcrd = AmberInpcrdFile(f'{filename}.crd')
        fg_positions = inpcrd.getPositions(asNumpy=True)

        fg_system = fg_prmtop.createSystem(implicitSolvent=None,
                                           constraints=None,
                                           nonbondedCutoff=None,
                                           rigidWater=True,
                                           hydrogenMass=None)
        if indices_to_keep is None:
            indices_to_keep = list(range(fg_topology.getNumAtoms()))
        modeller = Modeller(fg_topology,
                            fg_positions)
        modeller.delete([atom for atom in modeller.topology.atoms()
                         if atom.index not in indices_to_keep])

        topology = modeller.getTopology()
        positions = modeller.getPositions()

        system = self._init_system(fg_system=fg_system,
                                   topology=topology,
                                   indices_to_keep=indices_to_keep,
                                   add_harmonic_bond_force=prior_force_args["add_harmonic_bond_force"],
                                   add_harmonic_angle_force=prior_force_args["add_harmonic_angle_force"],
                                   add_periodic_torsion_force=prior_force_args["add_periodic_torsion_force"],
                                   add_nonbonded_force=prior_force_args["add_nonbonded_force"])

        integrator = CustomIntegrator(0.1 * femtoseconds)
        platform = Platform.getPlatformByName(
            prior_force_args["prior_force_device"])
        self.simulation = Simulation(topology, system, integrator, platform)

    def _init_system(self, fg_system, topology,
                     indices_to_keep,
                     add_harmonic_bond_force=False,
                     add_harmonic_angle_force=False,
                     add_periodic_torsion_force=False,
                     add_nonbonded_force=False):
        """ Initializes an OpenMM system 
        Arguments:
            p: A Protein object from which to extract a subset system
            topology: An OpenMM topology object
            indices_to_keep: A list of sorted indices to keep 
        Returns:
            system: An OpenMM System object corresponding to the subset of the original system
        """
        system = System()
        # system.setDefaultPeriodicBoxVectors(*box_vectors)
        for atom in topology.atoms():
            system.addParticle(atom.element.mass)

        if add_harmonic_bond_force:
            harmonic_bond_force = self._get_harmonic_bond_force(fg_system, 
                                                                indices_to_keep)
            harmonic_bond_force.setForceGroup(0)
            system.addForce(harmonic_bond_force)

        if add_harmonic_angle_force:
            harmonic_angle_force = self._get_harmonic_angle_force(fg_system, 
                                                                  indices_to_keep)
            harmonic_angle_force.setForceGroup(0)
            system.addForce(harmonic_angle_force)

        if add_nonbonded_force:
            nonbonded_force = self._get_nonbonded_force(fg_system, 
                                                        indices_to_keep)
            nonbonded_force.setForceGroup(0)
            system.addForce(nonbonded_force)

        if add_periodic_torsion_force:
            periodic_torsion_force = self._get_periodic_torsion_force(fg_system, 
                                                                      indices_to_keep)
            periodic_torsion_force.setForceGroup(0)
            system.addForce(periodic_torsion_force)

        assert not system.usesPeriodicBoundaryConditions()
        return system

    def _get_harmonic_bond_force(self, fg_system, indices_to_keep):
        """Gets the harmonic bond force from a Protein object
        Arguments:
            fg_system: An OpenMM System object from which to extract a subset system
            indices_to_keep: A list of sorted indices to keep
        Returns:
            harmonic_bond_force: An OpenMM HarmonicBondForce object corresponding to the subset of the original system
        """
        harmonic_bond_force = HarmonicBondForce()
        for p_harmonic_bond in get_harmonic_bonds_subset(fg_system, indices_to_keep):
            p_atom_index_1, p_atom_index_2, length, k = p_harmonic_bond
            atom_index_1 = indices_to_keep.index(p_atom_index_1)
            atom_index_2 = indices_to_keep.index(p_atom_index_2)
            harmonic_bond = (atom_index_1, atom_index_2, length, k)
            harmonic_bond_force.addBond(*harmonic_bond)
        return harmonic_bond_force

    def _get_harmonic_angle_force(self, fg_system, indices_to_keep):
        """Gets the harmonic angle force from a Protein object
        Arguments:
            fg_system: An OpenMM System object from which to extract a subset system
            indices_to_keep: A list of sorted indices to keep
        Returns:
            harmonic_angle_force: An OpenMM HarmonicAngleForce object corresponding to the subset of the original system
        """
        harmonic_angle_force = HarmonicAngleForce()
        for p_harmonic_angle in get_harmonic_angles_subset(fg_system, indices_to_keep):
            p_atom_index_1, p_atom_index_2, p_atom_index_3, angle, k = p_harmonic_angle
            atom_index_1 = indices_to_keep.index(p_atom_index_1)
            atom_index_2 = indices_to_keep.index(p_atom_index_2)
            atom_index_3 = indices_to_keep.index(p_atom_index_3)
            harmonic_angle = (atom_index_1, atom_index_2,
                              atom_index_3, angle, k)
            harmonic_angle_force.addAngle(*harmonic_angle)
        return harmonic_angle_force

    def _get_periodic_torsion_force(self, fg_system, indices_to_keep):
        """Gets the periodic torsion force from a Protein object
        Arguments:
            fg_system: An OpenMM System  object from which to extract a subset system
            indices_to_keep: A list of sorted indices to keep
        Returns:
            periodic_torsion_force: An OpenMM PeriodicTorsionForce object corresponding to the subset of the original system
        """
        periodic_torsion_force = PeriodicTorsionForce()
        for p_periodic_torsion in get_periodic_torsion_subset(fg_system, indices_to_keep):
            p_atom_index_1, p_atom_index_2, p_atom_index_3, p_atom_index_4, periodicity, phase, k = p_periodic_torsion
            atom_index_1 = indices_to_keep.index(p_atom_index_1)
            atom_index_2 = indices_to_keep.index(p_atom_index_2)
            atom_index_3 = indices_to_keep.index(p_atom_index_3)
            atom_index_4 = indices_to_keep.index(p_atom_index_4)
            periodic_torsion = (atom_index_1, atom_index_2,
                                atom_index_3, atom_index_4, periodicity, phase, k)
            periodic_torsion_force.addTorsion(*periodic_torsion)
        return periodic_torsion_force

    def _get_nonbonded_force(self, fg_system, indices_to_keep):
        """Gets the nonbonded force from a Protein object
        Arguments:
            fg_system: An OpenMM System object from which to extract a subset system
            indices_to_keep: A list of sorted indices to keep
        Returns:
            nonbonded_force: An OpenMM NonbondedForce object corresponding to the subset of the original system
        """
        nonbonded_force = NonbondedForce()
        for p_nonbonded in get_nonbonded_force_subset(fg_system, indices_to_keep):
            p_charge, p_sigma, p_epsilon = p_nonbonded
            charge, sigma, epsilon = p_charge, p_sigma, p_epsilon
            charge = 0 * charge
            nonbonded_force.addParticle(charge, sigma, epsilon)

        for p_nonbonded_exception in get_nonbonded_force_exception_subset(fg_system, indices_to_keep):
            p_atom_index_1, p_atom_index_2, chargeProd, sigma, epsilon = p_nonbonded_exception
            atom_index_1 = indices_to_keep.index(p_atom_index_1)
            atom_index_2 = indices_to_keep.index(p_atom_index_2)
            chargeProd = 0 * chargeProd
            nonbonded_force.addException(
                atom_index_1, atom_index_2, chargeProd, sigma, epsilon)
        return nonbonded_force

    def get_forces(self, positions):
        """Gets the forces on the subset of atoms
        Arguments:
            positions: A list of positions
        Returns:
            forces: A list of forces on the subset of atoms
        """

        return self.simulation.context.getState(getForces=True).getForces()
