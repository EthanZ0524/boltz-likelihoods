from openmm import HarmonicBondForce, HarmonicAngleForce, PeriodicTorsionForce, NonbondedForce

force_keys = {
    HarmonicBondForce: "harmonic_bond",
    HarmonicAngleForce: "harmonic_angle",
    PeriodicTorsionForce: "periodic_torsion",
    NonbondedForce: "nonbonded",
}


def get_force_labels(system):
    return [force_keys.get(type(force_type), "unknown")
            for force_type in system.getForces()]


def get_harmonic_bonds_subset(system, indices_to_keep):
    """Gets the harmonic bonds for a subset of atoms
    Arguments:
        system: An OpenMM System object from which to extract the harmonic bonds
        indices_to_keep: A list of indices to keep
    Returns:
        A generator of harmonic bonds
    """
    force_labels = get_force_labels(system)
    force_index = force_labels.index("harmonic_bond")
    harmonic_bond_force = system.getForce(force_index)
    num_bonds = harmonic_bond_force.getNumBonds()
    for i in range(num_bonds):
        harmonic_bond = harmonic_bond_force.getBondParameters(i)
        if harmonic_bond[0] in indices_to_keep and harmonic_bond[1] in indices_to_keep:
            yield harmonic_bond


def get_harmonic_angles_subset(system, indices_to_keep):
    """Gets the harmonic angles for a subset of atoms
    Arguments:
        system: An OpenMM System object from which to extract the harmonic angles
        indices_to_keep: A list of indices to keep
    Returns:
        A generator of harmonic angle forces
    """
    force_labels = get_force_labels(system)
    force_index = force_labels.index("harmonic_angle")
    harmonic_angle_force = system.getForce(force_index)
    num_angles = harmonic_angle_force.getNumAngles()
    for i in range(num_angles):
        harmonic_angle = harmonic_angle_force.getAngleParameters(i)
        if harmonic_angle[0] in indices_to_keep and harmonic_angle[1] in indices_to_keep and harmonic_angle[2] in indices_to_keep:
            yield harmonic_angle


def get_periodic_torsion_subset(system, indices_to_keep):
    """Gets the periodic torsions for a subset of atoms
    Arguments:
        system: An OpenMM System object from which to extract the periodic torsions
        indices_to_keep: A list of indices to keep
    Returns:
        A generator of periodic torsion forces
    """
    force_labels = get_force_labels(system)
    force_index = force_labels.index("periodic_torsion")
    periodic_torsion_force = system.getForce(force_index)
    num_torsions = periodic_torsion_force.getNumTorsions()
    for i in range(num_torsions):
        periodic_torsion = periodic_torsion_force.getTorsionParameters(i)
        if periodic_torsion[0] in indices_to_keep and periodic_torsion[1] in indices_to_keep and periodic_torsion[2] in indices_to_keep and periodic_torsion[3] in indices_to_keep:
            yield periodic_torsion


def get_nonbonded_force_subset(system, indices_to_keep):
    """Gets the nonbonded forces for a subset of atoms
    Arguments:
        system: An OpenMM System object from which to extract the nonbonded forces
        indices_to_keep: A list of indices to keep
    Returns:
        A generator of nonbonded parameters
    """
    force_labels = get_force_labels(system)
    force_index = force_labels.index("nonbonded")
    nonbonded_force = system.getForce(force_index)
    num_particles = nonbonded_force.getNumParticles()
    for i in range(num_particles):
        if i in indices_to_keep:
            yield nonbonded_force.getParticleParameters(i)


def get_nonbonded_force_exception_subset(system, indices_to_keep):
    """Gets the nonbonded force exceptions for a subset of atoms
    Arguments:
        system: An OpenMM System object object from which to extract the nonbonded force exceptions
        indices_to_keep: A list of indices to keep
    Returns:
        A generator of nonbonded exception parameters
    """
    force_labels = get_force_labels(system)
    force_index = force_labels.index("nonbonded")
    nonbonded_force = system.getForce(force_index)
    num_exceptions = nonbonded_force.getNumExceptions()
    for i in range(num_exceptions):
        nonbonded_exception = nonbonded_force.getExceptionParameters(i)
        if nonbonded_exception[0] in indices_to_keep and nonbonded_exception[1] in indices_to_keep:
            yield nonbonded_exception
