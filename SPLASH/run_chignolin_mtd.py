from openmm.app import *
from openmm import *
import openmm.unit as unit
import openmm.app.metadynamics as mtd
import matplotlib.pyplot as plt 
from sys import stdout

pdb = PDBFile('/home/ethanz/data_boltz_likelihood/md/chignolin_folded.pdb') # Hydrogens already added.
forcefield = ForceField('amber19-all.xml', 'implicit/gbn2.xml')
system = forcefield.createSystem(
    pdb.topology, 
    nonbondedMethod=NoCutoff, 
    nonbondedCutoff=1 * unit.nanometer, 
    constraints=HBonds
)

# Preparing bias.
# First, prepare the custom CV force. 
ca_indices = [atom.index for atom in pdb.topology.atoms() if atom.name == "CA"]
i, j = ca_indices[0], ca_indices[-1]

cv_force = CustomCVForce("e2e_dist")
cv_force.addCollectiveVariable("e2e_dist", CustomBondForce("r"))
cv_force.getCollectiveVariable(0).addBond(i, j)
bias_variable = mtd.BiasVariable(
    force=cv_force,
    minValue=0.0,
    maxValue=2.0, 
    biasWidth=0.1,
    periodic=False
)

# Running simulation
# Minimization - removing steric clashes from hydrogens.
temp_omm = 300 * unit.kelvin
integrator = LangevinIntegrator(
    temp_omm,
    1.0 / unit.picosecond,
    2.0 * unit.femtoseconds
)
simulation = Simulation(pdb.topology, system, integrator)
simulation.context.setPositions(pdb.getPositions())
tolerance = 0.1*unit.kilojoules_per_mole/unit.angstroms
simulation.minimizeEnergy(tolerance=tolerance,maxIterations=10000)
print('Finished with energy minimization.')

# Equilibration.
simulation.context.setVelocitiesToTemperature(temp_omm) # Only set velocities after minimization. 
simulation.reporters.append(StateDataReporter('mtd_equilibrate.log', 1000, step=True, temperature=True, potentialEnergy=True, totalEnergy=True, speed=True))
simulation.step(10000)
positions = simulation.context.getState(getPositions=True).getPositions()
simulation.saveCheckpoint('mtd_min_eq.chk')
print('Done equilibrating, saved checkpoint.')
# update the current context with changes in system
simulation.context.reinitialize()

# Creating Metadynamics object.
# Requires a System and list of BiasViariables. 
integrator = LangevinIntegrator(
    temp_omm,
    1.0 / unit.picosecond,
    2.0 * unit.femtoseconds
)
BIAS_DIR = '/home/ethanz/boltz-likelihoods/SPLASH/chignolin_bias'
temp_omm = 300 * unit.kelvin
os.makedirs(BIAS_DIR, exist_ok=True)
meta = mtd.Metadynamics(
    system=system,
    variables=[bias_variable],
    temperature=temp_omm,
    biasFactor=5.0,
    height=1.0 * unit.kilojoule_per_mole,
    frequency=500,
    saveFrequency=500,
    biasDir=BIAS_DIR
)

simulation = Simulation(pdb.topology, system, integrator)
simulation.context.setPositions(positions)
simulation.context.setVelocitiesToTemperature(temp_omm)
simulation.step(100)
print("100 step sanity check done.")

# Set simulation reporters.
simulation.reporters.append(DCDReporter('chignolin_mtd.dcd', 5000))
simulation.reporters.append(StateDataReporter('chignolin_mtd.out', 5000, step=True, 
    potentialEnergy=True, temperature=True, progress=True, remainingTime=True, 
    speed=True, totalSteps=1e8, separator='\t'))

meta.step(simulation, 1e8) # 2fs * 1e8 = 200 ns.

# Saving information.
import numpy as np
gridWidth = bias_variable.gridWidth
free_energies = meta.getFreeEnergy()
np.save(
    '/home/ethanz/boltz-likelihoods/SPLASH/mtd_free_energies.npy', 
    np.array(free_energies)
)
print('Grid width: ', gridWidth)