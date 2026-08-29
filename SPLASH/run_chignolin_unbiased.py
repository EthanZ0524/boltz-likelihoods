from openmm.app import *
from openmm import *
import openmm.unit as unit

pdb = PDBFile('/home/ethanz/data_boltz_likelihood/md/chignolin_folded.pdb') # Hydrogens already added.
forcefield = ForceField('amber19-all.xml', 'implicit/gbn2.xml')
system = forcefield.createSystem(
    pdb.topology, 
    nonbondedMethod=NoCutoff, 
    nonbondedCutoff=1 * unit.nanometer, 
    constraints=HBonds
)

# Minimization.
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
simulation.reporters.append(StateDataReporter('equilibrate.log', 1000, step=True, temperature=True, potentialEnergy=True, totalEnergy=True, speed=True))
simulation.step(10000)
positions = simulation.context.getState(getPositions=True).getPositions()
simulation.saveCheckpoint('unbiased_min_eq.chk')
print('Done equilibrating, saved checkpoint.')
# update the current context with changes in system
# simulation.context.reinitialize()

# Production run.
platform = Platform.getPlatformByName('CUDA')
integrator = LangevinIntegrator(
    temp_omm,
    1.0 / unit.picosecond,
    2.0 * unit.femtoseconds
)
simulation = Simulation(pdb.topology, system, integrator, platform)
simulation.context.setPositions(positions)
simulation.context.setVelocitiesToTemperature(temp_omm)

# Set simulation reporters.
simulation.reporters.append(DCDReporter('chignolin_unbiased.dcd', 5000))
simulation.reporters.append(StateDataReporter('chignolin_unbiased.out', 5000, step=True, 
    potentialEnergy=True, temperature=True, progress=True, remainingTime=True, 
    speed=True, totalSteps=1e8, separator='\t'))

# Off to the races!
simulation.step(1e8) # 2fs * 1e8 = 200 ns.