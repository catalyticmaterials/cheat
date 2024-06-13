from gpaw import GPAW, PW, Mixer, MixerDif, MixerSum, Davidson
from ase.db import connect
from ase.optimize import LBFGS
from time import sleep

while True:
    try:
        atoms = connect('../example_project_preview.db').get_atoms(slabId=1)
        break
    except:
        sleep(1)

calc = GPAW(mode=PW(400), xc='RPBE', kpts=(4, 4, 1), eigensolver=Davidson(3), mixer=MixerDif(beta=0.05, nmaxold=5, weight=50.0), parallel={'augment_grids': True, 'sl_auto': True}, txt='../txt/example_project_0001_slab.txt')
atoms.set_calculator(calc)
dyn = LBFGS(atoms, trajectory='../traj/example_project_0001_slab.traj')
dyn.run(fmax = 0.1)
atoms.get_potential_energy()
connect('../example_project_slab.db').write(atoms, slabId=1)

