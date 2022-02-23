from gpaw import GPAW, PW
from ase.io import Trajectory
from ase import Atoms
from ase.db import connect
from ase.optimize import QuasiNewton
import numpy as np
from time import sleep

while True:
    try:
        atoms = connect('../example_project_0002_preview.db').get_atoms(slabId=2)
        break
    except:
        sleep(1)

calc = GPAW(mode=PW(400), kpts=(4, 4, 1), xc='RPBE', txt='../txt/example_project_0002_slab.txt')
atoms.set_calculator(calc)
dyn = QuasiNewton(atoms, trajectory='../traj/example_project_0002_slab.traj')
dyn.run(fmax = 0.1)
atoms.get_potential_energy()
connect('../example_project_0002_slab.db').write(atoms, slabId=2)

