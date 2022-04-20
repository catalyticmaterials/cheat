from gpaw import GPAW, PW
from ase.io import Trajectory
from ase import Atoms
from ase.db import connect
from ase.optimize import QuasiNewton
import numpy as np
from time import sleep

while True:
    try:
        atoms = connect('../agirpdptru_preview.db').get_atoms(slabId=1)
        break
    except:
        sleep(1)

calc = GPAW(mode=PW(400), kpts=(4, 4, 1), xc='RPBE', txt='../txt/agirpdptru_0001_slab.txt')
atoms.set_calculator(calc)
dyn = QuasiNewton(atoms, trajectory='../traj/agirpdptru_0001_slab.traj')
dyn.run(fmax = 1.0)
atoms.get_potential_energy()
connect('../agirpdptru_slab.db').write(atoms, slabId=1)

