from gpaw import GPAW, PW
from ase.io import Trajectory
from ase import Atoms
from ase.db import connect
from ase.build import add_adsorbate
from ase.optimize import QuasiNewton
import numpy as np
from time import sleep

atoms = Trajectory('../traj/agirpdptru_0000_slab.traj')[-1]

atoms_2x2 = atoms.repeat((2,2,1))
positions = np.array([atom.position for atom in atoms_2x2 if atom.index in [41, 129, 44]])
x_pos = np.mean(positions[:,0])
y_pos = np.mean(positions[:,1])
ads_object = Atoms('O', ([0, 0, 0],))
add_adsorbate(atoms,ads_object,1.3,position=(x_pos,y_pos))
calc = GPAW(mode=PW(400), kpts=(4, 4, 1), xc='RPBE', txt='../txt/agirpdptru_0000_fcc_O_5.txt')
atoms.set_calculator(calc)
dyn = QuasiNewton(atoms, trajectory='../traj/agirpdptru_0000_fcc_O_5.traj')
dyn.run(fmax = 1.0)
atoms.get_potential_energy()
connect('../agirpdptru_fcc_O.db').write(atoms, slabId=0, adsId=5)
