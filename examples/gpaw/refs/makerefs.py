from ase.db import connect
from ase.build import fcc111, add_adsorbate
from ase.visualize import view
from ase import Atoms
from ase.constraints import FixAtoms
import numpy as np
from copy import copy

db = connect('refs.db')

L = 20
bond_len = [0.95,1.15,0.75,1.15,1.2]
lables = ['H2O','CO','H2','N2','NO']
t = np.pi / 180 * 104.51

for i, b in enumerate(bond_len):
    if i == 0:
        pos = [(L/2+b,L/2,L/2), (L/2+b*np.cos(t),L/2+b*np.sin(t),L/2), (L/2,L/2,L/2)]
    else:
        pos = [(L/2,L/2,L/2),(L/2+b,L/2,L/2)]
    
    atoms = Atoms(lables[i],pos,cell=[L,L,L],pbc=[False,False,False])
    db.write(atoms,ads=lables[i],relaxed=0)
	
