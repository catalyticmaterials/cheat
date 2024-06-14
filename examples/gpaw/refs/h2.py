from ase import Atoms
from gpaw import GPAW, PW, Mixer, Davidson
from ase.db import connect
from ase.optimize import QuasiNewton

db = connect('refs.db')
kpts = (1, 1, 1)
ecut = 400
row = db.get(relaxed=0, ads='H2')
atoms = db.get_atoms(row.id)
calc = GPAW(mode = PW(ecut),
			xc = 'RPBE',
			eigensolver=Davidson(3),
			kpts = kpts,
			txt = 'h2.txt')
atoms.set_calculator(calc)
dyn = QuasiNewton(atoms, trajectory='h2.traj')
dyn.run(fmax = 0.01)
atoms.get_potential_energy()
db.write(atoms, relaxed=1, ads=row.ads)
