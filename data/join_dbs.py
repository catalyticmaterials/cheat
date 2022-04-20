import sys
from ase.db import connect

filename = sys.argv[1]

new_db = connect(f'{filename}.db')

for row in connect(f'{filename}_slab.db').select():
	print(row.slabId)
	slab = connect(f'{filename}_slab.db').get_atoms(row.id)
	new_db.write(slab,slabId=row.slabId,ads='N/A',site='N/A',relaxed=1)

for row in connect(f'{filename}_ontop_OH.db').select():
	slab = connect(f'{filename}_ontop_OH.db').get_atoms(row.id)
	new_db.write(slab,slabId=row.slabId,ads='OH',site='ontop',relaxed=1)

for row in connect(f'{filename}_fcc_O.db').select():
	slab = connect(f'{filename}_fcc_O.db').get_atoms(row.id)
	new_db.write(slab,slabId=row.slabId,ads='O',site='fcc',relaxed=1)


