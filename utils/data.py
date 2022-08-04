# Made by Christian Møgelberg Clausen

# UPDATE 24/01 2022 - Adding adsorbates (CO)
# UPDATE 16/11 2021 - Restructuring, multiple sites, multiple adsorbates and dependency-submission

import numpy as np
import os
from ase.build import fcc100, fcc110, fcc111, bcc100, bcc110, bcc111, hcp0001, add_adsorbate
from ase.constraints import FixAtoms

lat_param_dict = {'Ag':4.2113,
                  'Al': 4.0674,
                  'Au': 4.2149,
                  'B': 2.8822,
                  'Be': 3.2022,
                  'Bi': 5.0699,
                  'Cd': 4.5795,
                  'Co': 3.5625,
                  'Cr': 3.6466,
                  'Cu': 3.6901,
                  'Fe': 3.6951,
                  'Ga': 4.2817,
                  'Ge': 4.304,
                  'Hf': 4.5321,
                  'In': 4.8536,
                  'Ir': 3.8841,
                  'Mg': 4.5673,
                  'Mn': 3.5371,
                  'Mo': 4.0204,
                  'Nb': 4.2378,
                  'Ni': 3.565,
                  'Os': 3.8645,
                  'Pb': 5.0942,
                  'Pd': 3.9814,
                  'Pt': 3.9936,
                  'Re': 3.9293,
                  'Rh': 3.8648,
                  'Ru': 3.8285,
                  'Sc': 4.6809,
                  'Si': 3.8935,
                  'Sn': 4.8612,
                  'Ta': 4.2504,
                  'Ti': 4.158,
                  'Tl': 5.0884,
                  'V': 3.8573,
                  'W': 4.0543,
                  'Y': 5.1289,
                  'Zn': 3.9871,
                  'Zr': 4.562}

mag_mom_dict = {'Co':1.7,
                'Fe':2.8,
                'Ni':0.7}

adsorbate_dict = {'O':"Atoms('O', ([0, 0, 0],))",
                  'H':"Atoms('H', ([0, 0, 0],))",
                  'N':"Atoms('N', ([0, 0, 0],))",
                  'C':"Atoms('C', ([0, 0, 0],))",
		     'CO':"Atoms('CO', ([0, 0, 0], [0, 0, 1.16]))",
                  'OH':"Atoms('OH', ([0, 0, 0], [0.65, 0.65, 0.40]))",
                  'N2_standing':"Atoms('NN', ([0, 0, 0], [0, 0, 1.13]))",
                  'N2_lying':"Atoms('NN', ([0, 0, 0], [1.13, 0, 0]))"}

def get_mag_mom(elements):
    mag_mom_dict = dict(np.loadtxt('magnetic_moments.csv', dtype='U10,f4', delimiter=','))
    mag_mom = [mag_mom_dict[metal] for metal in elements]
    return mag_mom

def write_hea_slab(facet, elements, composition, size, lattice = 'surface_adjusted', vacuum = 10, fix_bottom = 2):
    if facet not in ['fcc100','fcc110','fcc111','bcc100','bcc110','bcc111','hcp0001']:
        print("Please choose from the following facets: ['fcc100','fcc110','fcc111','bcc100','bcc110','bcc111','hcp0001']")
        raise NameError("Unsupported facet chosen.")

    lattice_parameters = [lat_param_dict[metal] for metal in elements]
    weighted_lat = np.sum(np.multiply(lattice_parameters,composition))
    atoms = globals()[facet]('Au', size=size, vacuum=vacuum, a=weighted_lat)
    rnd_symbols = np.random.choice(elements,np.product(size), p=composition)
    atoms.set_chemical_symbols(rnd_symbols)
    atoms.set_constraint(FixAtoms(indices=[atom.index for atom in atoms if atom.tag not in np.arange(size[2])[:-fix_bottom+1]]))

    if lattice == 'surface_adjusted':
        temp = []
        for symbol in [atom.symbol for atom in atoms if atom.tag == 1]:
            temp.append(np.array(lattice_parameters)[symbol == np.array(elements)][0])
        lat_scale = np.mean(temp)/weighted_lat
        atoms.set_cell([atoms.get_cell()[0]*lat_scale,atoms.get_cell()[1]*lat_scale,atoms.get_cell()[2]])

    if np.any(np.isin(list(mag_mom_dict.keys()),elements)):
        mag_moms = np.zeros(len(atoms))
        for j, symbol in enumerate(rnd_symbols):
            if symbol in list(mag_mom_dict.keys()):
                mag_moms[j] = mag_mom_dict[symbol]
            else: mag_moms[j] = 0.0
        atoms.set_initial_magnetic_moments(mag_moms)

    return atoms

def relax_slab_script(filename, slabId, distort_lim, max_force, ecut, kpts, xc):
    with open('py/' + filename + '_slab.py', 'w') as file:
        file.write("from gpaw import GPAW, PW\n" \
                   "from ase.io import Trajectory\n" \
                   "from ase import Atoms\n" \
                   "from ase.db import connect\n" \
                   "from ase.optimize import QuasiNewton\n" \
                   "import numpy as np\n" \
                   "from time import sleep\n" \
                   "\n")

        file.write("while True:\n" \
                   "    try:\n" \
                   f"        atoms = connect('../{filename[:-4]}preview.db').get_atoms(slabId={slabId})\n" \
                   "        break\n" \
                   "    except:\n" \
                   "        sleep(1)\n" \
                   "\n" \
                   f"calc = GPAW(mode=PW({ecut}), kpts={kpts}, xc='{xc}', txt='../txt/{filename}_slab.txt')\n" \
                   "atoms.set_calculator(calc)\n" \
                   f"dyn = QuasiNewton(atoms, trajectory='../traj/{filename}_slab.traj')\n" \
                   f"dyn.run(fmax = {max_force})\n" \
                   "atoms.get_potential_energy()\n" \
                   f"connect('../{filename[:-4]}slab.db').write(atoms, slabId={slabId})\n" \
                   "\n")

        if distort_lim != None:
            file.write(f"ur_atoms = Trajectory('../traj/{filename}_slab.traj')[0]\n"\
                        "tags = np.unique([atom.tag for atom in atoms])\n"\
                        "max_dis = []\n"\
                        "for slab in [ur_atoms, atoms]:\n"\
                        "   bot_z = np.array([atom.position[2] for atom in slab if atom.tag == tags[-1]])\n" \
                        "   top_z = np.array([atom.position[2] for atom in slab if atom.tag == tags[0]])\n" \
                        "   del_z = top_z - bot_z\n"\
                        "   max_dis.append(np.max(del_z))\n"\
                        "if max_dis[1] > distort_limit * max_dis[0]:\n"\
                        "   raise Exception('Relaxed slab distorted. Adsorbate calculations will not commence')\n")

        else:
            pass


def relax_ads_script(filename, slabId, adsId, facet, size, site, adsorbate, initial_bond_length, max_force, ecut, kpts, xc):
    ads_id_str = "+".join([str(Id) for Id in adsId])
    with open('py/' + filename + f'_{site}_{adsorbate}_{ads_id_str}.py', 'w') as file:
        file.write("from gpaw import GPAW, PW\n"\
                "from ase.io import Trajectory\n"\
                "from ase import Atoms\n"\
                "from ase.db import connect\n"\
                "from ase.build import add_adsorbate\n"\
                "from ase.optimize import QuasiNewton\n"\
                "import numpy as np\n"\
                "from time import sleep\n"\
                "\n")

        file.write(f"atoms = Trajectory('../traj/{filename}_slab.traj')[-1]\n"\
                   "\n"\
                   f"atoms_2x2 = atoms.repeat((2,2,1))\n")

        adsId_list = []
        for set in adsId:
            if isinstance(set, int):
                try:
                    adsId_list.append(get_site_ids(facet, site, size)[set])
                except IndexError:
                    print(f'Slab not large enough to support adsId {adsId}.')
                    os.system(f"(rm py/{filename}_{site}_{adsorbate}_{ads_id_str}.py)")
                    return

            if isinstance(set, list):
                temp = []
                for id in set:
                    try:
                        temp.append(get_site_ids(facet, site, size)[id][0])
                    except IndexError:
                        print(f'Slab not large enough to support adsIds {adsId}.')
                        os.system(f"(rm py/{filename}_{site}_{adsorbate}_{ads_id_str}.py)")
                        return
                adsId_list.append(temp)

        for set in adsId_list:
            file.write(f"positions = np.array([atom.position for atom in atoms_2x2 if atom.index in {set}])\n"\
                       f"x_pos = np.mean(positions[:,0])\n"\
                       f"y_pos = np.mean(positions[:,1])\n")

            file.write(f"ads_object = {adsorbate_dict[adsorbate]}\n"\
                       f"add_adsorbate(atoms,ads_object,{initial_bond_length},position=(x_pos,y_pos))\n")

        file.write(f"calc = GPAW(mode=PW({ecut}), kpts={kpts}, xc='{xc}', txt='../txt/{filename}_{site}_{adsorbate}_{ads_id_str}.txt')\n"\
                   "atoms.set_calculator(calc)\n"\
                   f"dyn = QuasiNewton(atoms, trajectory='../traj/{filename}_{site}_{adsorbate}_{ads_id_str}.traj')\n"\
                   f"dyn.run(fmax = {max_force})\n"\
                   "atoms.get_potential_energy()\n")
        if len(adsId) == 1:
            file.write(f"connect('../{filename[:-4]}{site}_{adsorbate}.db').write(atoms, slabId={slabId}, adsId={ads_id_str})\n")
        else:
            file.write(f"connect('../{filename[:-4]}{site}_{adsorbate}.db').write(atoms, slabId={slabId}, adsId='{ads_id_str}')\n")

def get_site_ids(facet, site, size):
    ads_id_sets = []

    if site == 'ontop':
        for id in np.arange(np.product(size))[-np.product(size[:2]):]:
            ads_id_sets.append([id])

    elif site == 'bridge':
        for i, id in enumerate(np.arange(np.product(size))[-np.product(size[:2]):]):
            if (i+1) % size[0] == 0:
                ads_id_sets.append([id,id+1-size[0]+2*np.product(size)])
            else:
                ads_id_sets.append([id, id + 1])

    elif site == 'hollow':
        if facet in ['bcc111','bcc110']:
            for i, id in enumerate(np.arange(np.product(size))[-np.product(size[:2]):]):
                if (i + 1) > size[0] * (size[1] - 1) and (i + 1) % size[0] == 0:
                    ads_id_sets.append(
                        [id, id + 1 - size[0] + 2 * np.product(size), id + np.product(size) - size[0] * (size[1] - 1)])

                elif (i + 1) > size[0] * (size[1] - 1):
                    ads_id_sets.append([id, id + 1, id + np.product(size) - size[0] * (size[1] - 1)])

                elif (i + 1) % size[0] == 0:
                    ads_id_sets.append([id, id + 1 - size[0] + 2 * np.product(size), id + size[0]])
                else:
                    ads_id_sets.append([id, id + 1, id + size[0]])

        if facet in ['fcc100','fcc110','bcc100']:
            for i, id in enumerate(np.arange(np.product(size))[-np.product(size[:2]):]):
                if (i + 1) > size[0] * (size[1] - 1) and (i + 1) % size[0] == 0:
                    ads_id_sets.append(
                        [id, id + 1 - size[0] + 2 * np.product(size), id + np.product(size) - size[0] * (size[1] - 1), id + 3 * np.product(size) - np.product(size[:2]) + 1])

                elif (i + 1) > size[0] * (size[1] - 1):
                    ads_id_sets.append([id, id + 1, id + np.product(size) - size[0] * (size[1] - 1), id + np.product(size) - size[0] * (size[1] - 1) + 1])

                elif (i + 1) % size[0] == 0:
                    ads_id_sets.append([id, id + 1 - size[0] + 2 * np.product(size), id + size[0], id + 2 * np.product(size) + 1])
                else:
                    ads_id_sets.append([id, id + 1, id + size[0], id + size[0] + 1])


    elif site  == 'shortbridge':
        if facet == 'fcc110':
            for i, id in enumerate(np.arange(np.product(size))[-np.product(size[:2]):]):
                if (i+1) > size[0]*(size[1]-1):
                    ads_id_sets.append([id, id - size[0] + np.product(size)])
                else:
                    ads_id_sets.append([id, id + size[0]])

    elif site == 'longbridge':
        if facet == 'fcc110':
            for i, id in enumerate(np.arange(np.product(size))[-np.product(size[:2]):]):
                if (i + 1) % size[0] == 0:
                    ads_id_sets.append([id, id + 1 - size[0] + 2 * np.product(size)])
                else:
                    ads_id_sets.append([id, id + 1])

    elif site == 'fcc':
        if facet in ['fcc111','hcp0001']:
            for i, id in enumerate(np.arange(np.product(size))[-np.product(size[:2]):]):
                if (i+1) > size[0]*(size[1]-1) and (i+1) % size[0] == 0:
                    ads_id_sets.append([id, id + 1 - size[0] + 2 * np.product(size), id + np.product(size) - size[0]*(size[1]-1)])

                elif (i+1) > size[0]*(size[1]-1):
                    ads_id_sets.append([id, id + 1, id + np.product(size) - size[0]*(size[1]-1)])

                elif (i+1) % size[0] == 0:
                    ads_id_sets.append([id, id + 1 - size[0] + 2 * np.product(size), id+size[0]])
                else:
                    ads_id_sets.append([id, id + 1, id+size[0]])

    elif site == 'hcp':
        if facet in ['fcc111','hcp0001']:
            for i, id in enumerate(np.arange(np.product(size))[-np.product(size[:2]):]):
                if (i + 1) > size[0] * (size[1] - 1) and (i + 1) % size[0] == 0:
                    ads_id_sets.append(
                        [id + 1 - size[0] + 2 * np.product(size), id + np.product(size) - size[0] * (size[1] - 1), id + 3 * np.product(size) - np.product(size[:2]) + 1])

                elif (i + 1) > size[0] * (size[1] - 1):
                    ads_id_sets.append([id + 1, id + np.product(size) - size[0] * (size[1] - 1), id + np.product(size) - size[0] * (size[1] - 1) + 1])

                elif (i + 1) % size[0] == 0:
                    ads_id_sets.append([id + 1 - size[0] + 2 * np.product(size), id + size[0], id + 1 + 2 * np.product(size)])

                else:
                    ads_id_sets.append([id + 1, id + size[0], id + size[0] + 1])

    return ads_id_sets

def SLURM_script(filename, partition, nodes, ntasks, ntasks_per_core, mem_per_cpu, constraint, nice, exclude, dependency):
    with open('sl/' + filename + '.sl', 'w') as f:
        f.write("#!/bin/bash\n"\
                "\n"\
                f"#SBATCH --job-name={filename}\n"\
                f"#SBATCH --partition={partition}\n" \
                f"#SBATCH --error='../err/{filename}.err'\n" \
                f"#SBATCH --output='../log/{filename}.log'\n" \
                f"#SBATCH --nodes={nodes}\n" \
                f"#SBATCH --ntasks={ntasks}\n" \
                f"#SBATCH --ntasks-per-core={ntasks_per_core}\n" \
                f"#SBATCH --mem-per-cpu={mem_per_cpu}\n" \
                f"#SBATCH --constraint={constraint}\n" \
                f"#SBATCH --nice={nice}\n")

        if exclude != None:
            f.write(f"#SBATCH --exclude={exclude}\n")

        if dependency != None:
            f.write(f"#SBATCH --dependency=afterok:{dependency}\n")

        f.write(f"srun gpaw-python ../py/{filename}.py")
