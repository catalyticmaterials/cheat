import os, json
import numpy as np
from ase import Atoms
from ase.build import fcc100, fcc110, fcc111, bcc100, bcc110, bcc111, hcp0001, add_adsorbate
from ase.constraints import FixAtoms
from .utils import get_lattice, get_magmom, get_ads

def make_slab(facet, composition, size, lattice = 'surface_adjusted', vacuum = 10, fix_bottom = 2, skin=None, spin_polarized=False):
    if facet not in ['fcc100','fcc110','fcc111','bcc100','bcc110','bcc111','hcp0001']:
        print("Please choose from the following facets: ['fcc100','fcc110','fcc111','bcc100','bcc110','bcc111','hcp0001']")
        raise NameError("Unsupported facet chosen.")
    
    # Vegards law determined lattice constant for the alloy composition
    #lat_params = [lat_dict[e] for e in composition.keys()]
    weighted_lat = np.sum([get_lattice(e) * f for e, f in composition.items()])

    # initiate atoms object and randomize symbols
    atoms = globals()[facet]('Au', size=size, vacuum=vacuum, a=weighted_lat)    
    rnd_symbols = np.random.choice(list(composition.keys()), np.prod(size), p=list(composition.values()))
    atoms.set_chemical_symbols(rnd_symbols)
    
    # fix bottom layers
    atoms.set_constraint(FixAtoms(indices=[atom.index for atom in atoms if atom.tag not in np.arange(size[2])[:-fix_bottom+1]]))
    
    # replace surface layer with specified skin element
    if skin != None:
        for j in [a.index for a in atoms if a.tag == 1]:
            atoms[j].symbol = skin

    # adjust x,y dimension of cell if surface adjusted lattice is chosen
    if lattice == 'surface_adjusted':
        temp = []
        lat_scale = np.mean([get_lattice(a.symbol) for a in atoms if a.tag == 1]) / weighted_lat
        cell = atoms.get_cell()
        atoms.set_cell([cell[0]*lat_scale,cell[1]*lat_scale,cell[2]], scale_atoms=True)

    # set initial magnetic moments if spin polarization is chosen 
    if spin_polarized:
        magmoms = [get_magmom(a.symbol) if a.symbol in ['Co','Fe','Ni'] else 0.0 for a in atoms]
        atoms.set_initial_magnetic_moments(magmoms)

    return atoms

def relax_slab(filename, slabId, fmax, distort_lim, gpaw_kwargs):
    with open('py/' + filename + '_slab.py', 'w') as file:
        file.write("from gpaw import GPAW, PW, Mixer, MixerDif, MixerSum, Davidson\n" \
                   "from ase.db import connect\n" \
                   "from ase.optimize import LBFGS\n" \
                   "from time import sleep\n" \
                   "\n")

        file.write("while True:\n" \
                   "    try:\n" \
                   f"        atoms = connect('../{filename[:-4]}preview.db').get_atoms(slabId={slabId})\n" \
                   "        break\n" \
                   "    except:\n" \
                   "        sleep(1)\n" \
                   "\n" \
                   f"calc = GPAW({', '.join(f'{k}={v}' for k, v in gpaw_kwargs.items())}, txt='../txt/{filename}_slab.txt')\n" \
                   "atoms.set_calculator(calc)\n" \
                   f"dyn = LBFGS(atoms, trajectory='../traj/{filename}_slab.traj')\n" \
                   f"dyn.run(fmax = {fmax})\n" \
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
                        f"if max_dis[1] > {distort_lim} * max_dis[0]:\n"\
                        "   raise Exception('Relaxed slab distorted. Adsorbate calculations will not commence')\n")


def relax_ads(filename, slabId, adsId, facet, size, site, adsorbate, initial_bond_length, arrayId, fmax, gpaw_kwargs):
    
    filename_w_arrId = filename + f'_ads{arrayId}'
    
    with open(f"py/{filename_w_arrId}.py", 'w') as file:
        file.write("from gpaw import GPAW, PW, Mixer, MixerDif, MixerSum, Davidson\n"\
                "from ase.io import read\n"\
                "from ase.db import connect\n"\
                "from ase.optimize import LBFGS\n"\
                "from cheatools.dftsampling import add_ads\n"\
                "\n")

        file.write(f"atoms = read('../traj/{filename}_slab.traj',-1)\n"\
                   "\n"\
                   )
        
        if isinstance(adsId, list):
            file.write(f"for i in {adsId}:\n"\
                       f"    atoms = add_ads(atoms, '{facet}', {size}, '{site}', '{adsorbate}', {initial_bond_length}, i)\n"\
                      )
        elif isinstance(adsId, int):
            file.write(f"atoms = add_ads(atoms, '{facet}', {size}, '{site}', '{adsorbate}', {initial_bond_length}, {adsId})\n"\
                      )

        file.write(f"calc = GPAW({', '.join(f'{k}={v}' for k, v in gpaw_kwargs.items())}, txt='../txt/{filename_w_arrId}.txt')\n"\
                   "atoms.set_calculator(calc)\n"\
                   f"dyn = LBFGS(atoms, trajectory='../traj/{filename_w_arrId}.traj')\n"\
                   f"dyn.run(fmax = {fmax})\n"\
                   "atoms.get_potential_energy()\n")
        
        if isinstance(adsId, list):
            file.write(f"connect('../{filename[:-4]}{site}_{adsorbate}.db').write(atoms, slabId={slabId}, adsId='{'+'.join([str(i) for i in adsId])}', arrayId={arrayId})\n")
        elif isinstance(adsId, int):
            file.write(f"connect('../{filename[:-4]}{site}_{adsorbate}.db').write(atoms, slabId={slabId}, adsId={adsId}, arrayId={arrayId})\n")

def add_ads(atoms, facet, size, site, adsorbate, initial_bond_length, adsId):

    atoms_2x2 = atoms.repeat((2,2,1))

    adsIds = get_site_ids(facet, site, size)[adsId]

    positions = np.array([atom.position for atom in atoms_2x2 if atom.index in adsIds])
    x_pos = np.mean(positions[:,0])
    y_pos = np.mean(positions[:,1])

    ads_object = get_ads(adsorbate)
    
    add_adsorbate(atoms,ads_object,initial_bond_length,position=(x_pos,y_pos))

    return atoms

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


def SLURM_script(filename, partition, nodes, ntasks, ntasks_per_core, mem_per_cpu, constraint, nice, exclude, dependency, array_len=None):
    """
    Writes submission sbatch script for SLURM. 
    -------
    !DISCLAIMER!
    This function is highly personalized and should be modified accordingly to fit your own HPC protocols.
    """
    with open('sl/' + filename + '.sl', 'w') as f:
        f.write("#!/bin/bash\n"\
                "\n"\
                f"#SBATCH --job-name={filename}\n"\
                f"#SBATCH --partition={partition}\n" \
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
        
        if array_len != None:
            f.write(f"#SBATCH --array=0-{array_len-1}\n" \
                    f"#SBATCH --error='../err/{filename}%a.err'\n" \
                    f"#SBATCH --output='../log/{filename}%a.log'\n")
        else:
            f.write(f"#SBATCH --error='../err/{filename}.err'\n" \
                    f"#SBATCH --output='../log/{filename}.log'\n")

        f.write(f"module purge\n"\
                 '. "/groups/kemi/clausen/miniconda3/etc/profile.d/conda.sh"\n'
                 "conda activate gpaw22\n" \
                 "expand_node () {\n" \
                 'eval echo $(echo $1 | sed "s|\([[:digit:]]\{3\}\)-\([[:digit:]]\{3\}\)|{^A..^B}|g;s|\[|\{|g;s|\]|,\}|g") | sed "s/ node$//g;s/ /|/g"\n' \
                 "}\n" \
                 "\n" \
                 "v5_nodes=$(expand_node node[024-030])\n" \
                 "used_nodes=$(expand_node $SLURM_NODELIST)\n" \
                 "if [[ ! $used_nodes =~ \| || $used_nodes =~ $v5_nodes ]]; then\n" \
                 'export OMPI_MCA_pml="^ucx"\n' \
                 'export OMPI_MCA_osc="^ucx"\n' \
                 "fi\n" \
                 "if [[  $used_nodes =~ \| && $used_nodes =~ $v5_nodes ]]; then\n" \
                 "export OMPI_MCA_btl_openib_rroce_enable=1\n" \
                 "fi\n" \
               )

        if array_len != None:
            f.write(f"mpirun gpaw python ../py/{filename}$SLURM_ARRAY_TASK_ID.py")

        else:
            f.write(f"mpirun gpaw python ../py/{filename}.py")

