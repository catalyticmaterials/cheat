import numpy as np
from ase.build import fcc100, fcc110, fcc111, bcc100, bcc110, bcc111, hcp0001, add_adsorbate
from ase.constraints import FixAtoms
from .utils import get_lattice, get_magmom, get_ads

def make_slab(facet, composition, size, surf_adj_lat = True, vacuum = 10, fix_bottom = 2, skin=None, spin_polarized=False):
    """
    Generates a randomized slab with a specified facet and composition.
    -------
    The specified vacuum is added on top and below the slab and the specified number of bottom layers are fixed.
    If the lattice is set to 'surface_adjusted' the x,y dimensions of the cell will be adjusted to the average lattice constant of the surface atoms.
    If skin is specified, the surface layer will be overridden with the specified element not taking into account the composition.
    If spin_polarized is True, the magnetic moments of the atoms will be set according to the elements in the composition.
    
    Returns: 
    -------
    Atoms object
    """


    if facet not in ['fcc100','fcc110','fcc111','bcc100','bcc110','bcc111','hcp0001']:
        print("Please choose from the following facets: ['fcc100','fcc110','fcc111','bcc100','bcc110','bcc111','hcp0001']")
        raise NameError("Unsupported facet chosen.")
    
    # Vegards law determined lattice constant for the alloy composition
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
    if surf_adj_lat == True:
        lat_scale = np.mean([get_lattice(a.symbol) for a in atoms if a.tag == 1]) / weighted_lat
        cell = atoms.get_cell()
        atoms.set_cell([cell[0]*lat_scale,cell[1]*lat_scale,cell[2]], scale_atoms=True)

    # set initial magnetic moments if spin polarization is chosen 
    if spin_polarized:
        magmoms = [get_magmom(a.symbol) if a.symbol in ['Co','Fe','Ni'] else 0.0 for a in atoms]
        atoms.set_initial_magnetic_moments(magmoms)

    return atoms

def relax_slab(filename, slabId, fmax, distort_lim, gpaw_kwargs):
    """
    Generates a separate script for relaxing a slab.
    -------
    The script pulls the slab from the preview database and writes the relaxed slab to the slab database.
    distor_lim can be used to check if the slab has been distorted too much during relaxation. If so an exception is raised and adsorbate calculations will not commence.
    gpaw_kwargs are the keyword arguments in dict format for the GPAW calculator.
    """

    # open file
    with open('py/' + filename + '_slab.py', 'w') as file:
        # write imports
        file.write("from gpaw import GPAW, PW, Mixer, MixerDif, MixerSum, Davidson\n" \
                   "from ase.db import connect\n" \
                   "from ase.optimize import LBFGS\n" \
                   "from time import sleep\n" \
                   "\n")
        # write while loop to connect to preview database, set up GPAW calculator and run relaxation
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
        # check if slab has been distorted too much
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
    """
    Generates a separate script for adding an adsorbate to a slab and relaxing the system.
    -------
    The script pulls the slab from the slab database and writes the relaxed slab to the specified adsorbate database.
    facet, size, site, adsorbate and initial_bond_length are necessesary for enumeration of the binding sites and adding the adsorbate.
    Unsupported adsorbates can be added to the get_ads function in utils.py.
    adsId is the index of the binding site(s) to add the adsorbate to.
    arrayId is used for SLURM array job submission.
    gpaw_kwargs are the keyword arguments in dict format for the GPAW calculator.
    """

    filename_w_arrId = filename + f'_ads{arrayId}' # filename formatted with arrayId

    # open file
    with open(f"py/{filename_w_arrId}.py", 'w') as file:
        # write imports
        file.write("from gpaw import GPAW, PW, Mixer, MixerDif, MixerSum, Davidson\n"\
                "from ase.io import read\n"\
                "from ase.db import connect\n"\
                "from ase.optimize import LBFGS\n"\
                "from cheatools.dftsampling import add_ads\n"\
                "\n")

        # fetch relaxed slab from slab database
        file.write(f"atoms = read('../traj/{filename}_slab.traj',-1)\n"\
                   "\n"\
                   )
        
        # add adsorbate to slab taking into account the possibility of multiple binding sites
        if isinstance(adsId, list):
            file.write(f"for i in {adsId}:\n"\
                       f"    atoms = add_ads(atoms, '{facet}', {size}, '{site}', '{adsorbate}', {initial_bond_length}, i)\n"\
                      )
        elif isinstance(adsId, int):
            file.write(f"atoms = add_ads(atoms, '{facet}', {size}, '{site}', '{adsorbate}', {initial_bond_length}, {adsId})\n"\
                      )

        # set up GPAW calculator and run relaxation
        file.write(f"calc = GPAW({', '.join(f'{k}={v}' for k, v in gpaw_kwargs.items())}, txt='../txt/{filename_w_arrId}.txt')\n"\
                   "atoms.set_calculator(calc)\n"\
                   f"dyn = LBFGS(atoms, trajectory='../traj/{filename_w_arrId}.traj')\n"\
                   f"dyn.run(fmax = {fmax})\n"\
                   "atoms.get_potential_energy()\n")
        
        # write relaxed slab to adsorbate database with specified adsId(s)
        if isinstance(adsId, list):
            file.write(f"connect('../{filename[:-4]}{site}_{adsorbate}.db').write(atoms, slabId={slabId}, adsId='{'+'.join([str(i) for i in adsId])}', arrayId={arrayId})\n")
        elif isinstance(adsId, int):
            file.write(f"connect('../{filename[:-4]}{site}_{adsorbate}.db').write(atoms, slabId={slabId}, adsId={adsId}, arrayId={arrayId})\n")

def add_ads(atoms, facet, size, site, adsorbate, initial_bond_length, adsId):
    """
    Adds an adsorbate to a slab.
    -------
    adsId specifies the binding site to add the adsorbate to.
    facet, size, site, adsorbate and initial_bond_length are necessesary for enumeration of the binding sites and adding the adsorbate.

    Returns:
    -------
    Atoms object
    """

    atoms_2x2 = atoms.repeat((2,2,1)) # repeat slab to account for edge binding sites

    adsIds = get_site_ids(facet, site, size)[adsId] # get binding site indices

    # calculate average position of binding sites
    positions = np.array([atom.position for atom in atoms_2x2 if atom.index in adsIds])
    x_pos = np.mean(positions[:,0])
    y_pos = np.mean(positions[:,1])

    # add adsorbate to binding site
    ads_object = get_ads(adsorbate)
    add_adsorbate(atoms,ads_object,initial_bond_length,position=(x_pos,y_pos))

    return atoms

def get_site_ids(facet, site, size):
    """
    Enumerates the binding sites of a slab.
    -------
    Uses the positions of the surface atoms to determine the binding sites and thus takes into accound in lattice distortions.
    The slab must adher to the ASE id convention with ids starting from 0 and increasing along the x, then y, then z directions.
    
    Returns:
    -------
    List of lists with binding site indices
    """

    ads_id_sets = [] # initiate list of binding site indices

    # ontop sites
    if site == 'ontop':
        for id in np.arange(np.prod(size))[-np.prod(size[:2]):]:
            ads_id_sets.append([id])

    # horizontal bridge sites
    elif site == 'bridge':
        for i, id in enumerate(np.arange(np.prod(size))[-np.prod(size[:2]):]):
            if (i+1) % size[0] == 0:
                ads_id_sets.append([id,id+1-size[0]+2*np.prod(size)])
            else:
                ads_id_sets.append([id, id + 1])

    # hollow sites
    elif site == 'hollow':
        if facet in ['bcc111','bcc110']:
            for i, id in enumerate(np.arange(np.prod(size))[-np.prod(size[:2]):]):
                if (i + 1) > size[0] * (size[1] - 1) and (i + 1) % size[0] == 0:
                    ads_id_sets.append(
                        [id, id + 1 - size[0] + 2 * np.prod(size), id + np.prod(size) - size[0] * (size[1] - 1)])

                elif (i + 1) > size[0] * (size[1] - 1):
                    ads_id_sets.append([id, id + 1, id + np.prod(size) - size[0] * (size[1] - 1)])

                elif (i + 1) % size[0] == 0:
                    ads_id_sets.append([id, id + 1 - size[0] + 2 * np.prod(size), id + size[0]])
                else:
                    ads_id_sets.append([id, id + 1, id + size[0]])

        if facet in ['fcc100','fcc110','bcc100']:
            for i, id in enumerate(np.arange(np.prod(size))[-np.prod(size[:2]):]):
                if (i + 1) > size[0] * (size[1] - 1) and (i + 1) % size[0] == 0:
                    ads_id_sets.append(
                        [id, id + 1 - size[0] + 2 * np.prod(size), id + np.prod(size) - size[0] * (size[1] - 1), id + 3 * np.prod(size) - np.prod(size[:2]) + 1])

                elif (i + 1) > size[0] * (size[1] - 1):
                    ads_id_sets.append([id, id + 1, id + np.prod(size) - size[0] * (size[1] - 1), id + np.prod(size) - size[0] * (size[1] - 1) + 1])

                elif (i + 1) % size[0] == 0:
                    ads_id_sets.append([id, id + 1 - size[0] + 2 * np.prod(size), id + size[0], id + 2 * np.prod(size) + 1])
                else:
                    ads_id_sets.append([id, id + 1, id + size[0], id + size[0] + 1])

    # short bridge sites
    elif site  == 'shortbridge':
        if facet == 'fcc110':
            for i, id in enumerate(np.arange(np.prod(size))[-np.prod(size[:2]):]):
                if (i+1) > size[0]*(size[1]-1):
                    ads_id_sets.append([id, id - size[0] + np.prod(size)])
                else:
                    ads_id_sets.append([id, id + size[0]])

    # long bridge sites
    elif site == 'longbridge':
        if facet == 'fcc110':
            for i, id in enumerate(np.arange(np.prod(size))[-np.prod(size[:2]):]):
                if (i + 1) % size[0] == 0:
                    ads_id_sets.append([id, id + 1 - size[0] + 2 * np.prod(size)])
                else:
                    ads_id_sets.append([id, id + 1])

    # fcc sites
    elif site == 'fcc':
        if facet in ['fcc111','hcp0001']:
            for i, id in enumerate(np.arange(np.prod(size))[-np.prod(size[:2]):]):
                if (i+1) > size[0]*(size[1]-1) and (i+1) % size[0] == 0:
                    ads_id_sets.append([id, id + 1 - size[0] + 2 * np.prod(size), id + np.prod(size) - size[0]*(size[1]-1)])

                elif (i+1) > size[0]*(size[1]-1):
                    ads_id_sets.append([id, id + 1, id + np.prod(size) - size[0]*(size[1]-1)])

                elif (i+1) % size[0] == 0:
                    ads_id_sets.append([id, id + 1 - size[0] + 2 * np.prod(size), id+size[0]])
                else:
                    ads_id_sets.append([id, id + 1, id+size[0]])

    # hcp sites
    elif site == 'hcp':
        if facet in ['fcc111','hcp0001']:
            for i, id in enumerate(np.arange(np.prod(size))[-np.prod(size[:2]):]):
                if (i + 1) > size[0] * (size[1] - 1) and (i + 1) % size[0] == 0:
                    ads_id_sets.append(
                        [id + 1 - size[0] + 2 * np.prod(size), id + np.prod(size) - size[0] * (size[1] - 1), id + 3 * np.prod(size) - np.prod(size[:2]) + 1])

                elif (i + 1) > size[0] * (size[1] - 1):
                    ads_id_sets.append([id + 1, id + np.prod(size) - size[0] * (size[1] - 1), id + np.prod(size) - size[0] * (size[1] - 1) + 1])

                elif (i + 1) % size[0] == 0:
                    ads_id_sets.append([id + 1 - size[0] + 2 * np.prod(size), id + size[0], id + 1 + 2 * np.prod(size)])

                else:
                    ads_id_sets.append([id + 1, id + size[0], id + size[0] + 1])

    return ads_id_sets


def SLURM_script(filename, slurm_kwargs, dependency, array_len=None):
    """
    Writes submission sbatch script for SLURM. 
    -------
    !DISCLAIMER!
    This function is highly personalized and should be modified accordingly to fit your own HPC protocols.
    """

    with open('sl/' + filename + '.sl', 'w') as f:
        f.write("#!/bin/bash\n"\
                "\n"\
                f"#SBATCH --job-name={filename}\n")
        
        for k, v in slurm_kwargs.items():
             f.write(f"#SBATCH --{k}={v}\n")

        if dependency != None:
            f.write(f"#SBATCH --dependency=afterok:{dependency}\n")
        
        if array_len != None:
            f.write(f"#SBATCH --array=0-{array_len-1}\n" \
                    f"#SBATCH --error='../err/{filename}%a.err'\n" \
                    f"#SBATCH --output='../log/{filename}%a.log'\n")
        else:
            f.write(f"#SBATCH --error='../err/{filename}.err'\n" \
                    f"#SBATCH --output='../log/{filename}.log'\n")

        # NB! specific to own HPC user and conda environment -> change accordingly
        f.write(f"module purge\n" \
                 '. "/groups/kemi/clausen/miniconda3/etc/profile.d/conda.sh"\n' \
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

