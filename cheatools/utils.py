import numpy as np
from ase import Atoms

def below_fmax(atoms,threshold):
    """
    Check fmax threshold of atoms object 
    """
    f = atoms.get_forces()
    fmax = np.max(np.sqrt(np.sum(f**2,axis=1)))
    return fmax <= threshold

def get_lattice(symbol):
    """
    Get fcc lattice parameter of element
    -------
    Equation of state calculations are available here: https://sid.erda.dk/share_redirect/D7KKongrWv
    """
    return {"Ag": 4.2113,
            "Al": 4.0674,
            "Au": 4.2149,
            "B": 2.8822,
            "Be": 3.2022,
            "Bi": 5.0699,
            "Cd": 4.5795,
            "Co": 3.5625,
            "Cr": 3.6466,
            "Cu": 3.6901,
            "Fe": 3.6951,
            "Ga": 4.2817,
            "Ge": 4.304,
            "Hf": 4.5321,
            "In": 4.8536,
            "Ir": 3.8841,
            "Mg": 4.5673,
            "Mn": 3.5371,
            "Mo": 4.0204,
            "Nb": 4.2378,
            "Ni": 3.565,
            "Os": 3.8645,
            "Pb": 5.0942,
            "Pd": 3.9814,
            "Pt": 3.9936,
            "Re": 3.9293,
            "Rh": 3.8648,
            "Ru": 3.8285,
            "Sc": 4.6809,
            "Si": 3.8935,
            "Sn": 4.8612,
            "Ta": 4.2504,
            "Ti": 4.158,
            "Tl": 5.0884,
            "V": 3.8573,
            "W": 4.0543,
            "Y": 5.1289,
            "Zn": 3.9871,
            "Zr": 4.562
           }[symbol]

def get_magmom(symbol):
    """
    Get magnetic moment of Co, Fe, or Ni
    -------
    Taken as mean moment of equation of state calculations are available here: https://sid.erda.dk/share_redirect/D7KKongrWv 
    """
    return {"Co": 1.6856908166343516,
            "Fe": 2.65971865776883,
            "Ni": 0.6512631627385174
           }[symbol]

def get_ads(ads):
    """
    Get adsorbate geometry
    """
    return {'O':Atoms('O', ([0, 0, 0],)),
            'H':Atoms('H', ([0, 0, 0],)),
            'N':Atoms('N', ([0, 0, 0],)),
            'C':Atoms('C', ([0, 0, 0],)),
            'CH':Atoms('CH', ([0, 0, 0],[0,0,1.1])),
            'CH2':Atoms('CH2', ([0, 0, 0],[-0.90,0,0.63],[0.90,0,0.63])),
            'CH3':Atoms('CH3', ([0, 0, 0],[-0.90,0.52,0.36],[0.90,0.52,0.36],[0,-1.04,0.36])),
            'CO':Atoms('CO', ([0, 0, 0], [0, 0, 1.16])),
            'OH':Atoms('OH', ([0, 0, 0], [0.65, 0.65, 0.40])),
            'N2_standing':Atoms('NN', ([0, 0, 0], [0, 0, 1.13])),
            'N2_lying':Atoms('NN', ([0, 0, 0], [1.13, 0, 0])),
            'NO':Atoms('NO', ([0, 0, 0], [0, 0, 1.14]))
           }[ads]

def saferound(x, decimals=2):
    """Numpy implementation of saferound function from https://stackoverflow.com/a/74044227"""
    x = x * 10**decimals
    N = np.round(np.sum(x)).astype(int) # true sum
    y = x.astype(int)
    M = np.sum(y) # rounded sum
    K = N - M # difference
    z = y-x # difference between rounded and true values
    if K!=0: 
        idx = np.argpartition(z,K)[:K] # indices of the largest differences
        y[idx] += 1 # add 1 to the largest differences
    return y / float(10**decimals)
