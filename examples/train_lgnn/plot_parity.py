import sys
from ase.io.trajectory import Trajectory
import ase.io
import numpy as np
import pickle
from utils.plot import plot_parity_single, plot_parity_array
import matplotlib.pyplot as plt
from copy import deepcopy

result_file = sys.argv[1]

true_dict = {ads: [] for ads in ['O','OH']}

pred_dict = deepcopy(true_dict)

with open(f'{result_file}', 'rb') as input:
    results = pickle.load(input)

for i, ads in enumerate(results['ads']):
    true_dict[ads].append(results['true'][i])
    pred_dict[ads].append(results['pred'][i])
     
colors = ['firebrick','steelblue']

arr = zip(true_dict.values(),pred_dict.values())
s = r'LeanGNN IS2RE' 
fig = plot_parity_array(arr,s,colors,list(true_dict.keys()),[-0.75,2.25])
fig.savefig(f'parity/{result_file.split("/")[-1].split(".")[0]}.png')
plt.close()
    
