import sys, pickle
from cheatools.plot import plot_parity
from copy import deepcopy

# dictionaries to store true and predicted values for each adsorbate
true_dict = {ads: [] for ads in ['O','OH']}
pred_dict = deepcopy(true_dict)

# load results file
result_file = sys.argv[1]
with open(f'{result_file}', 'rb') as input:
    results = pickle.load(input)

# fetch true and predicted values for all adsorbates
for i, ads in enumerate(results['ads']):
    true_dict[ads].append(results['true'][i])
    pred_dict[ads].append(results['pred'][i])
     
# plot parity plot
colors = ['firebrick','steelblue']
arr = zip(true_dict.values(),pred_dict.values())
header = r'EquiformerV2-31M IS2RE' 

fig = plot_parity(true_dict, pred_dict, colors, header, [-0.75,2.25])
fig.savefig(f'parity/{result_file.split("/")[-1].split(".")[0]}.png')
