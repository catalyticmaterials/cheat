import pickle
import numpy as np
from torch_geometric.loader import DataLoader
from cheatools.lgnn import lGNN

# load test set
for s in ['test']:
    with open(f'graphs/{s}.graphs', 'rb') as input:
        globals()[f'{s}_graphs'] = pickle.load(input)

filename = 'lGNN'

# load trained state
with open(f'{filename}.state', 'rb') as input:
    regressor = lGNN(trained_state=pickle.load(input))

# predict on test set and save to results-file
for s in ['test']:
    test_loader = DataLoader(globals()[f'{s}_graphs'], batch_size=len(globals()[f'{s}_graphs']), drop_last=True, shuffle=False)

    _, pred, true, ads = regressor.test(test_loader, len(globals()[f'{s}_graphs']))

    results_dict = {'true':true,'pred':pred,'ads':ads}
    with open(f'results/{filename}_{s}.results', 'wb') as output:
        pickle.dump(results_dict, output)

