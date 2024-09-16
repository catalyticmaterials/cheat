import pickle
from torch_geometric.loader import DataLoader
from cheatools.lgnn import lGNN
from cheatools.plot import plot_parity

filename = 'lGNN' # name of model

# load trained state
with open(f'{filename}.state', 'rb') as input:
    regressor = lGNN(trained_state=pickle.load(input))

# loop over multiple test sets if necessary
for s in ['test']:
    # load and predict on test set
    with open(f'graphs/{s}.graphs', 'rb') as input:
        test_set = pickle.load(input)
    test_loader = DataLoader(test_set, batch_size=len(test_set), drop_last=True, shuffle=False)
    pred, true, ads = regressor.test(test_loader, len(test_set))

    # dictionaries to store true and predicted values for each adsorbate
    true_dict = {ads: [] for ads in ['O','OH']}
    pred_dict = {ads: [] for ads in ['O','OH']}

    # sort predictions according to adsorbate
    for i, p in enumerate(pred):
        true_dict[ads[i]].append(true[i])
        pred_dict[ads[i]].append(pred[i])    

    # plot parity plot
    colors = ['firebrick','steelblue']
    arr = zip(true_dict.values(),pred_dict.values())
    header = r'LeanGNN IS2RE'

    fig = plot_parity(true_dict, pred_dict, colors, header, [-0.75,2.25])
    fig.savefig(f'parity/{filename}_{s}.png')

