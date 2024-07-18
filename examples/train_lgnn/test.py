import pickle
from torch_geometric.loader import DataLoader
from cheatools.lgnn import lGNN

filename = 'lGNN' # name of model

# load trained state
with open(f'{filename}.state', 'rb') as input:
    regressor = lGNN(trained_state=pickle.load(input))

# load and predict on test set, then save to results file
for s in ['test']:
    with open(f'graphs/{s}.graphs', 'rb') as input:
        test_set = pickle.load(input)

    test_loader = DataLoader(test_set, batch_size=len(test_set), drop_last=True, shuffle=False)

    pred, true, ads = regressor.test(test_loader, len(test_set))

    results_dict = {'true':true,'pred':pred,'ads':ads}
    with open(f'results/{filename}_{s}.results', 'wb') as output:
        pickle.dump(results_dict, output)

