import pickle
from torch_geometric.loader import DataLoader
from cheatools.fairchem import OCPbatchpredictor
from fairchem.core.datasets import LmdbDataset

# load OCP batch predictor
checkpoint = f"checkpoints/AI2PR-dft-S2EF153M.pt"
model = OCPbatchpredictor(checkpoint_path=checkpoint, batch_size=6, cpu=False, seed=42)

# load and predict on test set, then save to results file
for s in ['test']:
    test_set = LmdbDataset({"src": f"lmdbs/{s}.lmdb"})
    pred = model.predict(test_set)
    true = [d.y_relaxed for d in test_set]
    ads = [d.ads for d in test_set]

    results_dict = {'true':true,'pred':pred,'ads':ads}
    with open(f'results/{s}.results', 'wb') as output:
        pickle.dump(results_dict, output)

