import pickle
from torch_geometric.loader import DataLoader
from cheatools.fairchem import OCPbatchpredictor
from fairchem.core.datasets import LmdbDataset

# load test set
for s in ['test']:
    globals()[f'{s}_lmdb'] = LmdbDataset({"src": f"lmdbs/{s}.lmdb"})

# load OCP batch predictor
checkpoint = f"checkpoints/AI2PR-dft-IS2RE31M.pt"
model = OCPbatchpredictor(checkpoint_path=checkpoint, batch_size=16, cpu=False, seed=42)

# predict on test set and save to results file
for s in ['test']:
    pred = model.predict(globals()[f'{s}_lmdb'])
    true = [d.y_relaxed for d in globals()[f'{s}_lmdb']]
    ads = [d.ads for d in globals()[f'{s}_lmdb']]

    results_dict = {'true':true,'pred':pred,'ads':ads}
    with open(f'results/{s}.results', 'wb') as output:
        pickle.dump(results_dict, output)

