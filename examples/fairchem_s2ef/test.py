from cheatools.fairchem import OCPbatchpredictor
from cheatools.plot import plot_parity
from fairchem.core.datasets import LmdbDataset

# load OCP batch predictor
checkpoint = f"checkpoints/AI2PR-dft-S2EF153M.pt"
model = OCPbatchpredictor(checkpoint_path=checkpoint, batch_size=6, cpu=True, seed=42)

# loop over multiple test sets if necessary
for s in ['test']:
    # load and predict on test set
    test_set = LmdbDataset({"src": f"lmdbs/{s}.lmdb"})
    pred = model.predict(test_set)
    
    # dictionaries to store true and predicted values for each adsorbate
    true_dict = {ads: [] for ads in ['O','OH']}
    pred_dict = {ads: [] for ads in ['O','OH']}

    # sort predictions according to adsorbate 
    for i, p in enumerate(pred):
        true_dict[test_set[i].ads].append(test_set[i].y)
        pred_dict[test_set[i].ads].append(p)

    # plot parity plot
    colors = ['firebrick','steelblue']
    arr = zip(true_dict.values(),pred_dict.values())
    header = r'EquiformerV2-153M S2EF'

    fig = plot_parity(true_dict, pred_dict, colors, header, [-0.75,2.25])
    fig.savefig(f'parity/{s}.png')
