#### Open Catalyst Project / FAIR Chemistry - S2EF

This folder contains an example of how to finetune a machine learning potential from [FAIR Chemistry](https://github.com/FAIR-Chem/fairchem) for "Structure to Relaxed Energy and Forces" (S2EF) on the provided High-Entropy Alloy dataset. This model can then be used as a surrogate DFT calculator to rapidly perform structure relaxations.

To run the contents of this folder, you should follow these [installation instructions](https://github.com/catalyticmaterials/fairchem) to set up a conda environment and subsequently install a stable version of the fairchem-core package. Next install *cheatools* from the [main folder](../../).

This example uses Lightning Memory-Mapped Databases (LMDBs) as sources for the training, validation and testing of the model. To create these run `dft2lmdbs.py` which transforms all images in the DFT trajectories to graphs data objects and saves them to LMDBs.

To avoid training a model from scratch we need a checkpoint file to initilize the pre-trained model. The checkpoint used in this example is the EquiformerV2-153M model trained on the OC20-dataset which can be fetched in the [checkpoints folder](checkpoints). After fetching the checkpoint file, you have the option of setting up a wandb profile to monitor the finetuning process (see the [config file](configs/equiformer_v2_N@20_L@6_M@3_153M.yml). The finetuning is initialized by running the *finetune.py* wrapper script. Be mindful that this should be done on a GPU supported machine. For convenience, an already finetuned checkpoint file can also be fetched in the checkpoints folder. 

Run `test.py` to obtain a parity plot of the test results.

Finally `run_relaxations.py` showcases writing initial structures to an ASE database and relaxing the structures using the finetuned machine learning potential.

