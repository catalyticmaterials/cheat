#### Open Catalyst Project / FAIR Chemistry - IS2RE

This folder contains an example of how to finetune a machine learning potential from [FAIR Chemistry](https://github.com/FAIR-Chem/fairchem) to perform direct adsorption energy inference or "Initial Structure to Relaxed Energy" (IS2RE) on the provided High-Entropy Alloy dataset. Additionally, it showcases the implementation of the model in the surrogate surface simulation.

To run the contents of this folder, you should follow these [installation instructions](https://github.com/catalyticmaterials/fairchem) to set up a conda environment and subsequently install a stable version of the fairchem-core package. Next install *cheatools* from the [main folder](../../).

This example uses Lightning Memory-Mapped Databases (LMDBs) as sources for the training, validation and testing of the inference model. To create these run `dft2lmdbs.py` which reduces the relaxed atomic structures to template structures as described in [Clausen et al. *J. Phys. Chem. C* 2024](https://doi.org/10.1021/acs.jpcc.4c01704) and save them to LMDBs. 

To avoid training a model from scratch we need a checkpoint file to initilize the pre-trained model. The checkpoint used in this example is the EquiformerV2-31M model trained on the OC20-dataset which can be fetched in the [checkpoints folder](checkpoints). After fetching the checkpoint file, you have the option of setting up a wandb profile to monitor the finetuning process (see the [config file](configs/equiformer_v2_N@8_L@4_M@2_31M.yml)). The finetuning is initialized by running the `finetune.py` wrapper script.

Be mindful that this should be done on a GPU supported machine. For convenience, an already finetuned checkpoint file can also be fetched in the [checkpoints folder](checkpoints). Run `test.py` to obtain a parity plot of the test results.

Finally `simulate_surface.py` showcases the surrogate surface simulation identical to the one found in [surface_simulation](../surface_simulation) but employs the finetuned EquiformerV2 model to infer the adsorption energies of the surface.

