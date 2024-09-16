"""
This code is originally sourced from https://github.com/ulissigroup/catlas and the Open Catalyst Project (OCP). 

Source code relating to OCP/FAIR-chem is licensed under the MIT license found in the
LICENSE file in https://github.com/FAIR-Chem/fairchem/tree/main with copyright (c) Meta, Inc. and its affiliates.
"""

import copy, logging, torch, ase.build, os, subprocess
import numpy as np
from torch.utils.data import Dataset
from tqdm import tqdm
from typing import Dict, Optional, NamedTuple
from numpy.typing import ArrayLike
from .dftsampling import add_ads
from .graphtools import ase2ocp_tags
from copy import deepcopy
from fairchem.core.common import distutils
from fairchem.core.common.registry import registry
from fairchem.core.common.utils import load_config, setup_imports, update_config
from fairchem.core.preprocessing import AtomsToGraphs

class OCPtemplater():
    def __init__(self,facet,adsorbates,sites,n_layers=5):
        self.template_dict = {}
        height = {'ontop':2.0,'bridge':1.8,'fcc':1.3,'hcp':1.5}
        atoms = ase.build.fcc111('Au', size=(3,3,5), vacuum=10, a=3.9)
        a2g = AtomsToGraphs()
        for ads, site in zip(adsorbates,sites):
            ads_id = 3 if site == 'hcp' else 4
            temp_atoms = add_ads(copy.deepcopy(atoms), 'fcc111', (3,3,5), site, ads, height[site], ads_id)
            temp_atoms = ase2ocp_tags(temp_atoms)
            data_object = a2g.convert_all([temp_atoms], disable_tqdm=True)[0]
            
            self.template_dict[(ads,site)] = data_object

    def fill_template(self,symbols,adsorbate,site):
        cell = deepcopy(self.template_dict[(adsorbate,site)])
        cell.atomic_numbers[:len(symbols)] = torch.tensor([ase.data.atomic_numbers[s] for s in symbols])
        return cell

class DatasetMetadata(NamedTuple):
    natoms: ArrayLike | None = None

class GraphsListDataset(Dataset):
    """
    Make a list of graphs to feed into OCP dataloader

    Extends:
        torch.utils.data.Dataset: a torch Dataset
    """

    def __init__(self, graphs_list):
        self.graphs_list = graphs_list
        self._metadata = DatasetMetadata([g.natoms for g in graphs_list])  
    
    def __len__(self):
        return len(self.graphs_list)

    def __getitem__(self, idx):
        graph = self.graphs_list[idx]
        return graph

    def metadata_hasattr(self, attr) -> bool:
        if self._metadata is None:
            return False
        return hasattr(self._metadata, attr)

    def get_metadata(self, attr, idx):
        if self._metadata is not None:
            metadata_attr = getattr(self._metadata, attr)
            if isinstance(idx, list):
                return [metadata_attr[_idx] for _idx in idx]
            return metadata_attr[idx]
        return None

class OCPbatchpredictor():
    def __init__(
        self,
        config_yml: Optional[str] = None,
        checkpoint_path: Optional[str] = None,
        batch_size: Optional[int] = 1,
        trainer: Optional[str] = None,
        cpu: bool = True,
        seed: Optional[int] = None,
    ) -> None:
        """
        Batch prediction class for fairchem IS2RE models

        Args:
            config_yml (str):
                Path to yaml config or could be a dictionary.
            checkpoint_path (str):
                Path to trained checkpoint.
            cpu (bool):
                Whether to load and run the model on CPU. Set `False` for GPU.
        """
        setup_imports()

        # Either the config path or the checkpoint path needs to be provided
        assert config_yml or checkpoint_path is not None

        checkpoint = None
        if config_yml is not None:
            if isinstance(config_yml, str):
                config, duplicates_warning, duplicates_error = load_config(
                    config_yml
                )
                if len(duplicates_warning) > 0:
                    logging.warning(
                        f"Overwritten config parameters from included configs "
                        f"(non-included parameters take precedence): {duplicates_warning}"
                    )
                if len(duplicates_error) > 0:
                    raise ValueError(
                        f"Conflicting (duplicate) parameters in simultaneously "
                        f"included configs: {duplicates_error}"
                    )
            else:
                config = config_yml

            # Only keeps the train data that might have normalizer values
            if isinstance(config["dataset"], list):
                config["dataset"] = config["dataset"][0]
            elif isinstance(config["dataset"], dict):
                config["dataset"] = config["dataset"].get("train", None)
        else:
            # Loads the config from the checkpoint directly (always on CPU).
            checkpoint = torch.load(
                checkpoint_path, map_location=torch.device("cpu")
            )
            config = checkpoint["config"]

        if trainer is not None:
            config["trainer"] = trainer
        else:
            config["trainer"] = config.get("trainer", "ocp")

        if "model_attributes" in config:
            config["model_attributes"]["name"] = config.pop("model")
            config["model"] = config["model_attributes"]

        # for checkpoints with relaxation datasets defined, remove to avoid
        # unnecesarily trying to load that dataset
        if "relax_dataset" in config["task"]:
            del config["task"]["relax_dataset"]

        # Calculate the edge indices on the fly
        config["model"]["otf_graph"] = True

        # Save config so obj can be transported over network (pkl)
        config = update_config(config)
        self.config = copy.deepcopy(config)
        self.config["checkpoint"] = checkpoint_path
        del config["dataset"]["src"]
        
        self.trainer = registry.get_trainer_class(config["trainer"])(
            task=config.get("task", {}),
            model=config["model"],
            dataset=[config["dataset"]],
            outputs=config["outputs"],
            loss_functions=config["loss_functions"],
            evaluation_metrics=config["evaluation_metrics"],
            optimizer=config["optim"],
            identifier="",
            slurm=config.get("slurm", {}),
            local_rank=config.get("local_rank", 0),
            is_debug=config.get("is_debug", True),
            cpu=cpu,
            amp=config.get("amp", False),
        )
        
        if checkpoint_path is not None:
            self.load_checkpoint(
                checkpoint_path=checkpoint_path, checkpoint=checkpoint
            )

        seed = seed if seed is not None else self.trainer.config["cmd"]["seed"]
        if seed is None:
            logging.warning(
                "No seed has been set in modelcheckpoint or OCPCalculator! Results may not be reproducible on re-run"
            )
        else:
            self.trainer.set_seed(seed)

        self.batch_size = batch_size

    def make_dataloader(self, graphs_list):
        """
        Make the dataloader used to feed graphs into the OCP model.

        Args:
            graphs_list (Iterable[torch_geometric.data.Data]): structures to run predictions on.

        Returns:
            torch.utils.data.DataLoader: an object that feeds data into pytorch models.
        """
        # Make a dataset
        graphs_list_dataset = GraphsListDataset(graphs_list)
        
        # Make a loader
        data_loader = self.trainer.get_dataloader(
            graphs_list_dataset,
            self.trainer.get_sampler(
                graphs_list_dataset, self.batch_size, shuffle=False
            ),
        )

        return data_loader

    def load_checkpoint(
        self, checkpoint_path: str, checkpoint: Dict = {}
    ) -> None:
        """
        Load existing trained model

        Args:
            checkpoint_path: string
                Path to trained model
        """
        try:
            self.trainer.load_checkpoint(checkpoint_path, checkpoint)
        except NotImplementedError:
            logging.warning("Unable to load checkpoint!")

    def predict(self, graphs_list, tqdm_bool=True):
        """
        Run direct energy predictions on a list of graphs. Predict the relaxed
        energy without running ML relaxations.

        Args:
            graphs_list (Iterable[torch_geometric.data.Data]): structures to run
                inference on.

        Returns:
            Iterable[float]: predicted energies of the input structures.
        """
        data_loader = self.make_dataloader(graphs_list)
        predictions = np.array([])
        rank = distutils.get_rank()
        for i, batch in tqdm(
            enumerate(data_loader),
            total=len(data_loader),
            position=rank,
            desc="device {}".format(rank),
            disable= not tqdm_bool,
        ):
             
            # Batch inference
            p = self.trainer.predict(batch, per_image=False, disable_tqdm=True)
            predictions = np.append(predictions,p['energy'].cpu().detach().numpy())
        
        return predictions
        
