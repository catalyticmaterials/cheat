import os
from fairchem.core.common.tutorial_utils import fairchem_main
from cheatools.fairchem import run_fairchem

# model identifier
identifier = 'AI2PR-dft-S2EF153M'

# path to model checkpoint
checkpoint_path = 'checkpoints/eq2_153M_ec4_allmd.pt'

# path to config file
config_path = 'configs/equiformer_v2_N@20_L@6_M@3_153M.yml'

# initialize fine-tuning
os.system(f'python {fairchem_main()} --mode train --identifier {identifier} --config-yml {config_path} --checkpoint {checkpoint_path}')

