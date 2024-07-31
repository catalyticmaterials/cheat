import os
from fairchem.core.common.tutorial_utils import fairchem_main

# model identifier
identifier = 'AI2PR-dft-IS2RE31M'

# path to model checkpoint
checkpoint_path = 'checkpoints/eq2_31M_ec4_allmd.pt'

# path to config file
config_path = 'configs/equiformer_v2_N@8_L@4_M@2_31M.yml'

# initialize relaxations
os.system(f'python {fairchem_main()} --mode train --identifier {identifier} --config-yml {config_path} --checkpoint {checkpoint_path}')

