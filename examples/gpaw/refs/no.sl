#!/bin/bash

#SBATCH --job-name=no
#SBATCH --partition=katla_short
#SBATCH --nodes=1-1
#SBATCH --ntasks=16
#SBATCH --ntasks-per-core=2
#SBATCH --mem-per-cpu=2G
#SBATCH --constraint=[v1|v2|v3|v4|v5]
#SBATCH --error='no.err'
#SBATCH --output='no.log'
module purge
. "/groups/kemi/clausen/miniconda3/etc/profile.d/conda.sh"
conda activate gpaw22
mpirun --mca btl_openib_rroce_enable 1 gpaw python no.py
