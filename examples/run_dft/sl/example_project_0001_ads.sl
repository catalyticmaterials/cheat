#!/bin/bash

#SBATCH --job-name=example_project_0001_ads
#SBATCH --partition=katla_day
#SBATCH --nodes=1-1
#SBATCH --ntasks=24
#SBATCH --ntasks-per-core=2
#SBATCH --mem-per-cpu=2G
#SBATCH --constraint=[v1|v2|v3|v4|v5]
#SBATCH --nice=0
#SBATCH --dependency=afterok:45227116
#SBATCH --array=0-5
#SBATCH --error='../err/example_project_0001_ads%a.err'
#SBATCH --output='../log/example_project_0001_ads%a.log'
module purge
. "/groups/kemi/clausen/miniconda3/etc/profile.d/conda.sh"
conda activate gpaw22
expand_node () {
eval echo $(echo $1 | sed "s|\([[:digit:]]\{3\}\)-\([[:digit:]]\{3\}\)|{^A..^B}|g;s|\[|\{|g;s|\]|,\}|g") | sed "s/ node$//g;s/ /|/g"
}

v5_nodes=$(expand_node node[024-030])
used_nodes=$(expand_node $SLURM_NODELIST)
if [[ ! $used_nodes =~ \| || $used_nodes =~ $v5_nodes ]]; then
export OMPI_MCA_pml="^ucx"
export OMPI_MCA_osc="^ucx"
fi
if [[  $used_nodes =~ \| && $used_nodes =~ $v5_nodes ]]; then
export OMPI_MCA_btl_openib_rroce_enable=1
fi
mpirun gpaw python ../py/example_project_0001_ads$SLURM_ARRAY_TASK_ID.py