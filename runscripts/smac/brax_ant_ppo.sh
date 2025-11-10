#!/bin/bash


#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=2000M
#SBATCH --job-name=smac_brax_ant_ppo
#SBATCH --time 01:00:00                                 
#SBATCH --mail-type fail
#SBATCH --mail-user stamatios.chrysanthidis@rwth-aachen.de
#SBATCH --output smac_mf/log/smac_brax_ant_ppo_%A.out
#SBATCH --error smac_mf/log/smac_brax_ant_ppo_%A.err
#SBATCH --array 3-4%1

cd ..
source /rwthfs/rz/cluster/home/aq055502/projects/arlbench-smac-hyper/arlbench/.venv/bin/activate
python runscripts/run_arlbench.py -m --config-name=tune_smac_cost_aware_rf experiments=brax_ant_ppo cluster=claix_gpu_h100 smac_seed=$SLURM_ARRAY_TASK_ID

