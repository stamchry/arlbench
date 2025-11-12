#!/bin/zsh


#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=1000M
#SBATCH --job-name=smac_brax_ant_ppo
#SBATCH -t 01:00:00                                 
#SBATCH --mail-type fail,end
#SBATCH --mail-user stamatios.chrysanthidis@rwth-aachen.de
#SBATCH --output smac/log/smac_brax_ant_ppo_%A.out
#SBATCH --error smac/log/smac_brax_ant_ppo_%A.err
#SBATCH --array 1,3

source .venv3.11/bin/activate
python runscripts/run_arlbench.py -m --config-name=tune_smac_cost_aware_rf experiments=brax_ant_ppo cluster=claix_gpu_h100 smac_seed=$SLURM_ARRAY_TASK_ID +sb_zoo=brax_ant_ppo environment=brax_ant

