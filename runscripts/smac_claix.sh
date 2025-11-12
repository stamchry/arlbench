#!/bin/zsh

# USAGE run_rs.sh EXPERIMENT      CLUSTER 
# USAGE run_rs.sh cc_cartpole_dqn local  

directory="smac"

mkdir -p "$directory/log"

echo "#!/bin/zsh

#SBATCH --job-name=smac_${1}
#SBATCH -t 01:00:00                                                
#SBATCH --mail-type fail,end
#SBATCH --mail-user stamatios.chrysanthidis@rwth-aachen.de
#SBATCH --output $directory/log/smac_${1}_%A.out
#SBATCH --error $directory/log/smac_${1}_%A.err

source .venv3.11/bin/activate
python runscripts/run_arlbench.py -m --config-name=tune_smac_cost_aware_rf experiments=$1 cluster=$2 smac_seed=1
" > $directory/${1}.sh
echo "Submitting $directory for $1"
chmod +x $directory/${1}.sh
sbatch --begin=now $directory/${1}.sh
