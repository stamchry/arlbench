#!/bin/zsh

# USAGE: ./smac_claix.sh EXPERIMENT      CLUSTER      SEARCH_SPACE      CONFIG_NAME
# e.g.:  ./smac_claix.sh brax_ant_ppo claix_gpu_h100 ppo_gpu_hybrid tune_smac_cost_aware_rf

if [ "$#" -ne 4 ]; then
    echo "Illegal number of parameters. Usage: $0 EXPERIMENT CLUSTER SEARCH_SPACE CONFIG_NAME"
    exit 1
fi

EXPERIMENT=$1
CLUSTER=$2
SEARCH_SPACE=$3
CONFIG_NAME=$4
JOB_NAME="smac_${EXPERIMENT}_${SEARCH_SPACE}"
DIRECTORY="smac/${EXPERIMENT}/${SEARCH_SPACE}/${CONFIG_NAME}"

# Create a dedicated directory for this specific experiment run
mkdir -p "$DIRECTORY/log"

# Use a 'here document' (cat <<EOF) for better readability and maintenance
cat > "$DIRECTORY/submit.sh" <<EOF
#!/bin/zsh

#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=2000M
#SBATCH --job-name=${JOB_NAME}
#SBATCH -t 08:00:00
#SBATCH --mail-type fail,end
#SBATCH --mail-user stamatios.chrysanthidis@rwth-aachen.de
#SBATCH --output $DIRECTORY/log/%A.out
#SBATCH --error $DIRECTORY/log/%A.err
#SBATCH --array 1,3

# Change to the project directory
cd /home/aq055502/projects/arlbench-smac-hyper/arlbench

module purge
module load GCCcore/12.2.0
module load Python/3.10.8
source .venv12/bin/activate

# First, run with default parameters to get the cost
echo "Determining cost of default configuration..."
# Run the script, find the specific log line with the cost, and extract the number.
COST=\$(python runscripts/run_arlbench.py experiments=$EXPERIMENT cluster=$CLUSTER 2>&1 | grep "Returning objectives for Hypersweeper" | sed -n "s/.*'cost': \([0-9.]*\).*/\1/p")
echo "Default cost determined: \$COST"

# Now, run the multi-node SMAC optimization, passing the cost
python runscripts/run_arlbench.py -m \\
    --config-name=$CONFIG_NAME \\
    experiments=$EXPERIMENT \\
    cluster=$CLUSTER \\
    search_space=$SEARCH_SPACE \\
    smac_seed=\$SLURM_ARRAY_TASK_ID \\
    +default_hyperparameter.running_time=\$COST
EOF

echo "Generated submission script in $DIRECTORY/submit.sh"
chmod +x "$DIRECTORY/submit.sh"
sbatch --begin=now "$DIRECTORY/submit.sh"