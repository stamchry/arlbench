import os
import pandas as pd
import yaml
import shutil

def find_config_file(experiment_name, config_root):
    """
    Maps experiment folder name to config file.
    Example: ppo_CartPole-v1_ppo_cpu_algo -> cc_cartpole_ppo.yaml
    """
    exp_lower = experiment_name.lower()
    
    # Specific mapping for the example provided
    if "cartpole" in exp_lower and "ppo" in exp_lower:
        return os.path.join(config_root, "cc_cartpole_ppo.yaml")
        
    # Generic search in config folder
    if os.path.exists(config_root):
        for f in os.listdir(config_root):
            if f.endswith(".yaml"):
                # Check if parts of the filename match the experiment name
                # e.g. cc_acrobot_ppo.yaml matches ppo_Acrobot-v1...
                fname_no_ext = f.replace(".yaml", "")
                parts = fname_no_ext.split('_')
                
                match_count = 0
                for part in parts:
                    if part in ['cc']: continue
                    if part in exp_lower:
                        match_count += 1
                
                # If we matched enough parts (env and algo), assume it's the one
                if match_count >= 2:
                    return os.path.join(config_root, f)
                    
    return None

def process_rs_folder(folder_path, config_root):
    runhistory_path = os.path.join(folder_path, "runhistory.csv")
    incumbent_path = os.path.join(folder_path, "incumbent.csv")
    
    if not os.path.exists(runhistory_path) or not os.path.exists(incumbent_path):
        return

    # Infer experiment name from path
    # Path: .../results/rs/EXPERIMENT_NAME/SEED/ID
    try:
        norm_path = os.path.normpath(folder_path)
        path_parts = norm_path.split(os.sep)
        
        if 'rs' in path_parts:
            rs_idx = path_parts.index('rs')
            if len(path_parts) > rs_idx + 1:
                experiment_name = path_parts[rs_idx + 1]
            else:
                return
        else:
            return
    except ValueError:
        return

    config_file = find_config_file(experiment_name, config_root)
    if not config_file or not os.path.exists(config_file):
        print(f"Warning: Config file not found for {experiment_name} in {folder_path}")
        return

    try:
        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)
        running_time = config.get('default_hyperparameter', {}).get('running_time')
        if running_time is None:
            print(f"Warning: 'running_time' not found in {config_file}")
            return
    except Exception as e:
        print(f"Error reading config {config_file}: {e}")
        return

    # Read runhistory
    df_rh = pd.read_csv(runhistory_path)
    
    # Determine seeds from columns
    seed_cols = [c for c in df_rh.columns if c.startswith("performance_seed_")]
    num_seeds = len(seed_cols)
    if num_seeds == 0:
        print(f"Warning: No seed columns found in {runhistory_path}")
        return

    # Calculate budget limit
    # Formula: running_time * num_seeds * 50
    total_resource_budget = running_time * num_seeds * 50
    
    # Calculate cumulative usage using COST (time) instead of budget (steps)
    if 'cost' not in df_rh.columns:
        print(f"Warning: 'cost' column missing in {runhistory_path}")
        return
        
    # Assuming 'cost' is the average cost across seeds, total time for this config is cost * num_seeds
    df_rh['resource_usage'] = df_rh['cost'] * num_seeds
    df_rh['cumulative_resource'] = df_rh['resource_usage'].cumsum()
    
    # Filter runhistory
    # We want to keep all rows within budget, PLUS the first one that exceeds it.
    valid_mask = df_rh['cumulative_resource'] <= total_resource_budget
    valid_count = valid_mask.sum()
    
    if valid_count < len(df_rh):
        # We include all valid rows plus the first one that exceeds the budget
        df_rh_filtered = df_rh.iloc[:valid_count + 1].copy()
    else:
        df_rh_filtered = df_rh.copy()
    
    rows_removed = len(df_rh) - len(df_rh_filtered)
    
    if rows_removed > 0:
        print(f"Pruning {folder_path}: Removing {rows_removed} rows. Limit: {total_resource_budget}")
        
        # Get valid config_ids from the filtered runhistory
        valid_ids = set(df_rh_filtered['config_id'].unique())

        # Filter incumbent
        df_inc = pd.read_csv(incumbent_path)
        
        # Filter based on whether the config_id exists in the filtered runhistory
        if 'config_id' in df_inc.columns:
            df_inc_filtered = df_inc[df_inc['config_id'].isin(valid_ids)].copy()
        else:
            print(f"Warning: 'config_id' column missing in {incumbent_path}, skipping incumbent pruning.")
            df_inc_filtered = df_inc.copy()

        # Backup
        shutil.move(runhistory_path, os.path.join(folder_path, "runhistory_full.csv"))
        shutil.move(incumbent_path, os.path.join(folder_path, "incumbent_full.csv"))
        
        # Save
        df_rh_filtered.drop(columns=['resource_usage', 'cumulative_resource'], inplace=True)
        df_rh_filtered.to_csv(runhistory_path, index=False)
        df_inc_filtered.to_csv(incumbent_path, index=False)
    else:
        print(f"No pruning needed for {folder_path}")
        pass

def main():
    project_root = os.getcwd()
    results_rs_dir = os.path.join(project_root, "results/rs")
    config_root = os.path.join(project_root, "runscripts/configs/experiments")
    
    if not os.path.exists(results_rs_dir):
        print(f"Results directory not found: {results_rs_dir}")
        return

    for root, dirs, files in os.walk(results_rs_dir):
        if "runhistory.csv" in files:
            process_rs_folder(root, config_root)

if __name__ == "__main__":
    main()