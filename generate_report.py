import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from pathlib import Path
from matplotlib.backends.backend_pdf import PdfPages

# ---------------------------------------------------------
# 1. CONFIGURATION
# ---------------------------------------------------------
BASE_DIR = Path("results/evaluation") 
DOWNLOAD_DIR = Path("results") 

METHOD_ORDER = ['rs', 'smac', 'smac_mf', 'smac_mf_hand_crafted', 'smac_hand_crafted', 'smac_rf', 'smac_gp', 'smac_linear_crafted']
METHOD_LABELS = {
    'rs': 'Random Search',
    'smac': 'SMAC',
    'smac_mf': 'SMAC Multifidelity',
    'smac_hand_crafted': 'Cost-Aware SMAC\n(Hand-crafted)',
    'smac_rf': 'Cost-Aware SMAC\n(Random Forest)',
    'smac_gp': 'Cost-Aware SMAC\n(Gaussian Process)',
    'smac_linear_crafted': 'Cost-Aware SMAC\n(Linear-crafted)',
    'smac_mf_hand_crafted': 'Cost-Aware SMAC MF\n(Hand-crafted)',
}

SEARCH_SPACE_LABELS = {
    'ppo_cpu_algo': 'Non cost-aware parameters',
    'ppo_gpu_algo': 'Non cost-aware parameters',
    'ppo_cpu_hybrid': 'Cost aware + non cost aware parameters',
    'ppo_gpu_hybrid': 'Cost aware + non cost aware parameters'
}

EXPERIMENT_MAPPING = {
    'ppo_LunarLander-v2': 'box2d_lunar_lander_ppo',
    'ppo_ant': 'brax_ant_ppo',
    'ppo_CartPole-v1': 'cc_cartpole_ppo',
}

# Define a consistent color palette for all methods
_method_colors = sns.color_palette("Dark2", len(METHOD_LABELS))
METHOD_PALETTE = dict(zip(METHOD_LABELS.values(), _method_colors))

sns.set_style("whitegrid")

# ---------------------------------------------------------
# 2. DATA LOADING
# ---------------------------------------------------------

def load_results_data(base_paths):
    """Loads final evaluation results (boxplots)."""
    print("Loading results.csv data...")
    all_data = []
    
    paths = base_paths if isinstance(base_paths, list) else [base_paths]
    
    for path in paths:
        if not os.path.exists(path):
            continue
        for root, dirs, files in os.walk(path):
            if 'results.csv' in files:
                file_path = os.path.join(root, 'results.csv')
                try:
                    df = pd.read_csv(file_path)
                    if 'search_space' not in df.columns:
                        df['search_space'] = os.path.basename(root)
                    all_data.append(df)
                except Exception as e:
                    print(f"Error loading {file_path}: {e}")
    
    if not all_data: return pd.DataFrame()
    
    df = pd.concat(all_data, ignore_index=True)
    if 'experiment' in df.columns:
        df['experiment'] = df['experiment'].replace(EXPERIMENT_MAPPING)
    
    df['optimization_method_label'] = df['optimization_method'].map(METHOD_LABELS).fillna(df['optimization_method'])
    df['search_space_label'] = df['search_space'].map(SEARCH_SPACE_LABELS).fillna(df['search_space'])
    return df

def load_runhistory_data(base_dir):
    """Loads runhistory for trajectory plots."""
    print("Loading runhistory.csv data...")
    all_data = []
    
    for csv_path in base_dir.rglob("runhistory.csv"):
        try:
            parts = csv_path.parts
            if "results" in parts:
                idx = parts.index("results")
                if len(parts) < idx + 5: continue
                method = parts[idx + 1]
                folder_name = parts[idx + 2]
                opt_id = parts[idx + 3]
                seed = parts[idx + 4]
            else:
                continue

            search_space = next((key for key in SEARCH_SPACE_LABELS if key in folder_name), None)
            experiment = folder_name.replace(f"_{search_space}", "") if search_space else folder_name
            experiment = EXPERIMENT_MAPPING.get(experiment, experiment)

            df = pd.read_csv(csv_path)
            
            if 'config_id' in df.columns:
                df = df.sort_values(by='config_id').reset_index(drop=True)
            
            # --- PRE-CALCULATION FOR SPEED ---
            # We calculate these here so we don't have to do it inside the plotting loop
            if 'performance' in df.columns:
                # Assumes higher is better
                df['incumbent_performance'] = df['performance'].cummax()
            else:
                continue

            if 'cost' in df.columns:
                df['total_wallclock_time'] = df['cost'].cumsum()
            else:
                df['total_wallclock_time'] = np.arange(1, len(df) + 1)
            
            df['method'] = method
            df['search_space'] = search_space
            df['experiment'] = experiment
            df['opt_id'] = opt_id
            df['seed'] = seed
            
            all_data.append(df)
        except Exception as e:
            print(f"Error processing {csv_path}: {e}")

    if not all_data: return pd.DataFrame()
    
    df = pd.concat(all_data, ignore_index=True)
    df['optimization_method_label'] = df['method'].map(METHOD_LABELS).fillna(df['method'])
    df['search_space_label'] = df['search_space'].map(SEARCH_SPACE_LABELS).fillna(df['search_space'])
    return df

# ---------------------------------------------------------
# 3. PLOTTING FUNCTIONS
# ---------------------------------------------------------

def plot_incumbent_trajectories(pdf, df_runhistory, env):
    """
    Plots the incumbent performance over Wallclock Time (aggregated over seeds).
    Uses vectorized resampling (step-function) to align disjoint time series.
    """
    PLOT_RESOLUTION = 250  # Number of points on the X-axis grid
    
    env_data = df_runhistory[df_runhistory['experiment'] == env]
    if env_data.empty: return

    unique_spaces = env_data['search_space'].unique()

    for space_code in unique_spaces:
        # Filter for this specific plot
        space_subset = env_data[env_data['search_space'] == space_code]
        space_label = SEARCH_SPACE_LABELS.get(space_code, space_code)
        
        # 1. Prepare Data Buckets for Resampling
        # We need to group by (Method, Seed) to get individual trajectories
        # Using a dictionary is faster than repeated pandas filtering
        trajectories = []
        
        # Group by unique run identifiers
        groups = space_subset.groupby(['method', 'opt_id', 'seed'])
        
        global_min_time = float('inf')
        global_max_time = 0

        for (method, opt, seed), group in groups:
            # Get arrays
            times = group['total_wallclock_time'].values
            values = group['incumbent_performance'].values
            
            if len(times) == 0: continue
            
            trajectories.append({
                'times': times,
                'values': values,
                'method': method
            })
            
            # Track global time bounds for the grid
            if times[-1] > global_max_time: global_max_time = times[-1]
            if times[0] < global_min_time: global_min_time = times[0]

        if not trajectories: continue

        # 2. Create Common Time Grid (Log-spaced looks better for cost/time)
        # Avoid log(0) issues by using a small epsilon if min_time is 0
        start_time = max(global_min_time, 1e-2) 
        # Create a grid that is logarithmic to capture early details and late convergence equally well
        grid_x = np.geomspace(start_time, global_max_time, PLOT_RESOLUTION)
        
        resampled_rows = []

        # 3. Vectorized Resampling
        for traj in trajectories:
            r_times = traj['times']
            r_vals = traj['values']
            method_name = traj['method']

            # Find indices where grid_x falls into r_times
            # side='right' + minus 1 gives the "last valid time" index (Step Function)
            indices = np.searchsorted(r_times, grid_x, side='right') - 1
            
            # Handle points before the first recorded time (index -1)
            # Strategy: Forward fill. If grid point is before first run, it's effectively "undefined" 
            # or the initial value. We clamp to 0.
            valid_indices = np.maximum(indices, 0)
            
            # Map values
            resampled_y = r_vals[valid_indices]
            
            # Optional: Set values before start time to NaN if strict correctness is needed
            # But usually extending the first value backwards is preferred for visualization
            # resampled_y[indices == -1] = np.nan 

            # Create data rows
            # Using list comprehension for speed
            resampled_rows.extend([
                {
                    'wallclock_time': t,
                    'performance': v,
                    'optimization_method_label': METHOD_LABELS.get(method_name, method_name)
                }
                for t, v in zip(grid_x, resampled_y)
            ])

        # 4. Plotting
        df_plot = pd.DataFrame(resampled_rows)
        
        plt.figure(figsize=(12, 7))
        
        # Identify available methods to sort the legend correctly
        current_methods = df_plot['optimization_method_label'].unique()
        plot_order = [METHOD_LABELS[m] for m in METHOD_ORDER if METHOD_LABELS[m] in current_methods]

        sns.lineplot(
            data=df_plot,
            x='wallclock_time',
            y='performance',
            hue='optimization_method_label',
            hue_order=plot_order,
            palette=METHOD_PALETTE,
            linewidth=2
        )

        plt.xscale('log') # Log scale for Time
        plt.title(f"Incumbent Performance vs. Wallclock Time: {env}\n({space_label})", fontsize=14)
        plt.xlabel("Cumulative Wallclock Time (seconds) [Log Scale]", fontsize=12)
        plt.ylabel("Incumbent Performance", fontsize=12)
        plt.legend(title="Optimization Method", loc='lower right')
        plt.grid(True, which="both", ls="-", alpha=0.5)
        
        plt.tight_layout()
        pdf.savefig()
        plt.close()
        print(f"   > Plotted incumbent trajectory for {space_code}")

def plot_boxplots(pdf, df_results, env):
    """Generates Mean and Median boxplots."""
    env_data = df_results[df_results['experiment'] == env]
    if env_data.empty:
        return

    # Aggregations
    mean_agg = env_data.groupby(['search_space_label', 'optimization_method_label', 'opt_id'])['performance'].mean().reset_index()
    median_agg = env_data.groupby(['search_space_label', 'optimization_method_label', 'opt_id'])['performance'].median().reset_index()

    # Order for plotting
    plot_order = [METHOD_LABELS[m] for m in METHOD_ORDER if METHOD_LABELS[m] in env_data['optimization_method_label'].unique()]

    # 1. Mean Plot
    plt.figure(figsize=(14, 8))
    sns.boxplot(
        data=mean_agg, x='optimization_method_label', y='performance', hue='search_space_label',
        palette="Set2", showfliers=True, order=plot_order
    )
    plt.title(f'Mean Performance Comparison: {env}', fontsize=15)
    plt.ylabel('Mean Performance', fontsize=12)
    plt.xlabel('Optimization Method', fontsize=12)
    plt.legend(title='Search Space')
    plt.xticks(rotation=0, ha='center', fontsize=11, wrap=True)
    plt.tight_layout()
    pdf.savefig()  # Save to PDF
    plt.close()

    # 2. Median Plot
    plt.figure(figsize=(14, 8))
    sns.boxplot(
        data=median_agg, x='optimization_method_label', y='performance', hue='search_space_label',
        palette="Set2", showfliers=True, order=plot_order
    )
    plt.title(f'Median Performance Comparison: {env}', fontsize=15)
    plt.ylabel('Median Performance', fontsize=12)
    plt.xlabel('Optimization Method', fontsize=12)
    plt.legend(title='Search Space')
    plt.xticks(rotation=0, ha='center', fontsize=11, wrap=True)
    plt.tight_layout()
    pdf.savefig()
    plt.close()

    print(f"   > Plotted boxplots for {env}")

def plot_costs(pdf, df_runhistory, env):
    """Comparison of individual trial costs (runtimes)."""
    env_data = df_runhistory[df_runhistory['experiment'] == env]
    if env_data.empty: return
    unique_spaces = env_data['search_space'].dropna().unique()

    for space_code in unique_spaces:
        subset = env_data[env_data['search_space'] == space_code].copy()
        # Ensure we use labels for consistent coloring
        subset['method_label'] = subset['method'].map(METHOD_LABELS).fillna(subset['method'])
        space_label = SEARCH_SPACE_LABELS.get(space_code, space_code)
        
        # Separate all methods containing 'mf'
        mf_data = subset[subset['method'].str.contains('mf')]
        std_data = subset[~subset['method'].str.contains('mf')]

        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        if not std_data.empty:
            sns.lineplot(data=std_data, x='config_id', y='cost', hue='method_label', palette=METHOD_PALETTE, ax=axes[0])
            axes[0].set_title("Standard Methods Cost per Trial")
            axes[0].set_ylabel("Cost (s)")
        
        if not mf_data.empty:
            sns.lineplot(data=mf_data, x='config_id', y='cost', hue='method_label', palette=METHOD_PALETTE, ax=axes[1])
            axes[1].set_title("Multifidelity Methods Cost per Trial")
        
        plt.suptitle(f"Trial Runtime Comparison: {env}\n({space_label})", fontsize=16)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        pdf.savefig()
        plt.close()
        print(f"   > Plotted cost comparison for {space_code}")

def plot_number_of_configurations_boxplots(pdf, df_runhistory, env):
    """Generates boxplots comparing the number of evaluated configurations."""
    env_data = df_runhistory[df_runhistory['experiment'] == env]
    if env_data.empty:
        return

    # Count configurations per run (identified by method, search_space, opt_id, seed)
    counts = env_data.groupby(['search_space_label', 'optimization_method_label', 'opt_id', 'seed']).size().reset_index(name='num_configs')

    # Order for plotting
    current_methods = counts['optimization_method_label'].unique()
    plot_order = [METHOD_LABELS[m] for m in METHOD_ORDER if METHOD_LABELS[m] in current_methods]

    plt.figure(figsize=(14, 8))
    sns.boxplot(
        data=counts, x='optimization_method_label', y='num_configs', hue='search_space_label',
        palette="Set2", showfliers=True, order=plot_order
    )
    plt.title(f'Number of Evaluated Configurations: {env}', fontsize=15)
    plt.ylabel('Number of Configurations', fontsize=12)
    plt.xlabel('Optimization Method', fontsize=12)
    plt.legend(title='Search Space')
    plt.xticks(rotation=0, ha='center', fontsize=11, wrap=True)
    plt.tight_layout()
    pdf.savefig()
    plt.close()

    print(f"   > Plotted configuration counts for {env}")

# ---------------------------------------------------------
# 4. MAIN EXECUTION
# ---------------------------------------------------------

def main():
    df_results = load_results_data([BASE_DIR])
    df_runhistory = load_runhistory_data(DOWNLOAD_DIR)

    if df_results.empty and df_runhistory.empty:
        print("No data found in current or download directories.")
        return

    all_envs = set()
    if not df_results.empty: 
        all_envs.update(df_results['experiment'].unique())
    if not df_runhistory.empty: 
        all_envs.update(df_runhistory['experiment'].unique())

    for env in all_envs:
        filename = f"report_{env}.pdf"
        print(f"Generating report: {filename}...")
        
        with PdfPages(filename) as pdf:
            # 1. Boxplots (Final Performance)
            if not df_results.empty: 
                plot_boxplots(pdf, df_results, env)
            
            # 2. Incumbent Trajectories (Performance vs Log-Time)
            if not df_runhistory.empty: 
                plot_incumbent_trajectories(pdf, df_runhistory, env)
            
            # 3. Cost Analysis (Debug info)
            if not df_runhistory.empty: 
                plot_costs(pdf, df_runhistory, env)

            # 4. Number of Configurations Evaluated
            if not df_runhistory.empty:
                plot_number_of_configurations_boxplots(pdf, df_runhistory, env)

    print("\nDone. Reports generated.")

if __name__ == "__main__":
    main()