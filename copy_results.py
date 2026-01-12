import os
import shutil

def collect_results(base_dir="results", target_dir="download"):
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    # Files we are interested in
    target_filenames = [
        "incumbent.csv", 
        "runhistory.csv",
        "results.csv",
        "runhistory.json",
        "configspace.json",
        "intensifier.json",
        "optimization.json",
        "scenario.json",
        "final_config.yaml"
    ]

    print(f"Scanning {base_dir}...")
    
    # Walk through the entire results directory
    for root, dirs, files in os.walk(base_dir):
        for file in files:
            if file in target_filenames:
                # Get the relative path from the base_dir (e.g., 'smac/ppo_CartPole-v1_algo/1/7')
                rel_path = os.path.relpath(root, base_dir)
                
                # Construct the destination directory path
                dest_dir = os.path.join(target_dir, rel_path)
                
                # Create the destination directory if it doesn't exist
                os.makedirs(dest_dir, exist_ok=True)
                
                # Copy the file while preserving metadata
                src_file = os.path.join(root, file)
                dest_file = os.path.join(dest_dir, file)
                
                shutil.copy2(src_file, dest_file)

    print(f"Done! Files copied to '{target_dir}' preserving the original structure.")

if __name__ == "__main__":
    collect_results()