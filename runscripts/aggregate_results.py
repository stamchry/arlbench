import argparse
import json
import csv
import pandas as pd
from pathlib import Path

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder", required=True, help="Folder containing seed_*.json files")
    args = parser.parse_args()
    
    folder = Path(args.folder)
    if not folder.exists():
        print(f"Folder {folder} does not exist.")
        return

    json_files = list(folder.glob("seed_*.json"))
    if not json_files:
        print(f"No JSON files found in {folder}")
        return

    data = []
    for json_file in json_files:
        try:
            with open(json_file, 'r') as f:
                content = json.load(f)
                
                # Flatten the structure for CSV
                row = {
                    "experiment": content.get("experiment"),
                    "search_space": content.get("search_space"),
                    "optimization_method": content.get("optimization_method"),
                    "opt_id": content.get("opt_id"),
                    "test_seed": content.get("test_seed"),
                    "performance": content["result"]["performance"],
                    "cost": content["result"]["cost"]
                }
                data.append(row)
        except Exception as e:
            print(f"Error reading {json_file}: {e}")

    if data:
        df = pd.DataFrame(data)
        # Sort by seed for tidiness
        df = df.sort_values(by="test_seed")
        
        output_csv = folder / "results.csv"
        df.to_csv(output_csv, index=False)
        print(f"Aggregated {len(data)} results into {output_csv}")
    else:
        print("No valid data found to aggregate.")

if __name__ == "__main__":
    main()