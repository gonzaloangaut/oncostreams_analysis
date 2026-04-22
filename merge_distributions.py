import pandas as pd
from pathlib import Path
import os

# Define environment paths
CONFIG = {
    "base_data_path": Path("data"), 
}

# Logic to merge _1 and _2 parts
def merge_all_distributions_for_N(n_cells, densities):
    """
    Iterates through densities for a given N, merging both 'final' and 'no_giant' 
    cluster distribution parts (_1 and _2) into single unified CSV files.
    """
    num_cells_folder = f"N={n_cells:_}"
    base_data_path = CONFIG["base_data_path"] / num_cells_folder

    print(f"--- Starting Merge Process for {num_cells_folder} ---")

    for rho in densities:
        rho_str = f"{rho:.3f}"
        dens_folder = rho_str.replace(".", "_")
        
        # Paths for regular and 'no_giant' distributions
        folders_to_process = {
            "regular": {
                "path": base_data_path / dens_folder / "dat_clusters" / "cluster_distributions_final",
                "prefix": "cluster_size_distribution_cells="
            },
            "no_giant": {
                "path": base_data_path / dens_folder / "dat_clusters" / "cluster_distributions_final_no_giant",
                "prefix": "cluster_size_distribution_no_giant_cells="
            }
        }

        print(f"Processing density: {rho_str}...")

        for key, info in folders_to_process.items():
            folder = info["path"]
            prefix = info["prefix"]
            
            file_v1 = folder / f"{prefix}{n_cells}_density={rho_str}_1.csv"
            file_v2 = folder / f"{prefix}{n_cells}_density={rho_str}_2.csv"
            output_file = folder / f"{prefix}{n_cells}_density={rho_str}.csv"

            if file_v1.exists() and file_v2.exists():
                df1 = pd.read_csv(file_v1)
                df2 = pd.read_csv(file_v2)
                
                df_combined = pd.concat([df1, df2], ignore_index=True)
                df_combined.to_csv(output_file, index=False)
                
                print(f"  [OK] Merged {key} distribution.")
            else:
                if not file_v1.exists(): print(f"  [Warning] Missing part 1 for {key} at {rho_str}")
                if not file_v2.exists(): print(f"  [Warning] Missing part 2 for {key} at {rho_str}")

    print(f"--- Finished Merge Process for {num_cells_folder} ---\n")

# Execute
if __name__ == "__main__":
    # Define the densities you want to process for N=1000
    n_cells_target = 1000
    densities_target = [
        0.4, 0.5, 0.6, 0.7, 0.75, 0.76, 0.77, 0.78, 0.79, 0.8,
        0.81, 0.82, 0.83, 0.84, 0.843, 0.845, 0.848, 0.85, 0.852,
        0.855, 0.86, 0.87, 0.88, 0.9 
    ]

    merge_all_distributions_for_N(n_cells_target, densities_target)