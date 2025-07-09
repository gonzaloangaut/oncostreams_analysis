import numpy as np
import pandas as pd
import os

def save_individual_cluster_files(num_cells, max_step, dens, step, rng_seed):
    """
    Function to save the files containing the cluster information for each seed and step
    """
    dens_folder = f"{dens:.2f}".replace(".", "_")
    # Initialize the last step as the max
    last_step = max_step
    for tic in range(0, max_step + 1, step):
        for seed in rng_seed:
            # Read every .dat
            dat_actual = (
                f"{dens_folder}/dat_labels/culture_initial_number_of_cells={num_cells}_density={dens}_"
                f"force=Anisotropic_Grosmann_k=3.33_gamma=3_With_Noise_eta=0.033_With_Shrinking_"
                f"rng_seed={seed}_step={tic:05}.dat"
            )
            if os.path.exists(dat_actual):
                # Create the pandas df
                df_tic = pd.read_csv(dat_actual)
                # Separate in elongated and round cells
                mask = np.isclose(df_tic['aspect_ratio'], 1)
                df_tic_round = df_tic[mask]
                df_tic_elongated = df_tic[~mask]
                
                # Take the number of clusters and size for each df
                number_clusters_round = df_tic_round['label'].nunique()
                size_biggest_cluster_round = (
                    df_tic_round['label'].value_counts().max() if not df_tic_round.empty else 0
                )
                number_clusters_elongated = df_tic_elongated['label'].nunique()
                size_biggest_cluster_elongated = (
                    df_tic_elongated['label'].value_counts().max() if not df_tic_elongated.empty else 0
                )
                
                # The same using the other label
                number_clusters_round_2 = df_tic_round['label2'].nunique()
                size_biggest_cluster_round_2 = (
                    df_tic_round['label2'].value_counts().max() if not df_tic_round.empty else 0
                )
                number_clusters_elongated_2 = df_tic_elongated['label2'].nunique()
                size_biggest_cluster_elongated_2 = (
                    df_tic_elongated['label2'].value_counts().max() if not df_tic_elongated.empty else 0
                )
                # Open and write the new .dat
                output_file = f"{dens_folder}/dat_clusters/clusters_culture_initial_number_of_cells={num_cells}_density={dens}_force=Anisotropic_Grosmann_k=3.33_gamma=3_With_Noise_eta=0.033_With_Shrinking_rng_seed={seed}_step={tic:05}.dat"
                with open(output_file, "w") as f:
                    f.write("n_round,max_round,n_elongated,max_elongated,n_round_2,max_round_2,n_elongated_2,max_elongated_2\n")
                    f.write(f"{number_clusters_round},{size_biggest_cluster_round},{number_clusters_elongated},{size_biggest_cluster_elongated},{number_clusters_round_2},{size_biggest_cluster_round_2},{number_clusters_elongated_2},{size_biggest_cluster_elongated_2}\n")
            else:
                # If the file does not exist, update the last step
                last_step = tic-step
                return last_step
    return last_step
            

def save_distribution_last_step(num_cells, dens, rng_seed, last_step):
    """
    Function to save a unique csv with all the cell distribution in the last step
    for every seed.
    """
    # Create the outputdir
    dens_folder = f"{dens:.2f}".replace(".", "_")
    output_dir = f"{dens_folder}/cluster_distributions_final"
    os.makedirs(output_dir, exist_ok=True)

    # Acumulate the sizes
    round_sizes = []
    elongated_sizes = []
    round_sizes_2 = []
    elongated_sizes_2 = []

    for seed in rng_seed:
        # Read the .dat
        final_dat = (
            f"{dens_folder}/dat_labels/culture_initial_number_of_cells={num_cells}_density={dens}_"
            f"force=Anisotropic_Grosmann_k=3.33_gamma=3_With_Noise_eta=0.033_With_Shrinking_"
            f"rng_seed={seed}_step={last_step:05}.dat"
        )
        if not os.path.exists(final_dat):
            continue
        # Create the pandas df
        final_df = pd.read_csv(final_dat)
        # Separate in elongated and round cells
        mask = np.isclose(final_df['aspect_ratio'], 1)
        df_round = final_df[mask]
        df_elongated = final_df[~mask]
        
        # Count the cluster sizes for label
        cluster_sizes_round = df_round['label'].value_counts().values
        cluster_sizes_elongated = df_elongated['label'].value_counts().values
        # Add them to the list
        round_sizes.extend(cluster_sizes_round)
        elongated_sizes.extend(cluster_sizes_elongated)
        # Same for label2
        cluster_sizes_round_2 = df_round['label2'].value_counts().values
        cluster_sizes_elongated_2 = df_elongated['label2'].value_counts().values
        # Add them to the list
        round_sizes_2.extend(cluster_sizes_round_2)
        elongated_sizes_2.extend(cluster_sizes_elongated_2)
    # fill the lists with less values with nans 
    max_len = max(len(round_sizes), len(elongated_sizes), len(round_sizes_2), len(elongated_sizes_2))
    round_sizes += [np.nan] * (max_len - len(round_sizes))
    elongated_sizes += [np.nan] * (max_len - len(elongated_sizes))
    round_sizes_2 += [np.nan] * (max_len - len(round_sizes_2))
    elongated_sizes_2 += [np.nan] * (max_len - len(elongated_sizes_2))
    # Create the dataframe
    df = pd.DataFrame({
        'sizes_round': round_sizes,
        'sizes_elongated': elongated_sizes,
        'sizes_round_2': round_sizes_2,
        'sizes_elongated_2': elongated_sizes_2
    })
    # Save it
    output_file = f"{output_dir}/cluster_size_distribution_cells={num_cells}_density={dens:.2f}.csv"
    df.to_csv(output_file, index=False)



# Parameters
density_list = [0.9]
nc = 10_000
max_step = 50_000
step = 100
number_of_realizations = 64

seed_1 = 0x87351080E25CB0FAD77A44A3BE03B491
rng_1 = np.random.default_rng(seed_1)
rng_seed = rng_1.integers(low=2**20, high=2**50, size=number_of_realizations)

for dens in density_list:
    dens_folder = f"{dens:.2f}".replace(".", "_")
    os.makedirs(f"{dens_folder}/dat_clusters", exist_ok=True)
    last_step = save_individual_cluster_files(nc, max_step, dens, step, rng_seed)
    save_distribution_last_step(nc, dens, rng_seed, last_step)
