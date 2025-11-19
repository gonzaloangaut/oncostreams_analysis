import numpy as np
import pandas as pd
import os

def save_individual_cluster_files(num_cells, max_step, dens, step, rng_seed):
    """
    Function to save the files containing the cluster information for each seed and step
    """
    dens_folder = f"{dens:.3f}".replace(".", "_")
    # Initialize the last step as the max
    last_step = max_step
    for tic in range(0, max_step + 1, step):
        for seed in rng_seed:
            # Read every .dat
            dat_actual = (
                f"data/N={num_cells:_}/{dens_folder}/dat_labels/culture_initial_number_of_cells={num_cells}_density={dens}_"
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
                output_file = f"data/N={num_cells:_}/{dens_folder}/dat_clusters/max_number/clusters_culture_initial_number_of_cells={num_cells}_density={dens}_force=Anisotropic_Grosmann_k=3.33_gamma=3_With_Noise_eta=0.033_With_Shrinking_rng_seed={seed}_step={tic:05}.dat"
                with open(output_file, "w") as f:
                    f.write("n_round,max_round,n_elongated,max_elongated,n_round_2,max_round_2,n_elongated_2,max_elongated_2\n")
                    f.write(f"{number_clusters_round},{size_biggest_cluster_round},{number_clusters_elongated},{size_biggest_cluster_elongated},{number_clusters_round_2},{size_biggest_cluster_round_2},{number_clusters_elongated_2},{size_biggest_cluster_elongated_2}\n")
            else:
                # If the file does not exist, update the last step
                last_step = tic-step
                return last_step
    return last_step
            

def get_s_list(num_cells):
    """
    Return 10 values of s depending on system size,
    spaced roughly logarithmically.
    """
    if num_cells == 1000:
        return [1, 2, 3, 5, 8, 13, 20, 40, 80, 160]
    elif num_cells == 5000:
        return [1, 3, 5, 10, 20, 40, 80, 200, 500, 1000]
    elif num_cells == 10000:
        return [1, 3, 5, 10, 30, 60, 120, 300, 800, 1600]
    else:
        # fallback: 10 log-spaced s values up to N/5
        import numpy as np
        return np.unique(
            np.round(np.logspace(0, np.log10(max(10, num_cells//5)), 10)).astype(int)
        ).tolist()


# def save_cluster_size_distribution_10_values(num_cells, max_step, dens, step, rng_seed):
#     """
#     Function to save cluster size distributions P(s,t) for 10 chosen s values.
#     Output: a .dat file per seed and step with the counts for each s.
#     """
#     dens_folder = f"{dens:.3f}".replace(".", "_")
#     last_step = max_step

#     # get s list for this system size
#     s_list = get_s_list(num_cells)

#     for tic in range(0, max_step + 1, step):
#         for seed in rng_seed:

#             dat_actual = (
#                 f"data/N={num_cells:_}/{dens_folder}/dat_labels/"
#                 f"culture_initial_number_of_cells={num_cells}_density={dens}_"
#                 f"force=Anisotropic_Grosmann_k=3.33_gamma=3_With_Noise_eta=0.033_With_Shrinking_"
#                 f"rng_seed={seed}_step={tic:05}.dat"
#             )

#             if not os.path.exists(dat_actual):
#                 last_step = tic - step
#                 return last_step

#             # read dataframe
#             df_tic = pd.read_csv(dat_actual)

#             # get cluster sizes from label column
#             sizes = df_tic['label'].value_counts().values

#             # quick histogram
#             max_s = sizes.max() if len(sizes) > 0 else 0
#             hist = np.bincount(sizes, minlength=max_s+1)

#             # count how many clusters have size exactly s in s_list
#             counts = []
#             for s in s_list:
#                 if s <= max_s:
#                     counts.append(hist[s])
#                 else:
#                     counts.append(0)

#             # output file
#             output_file = (
#                 f"data/N={num_cells:_}/{dens_folder}/dat_clusters/P_sizes/"
#                 f"Psizes_culture_initial_number_of_cells={num_cells}_density={dens}_"
#                 f"rng_seed={seed}_step={tic:05}.dat"
#             )

#             # write file similar in style to your original function
#             with open(output_file, "w") as f:
#                 header = ",".join([f"s_{s}" for s in s_list])
#                 f.write(header + "\n")
#                 f.write(",".join(str(c) for c in counts) + "\n")

#     return last_step


def save_distribution_last_step(num_cells, dens, rng_seed, last_step):
    """
    Function to save a unique csv with all the cell distribution in the last step
    for every seed.
    """
    # Create the outputdir
    dens_folder = f"{dens:.3f}".replace(".", "_")
    output_dir = f"data/N={num_cells:_}/{dens_folder}/dat_clusters/cluster_distributions_final"
    os.makedirs(output_dir, exist_ok=True)

    # Acumulate the sizes
    round_sizes = []
    elongated_sizes = []
    round_sizes_2 = []
    elongated_sizes_2 = []

    for seed in rng_seed:
        # Read the .dat
        final_dat = (
            f"data/N={num_cells:_}/{dens_folder}/dat_labels/culture_initial_number_of_cells={num_cells}_density={dens}_"
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
    output_file = f"{output_dir}/cluster_size_distribution_cells={num_cells}_density={dens:.3f}.csv"
    df.to_csv(output_file, index=False)


def save_distribution_last_step_no_giant(num_cells, dens, rng_seed, last_step):
    """
    Same as save_distribution_last_step, but removing the giant cluster
    for each seed, separately for round/elongated and for label/label2.
    """
    # Create the outputdir
    dens_folder = f"{dens:.3f}".replace(".", "_")
    output_dir = f"data/N={num_cells:_}/{dens_folder}/dat_clusters/cluster_distributions_final_no_giant"
    os.makedirs(output_dir, exist_ok=True)

    # Accumulate the sizes
    round_sizes = []
    elongated_sizes = []
    round_sizes_2 = []
    elongated_sizes_2 = []

    for seed in rng_seed:
        # Read the .dat
        final_dat = (
            f"data/N={num_cells:_}/{dens_folder}/dat_labels/"
            f"culture_initial_number_of_cells={num_cells}_density={dens}_"
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

        # --- label: round ---
        vc_r = df_round['label'].value_counts()
        if len(vc_r) > 0:
            giant_r = vc_r.idxmax()
            cluster_sizes_round = vc_r[vc_r.index != giant_r].values
            round_sizes.extend(cluster_sizes_round)

        # --- label: elongated ---
        vc_e = df_elongated['label'].value_counts()
        if len(vc_e) > 0:
            giant_e = vc_e.idxmax()
            cluster_sizes_elongated = vc_e[vc_e.index != giant_e].values
            elongated_sizes.extend(cluster_sizes_elongated)

        # --- label2: round ---
        vc_r2 = df_round['label2'].value_counts()
        if len(vc_r2) > 0:
            giant_r2 = vc_r2.idxmax()
            cluster_sizes_round_2 = vc_r2[vc_r2.index != giant_r2].values
            round_sizes_2.extend(cluster_sizes_round_2)

        # --- label2: elongated ---
        vc_e2 = df_elongated['label2'].value_counts()
        if len(vc_e2) > 0:
            giant_e2 = vc_e2.idxmax()
            cluster_sizes_elongated_2 = vc_e2[vc_e2.index != giant_e2].values
            elongated_sizes_2.extend(cluster_sizes_elongated_2)

    # Fill the lists with NaNs so all have same length
    max_len = max(
        len(round_sizes),
        len(elongated_sizes),
        len(round_sizes_2),
        len(elongated_sizes_2),
    )

    round_sizes += [np.nan] * (max_len - len(round_sizes))
    elongated_sizes += [np.nan] * (max_len - len(elongated_sizes))
    round_sizes_2 += [np.nan] * (max_len - len(round_sizes_2))
    elongated_sizes_2 += [np.nan] * (max_len - len(elongated_sizes_2))

    # Create the dataframe
    df = pd.DataFrame({
        'sizes_round_no_giant': round_sizes,
        'sizes_elongated_no_giant': elongated_sizes,
        'sizes_round_2_no_giant': round_sizes_2,
        'sizes_elongated_2_no_giant': elongated_sizes_2
    })

    # Save it
    output_file = (
        f"{output_dir}/cluster_size_distribution_no_giant_cells={num_cells}_density={dens:.3f}.csv"
    )
    df.to_csv(output_file, index=False)


# Parameters
density_list = [0.865]
nc = 10_000
max_step = 80_000
step = 100
number_of_realizations = 64

seed_1 = 0x87351080E25CB0FAD77A44A3BE03B491
rng_1 = np.random.default_rng(seed_1)
rng_seed = rng_1.integers(low=2**20, high=2**50, size=number_of_realizations)

for dens in density_list:
    dens_folder = f"{dens:.3f}".replace(".", "_")
    os.makedirs(f"data/N={nc:_}/{dens_folder}/dat_clusters", exist_ok=True)
    os.makedirs(f"data/N={nc:_}/{dens_folder}/dat_clusters/max_number", exist_ok=True)
    # os.makedirs(f"data/N={nc:_}/{dens_folder}/dat_clusters/P_sizes", exist_ok=True)
    last_step = save_individual_cluster_files(nc, max_step, dens, step, rng_seed)
    # save_cluster_size_distribution_10_values(nc, max_step, dens, step, rng_seed)
    save_distribution_last_step(nc, dens, rng_seed, last_step)
    save_distribution_last_step_no_giant(nc, dens, rng_seed, last_step)
