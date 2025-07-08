import numpy as np
import pandas as pd
import os

def save_individual_cluster_files(num_cells, max_step, dens, step, rng_seed):
    dens_folder = f"{dens:.2f}".replace(".", "_")
    for tic in range(0, max_step + 1, step):
        for seed in rng_seed:
            dat_actual = (
                f"{dens_folder}/dat_labels/culture_initial_number_of_cells={num_cells}_density={dens}_"
                f"force=Anisotropic_Grosmann_k=3.33_gamma=3_With_Noise_eta=0.033_With_Shrinking_"
                f"rng_seed={seed}_step={tic:05}.dat"
            )
            if os.path.exists(dat_actual):
                df_tic = pd.read_csv(dat_actual)

                number_clusters = df_tic['label'].nunique()
                size_biggest_cluster = df_tic['label'].value_counts().max()

                number_clusters_2 = df_tic['label2'].nunique()
                size_biggest_cluster_2 = df_tic['label2'].value_counts().max()

                output_file = f"{dens_folder}/dat_clusters/clusters_culture_initial_number_of_cells={num_cells}_density={dens}_force=Anisotropic_Grosmann_k=3.33_gamma=3_With_Noise_eta=0.033_With_Shrinking_rng_seed={seed}_step={tic:05}.dat"
                with open(output_file, "w") as f:
                    f.write("number_clusters,size_biggest_cluster,number_clusters2,size_biggest_cluster2\n")
                    f.write(f"{number_clusters},{size_biggest_cluster},{number_clusters_2},{size_biggest_cluster_2}\n")
            else:
                pass

# Parámetros
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
    save_individual_cluster_files(nc, max_step, dens, step, rng_seed)
