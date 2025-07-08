import numpy as np
import pandas as pd
import os

def save_individual_phenotype_files(num_cells, max_step, dens, step, rng_seed):
    dens_folder = f"{dens:.2f}".replace(".", "_")
    for tic in range(0, max_step + 1, step):
        for seed in rng_seed:
            dat_actual = (
                f"{dens_folder}/dat/culture_initial_number_of_cells={num_cells}_density={dens}_"
                f"force=Anisotropic_Grosmann_k=3.33_gamma=3_With_Noise_eta=0.033_With_Shrinking_"
                f"rng_seed={seed}_step={tic:05}.dat"
            )
            if os.path.exists(dat_actual):
                df_tic = pd.read_csv(dat_actual)

                N = len(df_tic)
                mean_ar = df_tic["aspect_ratio"].mean()
                fraction_elongated = np.isclose(df_tic["aspect_ratio"], 2.7).sum() / N
                fraction_round = np.isclose(df_tic["aspect_ratio"], 1.0).sum() / N
                fraction_binary = 1 - fraction_elongated - fraction_round

                output_file = f"{dens_folder}/dat_phenotype/phenotype_culture_initial_number_of_cells={num_cells}_density={dens}_force=Anisotropic_Grosmann_k=3.33_gamma=3_With_Noise_eta=0.033_With_Shrinking_rng_seed={seed}_step={tic:05}.dat"
                with open(output_file, "w") as f:
                    f.write("mean_aspect_ratio,fraction_elongated,fraction_round,fraction_binary\n")
                    f.write(f"{mean_ar:.5f},{fraction_elongated:.5f},{fraction_round:.5f},{fraction_binary:.5f}\n")
            else:
                pass

# Parámetros
#density_list = [0.5, 0.6, 0.7, 0.8, 0.85]
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
    os.makedirs(f"{dens_folder}/dat_phenotype", exist_ok=True)
    save_individual_phenotype_files(nc, max_step, dens, step, rng_seed)
