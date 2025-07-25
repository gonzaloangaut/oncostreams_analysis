import numpy as np
import pandas as pd
import os

def calculate_local_phenotype(df_filtered, number_cells_box):
    """
    Function to calculate the local phenotypes given a filtered dataframe  
    and the number of cells in the box.
    """
    # Return NaN if the number of cells is 0
    if number_cells_box == 0:
        return np.nan, np.nan, np.nan, np.nan
    # Calculate the phenotypes
    mean_ar = df_filtered["aspect_ratio"].mean()
    fraction_elongated = np.isclose(df_filtered["aspect_ratio"], 2.7).sum() / number_cells_box
    fraction_round = np.isclose(df_filtered["aspect_ratio"], 1.0).sum() / number_cells_box
    fraction_binary = 1 - fraction_elongated - fraction_round
    return mean_ar, fraction_elongated, fraction_round, fraction_binary

def save_local_phenotype(mean_ar, fraction_elongated, fraction_round, fraction_binary, dens, number_cells, box_length, seed, tic, dens_folder, number_cells_box):
    """
    Function to save the local phenotypes.
    """
    # Name of the output file
    output_file = (
        f"{dens_folder}/dat_local_phenotype/"
        f"local_phenotype_box_length={int(box_length)}_"
        f"culture_initial_number_of_cells={number_cells}_density={dens}_"
        f"force=Anisotropic_Grosmann_k=3.33_gamma=3_With_Noise_eta=0.033_"
        f"With_Shrinking_rng_seed={seed}_step={tic:05}.dat"
    )
    # Save the file
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, "w") as f:
        f.write("mean_aspect_ratio,fraction_elongated,fraction_round,fraction_binary,number_cells_box\n")
        f.write(f"{mean_ar:.5f},{fraction_elongated:.5f},{fraction_round:.5f},{fraction_binary:.5f}, {int(round(number_cells_box))}\n")

def calculate_local_order_parameters(df_filtered, number_cells_box):
    """
    Function to calculate the local order parameters given a filtered dataframe
    and the number of cells in the box.
    """
    # Return NaN if the number of cells is 0
    if number_cells_box == 0:
        return np.nan, np.nan, np.nan, np.nan
    # Filter for the not round cells
    df_not_round = df_filtered[~np.isclose(df_filtered["aspect_ratio"], 1.0)]
    number_not_round = len(df_not_round)
    # Calculate the order parameters with number of not round
    if number_not_round > 0:
        sin_phi = np.sin(df_not_round["orientation"])
        cos_phi = np.cos(df_not_round["orientation"])
        sin_2_phi = np.sin(2 * df_not_round["orientation"])
        cos_2_phi = np.cos(2 * df_not_round["orientation"])

        sum_sin = sin_phi.sum()
        sum_cos = cos_phi.sum()
        sum_sin_2 = sin_2_phi.sum()
        sum_cos_2 = cos_2_phi.sum()

        nematic = np.sqrt(sum_sin_2**2 + sum_cos_2**2) / number_not_round
        polar = np.sqrt(sum_sin**2 + sum_cos**2) / number_not_round
        nematic_2 = np.sqrt(sum_sin_2**2 + sum_cos_2**2) / number_cells_box
        polar_2 = np.sqrt(sum_sin**2 + sum_cos**2) / number_cells_box
    else:
        nematic = polar = nematic_2 = polar_2 = 0.0
    
    return nematic, polar, nematic_2, polar_2

# Función para guardar los parámetros de orden locales
def save_local_order_parameters(nematic, polar, nematic_2, polar_2, dens, number_cells, box_length, seed, tic, dens_folder, number_cells_box):
    """
    Function to save the local order parameters.
    """
    # Name of the output file
    output_file = (
        f"{dens_folder}/dat_local_op/"
        f"local_op_box_length={int(box_length)}_"
        f"culture_initial_number_of_cells={number_cells}_density={dens}_"
        f"force=Anisotropic_Grosmann_k=3.33_gamma=3_With_Noise_eta=0.033_"
        f"With_Shrinking_rng_seed={seed}_step={tic:05}.dat"
    )
    # Save the file
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, "w") as f:
        f.write("nematic,polar,nematic_2,polar_2,number_cells_box\n")
        f.write(f"{nematic:.5f},{polar:.5f},{nematic_2:.5f},{polar_2:.5f},{int(round(number_cells_box))}\n")

# Simulation parameters
density = [0.5, 0.6, 0.7, 0.8, 0.85, 0.9]
number_cells = 10_000
cell_area = np.pi
attempts = 10
max_step = 60_000
step = 100
number_of_realizations = 64

seed_1 = 0x87351080E25CB0FAD77A44A3BE03B491
rng_1 = np.random.default_rng(seed_1)
rng_seed = rng_1.integers(low=2**20, high=2**50, size=number_of_realizations)

# List with the box lengths we want to study
box_lengths = [5, 10, 15, 20, 25, 30]
# Save data for every density
for dens in density:
    # Calculate the side given the density
    side = np.sqrt(number_cells * cell_area / dens)
    dens_folder = f"{dens:.2f}".replace(".", "_")

    # Loop for every box length
    for length in box_lengths:
        # Loop for every step
        for tic in range(0, max_step + 1, step):
            # Loop for every seed
            for seed in rng_seed:
                # Initialize lists to save the parameters
                mean_ar_attempt = []
                fraction_elongated_attempt = []
                fraction_round_attempt = []
                fraction_binary_attempt = []
                nematic_attempt = []
                polar_attempt = []
                nematic_2_attempt = []
                polar_2_attempt = []
                number_cells_box_attempt = []
                # Read the file
                dat_actual = (
                    f"{dens_folder}/dat/"
                    f"culture_initial_number_of_cells={number_cells}_density={dens}_"
                    f"force=Anisotropic_Grosmann_k=3.33_gamma=3_With_Noise_eta=0.033_"
                    f"With_Shrinking_rng_seed={seed}_step={tic:05}.dat"
                )
                if not os.path.exists(dat_actual):
                    continue
                df_tic = pd.read_csv(dat_actual)

                # attempt is the quantity of boxes we choose in every seed and tic
                for i in range(attempts):
                    # Choose a random center
                    center = np.array([
                        rng_1.uniform(0, side),
                        rng_1.uniform(0, side),
                        0
                    ])
                    half_length = length / 2

                    # Apply boundary conditions (check)
                    df = df_tic.copy()
                    dx = (df["position_x"] - center[0] + side/2) % side - side/2
                    dy = (df["position_y"] - center[1] + side/2) % side - side/2

                    mask = (np.abs(dx) <= half_length) & (np.abs(dy) <= half_length)
                    df_filtered = df[mask].copy()

                    # Calculate local phenotypes and order parameters
                    number_cells_box = len(df_filtered)
                    mean_ar, fraction_elongated, fraction_round, fraction_binary = calculate_local_phenotype(df_filtered, number_cells_box)
                    mean_ar_attempt.append(mean_ar)
                    fraction_elongated_attempt.append(fraction_elongated)
                    fraction_round_attempt.append(fraction_round)
                    fraction_binary_attempt.append(fraction_binary)

                    nematic, polar, nematic_2, polar_2 = calculate_local_order_parameters(df_filtered, number_cells_box)
                    nematic_attempt.append(nematic)
                    polar_attempt.append(polar)
                    nematic_2_attempt.append(nematic_2)
                    polar_2_attempt.append(polar_2)

                    number_cells_box_attempt.append(number_cells_box)
                # Calculate the means
                mean_ar_mean = np.nanmean(mean_ar_attempt)
                fraction_elongated_mean = np.nanmean(fraction_elongated_attempt)
                fraction_round_mean = np.nanmean(fraction_round_attempt)
                fraction_binary_mean = np.nanmean(fraction_binary_attempt)

                nematic_mean = np.nanmean(nematic_attempt)
                polar_mean = np.nanmean(polar_attempt)
                nematic_2_mean = np.nanmean(nematic_2_attempt)
                polar_2_mean = np.nanmean(polar_2_attempt)

                number_cells_box_mean = np.mean(number_cells_box_attempt)
                # Save everything in the files
                save_local_phenotype(mean_ar_mean, fraction_elongated_mean, fraction_round_mean, fraction_binary_mean, dens, number_cells, length, seed, tic, dens_folder, number_cells_box_mean)
                save_local_order_parameters(nematic_mean, polar_mean, nematic_2_mean, polar_2_mean, dens, number_cells, length, seed, tic, dens_folder, number_cells_box_mean)

