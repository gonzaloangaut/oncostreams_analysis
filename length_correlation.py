import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os


def calculate_final_local_order_parameter(num_cells, max_step, dens, step, rng_seed, box_length):
    dens_folder = f"{dens:.2f}".replace(".", "_")
    frenar = False
    
    # Buscamos el último step común para todos los seeds
    ultimo_step = max_step
    for tic in range(100, max_step, step):
        for seed in rng_seed:
            dat_actual = f"{dens_folder}/dat_local_op/local_op_box_length={int(box_length)}_culture_initial_number_of_cells={num_cells}_density={dens}_force=Anisotropic_Grosmann_k=3.33_gamma=3_With_Noise_eta=0.033_With_Shrinking_rng_seed={seed}_step={tic:05}.dat"
            if not os.path.exists(dat_actual):
                frenar = True
                break
        if frenar:
            ultimo_step = tic - step
            break
    print(f"ultimo step = {ultimo_step}, para densidad = {dens}")

    nematic_order = []
    polar_order = []
    nematic_2_order = []
    polar_2_order = []    
    for seed in rng_seed:
        dat_actual = f"{dens_folder}/dat_local_op/local_op_box_length={int(box_length)}_culture_initial_number_of_cells={num_cells}_density={dens}_force=Anisotropic_Grosmann_k=3.33_gamma=3_With_Noise_eta=0.033_With_Shrinking_rng_seed={seed}_step={ultimo_step:05}.dat"
        if os.path.exists(dat_actual):
            df_tic = pd.read_csv(dat_actual)
        else:
            print("Error")
            break

        nematic_order.append(df_tic["nematic"].mean())
        polar_order.append(df_tic["polar"].mean())
        nematic_2_order.append(df_tic["nematic_2"].mean())
        polar_2_order.append(df_tic["polar_2"].mean())

    return {
        "mean": {
            "nematic_order": np.mean(nematic_order),
            "polar_order": np.mean(polar_order),
            "nematic_order_2": np.mean(nematic_2_order),
            "polar_order_2": np.mean(polar_2_order),
        },
        "std": {
            "nematic_order": np.std(nematic_order),
            "polar_order": np.std(polar_order),
            "nematic_order_2": np.std(nematic_2_order),
            "polar_order_2": np.std(polar_2_order),
        }
    }  


# Parámetros globales
# density = [0.5, 0.6, 0.7, 0.8, 0.85, 0.9]
density = [0.85]
nc = 10_000
cell_area = np.pi
max_step = 60_000
step = 100
number_of_realizations = 64

seed_1 = 0x87351080E25CB0FAD77A44A3BE03B491
rng_1 = np.random.default_rng(seed_1)
rng_seed = rng_1.integers(low=2**20, high=2**50, size=number_of_realizations)

box_lengths = [5, 10, 15, 20, 25, 30]

# Carpetas de salida
os.makedirs("graphs/local_order_vs_boxlength/order_parameters_1", exist_ok=True)
os.makedirs("graphs/local_order_vs_boxlength/order_parameters_2", exist_ok=True)

# Iteramos sobre densidades (y graficamos vs box_length)
for dens in density:
    side = np.sqrt(nc * cell_area / dens)
    
    df_mean = pd.DataFrame(columns=["box_length", "nematic_order", "polar_order", "nematic_order_2", "polar_order_2"])
    df_std = pd.DataFrame(columns=["box_length", "nematic_order_std", "polar_order_std", "nematic_order_2_std", "polar_order_2_std"])
    
    for box_length in box_lengths:
        result = calculate_final_local_order_parameter(nc, max_step + 1, dens, step, rng_seed, box_length)
        mean = result["mean"]
        std = result["std"]

        df_mean.loc[len(df_mean)] = [box_length, mean["nematic_order"], mean["polar_order"], mean["nematic_order_2"], mean["polar_order_2"]]
        df_std.loc[len(df_std)] = [box_length, std["nematic_order"], std["polar_order"], std["nematic_order_2"], std["polar_order_2"]]

    N = number_of_realizations
    error_bars = df_std / np.sqrt(N)

    # GRAFICO 1: Q y P
    fig, ax = plt.subplots()
    ax.errorbar(df_mean["box_length"], df_mean["nematic_order"],
                yerr=error_bars["nematic_order_std"], fmt='o-', color="blue", label="Nematic order (Q)")
    ax.errorbar(df_mean["box_length"], df_mean["polar_order"],
                yerr=error_bars["polar_order_std"], fmt='o-', color="green", label="Polar order (P)")
    ax.set_xlabel("Box length")
    ax.set_ylabel("Order parameters")
    ax.legend()
    plt.title(f"Order parameters vs box length\nDensity = {dens}")
    plt.savefig(f"graphs/local_order_vs_boxlength/order_parameters_1/order_vs_box_length_density={dens:.2f}.jpg", dpi=600)
    plt.close()

    # GRAFICO 2: Q̂ y P̂
    fig, ax = plt.subplots()
    ax.errorbar(df_mean["box_length"], df_mean["nematic_order_2"],
                yerr=error_bars["nematic_order_2_std"], fmt='o-', color="blue", label=r"Nematic order ($\hat{{Q}}$)")
    ax.errorbar(df_mean["box_length"], df_mean["polar_order_2"],
                yerr=error_bars["polar_order_2_std"], fmt='o-', color="green", label=r"Polar order ($\hat{{P}}$)")
    ax.set_xlabel("Box length")
    ax.set_ylabel("Order parameters")
    ax.legend()
    plt.title(f"Order parameters (2) vs box length\nDensity = {dens}")
    plt.savefig(f"graphs/local_order_vs_boxlength/order_parameters_2/order_2_vs_box_length_density={dens:.2f}.jpg", dpi=600)
    plt.close()
