import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os


def calculate_final_local_order_parameter(num_cells, max_step, dens, step, rng_seed, box_length):
    dens_folder = f"{dens:.2f}".replace(".", "_")
    frenar = False
    
    # Inicializamos el ultimo step como el maximo
    ultimo_step = max_step
    # Nos fijamos cual es el ultimo step que existe con todos los seed
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
        # leemos el archivo correspondiente
        dat_actual = f"{dens_folder}/dat_local_op/local_op_box_length={int(box_length)}_culture_initial_number_of_cells={num_cells}_density={dens}_force=Anisotropic_Grosmann_k=3.33_gamma=3_With_Noise_eta=0.033_With_Shrinking_rng_seed={seed}_step={ultimo_step:05}.dat"
        if os.path.exists(
            dat_actual
        ):
            df_tic = pd.read_csv(
                dat_actual
            )
        else:
            print("Error")
            break


        nematic = df_tic["nematic"].mean()
        polar = df_tic["polar"].mean()
        nematic_2 = df_tic["nematic_2"].mean()
        polar_2 = df_tic["polar_2"].mean()

        nematic_order.append(nematic)
        polar_order.append(polar)
        nematic_2_order.append(nematic_2)
        polar_2_order.append(polar_2)

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


density = [0.4, 0.5, 0.6, 0.7, 0.8, 0.81, 0.82, 0.83, 0.84, 0.85, 0.86, 0.87, 0.88, 0.9]
nc = 10_000
cell_area = np.pi
max_step = 60_000
step = 100
number_of_realizations = 64
seed_1 = 0x87351080E25CB0FAD77A44A3BE03B491
rng_1 = np.random.default_rng(seed_1)
rng_seed = rng_1.integers(low=2**20, high=2**50, size=number_of_realizations)

# ratios = [0.1, 0.2]
box_lengths = [5, 10, 15, 20, 25, 30]

os.makedirs(f"graphs/steady_state_local_order_parameters/order_parameters_1", exist_ok=True)
os.makedirs(f"graphs/steady_state_local_order_parameters/order_parameters_2", exist_ok=True)
# for ratio in ratios:
for box_length in box_lengths:
    #box_length_folder = f"{box_length:.2f}".replace(".", "_")

    # DataFrames para guardar medias y varianzas
    df_mean = pd.DataFrame(columns=["density", "nematic_order", "polar_order", "nematic_order_2", "polar_order_2"])
    df_std = pd.DataFrame(columns=["density", "nematic_order_std", "polar_order_std", "nematic_order_2_std", "polar_order_2_std"])
    for dens in density:
        side = np.sqrt(nc * cell_area / dens)
        # box_length = ratio*side
        dens_folder = f"{dens:.2f}".replace(".", "_")

        result = calculate_final_local_order_parameter(nc, max_step + 1, dens, step, rng_seed, box_length)
        mean = result["mean"]
        std = result["std"]

        df_mean.loc[len(df_mean)] = [dens, mean["nematic_order"], mean["polar_order"], mean["nematic_order_2"], mean["polar_order_2"]]
        df_std.loc[len(df_std)] = [dens, std["nematic_order"], std["polar_order"], std["nematic_order_2"], std["polar_order_2"]]

    # Número de muestras (para barras de error del promedio)
    N = number_of_realizations
    error_bars = df_std / np.sqrt(N)

    # ---------- GRAFICO 1: Medias con barras de error ----------
    fig, ax1 = plt.subplots()

    ax1.errorbar(df_mean["density"], df_mean["nematic_order"], 
                yerr=error_bars["nematic_order_std"], fmt='o-', color="blue", label="Nematic order (Q)")
    ax1.errorbar(df_mean["density"], df_mean["polar_order"], 
                yerr=error_bars["polar_order_std"], fmt='o-', color="green", label="Polar order (P)")

    ax1.set_xlabel("Density ρ")
    ax1.set_ylabel("Order parameters")

    lines, labels = ax1.get_legend_handles_labels()
    ax1.legend(lines, labels)

    plt.title(f"Steady State local order parameters, box length={box_length}")
    plt.savefig(f"graphs/steady_state_local_order_parameters/order_parameters_1/steady_state_local_order_parameters_box_length={box_length}.jpg", dpi=600)
    plt.close()

    # ---------- GRAFICO 2: Medias con barras de error (para las 2das variables) ----------
    fig, ax1 = plt.subplots()

    ax1.errorbar(df_mean["density"], df_mean["nematic_order_2"], 
                yerr=error_bars["nematic_order_2_std"], fmt='o-', color="blue", label=r"Nematic order ($\hat{Q}$)")
    ax1.errorbar(df_mean["density"], df_mean["polar_order_2"], 
                yerr=error_bars["polar_order_2_std"], fmt='o-', color="green", label=r"Polar order ($\hat{P}$)")

    ax1.set_xlabel("Density ρ")
    ax1.set_ylabel("Order parameters")

    lines, labels = ax1.get_legend_handles_labels()
    ax1.legend(lines, labels)

    plt.title(f"Steady State local order parameters (2), box length={box_length}")
    plt.savefig(f"graphs/steady_state_local_order_parameters/order_parameters_2/steady_state_local_order_parameters_2_box_length={box_length}.jpg", dpi=600)
    plt.close()

    # ---------- GRAFICO 3: Varianzas ----------
    plt.figure(figsize=(6, 4))
    plt.plot(df_mean["density"], df_std["nematic_order_std"]**2, label="Var(nematic order)", marker='o', color="blue")
    plt.plot(df_mean["density"], df_std["polar_order_std"]**2, label="Var(polar order)", marker='o', color="green")

    plt.xlabel("Density ρ")
    plt.ylabel("Variance")
    plt.title(f"Variance of local order parameters vs Density, box length={box_length}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"graphs/steady_state_local_order_parameters/order_parameters_1/steady_state_local_order_parameters_variance_box_length={box_length}.jpg", dpi=600)
    plt.close()


    # ---------- GRAFICO 4: Varianzas (los otros parametros) ----------
    plt.figure(figsize=(6, 4))
    plt.plot(df_mean["density"], df_std["nematic_order_2_std"]**2, label="Var(nematic order 2)", marker='o', color="blue")
    plt.plot(df_mean["density"], df_std["polar_order_2_std"]**2, label="Var(polar order 2)", marker='o', color="green")

    plt.xlabel("Density ρ")
    plt.ylabel("Variance")
    plt.title(f"Variance of local order parameters (2) vs Density, box length={box_length}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"graphs/steady_state_local_order_parameters/order_parameters_2/steady_state_local_order_parameters_2_variance_box_length={box_length}.jpg", dpi=600)
    plt.close()

