import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

def calculate_final_local_phenotype(num_cells, max_step, dens, step, rng_seed, box_length):
    dens_folder = f"{dens:.2f}".replace(".", "_")
    frenar = False
    ultimo_step = max_step
    for tic in range(100, max_step, step):
        for seed in rng_seed:
            dat_actual = f"{dens_folder}/dat_local_phenotype/local_phenotype_box_length={int(box_length)}_culture_initial_number_of_cells={num_cells}_density={dens}_force=Anisotropic_Grosmann_k=3.33_gamma=3_With_Noise_eta=0.033_With_Shrinking_rng_seed={seed}_step={tic:05}.dat"
            if not os.path.exists(dat_actual):
                frenar = True
                break
        if frenar:
            ultimo_step = tic - step
            break
    print(f"ultimo step = {ultimo_step}, para densidad = {dens}")

    fraction_elongated_total = []
    fraction_round_total = []
    fraction_binary_total = []
    mean_aspect_ratio_total = []

    for seed in rng_seed:
        dat_actual = f"{dens_folder}/dat_local_phenotype/local_phenotype_box_length={int(box_length)}_culture_initial_number_of_cells={num_cells}_density={dens}_force=Anisotropic_Grosmann_k=3.33_gamma=3_With_Noise_eta=0.033_With_Shrinking_rng_seed={seed}_step={ultimo_step:05}.dat"
        if os.path.exists(dat_actual):
            df_tic = pd.read_csv(dat_actual)
        else:
            print("Error")
            break

        mean_aspect_ratio_total.append(df_tic["mean_aspect_ratio"].mean())
        fraction_elongated_total.append(df_tic["fraction_elongated"].mean())
        fraction_round_total.append(df_tic["fraction_round"].mean())
        fraction_binary_total.append(df_tic["fraction_binary"].mean())

    return {
        "mean": {
            "mean_aspect_ratio": np.mean(mean_aspect_ratio_total),
            "fraction_elongated": np.mean(fraction_elongated_total),
            "fraction_round": np.mean(fraction_round_total),
            "fraction_binary": np.mean(fraction_binary_total),
        },
        "std": {
            "mean_aspect_ratio": np.std(mean_aspect_ratio_total),
            "fraction_elongated": np.std(fraction_elongated_total),
            "fraction_round": np.std(fraction_round_total),
            "fraction_binary": np.std(fraction_binary_total),
        }
    }


density = [0.5, 0.6, 0.7, 0.8, 0.85, 0.9]
nc = 10_000
max_step = 60_000
step = 100
number_of_realizations = 64
seed_1 = 0x87351080E25CB0FAD77A44A3BE03B491
rng_1 = np.random.default_rng(seed_1)
rng_seed = rng_1.integers(low=2**20, high=2**50, size=number_of_realizations)

box_lengths = [5, 10, 15, 20, 25, 30]

for box_length in box_lengths:
    #box_length_folder = f"{box_length:.2f}".replace(".", "_")
    os.makedirs(f"graphs/steady_state_local_cell_phenotype", exist_ok=True)
    # DataFrames para guardar medias y varianzas
    df_mean = pd.DataFrame(columns=["density", "mean_aspect_ratio", "fraction_elongated", "fraction_round", "fraction_binary"])
    df_std = pd.DataFrame(columns=["density", "mean_aspect_ratio_std", "fraction_elongated_std", "fraction_round_std", "fraction_binary_std"])

    for dens in density:
        result = calculate_final_local_phenotype(nc, max_step + 1, dens, step, rng_seed, box_length)
        mean = result["mean"]
        std = result["std"]

        df_mean.loc[len(df_mean)] = [dens, mean["mean_aspect_ratio"], mean["fraction_elongated"], mean["fraction_round"], mean["fraction_binary"]]
        df_std.loc[len(df_std)] = [dens, std["mean_aspect_ratio"], std["fraction_elongated"], std["fraction_round"], std["fraction_binary"]]

    # Número de muestras (para barras de error del promedio)
    N = number_of_realizations
    error_bars = df_std / np.sqrt(N)

    # ---------- GRAFICO 1: Medias con barras de error ----------
    plt.figure()
    fig, ax1 = plt.subplots()

    ax1.errorbar(df_mean["density"], df_mean["fraction_elongated"], 
                yerr=error_bars["fraction_elongated_std"], fmt='o-', color="red", label="Fraction of elongated cells")
    ax1.errorbar(df_mean["density"], df_mean["fraction_round"], 
                yerr=error_bars["fraction_round_std"], fmt='o-', color="blue", label="Fraction of round cells")
    ax1.errorbar(df_mean["density"], df_mean["fraction_binary"], 
                yerr=error_bars["fraction_binary_std"], fmt='o-', color="green", label="Fraction of binary cells")

    ax1.set_xlabel("Density ρ")
    ax1.set_ylabel("Fraction of each phenotype")

    ax2 = ax1.twinx()
    ax2.errorbar(df_mean["density"], df_mean["mean_aspect_ratio"], 
                yerr=error_bars["mean_aspect_ratio_std"], fmt='o-', color="orange", label="Mean aspect ratio")
    ax2.set_ylabel("Mean aspect ratio", color="orange")
    ax2.tick_params(axis='y', labelcolor="orange")

    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2, loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=2, fontsize=8)

    fig.subplots_adjust(bottom=0.2)
    plt.title(f"Steady State local Cell Phenotypes, box length={box_length}")
    plt.savefig(f"graphs/steady_state_local_cell_phenotype/steady_state_local_phenotype_box_length={box_length}.jpg", dpi=600)
    plt.close()

    # ---------- GRAFICO 2: Varianzas ----------
    plt.figure(figsize=(6, 4))
    plt.plot(df_mean["density"], df_std["fraction_elongated_std"]**2, label="Var(fraction elongated)", marker='o', color="red")
    plt.plot(df_mean["density"], df_std["fraction_round_std"]**2, label="Var(fraction round)", marker='o', color="blue")
    plt.plot(df_mean["density"], df_std["fraction_binary_std"]**2, label="Var(fraction binary)", marker='o', color="green")
    plt.plot(df_mean["density"], df_std["mean_aspect_ratio_std"]**2, label="Var(mean aspect ratio)", marker='o', color="orange")

    plt.xlabel("Density ρ")
    plt.ylabel("Variance")
    plt.title(f"Variance of local Phenotypes vs Density, box length={box_length}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"graphs/steady_state_local_cell_phenotype/steady_state_local_phenotype_variance_box_length={box_length}.jpg", dpi=600)
    plt.close()
