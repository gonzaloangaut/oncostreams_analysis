import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os


def calculate_phenotype(num_cells, max_step, dens, step, rng_seed):
    # para cada paso temporal calculamos el valor de los parametros de orden
    fraction_elongated_total = []
    fraction_round_total = []
    fraction_binary_total = []
    mean_aspect_ratio_total = []
    frenar = False
    steps_validos = []
    dens_folder = f"{dens:.2f}".replace(".", "_")
    for tic in range(100, max_step, step):
        # para cada paso. leemos el archivo
        # para cada paso de cada seed
        fraction_elongated_cells_seed = np.array([])
        fraction_round_cells_seed = np.array([])
        fraction_binary_cells_seed = np.array([])
        mean_aspect_ratio_seed = np.array([])

        for seed in rng_seed:
            # leemos el archivo correspondiente
            dat_actual = f"{dens_folder}/dat_phenotype/phenotype_culture_initial_number_of_cells={num_cells}_density={dens}_force=Anisotropic_Grosmann_k=3.33_gamma=3_With_Noise_eta=0.033_With_Shrinking_rng_seed={seed}_step={tic:05}.dat"
            if os.path.exists(
                dat_actual
            ):
                df_tic = pd.read_csv(
                    dat_actual
                )
            else:
                frenar = True
                break

            mean_aspect_ratio = df_tic["mean_aspect_ratio"].mean()
            fraction_elongated = df_tic["fraction_elongated"].mean()
            fraction_round = df_tic["fraction_round"].mean()
            fraction_binary = df_tic["fraction_binary"].mean()

            # y lo agregamos a la lista del seed
            mean_aspect_ratio_seed = np.append(mean_aspect_ratio_seed, mean_aspect_ratio)
            fraction_elongated_cells_seed = np.append(fraction_elongated_cells_seed, fraction_elongated)
            fraction_round_cells_seed = np.append(fraction_round_cells_seed, fraction_round)
            fraction_binary_cells_seed = np.append(fraction_binary_cells_seed, fraction_binary)
        if frenar is True:
            ultimo_step = tic
            print("Ultimo step = ", ultimo_step-step, " para dens = ", dens)
            break
        steps_validos.append(tic)
        # ahora calculamos el promedio de los parámetros para todas las seed
        mean_aspect_ratio_mean = np.mean(mean_aspect_ratio_seed)
        fraction_elongated_mean = np.mean(fraction_elongated_cells_seed)
        fraction_round_mean = np.mean(fraction_round_cells_seed)
        fraction_binary_mean = np.mean(fraction_binary_cells_seed)

        mean_aspect_ratio_total.append(mean_aspect_ratio_mean)
        fraction_elongated_total.append(fraction_elongated_mean)
        fraction_round_total.append(fraction_round_mean)
        fraction_binary_total.append(fraction_binary_mean)
    return mean_aspect_ratio_total, fraction_elongated_total, fraction_round_total, fraction_binary_total, steps_validos


# density = [0.5, 0.6, 0.7, 0.8, 0.85, 0.9]
density = [0.85]

nc = 10_000
cell_area = np.pi
max_step = 60_000
step = 100
delta_t = 0.05

# todos los seed
number_of_realizations=64

seed_1 = 0x87351080E25CB0FAD77A44A3BE03B491
rng_1 = np.random.default_rng(seed_1)

rng_seed = rng_1.integers(
            low=2**20, high=2**50, size=number_of_realizations
        )

# rng_seed_1 = rng_1.integers(
#             low=2**20, high=2**50, size=number_of_realizations
#         )

# seed_2 = 1
# rng_2 = np.random.default_rng(seed_2)

# rng_seed_2 = rng_2.integers(
#             low=2**20, high=2**50, size=number_of_realizations
#         )

# rng_seed = np.concatenate((rng_seed_1, rng_seed_2))

os.makedirs("graphs/cell_phenotype", exist_ok=True)

for dens in density:
    mean_aspect_ratio, fraction_elongated, fraction_round, fraction_binary, steps = calculate_phenotype(
        nc, max_step + 1, dens, step, rng_seed
    )
    # df = pd.DataFrame(
    #     {"steps": np.array(list(range(0, max_step+step, step))) * delta_t}
    # )

    df = pd.DataFrame({"steps": steps})
    df.insert(1, f"mean_aspect_ratio_nc={nc}_rho={dens}", mean_aspect_ratio)
    df.insert(2, f"fraction_elongated_nc={nc}_rho={dens}", fraction_elongated)
    df.insert(3, f"fraction_round_nc={nc}_rho={dens}", fraction_round)
    df.insert(4, f"fraction_binary_nc={nc}_rho={dens}", fraction_binary)
    
    # guardamos el dataframe
    #df.to_csv("order_parameters.csv", index=False)

    # Graficamos
    plt.figure()
    fig, ax1 = plt.subplots()

    # Eje primario (Q, P, f)
    ax1.plot(df["steps"], df[f"fraction_elongated_nc={nc}_rho={dens}"], color="red", label="Fraction of elongated cells", linewidth=1)
    ax1.plot(df["steps"], df[f"fraction_round_nc={nc}_rho={dens}"], color="blue", label="Fraction of round cells", linewidth=1)
    ax1.plot(df["steps"], df[f"fraction_binary_nc={nc}_rho={dens}"], color="green", label="Fraction of binary cells", linewidth=1)

    ax1.set_xlabel("Steps")
    ax1.set_ylabel("Fraction of each phenotype")
    ax1.tick_params(axis='y')

    # Eje secundario (aspect ratio)
    ax2 = ax1.twinx()
    ax2.plot(df["steps"], df[f"mean_aspect_ratio_nc={nc}_rho={dens}"], color="orange", label="Mean aspect ratio", linewidth=1)
    ax2.set_ylabel("Mean aspect ratio", color="orange")
    ax2.tick_params(axis='y', labelcolor="orange")

    # Leyenda combinada
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2, loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=2, fontsize=8)

    fig.subplots_adjust(bottom=0.2)  # Asegura espacio suficiente debajo

    plt.title(f"Phenotype and aspect ratio for ρ={dens}")
    plt.savefig(f"graphs/cell_phenotype/phenotype_aspect_ratio_rho={dens:.2f}.jpg", dpi=600)
    plt.close()

