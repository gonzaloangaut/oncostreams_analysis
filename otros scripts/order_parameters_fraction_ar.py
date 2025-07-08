import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os


def calculate_order_parameter(num_cells, max_step, dens, step, rng_seed):    
    # para cada paso temporal calculamos el valor de los parametros de orden
    nematic_order = []
    polar_order = []
    nematic_2_order = []
    polar_2_order = []
    fraction_elongated = []
    frenar = False
    for tic in range(0, max_step, step):
        # para cada paso. leemos el archivo
        # para cada paso de cada seed
        nematic_seed = np.array([])
        polar_seed = np.array([])
        nematic_2_seed = np.array([])
        polar_2_seed = np.array([])
        fraction_elongated_cells_seed = np.array([])

        for seed in rng_seed:
            # leemos el archivo correspondiente
            dat_actual = f"{dens}/dat_order_parameters/op_culture_initial_number_of_cells={num_cells}_density={dens}_force=Anisotropic_Grosmann_k=3.33_gamma=3_With_Noise_eta=0.033_With_Shrinking_rng_seed={seed}_step={tic:05}.dat"
            if os.path.exists(
                dat_actual
            ):
                df_tic = pd.read_csv(
                    dat_actual
                )
            else:
                #print("No existe el archivo: ", dat_actual)
                frenar = True
                break


            nematic = df_tic["nematic"]
            polar = df_tic["polar"]
            nematic_2 = df_tic["nematic_2"]
            polar_2 = df_tic["polar_2"]
            fraction_elongated_cells = df_tic["fraction_elongated"]

            # y lo agregamos a la lista del seed
            nematic_seed = np.append(nematic_seed, nematic)
            polar_seed = np.append(polar_seed, polar)
            nematic_2_seed = np.append(nematic_2_seed, nematic_2)
            polar_2_seed = np.append(polar_2_seed, polar_2)
            fraction_elongated_cells_seed = np.append(fraction_elongated_cells_seed, fraction_elongated_cells)
        if frenar is True:
            ultimo_step = tic
            print("Ultimo step = ", ultimo_step, " para dens = ", dens)
            break
        # ahora calculamos el promedio de los parámetros para todas las seed
        nematic_mean = np.mean(nematic_seed)
        polar_mean = np.mean(polar_seed)
        nematic_2_mean = np.mean(nematic_2_seed)
        polar_2_mean = np.mean(polar_2_seed)
        fraction_elongated_mean = np.mean(fraction_elongated_cells_seed)

        nematic_order.append(nematic_mean)
        polar_order.append(polar_mean)
        nematic_2_order.append(nematic_2_mean)
        polar_2_order.append(polar_2_mean)
        fraction_elongated.append(fraction_elongated_mean)
    return nematic_order, polar_order, nematic_2_order, polar_2_order, fraction_elongated


density = [0.5, 0.6, 0.7, 0.8]

nc = 10_000
cell_area = np.pi
max_step = 20_000
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

for dens in density:
    nematic_order, polar_order, nematic_2_order, polar_2_order, fraction_elongated = calculate_order_parameter(
        nc, max_step + 1, dens, step, rng_seed
    )
    # df = pd.DataFrame(
    #     {"steps": np.array(list(range(0, max_step+step, step))) * delta_t}
    # )
    # Tiempo real según longitud de resultados
    tiempo = np.array(range(0, step * len(nematic_order), step))

    df = pd.DataFrame({"steps": tiempo})
    df.insert(1, f"nematic_order_nc={nc}_rho={dens}", nematic_order)
    df.insert(2, f"polar_order_nc={nc}_rho={dens}", polar_order)
    df.insert(3, f"nematic_order_2_nc={nc}_rho={dens}", nematic_2_order)
    df.insert(4, f"polar_order_2_nc={nc}_rho={dens}", polar_2_order)
    df.insert(5, f"fraction_elongated_nc={nc}_rho={dens}", fraction_elongated)
    
    # Leer relación de aspecto media
    media_ar_df = pd.read_csv(f"{dens}/mean_ar_results/mean_ar_nc={nc}_rho={dens}.csv", index_col="step")

    # Alinear los valores con los pasos del tiempo
    aspect_ratios = [
        media_ar_df.loc[s, "mean_aspect_ratio"] if s in media_ar_df.index else None
        for s in tiempo
    ]

    df.insert(6, f"mean_ar_nc={nc}_rho={dens}", aspect_ratios)

    # guardamos el dataframe
    #df.to_csv("order_parameters.csv", index=False)

    # Primer gráfico
    plt.figure()
    fig, ax1 = plt.subplots()

    # Eje primario (Q, P, f)
    ax1.plot(df["steps"], df[f"fraction_elongated_nc={nc}_rho={dens}"], color="red", label="Fraction of elongated cells (f)", linewidth=1)
    ax1.plot(df["steps"], df[f"nematic_order_nc={nc}_rho={dens}"], color="blue", label="Nematic order (Q)", linewidth=1)
    ax1.plot(df["steps"], df[f"polar_order_nc={nc}_rho={dens}"], color="green", label="Polar order (P)", linewidth=1)

    ax1.set_xlabel("Steps")
    ax1.set_ylabel("Order parameters")
    ax1.tick_params(axis='y')

    # Eje secundario (aspect ratio)
    ax2 = ax1.twinx()
    ax2.plot(df["steps"], df[f"mean_ar_nc={nc}_rho={dens}"], color="orange", label="Mean aspect ratio", linewidth=1)
    ax2.set_ylabel("Mean aspect ratio", color="orange")
    ax2.tick_params(axis='y', labelcolor="orange")

    # Leyenda combinada
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2)

    plt.title(f"Order parameters and aspect ratio for ρ={dens}")
    plt.savefig(f"graphs/parameters_ar_rho={dens:.2f}.jpg", dpi=600)
    plt.close()

    # Segundo gráfico
    plt.figure()
    fig, ax1 = plt.subplots()

    # Eje primario (Q̂, P̂, f)
    ax1.plot(df["steps"], df[f"fraction_elongated_nc={nc}_rho={dens}"], color="red", label="Fraction of elongated cells (f)", linewidth=1)
    ax1.plot(df["steps"], df[f"nematic_order_2_nc={nc}_rho={dens}"], color="blue", label=r"Nematic order ($\hat{Q}$)", linewidth=1)
    ax1.plot(df["steps"], df[f"polar_order_2_nc={nc}_rho={dens}"], color="green", label=r"Polar order ($\hat{P}$)", linewidth=1)

    ax1.set_xlabel("Steps")
    ax1.set_ylabel("Order parameters")
    ax1.tick_params(axis='y')

    # Eje secundario (aspect ratio)
    ax2 = ax1.twinx()
    ax2.plot(df["steps"], df[f"mean_ar_nc={nc}_rho={dens}"], color="orange", label="Mean aspect ratio", linewidth=1)
    ax2.set_ylabel("Mean aspect ratio", color="orange")
    ax2.tick_params(axis='y', labelcolor="orange")

    # Leyenda combinada
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2)

    plt.title(f"Order parameters (2) and aspect ratio for ρ={dens}")
    plt.savefig(f"graphs/parameters_ar_rho={dens:.2f}_2.jpg", dpi=600)
    plt.close()
