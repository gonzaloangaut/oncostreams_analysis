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
    frenar = False
    steps_validos = []
    dens_folder = f"{dens:.2f}".replace(".", "_")
    for tic in range(100, max_step, step):
        # para cada paso. leemos el archivo
        # para cada paso de cada seed
        nematic_seed = np.array([])
        polar_seed = np.array([])
        nematic_2_seed = np.array([])
        polar_2_seed = np.array([])

        for seed in rng_seed:
            # leemos el archivo correspondiente
            dat_actual = f"{dens_folder}/dat_order_parameters/op_culture_initial_number_of_cells={num_cells}_density={dens}_force=Anisotropic_Grosmann_k=3.33_gamma=3_With_Noise_eta=0.033_With_Shrinking_rng_seed={seed}_step={tic:05}.dat"
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


            nematic = df_tic["nematic"].mean()
            polar = df_tic["polar"].mean()
            nematic_2 = df_tic["nematic_2"].mean()
            polar_2 = df_tic["polar_2"].mean()

            # y lo agregamos a la lista del seed
            nematic_seed = np.append(nematic_seed, nematic)
            polar_seed = np.append(polar_seed, polar)
            nematic_2_seed = np.append(nematic_2_seed, nematic_2)
            polar_2_seed = np.append(polar_2_seed, polar_2)
        if frenar is True:
            ultimo_step = tic
            print("Ultimo step = ", ultimo_step-step, " para dens = ", dens)
            break
        steps_validos.append(tic)
        # ahora calculamos el promedio de los parámetros para todas las seed
        nematic_mean = np.mean(nematic_seed)
        polar_mean = np.mean(polar_seed)
        nematic_2_mean = np.mean(nematic_2_seed)
        polar_2_mean = np.mean(polar_2_seed)

        nematic_order.append(nematic_mean)
        polar_order.append(polar_mean)
        nematic_2_order.append(nematic_2_mean)
        polar_2_order.append(polar_2_mean)
    return nematic_order, polar_order, nematic_2_order, polar_2_order, steps_validos


density = [0.4, 0.5, 0.6, 0.7, 0.8, 0.81, 0.82, 0.83, 0.84, 0.85, 0.86, 0.87, 0.88, 0.9]

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

os.makedirs("graphs/order_parameters", exist_ok=True)

for dens in density:
    nematic_order, polar_order, nematic_2_order, polar_2_order, steps = calculate_order_parameter(
        nc, max_step + 1, dens, step, rng_seed
    )

    df = pd.DataFrame({"steps": steps})
    df.insert(1, f"nematic_order_nc={nc}_rho={dens}", nematic_order)
    df.insert(2, f"polar_order_nc={nc}_rho={dens}", polar_order)
    df.insert(3, f"nematic_order_2_nc={nc}_rho={dens}", nematic_2_order)
    df.insert(4, f"polar_order_2_nc={nc}_rho={dens}", polar_2_order)
    
    # guardamos el dataframe
    #df.to_csv("order_parameters.csv", index=False)

    # Primer gráfico
    plt.figure()
    fig, ax1 = plt.subplots()

    # Eje primario (Q, P, f)
    ax1.plot(df["steps"], df[f"nematic_order_nc={nc}_rho={dens}"], color="blue", label="Nematic order (Q)", linewidth=1)
    ax1.plot(df["steps"], df[f"polar_order_nc={nc}_rho={dens}"], color="green", label="Polar order (P)", linewidth=1)

    ax1.set_xlabel("Steps")
    ax1.set_ylabel("Order parameters")
    ax1.tick_params(axis='y')


    # Leyenda combinada
    lines, labels = ax1.get_legend_handles_labels()
    ax1.legend(lines, labels)

    plt.title(f"Order parameters for ρ={dens}")
    plt.savefig(f"graphs/order_parameters/order_parameters_rho={dens:.2f}.jpg", dpi=600)
    plt.close()

    # Segundo gráfico
    plt.figure()
    fig, ax1 = plt.subplots()

    # Eje primario (Q̂, P̂, f)
    ax1.plot(df["steps"], df[f"nematic_order_2_nc={nc}_rho={dens}"], color="blue", label=r"Nematic order ($\hat{Q}$)", linewidth=1)
    ax1.plot(df["steps"], df[f"polar_order_2_nc={nc}_rho={dens}"], color="green", label=r"Polar order ($\hat{P}$)", linewidth=1)

    ax1.set_xlabel("Steps")
    ax1.set_ylabel("Order parameters")
    ax1.tick_params(axis='y')

    # Leyenda combinada
    lines, labels = ax1.get_legend_handles_labels()
    ax1.legend(lines, labels)

    plt.title(f"Order parameters (2) for ρ={dens}")
    plt.savefig(f"graphs/order_parameters/order_parameters_rho={dens:.2f}_2.jpg", dpi=600)
    plt.close()
