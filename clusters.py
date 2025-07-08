import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

# Para cada densidad podemos hacer distintos gráficos (para cada tipo de cluster):
# - Cantidad de clusters a traves del tiempo
# - Tamaño de cluster más grande a través del tiempo
# - Distribución de tamaños de cluster en escala log-log en el estado final

def calculate_clusters_vs_time(num_cells, dens):
    """
    Function to calculate both the amount of clusters and the biggest cluster
    against time for a given density
    """
    # For each step we calculate both the number_clusters and biggest_cluster
    number_cluster = []
    biggest_cluster = []
    frenar = False
    steps_validos = []
    dens_folder = f"{dens:.2f}".replace(".", "_")
    for tic in range(100, max_step, step):
        # For each seed we read the file
        number_cluster_seed = np.array([])
        biggest_cluster_seed = np.array([])

        for seed in rng_seed:
            # Read the .dat
            dat_actual = f"{dens_folder}/dat_labels/culture_initial_number_of_cells={num_cells}_density={dens}_force=Anisotropic_Grosmann_k=3.33_gamma=3_With_Noise_eta=0.033_With_Shrinking_rng_seed={seed}_step={tic:05}.dat"
            if os.path.exists(
                dat_actual
            ):
                df_tic = pd.read_csv(
                    dat_actual
                )
            else:
                frenar = True
                break
            
            # Calculate parameters
            cantidad_distintos = df_tic['label'].nunique()
            frecuencia_maxima = df_tic['label'].value_counts().max()

            # Add them to the seed's list
            number_cluster_seed = np.append(number_cluster_seed, cantidad_distintos)
            biggest_cluster_seed = np.append(biggest_cluster_seed, frecuencia_maxima)
        if frenar is True:
            ultimo_step = tic-step
            print("Ultimo step = ", ultimo_step, " para dens = ", dens)
            break
        steps_validos.append(tic)
        # We calculate the mean for every seed in that tic
        cantidad_distintos_mean = np.mean(number_cluster_seed)
        frecuencia_maxima_mean = np.mean(biggest_cluster_seed)

        # Add them to the global list
        number_cluster.append(cantidad_distintos_mean)
        biggest_cluster.append(frecuencia_maxima_mean)
    return number_cluster, biggest_cluster, steps_validos




def calculate_distribution_cluster_size(num_cells, dens, max_step):
    """
    Function to calculate the distribution of the cluster size at the steady
    state
    """
    dens_folder = f"{dens:.2f}".replace(".", "_")
    frenar = False
    
    # Initialize the las step as the max
    ultimo_step = max_step
    # We see which is the last step where all the seeds exist
    for tic in range(100, max_step, step):
        for seed in rng_seed:
            dat_actual = f"{dens_folder}/dat_labels/culture_initial_number_of_cells={num_cells}_density={dens}_force=Anisotropic_Grosmann_k=3.33_gamma=3_With_Noise_eta=0.033_With_Shrinking_rng_seed={seed}_step={tic:05}.dat"
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
        dat_actual = f"{dens_folder}/dat_order_parameters/op_culture_initial_number_of_cells={num_cells}_density={dens}_force=Anisotropic_Grosmann_k=3.33_gamma=3_With_Noise_eta=0.033_With_Shrinking_rng_seed={seed}_step={ultimo_step:05}.dat"
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


# Define the density
density = 0.85
# and other relevant parameters
nc = 10_000
cell_area = np.pi
max_step = 60_000
step = 100
delta_t = 0.05

# All the seeds
number_of_realizations=64

seed_1 = 0x87351080E25CB0FAD77A44A3BE03B491
rng_1 = np.random.default_rng(seed_1)

rng_seed = rng_1.integers(
            low=2**20, high=2**50, size=number_of_realizations
        )


os.makedirs(f"graphs/clusters/{density}", exist_ok=True)


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
