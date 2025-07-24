import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os


def calculate_local_order_parameter(num_cells, max_step, dens, step, rng_seed, box_length):
    """
    Calculates local order parameters (nematic, polar and their second-order variants)
    over multiple realizations for a given density and box length.

    Args:
        num_cells (int): Number of cells in the culture.
        max_step (int): Maximum simulation step to consider.
        dens (float): Density of the system.
        step (int): Time step interval to sample.
        rng_seed (array-like): Array of random seeds to loop over.
        box_length (float): Size of the local observation box.

    Returns:
        tuple: Lists of averaged nematic and polar order parameters, and their
               second-order versions, along with the list of valid steps.
    """
    nematic_order = []
    polar_order = []
    nematic_2_order = []
    polar_2_order = []
    steps_validos = []
    frenar = False

    dens_folder = f"{dens:.2f}".replace(".", "_")

    for tic in range(100, max_step, step):
        nematic_seed = []
        polar_seed = []
        nematic_2_seed = []
        polar_2_seed = []

        for seed in rng_seed:
            dat_actual = (
                f"{dens_folder}/dat_local_op/local_op_box_length={int(box_length)}"
                f"_culture_initial_number_of_cells={num_cells}_density={dens}"
                f"_force=Anisotropic_Grosmann_k=3.33_gamma=3_With_Noise_eta=0.033"
                f"_With_Shrinking_rng_seed={seed}_step={tic:05}.dat"
            )

            if not os.path.exists(dat_actual):
                print(f"[INFO] Último paso disponible: {tic - step} para densidad ρ = {dens}")
                frenar = True
                break

            df_tic = pd.read_csv(dat_actual)
            nematic_seed.append(df_tic["nematic"].mean())
            polar_seed.append(df_tic["polar"].mean())
            nematic_2_seed.append(df_tic["nematic_2"].mean())
            polar_2_seed.append(df_tic["polar_2"].mean())

        if frenar:
            break

        steps_validos.append(tic)
        nematic_order.append(np.mean(nematic_seed))
        polar_order.append(np.mean(polar_seed))
        nematic_2_order.append(np.mean(nematic_2_seed))
        polar_2_order.append(np.mean(polar_2_seed))

    return nematic_order, polar_order, nematic_2_order, polar_2_order, steps_validos


# === Main Execution ===
# density = [0.5, 0.6, 0.7, 0.8, 0.85, 0.9]
density = [0.4, 0.5, 0.6, 0.7, 0.8, 0.81, 0.82, 0.83, 0.84, 0.85, 0.86, 0.87, 0.88, 0.9]

nc = 10_000
cell_area = np.pi
max_step = 60_000
step = 100
delta_t = 0.05

number_of_realizations = 64
seed_1 = 0x87351080E25CB0FAD77A44A3BE03B491
rng_1 = np.random.default_rng(seed_1)

rng_seed = rng_1.integers(low=2**20, high=2**50, size=number_of_realizations)

os.makedirs("graphs/local_order_parameters", exist_ok=True)

box_lengths = [5, 10, 15, 20, 25, 30]
for dens in density:
    side = np.sqrt(nc * cell_area / dens)
    #box_lengths = [0.1 * side, 0.2 * side]
    dens_folder = f"{dens:.2f}".replace(".", "_")

    for length in box_lengths:
        nematic_order, polar_order, nematic_2_order, polar_2_order, steps = calculate_local_order_parameter(
            nc, max_step + 1, dens, step, rng_seed, length
        )

        df = pd.DataFrame({"steps": steps})
        df.insert(1, f"nematic_order_nc={nc}_rho={dens}", nematic_order)
        df.insert(2, f"polar_order_nc={nc}_rho={dens}", polar_order)
        df.insert(3, f"nematic_order_2_nc={nc}_rho={dens}", nematic_2_order)
        df.insert(4, f"polar_order_2_nc={nc}_rho={dens}", polar_2_order)

        # Guardar DataFrame
        # df.to_csv(
        #     f"graphs/local_order_parameters/op_data_rho={dens:.2f}_box_length={int(length)}.csv",
        #     index=False,
        # )

        # Primer gráfico (Q, P)
        fig, ax1 = plt.subplots()
        ax1.plot(df["steps"], df[f"nematic_order_nc={nc}_rho={dens}"], color="blue", label="Nematic order (Q)", linewidth=1)
        ax1.plot(df["steps"], df[f"polar_order_nc={nc}_rho={dens}"], color="green", label="Polar order (P)", linewidth=1)

        ax1.set_xlabel("Steps")
        ax1.set_ylabel("Order parameters")
        ax1.legend()
        plt.title(f"Order parameters for ρ={dens} and box length={int(length)}")
        plt.savefig(f"graphs/local_order_parameters/order_parameters_rho={dens:.2f}_box_length={int(length)}.jpg", dpi=600)
        plt.close()

        # Segundo gráfico (Q̂, P̂)
        fig, ax1 = plt.subplots()
        ax1.plot(df["steps"], df[f"nematic_order_2_nc={nc}_rho={dens}"], color="blue", label=r"Nematic order ($\hat{{Q}}$)", linewidth=1)
        ax1.plot(df["steps"], df[f"polar_order_2_nc={nc}_rho={dens}"], color="green", label=r"Polar order ($\hat{{P}}$)", linewidth=1)

        ax1.set_xlabel("Steps")
        ax1.set_ylabel("Order parameters")
        ax1.legend()
        plt.title(f"Order parameters (2) for ρ={dens} and box length={int(length)}")
        plt.savefig(f"graphs/local_order_parameters/order_parameters_rho={dens:.2f}_box_length={int(length)}_2.jpg", dpi=600)
        plt.close()
