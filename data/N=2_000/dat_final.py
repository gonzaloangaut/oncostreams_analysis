import os
import shutil
import pandas as pd
import numpy as np

# Suponiendo que ya tienes estas variables definidas
list_density = [0.81, 0.82, 0.83, 0.84, 0.86]  # ejemplo de densidades
num_cells = 2_000  # número de células
max_step = 60_000  # paso máximo
step = 100  # intervalo de pasos
delta_t = 0.05

# All the seeds
number_of_realizations=64

seed_1 = 0x87351080E25CB0FAD77A44A3BE03B491
rng_1 = np.random.default_rng(seed_1)

rng_seed = rng_1.integers(
            low=2**20, high=2**50, size=number_of_realizations
        )



for dens in list_density:
    dens_folder = f"{dens:.2f}".replace(".", "_")
    # Crear carpeta 'dat_finales' si no existe
    os.makedirs(f"{dens_folder}/dat_final", exist_ok=True)
    for seed in rng_seed:
        last_step = None
        last_file = None
        
        for tic in range(100, max_step, step):
            # Formato de nombre de archivo
            dat_actual = (
                f"{dens_folder}/dat/culture_initial_number_of_cells={num_cells}_density={dens}_"
                f"force=Anisotropic_Grosmann_k=3.33_gamma=3_With_Noise_eta=0.033_With_Shrinking_"
                f"rng_seed={seed}_step={tic:05}.dat"
            )
            
            if os.path.exists(dat_actual):
                last_step = tic
                last_file = dat_actual
        
        # Si encontramos el último archivo para esta densidad y semilla, copiarlo a 'dat_finales'
        if last_file:
            # Definir el nombre final y la ruta
            final_file_path = f"{dens_folder}/dat_final/{os.path.basename(last_file)}"
            shutil.copy(last_file, final_file_path)
            print(f"Archivo final para densidad {dens}, seed {seed}, paso {last_step}: {final_file_path}")
        else:
            print(f"No se encontró archivo para densidad {dens}, seed {seed}")
