# oncostreams_analysis

Software for analyzing data from oncostream simulations.

## Repository structure

- **`data/`**: Contains all the simulation data.  
  Inside, there are folders named after the number of cells. Each contains subdirectories for different densities, following this structure:  
  - **`dat/`**: `.dat` files with the position, orientation, and aspect ratio of each cell for every step and seed.  
  - **`dat_order_parameters/`**: Information about polar and nematic order parameters and the fraction of elongated cells for each step and seed.  
  - Example files are kept only for reference due to the large amount of data.  
  - **`steady_state_images/`**A folder with images of the steady state for each density is also included.

- **`graphs/`**: Stores all the figures generated during the analysis.  

- **`other_scripts/`**: Contains ideas and draft scripts for potential future use.

## Analysis scripts

There are two types of scripts used to analyze the data:

### Python scripts (`.py`)
Generate lighter, intermediate data to simplify the analysis:
- **`dat_clusters.py`**:  
  Generates clustering data. For every file in `dat_labels`, it outputs a new file with the number of clusters and the size of the largest cluster for each step and seed.  
  Also generates `.csv` files with all cluster sizes to study the final cluster size distribution.  
- **`dat_local_parameters.py`**:  
  Computes local order parameters and phenotypes for each step and seed at a given density.  
  The `box_lengths` list can be adjusted to study correlation lengths.  
- **`dat_ph_aspect_ratio.py`**:  
  Calculates and saves the quantity of each phenotype for every step and seed.

### Jupyter notebooks (`.ipynb`)
Used to visualize and analyze the processed data:
- **`analysis_clusters.ipynb`**:  
  Studies clustering dynamics. First analyzes the time evolution of the number of clusters and the largest cluster size, then the final cluster size distribution for each density.  
- **`analysis_order_parameters.ipynb`**:  
  Studies polar and nematic order parameters. Plots their evolution in time and the steady-state values versus density.  
- **`analysis_phenotypes.ipynb`**:  
  Analyzes phenotype evolution over time and steady-state values versus density.  
- **`analysis_local.ipynb`**:  
  Analyzes local properties. Computes local order parameters and phenotypes for various box sizes and compares local order for different spatial scales.



