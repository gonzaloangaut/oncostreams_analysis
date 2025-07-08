# oncostreams_analysis

Software for analyzing data from oncostream simulations

Description of directories:
  - `0_50`, `0_60`, `0_70`, `0_80`, `0_85`, `0_90`: Examples of how a density directory should be. Inside there is a folder called `dat` containing all the .dat files with the position, orientation and aspect ratio of each cell for each step and seed. There is also another folder, `dat_order_parameters`, that have information about the order parameters (polar and nematic) and the fraction of elongated cells for each step and seed. We have only keep one file in each folder as an example because of the huge amount of data.
  - `graphs`: Where all the graphs made in the analysis are saved.
  - `new`: Where new data can be stored before merging it with the global data.
  - `other scripts`: Ideas of scripts that can be used in the future.

There also are a huge variety of scripts to analyse the data. These are:
  - clusters.py
  - ...
