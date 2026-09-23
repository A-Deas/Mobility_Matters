# Mobility Matters

This repository contains the code and supporting files associated with the manuscript, Mobility matters: Quantifying differences between static and dynamic PM2.5 exposure estimates using a simulation-based framework in Utah.

## Environment setup

The analysis was developed in Python using the Conda environment specified in `environment.yml`, created with Miniforge.

To create the environment, run:

```bash
conda env create -f environment.yml
```

Then activate the environment:

```bash
conda activate utah_abm
```

## Working directory

The analysis scripts use relative file paths and are intended to be run from the root of the repository.

Before running the scripts, navigate to the `Mobility_Matters` directory:

```text
.../Mobility_Matters/
```

## Repository structure

### `Simulations/`

Contains the scripts used to run the exposure simulation and calculate the exposure-difference metrics.

A separate script is included for imputing PM2.5 exposure during scheduled travel activities.

### `Plots/`

Contains the scripts used to reproduce the figures presented in the manuscript.

Supporting spatial files required for figure generation are available in both shapefile and GeoPackage formats.

The analysis code uses shapefiles by default. Users who prefer GeoPackage files will need to update the corresponding input paths in the plotting scripts.

### `Shapefiles/`

Contains the shapefiles used by the plotting scripts.

### `GeoPackages/`

Contains GeoPackage versions of the spatial files provided in `Shapefiles/`.

These files are included as an alternative spatial-data format for users who prefer GeoPackage.

### `Logs/`

Contains output logs and summary results from the exposure simulation and exposure-difference calculations.

### `Dataframe_Heads/`

Contains dataframe heads showing the basic structure of the datasets used throughout the workflow.

These files are intended to help users understand the data formats.

### `Wildfire_Investigation/`

Contains supporting analyses and results used to investigate the wildfire smoke event examined in the study.

### `Mobility_Agent_Type_Comparison/`

Contains the scripts and supporting outputs used to compare exposure-difference metrics across agent types and alongside mobility measures.

### `Health_Impact_Calculations/`

Contains the scripts and outputs used to calculate and compare modeled attributable fractions under the static and dynamic exposure-estimation methods.

## Basic workflow

A typical reproduction workflow is:

1. Create and activate the conda environment using `environment.yml`.

2. Navigate to the root `Mobility_Matters` directory.

3. Run the main simulation via `Simulation/simulation.py`.

4. Impute exposures during scheduled travel hours via `Simulation/simulation_impute_travel.py`.

5. Calculate the exposure-difference metrics via `Simulation/calculate_difference_metrics.py`.

6. Run the downstream analyses described above.

As the primary datasets are not included in this repository, the required data must be obtained separately before the full workflow can be reproduced.

## Data availability

The datasets used in this study are not publicly hosted due to their size and storage requirements. Researchers interested in accessing the data or reproducing the analyses are encouraged to contact the corresponding author, who can provide information regarding data access and availability upon reasonable request.
