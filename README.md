Tropical Cyclone Precipitation Expands Inland

Contributors: Qian Xiang, De-Hui Ouyang, Dengxin He

This repository contains codes and data used for precipitation, tropical cyclone, and exposure analyses related to the manuscript. It is organized into separate directories for scripts, key datasets, and processed results.
Note: ERA5 and CMIP6 datasets, which are open access climate data, are not included here. Their sources are described in the Data Availability section of the manuscript.

## Repository Structure

- **codes/**  
  Contains Python scripts and notebooks used for data processing, calculations, and figure generation.
  
  1. **py_for_calculate/**  
     - calculate_rain_dist2land/: Scripts for calculating precipitation at different landmass thresholds.  
     - cacu_rain.py: Counts the frequency of heavy rainfall (≥ 30 mm) within each grid.  
     - city_exposure_0.1.py: Calculates exposure levels using the urban ratio and rainfall frequency.  
     - fig2_cal.py: Code for reproducing CMIP6-related calculations for Figure 2.  
     - transfer_utf8.py: Adjusts grid coordinates.  
  2. **figs.ipynb**: Generates the main figures.  
  3. **extended.ipynb**: Generates the extended data figures.

- **data/**  
  Contains core datasets required for calculation and validation. Subfolders include:
  
  1. **0.1rain_V2/**: Urban exposure maps and exposure–landward trend maps for the study areas.  
  2. **aftertreatment/**: CSV files after data cleaning and adding distance-to-land information.  
  3. **City/**: Grid-based city ratio results for different years, along with average city ratios and weighted offshore distances for each of the three study areas.
  4. **CMIP6/**：Trend results from CMIP6 models.
  5. **dist2land_files/**: Distance-to-land datasets.  
  6. **exposure_0yu_0.1/**: Exposure weights by distance from the shore in the three study areas.
  7. **flood/**: Information on flood centers caused by typhoons, filtered with DFO data.  
  8. **gshhg-shp-2/**: Global coastline dataset.
  9. **reanalysis/**: Reanalysis data from ERA5 and MERRA-2
  10. **shp/**: Shp files for calculation.
  11. **tc_track/**: IBTrACS dataset.
  12. **wrf_out/**: Model output from WRF simulations.


## System Requirements
- **Python versions tested:** 3.8, 3.11  
- **Operating systems tested:** Ubuntu, macOS  
- **Dependencies:**  
  📦 os  
  📦 pandas  
  📦 numpy  
  📦 xarray  
  📦 geopandas  
  📦 seaborn  
  📦 matplotlib  
  📦 cartopy  
  📦 scipy  
  📦 shapely  
  📦 netCDF4  
  📦 pathlib  

---
## Installation Guide

1. Clone the repository from GitHub.  

2. Download the dataset (split into subfolders due to the large size of the original archive) from **Zenodo**:  
   [10.5281/zenodo.17048203](https://doi.org/10.5281/zenodo.17048203)

3. In the same parent directory as `codes/`, create a new folder named `data/`.  

4. Extract the downloaded archive(s) into the `data/` folder. The final structure should look like:  
	project_root/  
	├── codes/  
	└── data/  
		├── 0.1rain_V2/  
		├── aftertreatment/  
		└── …  

## How to use
Please download the dataset and extract the data/ folder into the same directory containing codes/  
	•	Open figs.ipynb or extended.ipynb in Jupyter Notebook.  
	•	Run the code blocks sequentially.  
	•	Most code blocks finish within 1 minute on a normal desktop computer.  

## Notes on Data
	•	Due to the large size of the raw data, we provide only partial subsets as examples:
	•	Figure 1 (d–f): only the WNA region data is provided (coastal_rainfall/).
	•	Extended Data Fig. 4 (a–c): only the WNA region reanalysis data is provided (reanalysis/).
	•	All other scripts and notebooks are fully reproducible with the provided datasets.
	•	For complete replication, download the full dataset from Zenodo. If the full download is too large for your network, you may selectively download only the required subfolders for verification.


## Tips to Avoid Confusion
1. In the original dataset, offshore distance over the ocean is positive and over land is negative. For clarity in the main text, this sign convention is reversed (multiplied by -1).
2. In certain tropical cyclone–related datasets, the North Atlantic (NA) basin was mistakenly exported as NaN values. However, since basin indices are not used in calculations, this does not affect any results.
