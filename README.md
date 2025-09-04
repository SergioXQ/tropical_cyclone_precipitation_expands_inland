Tropical Cyclone Precipitation Expands Inland

Contributors: Qian Xiang, De-Hui Ouyang, Deng-xin He

This repository contains codes and data used for precipitation, tropical cyclone, and exposure analyses related to the manuscript. It is organized into separate directories for scripts, key datasets, and processed results.
Note: ERA5 and CMIP6 datasets, which are open access climate data, are not included here. Their sources are described in the Data Availability section of the manuscript.

Repository Structure

codes/
Contains Python scripts and notebooks used for data processing, calculations, and figure generation.
	1.	py_for_calculate/
	•	calculate_rain_dist2land/: Scripts for calculating precipitation at different landmass thresholds.
	•	cacu_rain.py: Counts the frequency of heavy rainfall (≥ 30 mm) within each grid.
	•	city_exposure_0.1.py: Calculates exposure levels using the urban ratio and rainfall frequency.
	•	fig2_cal.py: Code for reproducing CMIP6-related calculations for Figure 2.
	•	transfer_utf8.py: Adjusts grid coordinates.
	2.	figs.ipynb: Generates the main figures.
	3.	extended.ipynb: Generates the extended data figures.

key_data/
Contains core datasets required for calculation and validation. Subfolders include:
	1.	0.1rain_V2/: Urban exposure maps and exposure–landward trend maps for the study areas.
	2.	aftertreatment/: CSV files after data cleaning and adding distance-to-land information.
	3.	City/: Grid-based city ratio results for different years, along with average city ratios and weighted offshore distances for each of the three study areas.
	4.	dist2land_files/: Distance-to-land datasets.
	5.	exposure_0yu_0.1/: Exposure weights by distance from the shore in the three study areas.
	6.	flood/: Information on flood centers caused by typhoons, filtered with DFO data.
	7.	wrf_out/: Model output from WRF simulations.

## Full Dataset (Zenodo)

This dataset is publicly available on Zenodo:

**DOI:** [10.5281/zenodo.17048203](https://doi.org/10.5281/zenodo.17048203)

## How to use
Please download the dataset and extract the key_data/ folder into the same directory containing codes/

Tips to Avoid Confusion
1. In the original dataset, offshore distance over the ocean is positive and over land is negative. For clarity in the main text, this sign convention is reversed (multiplied by -1).
2. In certain tropical cyclone–related datasets, the North Atlantic (NA) basin was mistakenly exported as NaN values. However, since basin indices are not used in calculations, this does not affect any results.
