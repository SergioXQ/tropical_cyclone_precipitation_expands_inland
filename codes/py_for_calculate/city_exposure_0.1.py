import xarray as xr
import numpy as np
import os

countrys = ["East_Asia", "South_Asia", "USA"]
dis = [200, 300, 400]

for country in countrys:
    for di in dis:
        # **Folder paths**
        rainfall_folder = f"./{country}/{di}km/rounded_output/"  # Rainfall files
        urban_ratio_folder = f"./Fig.3/FCS30D_city_ratio_V2/"  # Urban ratio / impervious surface ratio files
        output_folder = f"./{country}/{di}km/FCSV2_exposure_0yu_0.1rain"  # Output exposure NetCDF files

        # **Ensure output folder exists**
        os.makedirs(output_folder, exist_ok=True)

        # **Set urban ratio threshold**
        CITY_RATIO_THRESHOLD = 0

        # **Urban ratio file mapping (five-year group before 2000, yearly after 2000)**
        multi_year_mapping = {
            1985: range(1981, 1986),
            1990: range(1986, 1991),
            1995: range(1991, 1996),
            2000: range(1996, 2001)
        }

        # **Loop through 1980–2023, calculating urban exposure each year**
        for year in range(1981, 2024):
            # **Determine which urban ratio file to use**
            if year <= 2000:
                urban_year = max([k for k in multi_year_mapping.keys() if year in multi_year_mapping[k]])
            else:
                urban_year = year

            urban_file = os.path.join(urban_ratio_folder, f"{urban_year}-30m.nc")

            # **Ensure urban ratio file exists**
            if not os.path.exists(urban_file):
                print(f" Skipping {year}, urban ratio file {urban_file} does not exist!")
                continue

            # **Determine rainfall data file**
            rainfall_file = os.path.join(rainfall_folder, f"rounded_{country}_{di}km_rain_{year}.nc")

            # **Ensure rainfall file exists**
            if not os.path.exists(rainfall_file):
                print(f" Skipping {year}, rainfall file {rainfall_file} does not exist!")
                continue

            print(f" Calculating exposure for {year} (using urban ratio from {urban_year})...")

            # **Load urban ratio data (0.1° resolution)**
            urban_ds = xr.open_dataset(urban_file)
            urban_da = urban_ds["city_ratio"]

            # **Create mask, keeping only grids where urban ratio > threshold**
            city_mask = urban_da >= CITY_RATIO_THRESHOLD

            # **Load rainfall frequency data (also 0.1° resolution now)**
            rainfall_ds = xr.open_dataset(rainfall_file)
            rainfall_da = rainfall_ds["precipitation"]

            # **Ensure coordinates are aligned**
            rainfall_da = rainfall_da.reindex_like(urban_da, method='nearest')
            print(f" Note: Coordinates aligned for rainfall data of {year}")

            # **Calculate exposure — multiply directly since resolutions match**
            urban_exposure_da = urban_da.where(city_mask) * rainfall_da
            urban_exposure_da.name = "urban_exposure"
            urban_exposure_da.attrs["description"] = "Urban exposure (urban ratio × rainfall frequency)"
            urban_exposure_da.attrs["units"] = "unitless"

            # **Create dataset and save**
            output_file = os.path.join(output_folder, f"exposure_{year}_0.1.nc")
            exposure_ds = urban_exposure_da.to_dataset(name="urban_exposure")
            exposure_ds.to_netcdf(output_file)

            print(f" Urban exposure for {year} calculated and saved: {output_file}")

            # **Close datasets**
            urban_ds.close()
            rainfall_ds.close()

        print(" All years processed!")
