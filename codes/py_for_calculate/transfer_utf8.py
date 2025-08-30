import xarray as xr
import numpy as np
import os
from glob import glob

# Define country and distance lists
countrys = ["East_Asia", "South_Asia", "USA"]
dis = [500]

# Iterate over each country and distance
for country in countrys:
    for di in dis:
        # Input and output folder paths
        input_folder = f"./{country}/{di}km/yearly_output/"
        output_folder = f"./{country}/{di}km/rounded_output/"

        # Ensure output folder exists
        os.makedirs(output_folder, exist_ok=True)

        # Print input folder path and matched file list
        print("Input folder path:", input_folder)
        file_list = glob(os.path.join(input_folder, "*.nc"))
        print("Matched file list:", file_list)

        # Iterate over all .nc files in the input folder
        for file_path in file_list:
            file_name = os.path.basename(file_path)
            print(f"Processing file: {file_name}")

            # Read NetCDF file using with statement
            with xr.open_dataset(file_path) as rain_ds:
                # Print file content
                print("File content:", rain_ds)

                # Get original lat/lon and precipitation data
                rain_lat = rain_ds["lat"].values
                rain_lon = rain_ds["lon"].values
                rain_data = rain_ds["precipitation"].values  # 2D data (lat, lon)

                # Round lat/lon to two decimal places
                rain_lat_rounded = np.round(rain_lat, 2)
                rain_lon_rounded = np.round(rain_lon, 2)

                # Create new xarray Dataset
                reshaped_ds = xr.Dataset(
                    {
                        "precipitation": (["lat", "lon"], rain_data)  # Keep dimensions unchanged
                    },
                    coords={
                        "lat": rain_lat_rounded,  # Use rounded latitude
                        "lon": rain_lon_rounded  # Use rounded longitude
                    }
                )

                # Generate new filename and save
                output_file = os.path.join(output_folder, f"rounded_{file_name}")
                try:
                    reshaped_ds.to_netcdf(output_file)
                    print(f"Processing complete, saved: {output_file}")
                except Exception as e:
                    print(f"Error saving file: {e}")

        print("ok")
