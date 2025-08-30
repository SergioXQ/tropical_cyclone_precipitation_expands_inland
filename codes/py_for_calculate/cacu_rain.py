import os
import numpy as np
import xarray as xr

# Country list and distance list
countries = ["USA", "South_Asia", "East_Asia", "Australia"]
distances = [400, 300, 200]

for country in countries:
    for distance in distances:
        # Input path
        copied_folder = f"./country/{distance}km/"  # Using f-string for path formatting
        output_folder = f"./country/{distance}km/yearly_output/"  # Output folder path

        # Create output folder (if it doesn’t exist)
        os.makedirs(output_folder, exist_ok=True)

        # Error file record list (re-initialized for each loop)
        error_files = []


        def load_full_grid(full_grid_file):
            """Load full grid data and return latitude/longitude arrays"""
            ds_full = xr.open_dataset(full_grid_file)
            grid_lat = ds_full.lat.values
            grid_lon = ds_full.lon.values
            ds_full.close()
            return grid_lat, grid_lon


        def regrid_compressed_dataset(ds_compressed, grid_lat, grid_lon):
            """
            Rebuild compressed data into a complete grid.
            Creates an array matching the size of the full grid and maps compressed data to the corresponding positions.
            """
            full_data = np.full((len(grid_lat), len(grid_lon)), np.nan, dtype=ds_compressed.precipitation.dtype)
            # Build mapping from lat/lon to index
            lat_to_idx = {lat: i for i, lat in enumerate(grid_lat)}
            lon_to_idx = {lon: j for j, lon in enumerate(grid_lon)}

            for idx in range(ds_compressed.dims['points']):
                lat_val = ds_compressed.lat.values[idx]
                lon_val = ds_compressed.lon.values[idx]
                i = lat_to_idx.get(lat_val)
                j = lon_to_idx.get(lon_val)
                if i is not None and j is not None:
                    full_data[i, j] = ds_compressed.precipitation.values[idx]

            ds_fullgrid = xr.Dataset(
                {'precipitation': (('lat', 'lon'), full_data)},
                coords={'lat': grid_lat, 'lon': grid_lon}
            )
            return ds_fullgrid


        # Get latitude/longitude information for the full grid
        grid_lat, grid_lon = load_full_grid('/mnt/data/precipitation/mswep/1984170.00.nc')

        # Group copied netCDF files by year (file name format: SID_YYYYDDD_HH_land_30.nc)
        year_files = {}
        for file_name in os.listdir(copied_folder):
            if file_name.endswith(".nc"):
                parts = file_name.split('_')
                if len(parts) < 3:
                    print(f"File name format is invalid, skipping: {file_name}")
                    continue
                year = parts[1][:4]  # First 4 characters are the year
                year_files.setdefault(year, []).append(file_name)

        # Process and accumulate data for each year
        for year in sorted(year_files.keys()):
            print(f"Processing year {year}, total {len(year_files[year])} files")
            accumulator = np.zeros((len(grid_lat), len(grid_lon)), dtype=int)
            for file_name in year_files[year]:
                full_file_path = os.path.join(copied_folder, file_name)
                try:
                    ds_compressed = xr.open_dataset(full_file_path)
                except Exception as e:
                    print(f"Failed to read file {file_name}: {e}")
                    error_files.append(file_name)
                    continue

                try:
                    ds_fullgrid = regrid_compressed_dataset(ds_compressed, grid_lat, grid_lon)
                    # Assign 1 if precipitation >= 30, otherwise 0, then accumulate
                    binary = (ds_fullgrid.precipitation.values >= 30).astype(int)
                    accumulator += binary
                except Exception as e:
                    print(f"Error processing file {file_name}: {e}")
                    error_files.append(file_name)
                finally:
                    ds_compressed.close()

            # Save the yearly accumulation result as a separate netCDF file
            output_file = os.path.join(output_folder, f"{country}_{distance}km_rain_{year}.nc")
            ds_year = xr.Dataset(
                {'precipitation': (('lat', 'lon'), accumulator)},
                coords={'lat': grid_lat, 'lon': grid_lon}
            )
            ds_year.to_netcdf(output_file)
            print(f"Year {year} precipitation data saved to {output_file}")

        # Output error files
        if error_files:
            print("The following files encountered errors during processing:")
            for err_file in error_files:
                print(" -", err_file)
        else:
            print("All files were successfully read and processed.")
