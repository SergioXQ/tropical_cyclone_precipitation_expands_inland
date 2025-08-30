import sys
import pandas as pd
import numpy as np
import xarray as xr
import os
import gc
import glob
from concurrent.futures import ProcessPoolExecutor, as_completed
from numba import njit, prange
from filelock import FileLock
import logging

# Constants and Paths
CSV_PATH = '../../data/tc_track/ibtracs.since1980.list.v04r01.csv'
DIST_GRID_PATH = '../../data/dist2land_files/mswep_790000_dist2land.nc'
NC_FOLDER = '../../data/precipitation/mswep'  # not provided in repo
INTERMEDIATE_SAVE_DIR = './intermediate_saves_790000/'
PRE_SAVE_DIR = './heavy_precipitation_790000/'
FINAL_SAVE_PATH = './intermediate_saves_790000/updated_TC_data_final.csv'
SAVE_INTERVAL = 2000
# NUM_WORKERS = os.cpu_count() - 1 or 1  # Dynamically set based on CPU cores
NUM_WORKERS = 8

# Define precipitation thresholds
thresholds = [1.5, 10, 20, 30, 40, 50]

# Define column name formats
columns = [
    'weighted_sum_land_', 'weight_land_', 
    'weighted_sum_ocean_', 'weight_ocean_', 
    'grid_num_land_', 'grid_num_ocean_'
]

# Ensure intermediate save directories exist
os.makedirs(INTERMEDIATE_SAVE_DIR, exist_ok=True)
os.makedirs(PRE_SAVE_DIR, exist_ok=True)

# Setup logging
logging.basicConfig(
    stream=sys.stdout,  # Log output directly to the console
    level=logging.INFO,
    format='%(asctime)s %(levelname)s:%(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

# Global variables for ProcessPoolExecutor
dist_grid_values_global = None
lat_grid_dist_global = None
lon_grid_dist_global = None

def initialize_worker(dist_grid_path):
    """
    Initialize global variables for each worker process.
    This function is called once per process.
    """
    global dist_grid_values_global, lat_grid_dist_global, lon_grid_dist_global
    dist_grid = xr.open_dataset(dist_grid_path)
    dist_grid = dist_grid.dist2land
    dist_grid = dist_grid[::-1]  # Align latitude and longitude
    dist_grid_values_global = dist_grid.values  # Extract as NumPy array
    lat_grid_dist_global = dist_grid['lat'].values
    lon_grid_dist_global = dist_grid['lon'].values
    dist_grid.close()
    del dist_grid
    gc.collect()

# NumPy vectorized Haversine function remains unchanged
def haversine_np(lat1, lon1, lat2, lon2):
    R = 6371.0  # Earth radius in kilometers
    lat1_rad = np.radians(lat1)
    lon1_rad = np.radians(lon1)
    lat2_rad = np.radians(lat2)
    lon2_rad = np.radians(lon2)

    delta_lat = lat2_rad - lat1_rad
    delta_lon = lon2_rad - lon1_rad

    a = np.sin(delta_lat / 2.0) ** 2 + \
        np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(delta_lon / 2.0) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    distance = R * c
    return distance

# Numba-optimized Haversine function remains unchanged
@njit(parallel=True)
def haversine_numba(lat1, lon1, lat2, lon2):
    R = 6371.0  # Earth radius in kilometers
    distance = np.empty(lat2.shape, dtype=np.float64)
    for i in prange(lat2.shape[0]):
        for j in prange(lat2.shape[1]):
            phi1 = lat1 * np.pi / 180.0
            lambda1 = lon1 * np.pi / 180.0
            phi2 = lat2[i, j] * np.pi / 180.0
            lambda2 = lon2[i, j] * np.pi / 180.0
            delta_phi = phi2 - phi1
            delta_lambda = lambda2 - lambda1
            a = np.sin(delta_phi / 2.0) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(delta_lambda / 2.0) ** 2
            c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
            distance[i, j] = R * c
    return distance

def load_and_preprocess_data(csv_path):
    """
    Load and preprocess the CSV data.
    """
    df = pd.read_csv(csv_path, keep_default_na=False, low_memory=False)[1:]
    df = df[df['USA_SSHS'] >= -1]
    df = df[['SID', 'SEASON', 'BASIN', 'ISO_TIME', 'USA_LAT', 'USA_LON', 'USA_WIND', 'USA_SSHS', 'DIST2LAND']].reset_index(drop=True)

    # Type conversion
    df[['USA_LAT', 'USA_LON']] = df[['USA_LAT', 'USA_LON']].astype(float)
    df[['SEASON', 'USA_WIND', 'USA_SSHS', 'DIST2LAND']] = df[['SEASON', 'USA_WIND', 'USA_SSHS', 'DIST2LAND']].astype(int)
    df['ISO_TIME'] = pd.to_datetime(df['ISO_TIME'])

    # Format datetime information for file path
    df = df.groupby('SID').filter(lambda g: not all(g['USA_SSHS'] == -1))
    df['year'] = df['ISO_TIME'].dt.year
    df['day_of_year'] = df['ISO_TIME'].dt.dayofyear.apply(lambda x: f"{x:03d}")
    df['hour'] = df['ISO_TIME'].dt.hour.apply(lambda x: f"{x:02d}")
    df['path'] = df.apply(lambda row: f"{row['year']}{row['day_of_year']}.{row['hour']}.nc", axis=1)
    df = df[df['hour'].astype(int) % 3 == 0].reset_index(drop=True)
    df = df[df['ISO_TIME'].dt.year <= 2023].reset_index(drop=True)
    # Uncomment the following line if you want to filter by distance to land
    # df = df[df['DIST2LAND'] <= 400].reset_index(drop=True)

    # Initialize columns with NaN
    for suffix in thresholds:
        for col in columns:
            df[f"{col}{suffix}"] = np.nan

    return df

def process_row(row):
    """
    Process a single row of the DataFrame.
    This function is intended to run in a separate process.
    """
    global dist_grid_values_global, lat_grid_dist_global, lon_grid_dist_global

    index = row['index']
    nc_file_path = os.path.join(NC_FOLDER, row['path'])
    result = {'index': index}

    if not os.path.exists(nc_file_path):
        # Return index with NaNs for all thresholds
        for suffix in thresholds:
            for col in columns:
                result[f"{col}{suffix}"] = np.nan
        return result

    try:
        # Open dataset and ensure it's closed after processing
        ds = xr.open_dataset(nc_file_path)
        da = ds.precipitation[0].loc[70.95:-71]
        tc_lat, tc_lon = row['USA_LAT'], row['USA_LON']
        lat = da['lat'].values
        lon = da['lon'].values

        # Calculate distance using numba-optimized Haversine
        lat_grid, lon_grid = np.meshgrid(lat, lon, indexing='ij')
        dist = haversine_numba(tc_lat, tc_lon, lat_grid, lon_grid)

        for suffix in thresholds:
            # ------------------ Land Precipitation Processing ------------------
            mask_land = (dist <= 500) & (da.values >= suffix) & (dist_grid_values_global < 0)
            if not np.any(mask_land):
                weighted_sum_land = np.nan
                weight_land = np.nan
                grid_num_land = np.nan
            else:
                pre_land = da.values.copy()
                pre_land[~mask_land] = np.nan
                weighted_sum_land = np.nansum(pre_land * dist_grid_values_global)
                weight_land = np.nansum(pre_land)
                grid_num_land = np.sum(~np.isnan(pre_land))

                # Extract valid precipitation and coordinates
                valid_mask_land = ~np.isnan(pre_land)
                valid_precipitation_land = pre_land[valid_mask_land]
                valid_lat_land = lat_grid[valid_mask_land]
                valid_lon_land = lon_grid[valid_mask_land]

                # Create Dataset for valid land data
                da_land_nonan = xr.Dataset(
                    {
                        'precipitation': ('points', valid_precipitation_land),
                    },
                    coords={
                        'lat': ('points', valid_lat_land),
                        'lon': ('points', valid_lon_land),
                    }
                )

                # Define filename and path
                filename_land = f"{row['SID']}_{row['year']}{row['day_of_year']}_{row['hour']}_land_{suffix}.nc"
                filepath_land = os.path.join(PRE_SAVE_DIR, filename_land)

                # Write to NetCDF (no lock needed as each process writes unique files)
                da_land_nonan.to_netcdf(filepath_land)

                # Clean up
                da_land_nonan.close()
                del da_land_nonan
                gc.collect()

            # ------------------ Ocean Precipitation Processing ------------------
            mask_ocean = (dist <= 500) & (da.values >= suffix) & (dist_grid_values_global > 0)
            if not np.any(mask_ocean):
                weighted_sum_ocean = np.nan
                weight_ocean = np.nan
                grid_num_ocean = np.nan
            else:
                pre_ocean = da.values.copy()
                pre_ocean[~mask_ocean] = np.nan
                weighted_sum_ocean = np.nansum(pre_ocean * dist_grid_values_global)
                weight_ocean = np.nansum(pre_ocean)
                grid_num_ocean = np.sum(~np.isnan(pre_ocean))

                # Extract valid precipitation and coordinates
                valid_mask_ocean = ~np.isnan(pre_ocean)
                valid_precipitation_ocean = pre_ocean[valid_mask_ocean]
                valid_lat_ocean = lat_grid[valid_mask_ocean]
                valid_lon_ocean = lon_grid[valid_mask_ocean]

                # Create Dataset for valid ocean data
                da_ocean_nonan = xr.Dataset(
                    {
                        'precipitation': ('points', valid_precipitation_ocean),
                    },
                    coords={
                        'lat': ('points', valid_lat_ocean),
                        'lon': ('points', valid_lon_ocean),
                    }
                )

                # Define filename and path
                filename_ocean = f"{row['SID']}_{row['year']}{row['day_of_year']}_{row['hour']}_ocean_{suffix}.nc"
                filepath_ocean = os.path.join(PRE_SAVE_DIR, filename_ocean)

                # Write to NetCDF (no lock needed as each process writes unique files)
                da_ocean_nonan.to_netcdf(filepath_ocean)

                # Clean up
                da_ocean_nonan.close()
                del da_ocean_nonan
                gc.collect()

            # Store results
            result[f'weighted_sum_land_{suffix}'] = weighted_sum_land
            result[f'weight_land_{suffix}'] = weight_land
            result[f'weighted_sum_ocean_{suffix}'] = weighted_sum_ocean
            result[f'weight_ocean_{suffix}'] = weight_ocean
            result[f'grid_num_land_{suffix}'] = grid_num_land
            result[f'grid_num_ocean_{suffix}'] = grid_num_ocean

        # Close the dataset
        ds.close()
        del ds
        gc.collect()

        return result

    except Exception as e:
        logging.error(f"Error processing index {index} (SID: {row['SID']}): {e}")
        # Return index with NaNs for all thresholds in case of error
        for suffix in thresholds:
            for col in columns:
                result[f"{col}{suffix}"] = np.nan
        return result

def update_dataframe(df, results):
    """
    Update the main DataFrame with the results from processing.
    """
    for res in results:
        index = res['index']
        for suffix in thresholds:
            df.at[index, f'weighted_sum_land_{suffix}'] = res.get(f'weighted_sum_land_{suffix}', np.nan)
            df.at[index, f'weight_land_{suffix}'] = res.get(f'weight_land_{suffix}', np.nan)
            df.at[index, f'weighted_sum_ocean_{suffix}'] = res.get(f'weighted_sum_ocean_{suffix}', np.nan)
            df.at[index, f'weight_ocean_{suffix}'] = res.get(f'weight_ocean_{suffix}', np.nan)
            df.at[index, f'grid_num_land_{suffix}'] = res.get(f'grid_num_land_{suffix}', np.nan)
            df.at[index, f'grid_num_ocean_{suffix}'] = res.get(f'grid_num_ocean_{suffix}', np.nan)

def save_intermediate(df, rows_processed, csv_lock):
    """
    Save intermediate results to CSV.
    """
    intermediate_save_path = os.path.join(INTERMEDIATE_SAVE_DIR, f"updated_TC_data_{rows_processed}.csv")
    with csv_lock:
        df.to_csv(intermediate_save_path, index=False)
    logging.info(f"Progress saved to {intermediate_save_path} (processed {rows_processed} rows)")

def get_last_saved_row(intermediate_save_dir):
    """
    Determine the last saved row from intermediate CSV files.
    """
    # Find all intermediate save files
    save_files = glob.glob(os.path.join(intermediate_save_dir, "updated_TC_data_*.csv"))
    if not save_files:
        return 0, None

    # Determine the latest save file by the number in the filename
    try:
        latest_file = max(
            save_files, 
            key=lambda x: int(os.path.splitext(os.path.basename(x))[0].split('_')[-1])
        )
        rows_processed = int(os.path.splitext(os.path.basename(latest_file))[0].split('_')[-1])
        logging.info(f"Continuing from row {rows_processed}, using file {latest_file}")

        # Load saved DataFrame
        saved_df = pd.read_csv(latest_file)

        return rows_processed, saved_df
    except Exception as e:
        logging.error(f"Error determining the latest save file: {e}")
        return 0, None

def process_and_save_parallel(df, save_interval=SAVE_INTERVAL, num_workers=NUM_WORKERS):
    """
    Process the DataFrame in parallel and save intermediate results.
    """
    csv_lock = FileLock(os.path.join(INTERMEDIATE_SAVE_DIR, "intermediate_save.lock"))

    # Determine starting point
    rows_processed, saved_df = get_last_saved_row(INTERMEDIATE_SAVE_DIR)
    if rows_processed > 0 and saved_df is not None:
        # Update main DataFrame with saved results
        df.update(saved_df)
        start_index = rows_processed
    else:
        start_index = 0

    total_rows = len(df)
    remaining_rows = total_rows - start_index

    logging.info(f"Starting processing from row {start_index} out of {total_rows} total rows.")

    results = []
    batch_size = save_interval  # Number of tasks to process before saving

    with ProcessPoolExecutor(max_workers=num_workers, initializer=initialize_worker, initargs=(DIST_GRID_PATH,)) as executor:
        # Create an iterator for the rows to process
        rows_iterator = df.iloc[start_index:].assign(index=df.index[start_index:]).to_dict(orient='records')

        # Submit tasks in batches to manage memory and avoid overloading
        future_to_index = {}
        for row in rows_iterator:
            future = executor.submit(process_row, row)
            future_to_index[future] = row['index']

            if len(future_to_index) >= batch_size:
                # Collect completed futures
                done, _ = as_completed(future_to_index), None
                for future in done:
                    result = future.result()
                    results.append(result)
                    if len(results) >= save_interval:
                        max_index = max(r['index'] for r in results)
                        update_dataframe(df, results)
                        save_intermediate(df, max_index + 1, csv_lock)
                        logging.info(f"Processed {max_index + 1} rows out of {total_rows}.")
                        results.clear()
                        break  # Exit to submit next batch

                # Clear completed futures
                future_to_index = {fut: idx for fut, idx in future_to_index.items() if not fut.done()}

        # Handle any remaining futures
        for future in as_completed(future_to_index):
            result = future.result()
            results.append(result)

        # Save any remaining results
        if results:
            max_index = max(r['index'] for r in results)
            update_dataframe(df, results)
            save_intermediate(df, max_index + 1, csv_lock)
            logging.info(f"Processed {max_index + 1} rows out of {total_rows}.")
            results.clear()
            gc.collect()

def main():
    """
    Main function to orchestrate the processing.
    """
    # Check if the final save file already exists to avoid overwriting
    if os.path.exists(FINAL_SAVE_PATH):
        logging.info(f"Final save file {FINAL_SAVE_PATH} already exists. To reprocess, please delete this file first.")
        return

    # Load and preprocess data
    df = load_and_preprocess_data(CSV_PATH)

    # Initialize locks
    csv_lock = FileLock(os.path.join(INTERMEDIATE_SAVE_DIR, "intermediate_save.lock"))
    nc_lock = FileLock(os.path.join(PRE_SAVE_DIR, "netcdf_save.lock"))  # Not used anymore as each process writes unique files

    # Process and save in parallel
    process_and_save_parallel(df, save_interval=SAVE_INTERVAL, num_workers=NUM_WORKERS)

    # Save final result with locking to ensure safe write
    with csv_lock:
        df.to_csv(FINAL_SAVE_PATH, index=False)
    logging.info(f"Final result saved to {FINAL_SAVE_PATH}")

if __name__ == "__main__":
    main()