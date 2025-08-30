# compute_trends_multi_region.py
import os
import numpy as np
import pandas as pd
import xarray as xr
from scipy.stats import linregress
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp

# ===== Base data directories and model/scenario settings =====
BASE_DIR = "../data/CMIP6"
MODELS = [
    "ACCESS-CM2","ACCESS-ESM1-5","BCC-CSM2-MR","CESM2","CNRM-CM6-1",
    "CanESM5","E3SM-2-0","FGOALS-g3","GFDL-ESM4","GISS-E2-1-G",
    "HadGEM3-GC31-LL","IPSL-CM6A-LR","MIROC6","MRI-ESM2-0","NorESM2-LM",
]
SCENARIOS = ["historical","hist-GHG","hist-aer","hist-nat"]

TIME_START, TIME_END, TIME_END_CMIP6 = "1980-01-01","2023-12-31","2014-12-31"

# ===== Region configurations =====
# id: unique region name
# lat_min/lat_max, lon_min/lon_max: spatial domain
# plev_min/plev_max: pressure level range
# d2l_mins/d2l_max: distance-to-land thresholds
# peak_months: months to select
# points_w/points_e: polyline segments for masking (None if no polyline mask)
REGIONS = [
    dict(
        id="WNA",
        lat_min=13, lat_max=40,
        lon_min=-102, lon_max=-70,
        plev_min=1000, plev_max=700,
        d2l_mins=[-200], d2l_max=0,
        peak_months=[8,9,10],
        points_w=[(-105, 30), (-98, 18), (-92, 16.6)],
        points_e=[(-92, 16.6), (-90, 15), (-86, 14.2), (-75, 22)],
    ),
    dict(
        id="BOB",
        lat_min=8, lat_max=28,
        lon_min=77, lon_max=100,
        plev_min=1000, plev_max=700,
        d2l_mins=[-200], d2l_max=0,
        peak_months=[5,10,11],
        points_w=None, points_e=None,
    ),
    dict(
        id="WNP",
        lat_min=8, lat_max=33,
        lon_min=103.5, lon_max=125,
        plev_min=1000, plev_max=700,
        d2l_mins=[-200], d2l_max=0,
        peak_months=[7,8,9,10],
        points_w=None, points_e=None,
    ),
    dict(
        id="AUS",
        lat_min=-25, lat_max=-10,
        lon_min=110, lon_max=155,
        plev_min=1000, plev_max=700,
        d2l_mins=[-200], d2l_max=0,
        peak_months=[1,2,3],
        points_w=None, points_e=None,
    ),
    dict(
        id="EAF",
        lat_min=-25, lat_max=-10,
        lon_min=30, lon_max=42,
        plev_min=1000, plev_max=700,
        d2l_mins=[-200], d2l_max=0,
        peak_months=[1,2,3],
        points_w=None, points_e=None,
    ),
]

# ===== Reanalysis dataset configurations =====
REAN_CONFIGS = {
    "ERA5": {
        "data_path": "../data/reanalysis/era5_specific_humidity_1980-2023.nc",
        "d2l_path": "/mnt/data/precipitation/era5_dist2land.nc",
        "var_name": "q", "time_name": "valid_time",
        "lat_name": "latitude", "lon_name": "longitude",
        "plev_name": "pressure",
    },
    "MERRA-2": {
        "data_path": "../data/reanalysis/merra2_QV_198001-202312.nc4",
        "d2l_path": "/mnt/data/MERRA2/dist2land.nc",
        "var_name": "QV", "time_name": "time",
        "lat_name": "lat", "lon_name": "lon",
        "plev_name": "lev",
    },
}

# ===== Helpers: math & masking =====
def calc_trend_and_pvalue(series_like):
    """Calculate linear trend per decade and p-value."""
    y = np.asarray(series_like, dtype=float)
    x = np.arange(len(y), dtype=float)
    slope, _, _, p, _ = linregress(x, y)
    return slope * 10.0, p.round(2)

def cross_z_2d(vx, vy, wx, wy):
    """Z-component of 2D cross product (avoids NumPy 2D deprecation)."""
    return vx * wy - vy * wx

def is_north_of_polyline(lon, lat, line_points, lon_min, lon_max):
    """Check if point is on the 'north/left' side of a polyline within longitude bounds."""
    if (lon < lon_min) or (lon > lon_max):
        return False
    for (x0,y0),(x1,y1) in zip(line_points[:-1], line_points[1:]):
        vx, vy = float(x1-x0), float(y1-y0)
        wx, wy = float(lon-x0), float(lat-y0)
        if cross_z_2d(vx, vy, wx, wy) <= 0:
            return False
    return True

# Optional: cache for polyline masks to avoid recomputing
_LINE_MASK_CACHE = {}
def build_line_mask(lons, lats, points_w, points_e, w_bounds, e_bounds, region_id=None):
    """Create a mask for points north of west/east polylines. Returns all True if both polylines are None."""
    if points_w is None and points_e is None:
        return np.ones((len(lats), len(lons)), dtype=bool)

    key = None
    if region_id is not None:
        # key depends on region + grid coordinates
        key = (region_id, tuple(np.asarray(lons)), tuple(np.asarray(lats)))
        if key in _LINE_MASK_CACHE:
            return _LINE_MASK_CACHE[key]

    lon_grid, lat_grid = np.meshgrid(lons, lats)
    mask_w = np.zeros_like(lon_grid, dtype=bool)
    mask_e = np.zeros_like(lon_grid, dtype=bool)
    for i in range(lat_grid.shape[0]):
        for j in range(lon_grid.shape[1]):
            lo, la = float(lon_grid[i,j]), float(lat_grid[i,j])
            if points_w is not None:
                mask_w[i,j] = is_north_of_polyline(lo, la, points_w, *w_bounds)
            if points_e is not None:
                mask_e[i,j] = is_north_of_polyline(lo, la, points_e, *e_bounds)
    out = mask_w | mask_e
    if key is not None:
        _LINE_MASK_CACHE[key] = out
    return out

def prep_d2l_to_q(d2l, q, lon_name, lat_name, region):
    """
    Align dist2land to q grid (robust to differing original coord names):
    - detect original coord names
    - rename to desired names FIRST
    - normalize orientation and longitude range
    - spatially subset to region box
    """
    # Detect existing coord names on d2l
    src_lat = ('latitude' if 'latitude' in d2l.coords else
               'lat' if 'lat' in d2l.coords else list(d2l.dims)[-2])
    src_lon = ('longitude' if 'longitude' in d2l.coords else
               'lon' if 'lon' in d2l.coords else list(d2l.dims)[-1])

    # Rename first so downstream code consistently uses lat_name/lon_name
    if (src_lat != lat_name) or (src_lon != lon_name):
        d2l = d2l.rename({src_lat: lat_name, src_lon: lon_name})

    # Ensure decreasing latitude (north->south) like ERA5 convention in this workflow
    if d2l[lat_name][0] < d2l[lat_name][-1]:
        d2l = d2l.sortby(lat_name, ascending=False)

    # Normalize longitude to [-180, 180]
    if d2l[lon_name].max() > 180:
        d2l = d2l.assign_coords({lon_name: ((d2l[lon_name] + 180) % 360) - 180})
    d2l = d2l.sortby(lon_name)

    # Subset to region box
    d2l = d2l.sel(
        **{
            lat_name: slice(region["lat_max"], region["lat_min"]),
            lon_name: slice(region["lon_min"], region["lon_max"]),
        }
    )
    return d2l

# ===== Reanalysis processing =====
def process_reanalysis(dataset, region, d2l_min):
    """Process one reanalysis dataset for one region; returns (slope, pvalue)."""
    cfg = REAN_CONFIGS[dataset]
    ds = xr.open_dataset(cfg["data_path"])
    q = ds[cfg["var_name"]]

    # Normalize orientation and longitude of q
    if q[cfg["lat_name"]][0] < q[cfg["lat_name"]][-1]:
        q = q.sortby(cfg["lat_name"], ascending=False)
    if q[cfg["lon_name"]].max() > 180:
        q = q.assign_coords({cfg["lon_name"]: ((q[cfg["lon_name"]] + 180) % 360) - 180})
    q = q.sortby(cfg["lon_name"])

    # Select subdomain + mean over pressure levels
    q = q.sel(
        **{
            cfg["plev_name"]: slice(region["plev_min"], region["plev_max"]),
            cfg["lat_name"]: slice(region["lat_max"], region["lat_min"]),
            cfg["lon_name"]: slice(region["lon_min"], region["lon_max"]),
            cfg["time_name"]: slice(TIME_START, TIME_END),
        }
    ).mean(dim=cfg["plev_name"])

    # Select months and average to one value per year (by grouping 3–4 months)
    q = q.sel(**{cfg["time_name"]: getattr(q, cfg["time_name"]).dt.month.isin(region["peak_months"])})
    q = q.coarsen(**{cfg["time_name"]: len(region["peak_months"])}, coord_func={cfg["time_name"]: "min"}).mean()

    # Load dist2land and align to q
    d2l = xr.open_dataset(cfg["d2l_path"])["dist2land"]
    d2l = prep_d2l_to_q(d2l, q, cfg["lon_name"], cfg["lat_name"], region)

    # Interpolate d2l to q grid to ensure identical shapes
    d2l = d2l.interp({cfg["lat_name"]: q[cfg["lat_name"]], cfg["lon_name"]: q[cfg["lon_name"]]}, method="nearest")

    # Polyline mask (if polylines exist)
    if region["id"] == "WNA" and (region["points_w"] or region["points_e"]):
        mid = -92.0 # Fixed split point for WNA
        mid = max(min(mid, region["lon_max"]), region["lon_min"])
        w_bounds = (region["lon_min"], mid)
        e_bounds = (mid, region["lon_max"])
    else:
        # No polyline
        w_bounds = (region["lon_min"], region["lon_min"])
        e_bounds = (region["lon_max"], region["lon_max"])

    line_mask = build_line_mask(
        q[cfg["lon_name"]].values, q[cfg["lat_name"]].values,
        region["points_w"], region["points_e"], w_bounds, e_bounds, region_id=region["id"]
    )

    dist_mask = (d2l >= d2l_min) & (d2l <= region["d2l_max"])
    combined = xr.DataArray(line_mask, dims=(cfg["lat_name"], cfg["lon_name"])) & dist_mask

    q = q.where(combined)

    # Area mean -> series -> guard for empty/short
    q_mean = q.mean(dim=[cfg["lat_name"], cfg["lon_name"]])
    series = q_mean.to_series()
    series = series.replace([np.inf, -np.inf], np.nan).dropna()
    if len(series) >= 2:
        slope, p_value = calc_trend_and_pvalue(series)
    else:
        slope, p_value = np.nan, np.nan
    return slope, p_value

# ===== CMIP6 processing =====
def process_cmip6_model_scenario(model, scenario, region, d2l_min):
    """Process one CMIP6 model/scenario for one region; returns (slope, pval, err_str)."""
    try:
        hus_path = os.path.join(BASE_DIR, "DAMIP_hus", f"hus_{model}_{scenario}_1850-2014.nc")
        d2l_path = os.path.join('../data/dust2land_files/', f"{model}_dist2land.nc")
        hus_ds = xr.open_dataset(hus_path)
        d2l_ds = xr.open_dataset(d2l_path)

        hus = hus_ds["hus"]  # (time, plev, lat, lon)

        # Ensure increasing lat/lon order for consistent slicing
        if hus.lat[0] > hus.lat[-1]:
            hus = hus.sortby("lat")
        if hus.lon[0] > hus.lon[-1]:
            hus = hus.sortby("lon")

        # Handle 0–360 longitude grids if present
        if hus.lon.max() > 180:
            lon_min_adj = (region["lon_min"] + 360) % 360
            lon_max_adj = (region["lon_max"] + 360) % 360
            # Convert polylines if any
            points_w = None if region["points_w"] is None else [(x % 360, y) for x, y in region["points_w"]]
            points_e = None if region["points_e"] is None else [(x % 360, y) for x, y in region["points_e"]]
            if region["id"] == "WNA" and (points_w or points_e):
                mid = (-92.0 % 360.0)  # 268.0
                mid = max(min(mid, lon_max_adj), lon_min_adj)
                w_bounds = (lon_min_adj, mid)
                e_bounds = (mid, lon_max_adj)
            else:
                w_bounds = (lon_min_adj, lon_min_adj)
                e_bounds = (lon_max_adj, lon_max_adj)
        else:
            lon_min_adj, lon_max_adj = region["lon_min"], region["lon_max"]
            points_w, points_e = region["points_w"], region["points_e"]
            if region["id"] == "WNA" and (points_w or points_e):
                mid = -92.0
                mid = max(min(mid, lon_max_adj), lon_min_adj)
                w_bounds = (lon_min_adj, mid)
                e_bounds = (mid, lon_max_adj)
            else:
                w_bounds = (lon_min_adj, lon_min_adj)
                e_bounds = (lon_max_adj, lon_max_adj)

        # Subset in space and pressure, then vertical mean
        hus = hus.loc[TIME_START:TIME_END_CMIP6, 100000:70000, region["lat_min"]:region["lat_max"], lon_min_adj:lon_max_adj].mean(dim="plev")

        # Select months and average to one value per year (by grouping months)
        hus = hus.sel(time=hus.time.dt.month.isin(region["peak_months"]))
        hus = hus.coarsen(time=len(region["peak_months"]), boundary="trim", coord_func={"time": "min"}).mean()

        # Prepare dist2land, normalize and subset
        d2l = d2l_ds["dist2land"]
        # Harmonize longitude domain for d2l to match hus domain
        if d2l.lon.max() <= 180 and hus.lon.max() > 180:
            d2l = d2l.assign_coords(lon=((d2l.lon + 360) % 360))
        elif d2l.lon.max() > 180 and hus.lon.max() <= 180:
            d2l = d2l.assign_coords(lon=(((d2l.lon + 180) % 360) - 180))
        if d2l.lat[0] > d2l.lat[-1]:
            d2l = d2l.sortby("lat")
        d2l = d2l.sortby("lon")

        d2l = d2l.sel(lat=slice(region["lat_min"], region["lat_max"]), lon=slice(lon_min_adj, lon_max_adj))

        # Interpolate dist2land exactly onto the model grid to avoid size mismatches
        d2l = d2l.interp(lat=hus.lat, lon=hus.lon, method="nearest")

        # Masks: distance + polyline
        dist_mask = (d2l >= d2l_min) & (d2l <= region["d2l_max"])
        line_mask = build_line_mask(hus.lon.values, hus.lat.values, points_w, points_e, w_bounds, e_bounds, region_id=region["id"])
        combined = dist_mask & xr.DataArray(line_mask, dims=("lat","lon"))

        # Yearly mean -> apply mask -> spatial mean per year with guards
        hus_yearly = hus.groupby("time.year").mean("time")
        mean_values = []
        for yr in hus_yearly.year.values:
            yearly = hus_yearly.sel(year=int(yr)).where(combined).where(lambda x: x >= 0)
            # If no valid cells this year, append NaN
            if yearly.count().sum().item() == 0:
                mean_values.append(np.nan)
            else:
                mean_values.append(float(yearly.mean(dim=("lat","lon"), skipna=True).values))

        # Drop NaNs and require at least 2 points for regression
        y = np.asarray(mean_values, dtype=float)
        y = y[np.isfinite(y)]
        if y.size >= 2:
            slope, pval = calc_trend_and_pvalue(y)
        else:
            slope, pval = np.nan, np.nan

        return slope, pval, ""

    except Exception as e:
        return np.nan, np.nan, f"{type(e).__name__}: {e}"

# ===== Parallel wrappers =====
def _cmip6_job(args):
    model, scenario, region, d2l_min = args
    slope, pval, err = process_cmip6_model_scenario(model, scenario, region, d2l_min)
    return {
        "region": region["id"], "model": model, "scenario": scenario,
        "d2l_min": d2l_min, "trend_per_decade": slope, "p_value": pval, "error": err
    }

def _rean_job(args):
    dataset, region, d2l_min = args
    slope, pval = process_reanalysis(dataset, region, d2l_min)
    return {
        "region": region["id"], "dataset": dataset,
        "trend_per_decade": slope, "p_value": pval, "d2l_min": d2l_min
    }

# ===== Main (parallel) driver =====
def run_and_save(cmip6_csv="../data/CMIP6/cmip6_trends.csv", rean_csv="../data/reanalysis/reanalysis_trends.csv"):
    # Build CMIP6 task list
    cmip6_tasks = []
    for region in REGIONS:
        for d2l_min in region["d2l_mins"]:
            for model in MODELS:
                for scenario in SCENARIOS:
                    cmip6_tasks.append((model, scenario, region, d2l_min))

    # Run CMIP6 in parallel
    rows = []
    max_workers = max(1, mp.cpu_count() - 1)
    with ProcessPoolExecutor(max_workers=max_workers) as ex:
        futures = [ex.submit(_cmip6_job, t) for t in cmip6_tasks]
        for fut in as_completed(futures):
            row = fut.result()
            if row.get("error"):
                print(f"[WARN] {row['region']} {row['model']}-{row['scenario']} -> {row['error']}")
            rows.append(row)
    pd.DataFrame(rows).to_csv(cmip6_csv, index=False)
    print(f"[OK] CMIP6 trends saved to {cmip6_csv}")

    # Build reanalysis task list
    rean_tasks = []
    for region in REGIONS:
        for d2l_min in region["d2l_mins"]:
            for dataset in ["ERA5","MERRA-2"]:
                rean_tasks.append((dataset, region, d2l_min))

    # Run reanalysis in parallel
    rrows = []
    with ProcessPoolExecutor(max_workers=max_workers) as ex:
        for row in ex.map(_rean_job, rean_tasks):
            rrows.append(row)
    pd.DataFrame(rrows).to_csv(rean_csv, index=False)
    print(f"[OK] Reanalysis trends saved to {rean_csv}")

if __name__ == "__main__":
    # (Optional) avoid over-threading inside BLAS/NetCDF libs
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    run_and_save()