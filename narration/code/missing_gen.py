import json
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr


def analyze_netcdf_file(file_path):
    """
    Analyze a single NetCDF file for missing values with daily granularity.
    """
    try:
        with xr.open_dataset(file_path, decode_times=True) as ds:
            var_name = Path(file_path).parent.name
            print(f"Processing {var_name} file: {Path(file_path).name}")

            # Find the main variable (excluding coordinates and metadata)
            exclude_vars = ['time', 'lat', 'lon', 'x', 'y', 'Lambert_Conformal',
                            'time_bnds', 'nbnds']
            main_vars = [v for v in ds.variables if v not in exclude_vars]

            if not main_vars:
                print(f"No main variables found in {file_path}")
                return None

            main_var = main_vars[0]
            data = ds[main_var].values

            # Get fill value or missing value
            fill_value = None
            if '_FillValue' in ds[main_var].attrs:
                fill_value = ds[main_var].attrs['_FillValue']
            elif 'missing_value' in ds[main_var].attrs:
                fill_value = ds[main_var].attrs['missing_value']

            # Initialize stats dictionary
            daily_stats = {}

            # Handle different data shapes
            if len(data.shape) == 3:  # time, y, x
                for time_idx in range(data.shape[0]):
                    time_slice = data[time_idx]

                    # Count missing values (either NaN or fill value)
                    missing_mask = np.isnan(time_slice)
                    if fill_value is not None:
                        missing_mask |= (time_slice == fill_value)

                    missing_count = np.count_nonzero(missing_mask)
                    total_points = time_slice.size

                    if missing_count > 0:
                        # Convert time to date string
                        date = pd.Timestamp(
                            ds.time.values[time_idx]).strftime('%Y-%m-%d')
                        daily_stats[date] = {
                            "missing_count": int(missing_count),
                            "total_points": int(total_points),
                            "missing_percentage": round(float(missing_count) / total_points * 100, 2)
                        }

            return {
                "variable": var_name,
                "filename": Path(file_path).name,
                "stats": daily_stats
            }

    except Exception as e:
        print(f"Error processing {file_path}: {str(e)}")
        return None


def create_daily_catalog(base_path):
    """
    Create a GitHub-style daily catalog of missing data.
    """
    catalog = {
        "metadata": {
            "created": datetime.now().isoformat(),
            "base_path": str(base_path),
            "variables_processed": []
        },
        "variables": {},
        "daily_summary": {}
    }

    # Process each variable folder
    for var_folder in ['air', 'hgt', 'omega', 'shum', 'tke', 'uwnd', 'vwnd']:
        folder_path = Path(base_path) / var_folder
        if not folder_path.exists():
            print(f"Folder not found: {folder_path}")
            continue

        print(f"\nProcessing {var_folder} folder...")
        catalog["metadata"]["variables_processed"].append(var_folder)
        catalog["variables"][var_folder] = {}

        # Process all NC files in the folder
        nc_files = list(folder_path.glob('*.nc'))
        with ProcessPoolExecutor() as executor:
            results = list(executor.map(analyze_netcdf_file, nc_files))

            # Aggregate results
            for result in results:
                if result is None:
                    continue

                var_name = result["variable"]
                daily_stats = result["stats"]

                # Update variable-specific stats
                catalog["variables"][var_name].update(daily_stats)

                # Update daily summary
                for date, stats in daily_stats.items():
                    if date not in catalog["daily_summary"]:
                        catalog["daily_summary"][date] = {
                            "total_missing": 0,
                            "variables_affected": set(),
                            "max_percentage": 0
                        }

                    summary = catalog["daily_summary"][date]
                    summary["total_missing"] += stats["missing_count"]
                    summary["variables_affected"].add(var_name)
                    summary["max_percentage"] = max(
                        summary["max_percentage"],
                        stats["missing_percentage"]
                    )

    # Convert sets to lists for JSON serialization
    for date in catalog["daily_summary"]:
        catalog["daily_summary"][date]["variables_affected"] = list(
            catalog["daily_summary"][date]["variables_affected"]
        )

    return catalog


def save_catalog(catalog, output_file='daily_missing_data_catalog.json'):
    with open(output_file, 'w') as f:
        json.dump(catalog, f, indent=2)


if __name__ == "__main__":
    base_path = "/media/bharxhav/valhalla/NARR/"
    print("Starting daily catalog creation...")
    catalog = create_daily_catalog(base_path)
    save_catalog(catalog)
    print("\nCatalog creation complete!")

    # Print summary
    print("\nProcessed variables:", catalog["metadata"]["variables_processed"])
    print("Total days with missing data:", len(catalog["daily_summary"]))
    for var in catalog["metadata"]["variables_processed"]:
        print(
            f"{var}: {len(catalog['variables'][var])} days with missing data")
