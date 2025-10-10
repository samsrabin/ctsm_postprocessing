"""
Trim a CTSM history file to a lat/lon bounding box
"""

import sys
import argparse
import os
import numpy as np
import xarray as xr

NEED_LON_0_360_MSG = "trim_latlon can only handle longitudes [0, 360)"


def parse_arguments():
    """
    Parse arguments
    """
    parser = argparse.ArgumentParser(description="Trim a dataset to a specified lat/lon box.")
    parser.add_argument("filename_in", type=str, help="Input filename")
    parser.add_argument("--radius", type=float, default=5, help="'Radius' of the box in degrees")
    parser.add_argument(
        "--center", type=float, nargs=2, default=[39, 265], help="Center of the box (lat, lon)"
    )
    parser.add_argument("--filename-out", type=str, help="Output filename")
    parser.add_argument(
        "--overwrite", action="store_true", help="Overwrite the output file if it exists"
    )
    args = parser.parse_args()
    return args


def main():
    # pylint: disable=missing-function-docstring

    # Argument parsing
    args = parse_arguments()
    filename_in = args.filename_in
    radius = args.radius
    center = args.center
    filename_out = args.filename_out
    overwrite = args.overwrite
    bnds_lat = center[0] + radius * np.array([-1, 1])
    bnds_lon = center[1] + radius * np.array([-1, 1])

    # Check that boundary longitudes are in [0, 360) format
    lon_min = np.min(bnds_lon)
    lon_max = np.max(bnds_lon)
    if lon_min < 0:
        raise NotImplementedError(f"Center lon - radius is {lon_min} < 0; {NEED_LON_0_360_MSG}")
    if lon_max >= 360:
        raise NotImplementedError(f"Center lon + radius is {lon_max} >= 360; {NEED_LON_0_360_MSG}")

    if not filename_out:
        filename_out = os.path.splitext(filename_in)[0] + "_trimmed.nc"
    if not overwrite and os.path.exists(filename_out):
        print(f"Error: {filename_out} already exists. Use --overwrite to overwrite it.")
        sys.exit(1)

    # Import
    ds = xr.open_dataset(filename_in, decode_timedelta=False)

    # Check that dataset longitudes are in [0, 360) format
    lon_min = np.min(ds["lon"].values)
    lon_max = np.max(ds["lon"].values)
    if lon_min < 0:
        raise NotImplementedError(f"Min dataset longitude {lon_min} < 0; {NEED_LON_0_360_MSG}")
    if lon_max >= 360:
        raise NotImplementedError(f"Max dataset longitude {lon_max} >= 360; {NEED_LON_0_360_MSG}")

    selection_dict = {}
    index1d_dict = {}

    # Get lat and lon indices
    selection_dict["lat"] = np.where(
        (ds["lat"].values >= bnds_lat[0]) & (ds["lat"].values <= bnds_lat[1])
    )[0]
    selection_dict["lon"] = np.where(
        (ds["lon"].values >= bnds_lon[0]) & (ds["lon"].values <= bnds_lon[1])
    )[0]

    # Get indices along each other dimension we care about
    our_dims = ["gridcell", "landunit", "column", "pft"]
    for dim in our_dims:
        # Find lat and lon variables
        da_lon = None
        for var in ds:
            if var.endswith("_lon") and (dim,) == ds[var].dims:
                da_lon = ds[var]
                da_lat = ds[var.replace("_lon", "_lat")]
                index1d_dict[dim] = var
                break
        if da_lon is None:
            raise RuntimeError(f"Unable to find lon/lat vars for {dim}")

        # Get cells in box along this dimension
        selection_dict[dim] = np.where(
            (da_lat >= bnds_lat[0])
            & (da_lat <= bnds_lat[1])
            & (da_lon >= bnds_lon[0])
            & (da_lon <= bnds_lon[1])
        )[0]
        if not selection_dict[dim].size:
            raise RuntimeError(f"Nothing in box along {dim} dimension")

    # Make output Dataset
    ds_out = xr.Dataset()
    ds_out.attrs = ds.attrs

    for var in ds:
        var_dims = ds[var].dims
        if any(dim in var_dims for dim in our_dims):
            for dim in var_dims:
                if dim not in our_dims:
                    continue
                ds_out[var] = ds[var].isel(
                    {
                        dim: selection_dict[dim],
                    }
                )
        elif set(var_dims) == set(("lat", "lon")):
            ds_out[var] = ds[var].sel(
                lat=slice(bnds_lat[0], bnds_lat[1] + 1e-6),
                lon=slice(bnds_lon[0], bnds_lon[1] + 1e-6),
            )
        else:
            ds_out[var] = ds[var]

    ds_out.attrs["trim_latlon_bnds_lat"] = str(bnds_lat)
    ds_out.attrs["trim_latlon_bnds_lon"] = str(bnds_lon)

    # Save the output dataset
    ds_out.to_netcdf(filename_out)
    print(f"Trimmed dataset saved to {filename_out}")


if __name__ == "__main__":
    main()
