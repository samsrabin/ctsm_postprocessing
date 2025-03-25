"""
Utility functions for handling crop data in the Community Terrestrial Systems Model (CTSM).

This module provides various utility functions to assist with processing and analyzing crop data,
including functions for calculating harvest indices, converting units, and handling crop-specific
variables.
"""


def get_cft_ds(ds, cft):
    """
    Get a dataset for a specific crop functional type (CFT).

    Parameters:
    ds (xarray.Dataset): Dataset containing crop data.
    cft (xarray.DataArray): DataArray containing crop functional type information.

    Returns:
    xarray.Dataset: Dataset for the specified crop functional type.
    """
    ds = ds.isel(pft=cft.where)
    ds["cft"] = cft.pft_num
    ds = ds.set_coords("cft")
    for var in ds:
        if "pft" in ds[var].dims:
            ds[var] = ds[var].assign_coords(cft=cft.pft_num)
            ds[var] = ds[var].expand_dims("cft")
    return ds
