"""
Utility functions for handling crop data in the Community Terrestrial Systems Model (CTSM).

This module provides various utility functions to assist with processing and analyzing crop data,
including functions for calculating harvest indices, converting units, and handling crop-specific
variables.
"""

from __future__ import annotations

import xarray as xr

from .cft import Cft


def get_cft_ds(ds: xr.Dataset, cft: Cft) -> xr.Dataset:
    """
    Get a dataset for a specific crop functional type (CFT).

    This function extracts data for a specific CFT from a dataset, adds CFT coordinate information,
    and expands dimensions appropriately.

    Parameters
    ----------
    ds : xarray.Dataset
        Dataset containing crop data with pft dimension.
    cft : Cft
        Cft object containing crop functional type information.

    Returns
    -------
    xarray.Dataset
        Dataset for the specified crop functional type with cft dimension and coordinates.
    """
    ds = ds.isel(pft=cft.get_where(ds))
    ds["cft"] = cft.pft_num
    ds = ds.set_coords("cft")
    for var in ds:
        if "pft" in ds[var].dims:
            ds[var] = ds[var].assign_coords(cft=cft.pft_num)
            ds[var] = ds[var].expand_dims("cft")
    return ds
