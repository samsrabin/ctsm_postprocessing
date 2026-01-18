"""
Module with functions for calculating crop biomass, ratios, etc.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import xarray as xr

try:
    # Attempt relative import if running as part of a package
    from .combine_cft_to_crop import combine_cft_to_crop
except ImportError:
    # Fallback to absolute import if running as a script
    sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    from crops.combine_cft_to_crop import combine_cft_to_crop


def _get_case_max_lai(cft_ds: xr.Dataset) -> xr.Dataset:
    if "MAX_TLAI_PERHARV" not in cft_ds:
        return cft_ds

    # Skip if Dataset already has both output variables
    out_var = "max_tlai"
    out_var_crop = out_var + "_crop"
    if all(v in cft_ds for v in [out_var, out_var_crop]):
        return cft_ds

    # List of variables to keep: Just what we need
    variables_to_keep = [
        "MAX_TLAI_PERHARV",
        "cft_crop",
        "pfts1d_ixy",
        "pfts1d_jxy",
        "cft_harv_area",
    ]

    # Get all data variables in the Dataset
    all_data_vars = list(cft_ds.data_vars.keys())

    # Drop everything except what's needed for gridding our result
    variables_to_drop = [var for var in all_data_vars if var not in variables_to_keep]
    tmp_ds = cft_ds.drop_vars(variables_to_drop)
    tmp_ds = tmp_ds.drop_vars("crop")

    da = tmp_ds["MAX_TLAI_PERHARV"].copy()
    da = da.where(da >= 0)
    da = da.mean(dim="mxharvests")
    assert not np.any(da < 0)

    # Combine CFTs to crops
    tmp_ds[out_var] = da
    tmp_ds = combine_cft_to_crop(
        tmp_ds,
        out_var,
        out_var_crop,
        method="mean",
        weights="cft_harv_area",
    )

    # This should be changed to happen automatically elsewhere!
    tmp_ds[out_var_crop].attrs["units"] = "m2/m2"

    cft_ds[out_var_crop] = tmp_ds[out_var_crop]

    return cft_ds


def get_crop_biomass_vars(cft_ds):
    """
    Get crop biomass variables for a case
    """
    case = _get_case_max_lai(cft_ds)

    return case


def get_caselist_crop_biomass_vars(case_list):
    """
    Loop through cases in CaseList, getting crop biomass variables for each
    """
    for case in case_list:
        case.cft_ds = get_crop_biomass_vars(case.cft_ds)

    return case_list
