"""
Module with functions for calculating crop biomass, ratios, etc.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import xarray as xr

from .combine_cft_to_crop import combine_cft_to_crop

if TYPE_CHECKING:
    from .crop_case_list import CropCaseList


def _get_case_max_lai(cft_ds: xr.Dataset) -> xr.Dataset:
    """
    Calculate maximum LAI (Leaf Area Index) for crops from CFT-level data.

    This function processes the MAX_TLAI_PERHARV variable to create crop-level maximum LAI
    variables. It averages across harvests within a year and combines CFTs to crops using harvest
    area weights.

    Parameters
    ----------
    cft_ds : xarray.Dataset
        Dataset containing CFT-level data with MAX_TLAI_PERHARV variable.

    Returns
    -------
    xarray.Dataset
        Dataset with added max_tlai_crop variable, or unchanged if MAX_TLAI_PERHARV is not
        present or output variables already exist.
    """
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


def get_crop_biomass_vars(cft_ds: xr.Dataset) -> xr.Dataset:
    """
    Get crop biomass variables for a case.

    Currently calculates maximum LAI for crops. May be extended to include additional
    biomass-related variables in the future.

    Parameters
    ----------
    cft_ds : xarray.Dataset
        Dataset containing CFT-level data.

    Returns
    -------
    xarray.Dataset
        Dataset with added crop biomass variables.
    """
    case = _get_case_max_lai(cft_ds)

    return case


def get_caselist_crop_biomass_vars(case_list: CropCaseList) -> CropCaseList:
    """
    Loop through cases in CropCaseList, getting crop biomass variables for each.

    Parameters
    ----------
    case_list : CropCaseList
        List of CropCase objects to process.

    Returns
    -------
    CropCaseList
        The same CropCaseList with updated cft_ds attributes containing crop biomass variables.
    """
    for case in case_list:
        case.cft_ds = get_crop_biomass_vars(case.cft_ds)

    return case_list
