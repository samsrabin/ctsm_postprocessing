"""
Calculate some extra area, prod, yield, etc. variables
"""

# TODO: Move this to CropCase?
from __future__ import annotations

import numpy as np
import xarray as xr

from .combine_cft_to_crop import get_cft_crop_da, get_all_cft_crop_das, combine_cft_to_crop


def extra_area_prod_yield_etc(crops_to_include, case, case_ds):
    """
    Calculate some extra area, prod, yield, etc. variables
    """

    # Calculate CFT area
    case_ds["cft_area"] = case_ds["pfts1d_gridcellarea"] * case_ds["pfts1d_wtgcell"]
    case_ds["cft_area"] *= 1e6  # Convert km2 to m2
    case_ds["cft_area"].attrs["units"] = "m2"
    crop_cft_area_da = get_all_cft_crop_das(crops_to_include, case, case_ds, "cft_area")

    # Calculate CFT production
    case_ds["cft_prod"] = case_ds["YIELD_ANN"] * case_ds["cft_area"]
    crop_cft_prod_da = get_all_cft_crop_das(crops_to_include, case, case_ds, "cft_prod")

    # Convert/set units
    crop_cft_prod_da.attrs["units"] = "g"

    # Save cft_crop variable
    cft_crop_da = get_cft_crop_da(crops_to_include, case, case_ds)
    case_ds["cft_crop"] = cft_crop_da

    # Add crop_cft_* variables to case_ds
    case_ds["crop_cft_area"] = crop_cft_area_da
    case_ds["crop_cft_prod"] = crop_cft_prod_da

    # Calculate CFT-level yield
    case_ds = _get_yield_and_croplevel_stats(case_ds)

    # Area harvested
    case_ds = _harvest_area_stats(case_ds)

    # Drop things we don't need anymore
    case_ds = case_ds.drop_vars(["cft_area", "cft_prod"])

    return case_ds


def _get_yield_and_croplevel_stats(case_ds):
    """
    Calculate yield, then consolidate CFT-level stats to crop-level
    """
    case_ds["crop_cft_yield"] = case_ds["crop_cft_prod"] / case_ds["crop_cft_area"]
    case_ds["crop_cft_yield"].attrs["units"] = (
        case_ds["crop_cft_prod"].attrs["units"] + "/" + case_ds["crop_cft_area"].attrs["units"]
    )

    # Collapse CFTs to individual crops
    case_ds = combine_cft_to_crop(case_ds, "crop_cft_area", "crop_area", "sum", keep_attrs=True)
    case_ds = combine_cft_to_crop(case_ds, "crop_cft_prod", "crop_prod", "sum", keep_attrs=True)

    # Calculate crop-level yield
    case_ds["crop_yield"] = case_ds["crop_prod"] / case_ds["crop_area"]
    case_ds["crop_yield"].attrs["units"] = (
        case_ds["crop_prod"].attrs["units"] + "/" + case_ds["crop_area"].attrs["units"]
    )

    return case_ds


def _harvest_area_stats(case_ds):
    hr = case_ds["HARVEST_REASON_PERHARV"]
    cft_planted_area = (case_ds["pfts1d_gridcellarea"] * case_ds["pfts1d_wtgcell"]).where(
        case_ds["pfts1d_wtgcell"] > 0,
    ) * 1e6  # convert km2 to m2
    cft_planted_area.attrs["units"] = "m2"
    case_ds["cft_harv_area"] = (cft_planted_area * (hr > 0)).sum(dim="mxharvests")
    case_ds["cft_harv_area_immature"] = (cft_planted_area * (hr > 1)).sum(
        dim="mxharvests",
    )
    case_ds["cft_harv_area_failed"] = (
        cft_planted_area * (1 - case_ds["VALID_HARVEST"]).where(hr > 0)
    ).sum(dim="mxharvests")
    case_ds["crop_harv_area"] = (
        case_ds["cft_harv_area"]
        .groupby(case_ds["cft_crop"])
        .sum(dim="cft")
        .rename({"cft_crop": "crop"})
    )
    case_ds["crop_harv_area_immature"] = (
        case_ds["cft_harv_area_immature"]
        .groupby(case_ds["cft_crop"])
        .sum(dim="cft")
        .rename({"cft_crop": "crop"})
    )
    case_ds["crop_harv_area_failed"] = (
        case_ds["cft_harv_area_failed"]
        .groupby(case_ds["cft_crop"])
        .sum(dim="cft")
        .rename({"cft_crop": "crop"})
    )

    return case_ds
