"""
Calculate some extra area, prod, yield, etc. variables
"""

# TODO: Move this to CropCase?
from __future__ import annotations

import re
import sys
import os

import numpy as np
import xarray as xr

try:
    from .combine_cft_to_crop import get_cft_crop_da, get_all_cft_crop_das, combine_cft_to_crop
    from .mark_crops_invalid import mark_crops_invalid
    from ..utils import food_grainc_to_harvested_tons_onecrop, ivt_int2str
except ImportError:
    # Fallback to absolute import if running as a script
    sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    from crops.combine_cft_to_crop import get_cft_crop_da, get_all_cft_crop_das, combine_cft_to_crop
    from crops.mark_crops_invalid import mark_crops_invalid
    from utils import food_grainc_to_harvested_tons_onecrop, ivt_int2str


def extra_area_prod_yield_etc(crops_to_include, case, case_ds):
    """
    Calculate some extra area, prod, yield, etc. variables
    """

    # Get crop products at various levels of maturity
    case_ds = _get_crop_products(case_ds)

    # Calculate CFT area
    case_ds["cft_area"] = case_ds["pfts1d_landarea"] * case_ds["pfts1d_wtgcell"]
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

    # cft_crop is often a groupby() variable, so computing it makes things more efficient.
    # Avoids DeprecationWarning that will become an error in xarray v2025.05.0+
    if hasattr(case_ds["cft_crop"].data, "compute"):
        case_ds["cft_crop"] = case_ds["cft_crop"].compute()

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


def _get_crop_products(cft_ds):
    """
    Get crop products for various levels of maturity
    """

    # Dictionary with keys the string to use in var names, values min. HUI (fraction) to qualify
    maturity_levels = {
        "ANY": 0.0,
        "MARKETABLE": "isimip3",
        "MATURE": 1.0,
    }

    for m, min_viable_hui in maturity_levels.items():

        # Create DataArray with zeroes where harvest is not viable (shouldn't be included in our
        # postprocessed yield) and ones elsewhere
        viable_harv_var = f"{m}_HARVEST"
        cft_ds[viable_harv_var] = mark_crops_invalid(cft_ds, min_viable_hui=min_viable_hui)

        # Mark invalid harvests as zero
        cft_ds = _mark_invalid_harvests_as_zero(cft_ds, m, viable_harv_var)

    # Calculate actual yield (wet matter) based on "marketable" harvests
    c_var = "GRAINC_TO_FOOD_MARKETABLE_PERHARV"
    if c_var not in cft_ds:
        print(f"WARNING: Will not calculate yield because {c_var} not in cft_ds")
        return cft_ds
    wm_arr = np.full_like(cft_ds[c_var].values, np.nan)
    for i, pft_int in enumerate(cft_ds["cft"].values):
        pft_str = ivt_int2str(pft_int)
        if cft_ds[c_var].dims[0] != "cft":
            raise NotImplementedError("Below code (wm_arr[i]) assumes cft is the 0th dimension")
        wm_arr[i] = cft_ds[c_var].sel(cft=pft_int)
        wm_arr[i] = food_grainc_to_harvested_tons_onecrop(wm_arr[i], pft_str)
    cft_ds["YIELD_PERHARV"] = xr.DataArray(
        data=wm_arr,
        coords=cft_ds[c_var].coords,
        dims=cft_ds[c_var].dims,
        attrs={
            "long_name": "marketable wet matter yield (minus losses) per harvest",
            "units": "g wet matter / m^2",
        },
    )
    cft_ds["YIELD_ANN"] = cft_ds["YIELD_PERHARV"].sum(dim="mxharvests", keep_attrs=True)
    long_name = cft_ds["YIELD_ANN"].attrs["long_name"]
    long_name = long_name.replace("per harvest", "per calendar year")
    return cft_ds


def _mark_invalid_harvests_as_zero(cft_ds, m, viable_harv_var):
    product_list = ["FOOD", "SEED"]
    for p in product_list:
        for v in cft_ds:
            if not re.match(rf"GRAIN[CN]_TO_{p}_PERHARV", v):
                continue

            # Change, e.g., GRAINC_TO_FOOD_PERHARV to GRAINC_TO_FOOD_MARKETABLE_PERHARV
            new_var = v.replace("_PERHARV", f"_{m}_PERHARV")
            da_new = cft_ds[v] * cft_ds[viable_harv_var]
            cft_ds[new_var] = da_new
            cft_ds[new_var].attrs["units"] = cft_ds[v].attrs["units"]
            long_name = f"grain C to {p.lower()} in {m} harvested organ per harvest"
            cft_ds[new_var].attrs["long_name"] = long_name

            # Get annual values
            new_var_ann = new_var.replace("PERHARV", "ANN")
            cft_ds[new_var_ann] = cft_ds[new_var].sum(dim="mxharvests")
            cft_ds[new_var_ann].attrs["long_name"] = long_name.replace(
                "per harvest", "per calendar year"
            )
    return cft_ds


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
    # Get CFT planted area
    hr = case_ds["HARVEST_REASON_PERHARV"]
    cft_planted_area = (case_ds["pfts1d_landarea"] * case_ds["pfts1d_wtgcell"]).where(
        case_ds["pfts1d_wtgcell"] > 0,
    ) * 1e6  # convert km2 to m2
    cft_planted_area.attrs["units"] = "m2"

    # Get CFT harvested area
    case_ds["cft_harv_area"] = (cft_planted_area * (hr > 0)).sum(dim="mxharvests")
    case_ds["cft_harv_area"].attrs["units"] = cft_planted_area.attrs["units"]
    case_ds["cft_harv_area_immature"] = (cft_planted_area * (hr > 1)).sum(
        dim="mxharvests",
    )
    case_ds["cft_harv_area_immature"].attrs["units"] = cft_planted_area.attrs["units"]
    case_ds["cft_harv_area_failed"] = (
        cft_planted_area * (1 - case_ds["MARKETABLE_HARVEST"]).where(hr > 0)
    ).sum(dim="mxharvests")
    case_ds["cft_harv_area_failed"].attrs["units"] = cft_planted_area.attrs["units"]
    case_ds["crop_harv_area"] = (
        case_ds["cft_harv_area"]
        .groupby(case_ds["cft_crop"])
        .sum(dim="cft")
        .rename({"cft_crop": "crop"})
    )
    case_ds["crop_harv_area"].attrs["units"] = cft_planted_area.attrs["units"]
    case_ds["crop_harv_area_immature"] = (
        case_ds["cft_harv_area_immature"]
        .groupby(case_ds["cft_crop"])
        .sum(dim="cft")
        .rename({"cft_crop": "crop"})
    )
    case_ds["crop_harv_area_immature"].attrs["units"] = cft_planted_area.attrs["units"]
    case_ds["crop_harv_area_failed"] = (
        case_ds["cft_harv_area_failed"]
        .groupby(case_ds["cft_crop"])
        .sum(dim="cft")
        .rename({"cft_crop": "crop"})
    )
    case_ds["crop_harv_area_failed"].attrs["units"] = cft_planted_area.attrs["units"]

    return case_ds
