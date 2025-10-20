"""
Function to combine CFTs into their corresponding crops
"""

import numpy as np
import xarray as xr


def get_cft_crop_da(crops_to_include, case, case_ds):
    """
    Get cft-dimensioned DataArray with string values of Crop
    """
    cft_crop_array = np.full(case_ds.sizes["cft"], "", dtype=object)
    for i, crop in enumerate(crops_to_include):
        for pft_num in case.crop_list[crop].pft_nums:
            cft_crop_array[np.where(case_ds["cft"].values == pft_num)] = crop
    cft_crop_da = xr.DataArray(
        data=cft_crop_array,
        dims=["cft"],
        coords={"cft": case_ds["cft"]},
    ).astype(str)

    return cft_crop_da


def _one_crop(
    case,
    case_ds,
    i,
    crop,
    crop_cft_area_da,
    crop_cft_prod_da,
):
    """
    Process things for one crop
    """
    cft_ds = case_ds.sel(cft=case.crop_list[crop].pft_nums)

    # Get area
    cft_area = cft_ds["pfts1d_gridcellarea"] * cft_ds["pfts1d_wtgcell"]

    # Get production
    cft_prod = cft_ds["YIELD_ANN"] * cft_area

    # Setup crop_cft_* variables or append to them
    if i == 0:
        # Define crop_cft_* variables
        crop_cft_area_da = xr.DataArray(
            data=cft_area,
        )
        crop_cft_prod_da = xr.DataArray(
            data=cft_prod,
        )
    else:
        # Append this crop's DataArrays to existing ones
        crop_cft_area_da = xr.concat(
            [crop_cft_area_da, cft_area],
            dim="cft",
        )
        crop_cft_prod_da = xr.concat(
            [crop_cft_prod_da, cft_prod],
            dim="cft",
        )

    return crop_cft_area_da, crop_cft_prod_da
