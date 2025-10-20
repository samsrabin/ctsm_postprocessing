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
    crop,
    var,
    crop_cft_da_io,
):
    """
    Process things for one crop
    """
    crop_cft_da = case_ds.sel(cft=case.crop_list[crop].pft_nums)[var]

    # Setup crop_cft_* variables or append to them
    if crop_cft_da_io is None:
        # Define crop_cft_* variables
        crop_cft_da_io = xr.DataArray(
            data=crop_cft_da,
        )
    else:
        # Append this crop's DataArrays to existing ones
        crop_cft_da_io = xr.concat(
            [crop_cft_da_io, crop_cft_da],
            dim="cft",
        )

    return crop_cft_da_io


def get_all_cft_crop_das(crops_to_include, case, case_ds, var):
    crop_cft_da = None
    for crop in crops_to_include:
        # Get data for CFTs of this crop
        crop_cft_da = _one_crop(
            case,
            case_ds,
            crop,
            var,
            crop_cft_da,
        )

    return crop_cft_da


def combine_cft_to_crop(ds, var_in, var_out, method, **kwargs):

    da_grouped = ds[var_in].groupby(ds["cft_crop"])

    # First, see if you can call the given method directly
    if callable(method):
        da = method(**kwargs)

    # Next, try to find the method in the Xarray DataArray
    elif hasattr(da_grouped, method) and callable(getattr(da_grouped, method)):
        da = getattr(da_grouped, method)(dim="cft", **kwargs)

    # Finally, check if it's a Numpy method
    elif hasattr(np, method) and callable(getattr(np, method)):
        da = getattr(np, method)(**kwargs)

    # If none of those worked, throw an error
    else:
        raise AttributeError(f"Method '{method}' not found")

    ds[var_out] = da.rename({"cft_crop": "crop"})

    return ds
