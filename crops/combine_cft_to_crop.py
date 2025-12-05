"""
Functions for combining CFTs (Crop Functional Types) into their corresponding crops.

This module provides utilities to process and combine CFT-level data into crop-level
data in CTSM (Community Terrestrial Systems Model) output. It handles the mapping
between CFTs and crops and provides methods for aggregating CFT-level variables.
"""

import numpy as np
import xarray as xr


def get_cft_crop_da(crops_to_include, case, case_ds):
    """
    Create a CFT-dimensioned DataArray mapping CFT indices to crop names.

    Parameters
    ----------
    crops_to_include : list
        List of crop names to process
    case : object
        Case object containing crop definitions and PFT mappings
    case_ds : xarray.Dataset
        Dataset containing CFT dimension information

    Returns
    -------
    xarray.DataArray
        DataArray with CFT dimension where each element contains the corresponding
        crop name as a string
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
    Process data for a single crop type.

    Parameters
    ----------
    case : object
        Case object containing crop definitions
    case_ds : xarray.Dataset
        Dataset containing CFT-dimensioned data
    crop : str
        Name of the crop to process
    var : str
        Variable name to process
    crop_cft_da_io : xarray.DataArray or None
        Existing DataArray to append to, or None if this is the first crop

    Returns
    -------
    xarray.DataArray
        DataArray containing the processed data for the specified crop,
        either as a new array or appended to the input array
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
    """
    Process CFT data for all specified crops.

    Parameters
    ----------
    crops_to_include : list
        List of crop names to process
    case : object
        Case object containing crop definitions
    case_ds : xarray.Dataset
        Dataset containing CFT-dimensioned data
    var : str
        Variable name to process

    Returns
    -------
    xarray.DataArray
        Combined DataArray containing processed data for all specified crops
    """
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


def combine_cft_to_crop(ds, var_in, var_out, method, weights=None, **kwargs):
    """
    Combine CFT-level data into crop-level data using the specified method.

    Parameters
    ----------
    ds : xarray.Dataset
        Input dataset containing CFT-dimensioned data
    var_in : str
        Name of the input variable to process
    var_out : str
        Name of the output variable to create
    method : callable or str
        Method to use for combining CFT data. Can be:
        - A callable function
        - Name of an xarray groupby method (e.g., 'mean', 'sum', 'std')
        - Name of a numpy method
    weights : str or xarray.DataArray, optional
        Weights to use for weighted mean operations. Can be:
        - Name of a variable in the dataset to use as weights
        - An xarray.DataArray with 'cft' dimension matching the input data
        Only supported when method='mean' (xarray groupby method).
        Raises NotImplementedError for other methods.
    **kwargs
        Additional keyword arguments passed to the combining method

    Returns
    -------
    xarray.Dataset
        Dataset with the new crop-level variable added

    Raises
    ------
    AttributeError
        If the specified method cannot be found or is not callable
    ValueError
        If weights are specified but incompatible with the data
    NotImplementedError
        If weights are provided with an unsupported method

    Examples
    --------
    >>> # Simple mean across CFTs within each crop
    >>> ds = combine_cft_to_crop(ds, 'YIELD', 'YIELD_crop', method='mean')
    >>>
    >>> # Weighted mean using area weights
    >>> ds = combine_cft_to_crop(ds, 'YIELD', 'YIELD_crop', method='mean',
    ...                          weights='AREA')
    """

    da_grouped = ds[var_in].groupby(ds["cft_crop"])

    # Handle weights if provided
    if weights is not None:
        # Get the weights DataArray
        if isinstance(weights, str):
            weights_da = ds[weights]
        else:
            weights_da = weights

        # Ensure weights have the cft dimension
        if "cft" not in weights_da.dims:
            raise ValueError("Weights must have 'cft' dimension")

    # For now, only implementing weighting for Xarray mean
    weights_not_implemented_msg = (
        "Weights are only supported for method='mean' (xarray groupby method)"
    )

    # First, see if you can call the given method directly
    if callable(method):
        if weights is not None:
            raise NotImplementedError(weights_not_implemented_msg)
        da = method(**kwargs)

    # Next, try to find the method in the Xarray DataArray groupby object
    elif hasattr(da_grouped, method) and callable(getattr(da_grouped, method)):
        if weights is not None:
            # Only support weights for 'mean' method
            if method != "mean":
                raise NotImplementedError(weights_not_implemented_msg)

            # Weighted mean: sum(values * weights) / sum(weights)
            weights_grouped = weights_da.groupby(ds["cft_crop"])
            weighted_sum = (
                (ds[var_in] * weights_da).groupby(ds["cft_crop"]).sum(dim="cft", **kwargs)
            )
            weight_sum = weights_grouped.sum(dim="cft", **kwargs)
            da = weighted_sum / weight_sum
        else:
            da = getattr(da_grouped, method)(dim="cft", **kwargs)

    # Finally, check if it's a Numpy method
    elif hasattr(np, method) and callable(getattr(np, method)):
        if weights is not None:
            raise NotImplementedError(weights_not_implemented_msg)
        da = da_grouped.apply(lambda x: getattr(np, method)(x.values, **kwargs))

    # If none of those worked, throw an error
    else:
        raise AttributeError(f"Method '{method}' not found")

    try:
        ds[var_out] = da.rename({"cft_crop": "crop"})
    except ValueError as e:
        if "the new name 'crop' conflicts" in str(e):
            raise ValueError(
                str(e)
                + ". This is probably due to calling combine_cft_to_crop on a cft_ds that has"
                + " already been sliced by crop."
            ) from e
        raise e

    return ds
