"""
Functions to mark crop seasons as invalid
"""

from __future__ import annotations

from types import MappingProxyType

import numpy as np
import xarray as xr

from .crop_defaults import DEFAULT_VAR_DICT
from .crop_secondary_variables import _handle_huifrac_where_gddharv_notpos
from ..utils import ivt_int2str

# Minimum marketable HUI under ISIMIP3-Agriculture protocol
MIN_HUIFRAC_CORN_ISIMIP3 = 0.8  # Lower than other crops to account for silage maize harvest
MIN_HUIFRAC_OTHER_ISIMIP3 = 0.9


def _get_min_viable_hui(
    ds: xr.Dataset, min_viable_hui: float | str, huifrac_var: str, this_pft: str | None = None
) -> float | np.ndarray:
    """
    Get minimum viable HUI values.

    Parameters
    ----------
    ds : xarray.Dataset
        Input dataset.
    min_viable_hui : float | str
        Minimum viable HUI value or a string identifier ('isimip3' or 'ggcmi3').
    huifrac_var : str
        Variable name for HUI fraction.
    this_pft : str | None, optional
        Specific PFT name to process. If None, processes all PFTs.

    Returns
    -------
    float | numpy.ndarray
        Minimum viable HUI values to use.

    Raises
    ------
    NotImplementedError
        If min_viable_hui is a string other than 'isimip3' or 'ggcmi3'.
    """
    if min_viable_hui in ["isimip3", "ggcmi3"]:
        min_viable_hui_touse = _get_isimip3_min_hui(ds, huifrac_var, this_pft=this_pft)
    elif isinstance(min_viable_hui, str):
        raise NotImplementedError(
            f"min_viable_hui {min_viable_hui} not recognized. Accepted strings are ggcmi3 or"
            " isimip3"
        )
    else:
        min_viable_hui_touse = min_viable_hui
    return min_viable_hui_touse


def _pft_or_patch(ds: xr.Dataset) -> str:
    """
    Determine whether dataset uses 'pft' or 'patch' dimension.

    Parameters
    ----------
    ds : xarray.Dataset
        Input dataset.

    Returns
    -------
    str
        Either 'pft' or 'patch'.

    Raises
    ------
    NotImplementedError
        If both 'patch' and 'pft' are found in ds.dims.
    KeyError
        If neither 'patch' nor 'pft' are found in ds.dims.
    """
    if all(x in ds.dims for x in ["patch", "pft"]):
        raise NotImplementedError("Both patch and pft found in ds.dims")
    if "patch" in ds.dims:
        pftpatch_dimname = "patch"
    elif "pft" in ds.dims:
        pftpatch_dimname = "pft"
    else:
        raise KeyError("Neither patch nor pft found in ds.dims")
    return pftpatch_dimname


def _get_itype_veg_str_varname(pftpatch_dimname: str) -> str:
    """
    Get the variable name for vegetation type string based on dimension name.

    Parameters
    ----------
    pftpatch_dimname : str
        Either 'pft' or 'patch'.

    Returns
    -------
    str
        Variable name for vegetation type string ('pfts1d_itype_veg_str' or
        'patches1d_itype_veg_str').

    Raises
    ------
    NotImplementedError
        If pftpatch_dimname is neither 'pft' nor 'patch'.
    """
    if pftpatch_dimname == "patch":
        itype_veg_str_varname = "patches1d_itype_veg_str"
    elif pftpatch_dimname == "pft":
        itype_veg_str_varname = "pfts1d_itype_veg_str"
    else:
        raise NotImplementedError(f"No itype_veg_str_varname for dim {pftpatch_dimname}.")
    return itype_veg_str_varname


def _get_isimip3_min_hui(
    ds: xr.Dataset, huifrac_var: str, this_pft: str | None = None
) -> np.ndarray:
    """
    Get minimum HUI values according to ISIMIP3 protocol.

    Parameters
    ----------
    ds : xarray.Dataset
        Input dataset.
    huifrac_var : str
        Variable name for HUI fraction.
    this_pft : str | None, optional
        Specific PFT name to process. If None, processes all PFTs.

    Returns
    -------
    numpy.ndarray
        Array of minimum viable HUI values, with corn PFTs set to MIN_HUIFRAC_CORN_ISIMIP3
        and others set to MIN_HUIFRAC_OTHER_ISIMIP3.

    Raises
    ------
    TypeError
        If this_pft is specified but is not a string.
    """
    # Check type of this_pft
    if this_pft is not None and not isinstance(this_pft, str):
        raise TypeError(f"If specified, this_pft must be str, not {type(this_pft)}")

    # Fill with non-corn value; will be replaced with corn value if needed
    min_viable_hui_touse = np.full_like(ds[huifrac_var].values, fill_value=MIN_HUIFRAC_OTHER_ISIMIP3)

    # Handle the simple case where we've given one specific PFT
    if this_pft is not None:
        if "corn" in this_pft:
            min_viable_hui_touse[:] = MIN_HUIFRAC_CORN_ISIMIP3
        return min_viable_hui_touse

    # Otherwise, handling depends on what kind of Dataset we're dealing with
    if "cft" in ds.coords:
        _get_isimip3_min_hui_corn = _get_isimip3_min_hui_corn_cftds
    else:
        _get_isimip3_min_hui_corn = _get_isimip3_min_hui_corn_regds
    min_viable_hui_touse = _get_isimip3_min_hui_corn(ds, huifrac_var, min_viable_hui_touse)

    return min_viable_hui_touse


def _get_isimip3_min_hui_corn_regds(
    ds: xr.Dataset, huifrac_var: str, min_viable_hui_touse: np.ndarray
) -> np.ndarray:
    """
    Fill corn PFTs in DataArray of minimum HUI for a regular (i.e., non-CFT-ized) Dataset.

    Parameters
    ----------
    ds : xarray.Dataset
        Input dataset.
    huifrac_var : str
        Variable name for HUI fraction.
    min_viable_hui_touse : numpy.ndarray
        Array to fill with corn-specific minimum HUI values.

    Returns
    -------
    numpy.ndarray
        Updated array with corn PFTs set to MIN_HUIFRAC_CORN_ISIMIP3.

    Raises
    ------
    NotImplementedError
        If the pft/patch dimension is not first or last in the huifrac variable.
    """

    pftpatch_dimname = _pft_or_patch(ds)
    itype_veg_str_varname = _get_itype_veg_str_varname(pftpatch_dimname)
    pft_list = np.unique(ds[itype_veg_str_varname].values)

    for veg_str in pft_list:
        if "corn" not in veg_str:
            # Skip, because the min_viable_hui_touse array is already set to other_value there
            continue
        is_thistype = np.where((ds[itype_veg_str_varname].values == veg_str))[0]
        pftpatch_index = list(ds[huifrac_var].dims).index(pftpatch_dimname)
        if pftpatch_index == 0:
            min_viable_hui_touse[is_thistype, ...] = MIN_HUIFRAC_CORN_ISIMIP3
        elif pftpatch_index == ds[huifrac_var].ndim - 1:
            min_viable_hui_touse[..., is_thistype] = MIN_HUIFRAC_CORN_ISIMIP3
        else:
            # Need patch to be either first or last dimension to allow use of ellipses
            raise NotImplementedError(
                "Temporarily rearrange min_viable_hui_touse so that"
                f" {pftpatch_dimname} dimension is"
                f" first (0) or last ({ds[huifrac_var].ndim - 1}), instead of"
                f" {pftpatch_index}."
            )
    return min_viable_hui_touse


def _get_isimip3_min_hui_corn_cftds(
    ds: xr.Dataset, huifrac_var: str, min_viable_hui_touse: np.ndarray
) -> np.ndarray:
    """
    Fill corn PFTs in DataArray of minimum HUI for a CFT-ized Dataset.

    Parameters
    ----------
    ds : xarray.Dataset
        Input dataset with 'cft' coordinate.
    huifrac_var : str
        Variable name for HUI fraction.
    min_viable_hui_touse : numpy.ndarray
        Array to fill with corn-specific minimum HUI values.

    Returns
    -------
    numpy.ndarray
        Updated array with corn CFTs set to MIN_HUIFRAC_CORN_ISIMIP3.

    Raises
    ------
    NotImplementedError
        If the cft dimension is not first or last in the huifrac variable.
    """

    # Where on the CFT dimension is it corn?
    pft_list = [ivt_int2str(pft_int) for pft_int in ds["cft"].values]
    corn_inds = [i for i, pft_name in enumerate(pft_list) if "corn" in pft_name]

    # Where is the CFT dimension in the dimension list?
    cft_index = list(ds[huifrac_var].dims).index("cft")

    # Assign corn value to corn points, depending on where CFT dim is in dim list.
    if cft_index == 0:
        min_viable_hui_touse[corn_inds, ...] = MIN_HUIFRAC_CORN_ISIMIP3
    elif cft_index == ds[huifrac_var].ndim - 1:
        min_viable_hui_touse[..., corn_inds] = MIN_HUIFRAC_CORN_ISIMIP3
    else:
        # Need patch to be either first or last dimension to allow use of ellipses
        raise NotImplementedError(
            "Temporarily rearrange min_viable_hui_touse so that"
            f" cft dimension is"
            f" first (0) or last ({ds[huifrac_var].ndim - 1}), instead of"
            f" {cft_index}."
        )

    return min_viable_hui_touse


def mark_invalid_hui_too_low(
    da_in: xr.DataArray,
    huifrac: np.ndarray,
    min_viable_hui_touse: np.ndarray | float,
    invalid_value: float = 0,
) -> xr.DataArray:
    """
    Mark data as invalid where HUI is too low.

    Parameters
    ----------
    da_in : xarray.DataArray
        Input DataArray.
    huifrac : numpy.ndarray
        HUI fraction values.
    min_viable_hui_touse : numpy.ndarray | float
        Minimum viable HUI values to use.
    invalid_value : float, optional
        Value to use for invalid data. Default is 0.

    Returns
    -------
    xarray.DataArray
        DataArray with invalid seasons marked with invalid_value.
    """
    tmp_da = da_in.copy()
    tmp = tmp_da.copy().values
    dont_include = (huifrac < min_viable_hui_touse) & (tmp > 0)
    tmp[np.where(dont_include)] = invalid_value
    # if "MATURE" in out_var:
    #     tmp[np.where(~dont_include & ~np.isnan(tmp))] = 1
    #     tmp_da.attrs["units"] = "fraction"
    da_out = xr.DataArray(data=tmp, attrs=tmp_da.attrs, coords=tmp_da.coords)
    return da_out


def mark_invalid_season_too_long(
    ds: xr.Dataset,
    da_in: xr.DataArray,
    mxmats: dict[str, float],
    gslen_var: str,
    invalid_value: float = 0,
    this_pft: str | None = None,
) -> xr.DataArray:
    # pylint: disable=too-many-positional-arguments
    """
    Mark too-long seasons as invalid.

    Parameters
    ----------
    ds : xarray.Dataset
        Input dataset.
    da_in : xarray.DataArray
        Input DataArray.
    mxmats : dict[str, float]
        Dictionary of maximum allowed season length. Format: {"crop": value}.
    gslen_var : str
        Variable name for growing season length.
    invalid_value : float, optional
        Value to use for invalid data. Default is 0.
    this_pft : str | None, optional
        Specific PFT name to process. If None, processes all PFTs.

    Returns
    -------
    xarray.DataArray
        DataArray with invalid seasons marked with invalid_value.
    """
    tmp_ra = da_in.copy().values

    # Handle the simple case where we've given one specific PFT
    if this_pft is not None:
        mxmat = mxmats[this_pft]
        tmp_ra[np.where(ds[gslen_var].values > mxmat)] = invalid_value

    # Handle cases where we need to look through all PFTs
    else:
        itype_veg_str_varname = _get_itype_veg_str_varname(_pft_or_patch(ds))
        for veg_str in np.unique(ds[itype_veg_str_varname].values):
            mxmat_veg_str = (
                veg_str.replace("soybean", "temperate_soybean")
                .replace("tropical_temperate", "tropical")
                .replace("temperate_temperate", "temperate")
            )
            mxmat = mxmats[mxmat_veg_str]
            where_invalid = np.where(
                (ds[itype_veg_str_varname].values == veg_str) & (ds[gslen_var].values > mxmat)
            )
            tmp_ra[where_invalid] = invalid_value

    da_out = xr.DataArray(data=tmp_ra, coords=da_in.coords, attrs=da_in.attrs)
    return da_out


def _get_ones_da(template_da: xr.DataArray) -> xr.DataArray:
    """
    Produce a DataArray that's like a template DataArray but with all ones.

    Parameters
    ----------
    template_da : xarray.DataArray
        Template DataArray to copy structure from.

    Returns
    -------
    xarray.DataArray
        DataArray with same dimensions and coordinates as template, but filled with ones.
    """
    tmp = np.ones_like(template_da.values)
    da_out = xr.DataArray(
        data=tmp,
        dims=template_da.dims,
        coords=template_da.coords,
    )
    return da_out


def mark_crops_invalid(
    ds: xr.Dataset,
    in_var: str | None = None,
    min_viable_hui: float | str | None = None,
    mxmats: dict[str, float] | None = None,
    var_dict: MappingProxyType = DEFAULT_VAR_DICT,
    invalid_value: float = 0,
    this_pft: str | None = None,
) -> xr.DataArray:
    # pylint: disable=too-many-positional-arguments
    """
    Mark a variable as invalid where minimum viable HUI wasn't reached or season was longer than
    maximum allowed length.

    Parameters
    ----------
    ds : xarray.Dataset
        Input dataset.
    in_var : str | None, optional
        Name of variable to process for invalidity. If None, will make an array with 1
        for valid harvests and 0 elsewhere.
    min_viable_hui : float | str | None, optional
        Minimum viable HUI value, a string identifier ('isimip3' or 'ggcmi3'), or None.
    mxmats : dict[str, float] | None, optional
        Dictionary of maximum allowed season length. Format: {"crop": value}.
    var_dict : MappingProxyType, optional
        Dictionary of variable names. Defaults to DEFAULT_VAR_DICT.
    invalid_value : float, optional
        Value to use for invalid data. Default is 0.
    this_pft : str | None, optional
        Specific PFT name to process. If None, processes all PFTs.

    Returns
    -------
    xarray.DataArray
        DataArray with invalid seasons marked with invalid_value.
    """
    mxmat_limited = bool(mxmats)

    if in_var is None:
        da_out = None
    else:
        da_out = ds[in_var].copy()

    # Mark as invalid where minimum viable HUI wasn't reached
    if min_viable_hui is not None:
        if da_out is None:
            da_out = _get_ones_da(ds[var_dict["huifrac_var"]])
        huifrac = _handle_huifrac_where_gddharv_notpos(
            ds[var_dict["huifrac_var"]], ds[var_dict["gddharv_var"]]
        )
        min_viable_hui_touse = _get_min_viable_hui(
            ds, min_viable_hui, var_dict["huifrac_var"], this_pft=this_pft
        )
        if np.any(huifrac < min_viable_hui_touse):
            da_out = mark_invalid_hui_too_low(
                da_out, huifrac, min_viable_hui_touse, invalid_value=invalid_value
            )
        da_out.attrs["min_viable_hui"] = min_viable_hui

    # Get variants with values set marked as invalid if season was longer than CLM PFT parameter
    # mxmat
    if mxmat_limited:
        if da_out is None:
            da_out = _get_ones_da(ds[var_dict["gslen_var"]])
        da_out = mark_invalid_season_too_long(
            ds,
            da_out,
            mxmats,
            var_dict["gslen_var"],
            invalid_value=invalid_value,
            this_pft=this_pft,
        )
        da_out.attrs["mxmat_limited"] = True

    return da_out
