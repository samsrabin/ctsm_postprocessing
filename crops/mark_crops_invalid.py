"""
Functions to mark crop seasons as invalid
"""

import os
import sys
import numpy as np
import xarray as xr

try:
    # Attempt relative import if running as part of a package
    from .crop_defaults import DEFAULT_VAR_DICT
    from .crop_secondary_variables import _handle_huifrac_where_gddharv_notpos
except ImportError:
    # Fallback to absolute import if running as a script
    sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    from crops.crop_defaults import DEFAULT_VAR_DICT
    from crops.crop_secondary_variables import _handle_huifrac_where_gddharv_notpos


def _get_min_viable_hui(ds, min_viable_hui, huifrac_var, this_pft=None):
    """
    Get minimum viable HUI values.

    Parameters:
    ds (xarray.Dataset): Input dataset.
    min_viable_hui (float or str): Minimum viable HUI value or a string identifier.
    huifrac_var (str): Variable name for HUI fraction.

    Returns:
    numpy.ndarray: Minimum viable HUI values to use.
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


def _pft_or_patch(ds):
    if all(x in ds.dims for x in ["patch", "pft"]):
        raise NotImplementedError("Both patch and pft found in ds.dims")
    if "patch" in ds.dims:
        pftpatch_dimname = "patch"
    elif "pft" in ds.dims:
        pftpatch_dimname = "pft"
    else:
        raise KeyError("Neither patch nor pft found in ds.dims")
    return pftpatch_dimname


def _get_itype_veg_str_varname(pftpatch_dimname):
    if pftpatch_dimname == "patch":
        itype_veg_str_varname = "patches1d_itype_veg_str"
    elif pftpatch_dimname == "pft":
        itype_veg_str_varname = "pfts1d_itype_veg_str"
    else:
        raise NotImplementedError(f"No itype_veg_str_varname for dim {pftpatch_dimname}.")
    return itype_veg_str_varname


def _get_isimip3_min_hui(ds, huifrac_var, this_pft=None):
    corn_value = 0.8  # Lower than other crops to account for silage maize harvest
    other_value = 0.9

    # Check type of this_pft
    if this_pft is not None and not isinstance(this_pft, str):
        raise TypeError(f"If specified, this_pft must be str, not {type(this_pft)}")

    # Fill with other_value; will be replaced with corn_value if needed
    min_viable_hui_touse = np.full_like(ds[huifrac_var].values, fill_value=other_value)

    # Handle the simple case where we've given one specific PFT
    if this_pft is not None:
        if "corn" in this_pft:
            min_viable_hui_touse[:] = corn_value
        return min_viable_hui_touse

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
            min_viable_hui_touse[is_thistype, ...] = corn_value
        elif pftpatch_index == ds[huifrac_var].ndim - 1:
            min_viable_hui_touse[..., is_thistype] = corn_value
        else:
            # Need patch to be either first or last dimension to allow use of ellipses
            raise NotImplementedError(
                "Temporarily rearrange min_viable_hui_touse so that"
                f" {pftpatch_dimname} dimension is"
                f" first (0) or last ({ds[huifrac_var].ndim - 1}), instead of"
                f" {pftpatch_index}."
            )
    return min_viable_hui_touse


def mark_invalid_hui_too_low(da_in, huifrac, min_viable_hui_touse, invalid_value=0):
    """
    Mark data as invalid where HUI is too low.

    Parameters:
    da_in (xarray.DataArray): Input DataArray.
    huifrac (numpy.ndarray): HUI fraction values.
    min_viable_hui_touse (numpy.ndarray): Minimum viable HUI values to use.

    Returns:
    xarray.DataArray: DataArray with invalid seasons marked as such.
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


def mark_invalid_season_too_long(ds, da_in, mxmats, gslen_var, invalid_value=0, this_pft=None):
    # pylint: disable=too-many-positional-arguments
    """
    Mark too-long seasons as invalid.

    Parameters:
    ds (xarray.Dataset): Input dataset.
    da_in (xarray.DataArray): Input DataArray.
    mxmats (dict): Dictionary of maximum allowed season length. Format: {"crop": value}.
    gslen_var (str): Variable name for growing season length.

    Returns:
    xarray.DataArray: DataArray with invalid seasons marked as such.
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


def mark_crops_invalid(
    ds,
    in_var,
    min_viable_hui=None,
    mxmats=None,
    var_dict=DEFAULT_VAR_DICT,
    invalid_value=0,
    this_pft=None,
):  # pylint: disable=too-many-positional-arguments
    """
    Mark a variable as invalid where minimum viable HUI wasn't reached or season was longer than
    maximum allowed length.

    Parameters:
    ds (xarray.Dataset): Input dataset.
    in_var (str): Name of variable to process for invalidity.
    min_viable_hui (float or str): Minimum viable HUI value or a string identifier.
    mxmats (dict): Dictionary of maximum allowed season length. Format: {"crop": value}.
    var_dict (dict): Dictionary of variable names.

    Returns:
    xarray.DataArray: DataArray with invalid seasons marked as such.
    """
    mxmat_limited = bool(mxmats)

    da_out = ds[in_var].copy()

    # Mark as invalid where minimum viable HUI wasn't reached
    if min_viable_hui is not None:
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
