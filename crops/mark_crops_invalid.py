"""
Functions to mark crop seasons as invalid
"""

import numpy as np
import xarray as xr

from .crop_defaults import DEFAULT_VAR_DICT
from .crop_secondary_variables import _handle_huifrac_where_gddharv_notpos
from ..utils import ivt_int2str

# Minimum marketable HUI under ISIMIP3-Agriculture protocol
MIN_HUIFRAC_CORN_ISIMIP3 = 0.8  # Lower than other crops to account for silage maize harvest
MIN_HUIFRAC_OTHER_ISIMIP3 = 0.9

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


def _get_isimip3_min_hui_corn_regds(ds, huifrac_var, min_viable_hui_touse):
    """
    Fill corn PFTs in DataArray of minimum HUI for a regular (i.e., non-CFT-ized) Dataset
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


def _get_isimip3_min_hui_corn_cftds(ds, huifrac_var, min_viable_hui_touse):
    """
    Fill corn PFTs in DataArray of minimum HUI for a CFT-ized Dataset
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


def _get_ones_da(template_da):
    """
    Produce a DataArray that's like a template DataArray but with all ones
    """
    tmp = np.ones_like(template_da.values)
    da_out = xr.DataArray(
        data=tmp,
        dims=template_da.dims,
        coords=template_da.coords,
    )
    return da_out



def mark_crops_invalid(
    ds,
    in_var=None,
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
    in_var (str): Name of variable to process for invalidity. If None, will make an array with 1
                  for valid harvests and 0 elsewhere.
    min_viable_hui (float or str): Minimum viable HUI value or a string identifier.
    mxmats (dict): Dictionary of maximum allowed season length. Format: {"crop": value}.
    var_dict (dict): Dictionary of variable names.

    Returns:
    xarray.DataArray: DataArray with invalid seasons marked as such.
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
