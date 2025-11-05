"""
Functions to generate useful crop variables not saved by CTSM
"""

import os
import sys
import numpy as np

try:
    # Attempt relative import if running as part of a package
    from .crop_defaults import DEFAULT_VAR_DICT
except ImportError:
    # Fallback to absolute import if running as a script
    sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    from crops.crop_defaults import DEFAULT_VAR_DICT


def _handle_huifrac_where_gddharv_notpos(da_huifrac, da_gddharv):
    # Error if any GDDHARV value is negative for non-NaN HUIFRAC
    if np.any((da_gddharv < 0) & ~np.isnan(da_huifrac)):
        raise NotImplementedError("How should negative GDDHARV affect HUIFRAC?")
    # TODO: Should there be an error for vice versa?

    huifrac = da_huifrac.values

    # If harvest threshold HUI is 0, mark huifrac as 1
    huifrac[np.where(da_gddharv.values == 0)] = 1
    return huifrac


def get_huifrac(ds, var_dict=DEFAULT_VAR_DICT):
    """
    Given a dataset, calculate HUIFRAC as hui_var/gddharv_var
    """
    hui_var = var_dict["hui_var"]
    gddharv_var = var_dict["gddharv_var"]

    da_hui = ds[hui_var]
    da_gddharv = ds[gddharv_var]

    # Mask where both are negative (indicating no harvest)
    both_negative = (da_hui < 0) & (da_gddharv < 0)
    da_hui = da_hui.where(~both_negative)
    da_gddharv = da_gddharv.where(~both_negative)

    # Calculate HUIFRAC
    da_huifrac = da_hui / da_gddharv

    # Handle HUIFRAC where GDDHARV (denominator) is zero or negative
    huifrac = _handle_huifrac_where_gddharv_notpos(da_huifrac, da_gddharv)
    da_huifrac.data = huifrac

    da_huifrac.attrs["units"] = "Fraction of required"
    return da_huifrac


def _calendar_has_leapdays(time_da):
    """
    Returns True if da has has a calendar and it definitely has leap days
    """
    # Handle trivial cases where time is missing, empty, or a plain numpy type
    if "time" not in time_da.dims:
        return False
    if len(time_da) == 0:
        return False
    time0 = time_da.values[0]
    if hasattr(time0, "dtype"):
        return False

    # Get day of year for Dec. 31 of a leap year
    if not hasattr(time0, "dayofyr"):
        raise NotImplementedError(f"Calendar type {type(time0)}")
    dec_31_leapyr = type(time0)(2024, 12, 31).dayofyr

    return dec_31_leapyr > 365


def get_gslen(ds):
    """
    Given a dataset, calculate growing season length as HDATES - SDATES_PERHARV
    """
    var_hdates = "HDATES"
    var_sdates = "SDATES_PERHARV"
    if not var_hdates in ds and var_sdates in ds:
        return None

    da_hdates = ds[var_hdates]
    da_sdates = ds[var_sdates]

    # Check for weirdness
    if np.any(da_hdates < 1):
        raise ValueError(f"Unexpected {var_hdates} value(s) < 1")
    if np.any(da_sdates < 1):
        raise ValueError(f"Unexpected {var_sdates} value(s) < 1")
    if np.any(da_hdates > 366):
        raise ValueError(f"Unexpected {var_hdates} value(s) > 366")
    if np.any(da_sdates > 366):
        raise ValueError(f"Unexpected {var_sdates} value(s) > 366")
    if not np.array_equal(np.isnan(da_hdates), np.isnan(da_sdates)):
        raise ValueError(f"Unexpected NaN mismatch between {var_hdates} and {var_sdates}")

    # Check for no leap years
    if (
        "time" in ds
        and any("time" in x.dims for x in [da_hdates, da_sdates])
        and _calendar_has_leapdays(ds["time"])
    ):
        raise NotImplementedError("Unexpected calendar with leap days")
    if np.any(da_hdates == 366):
        raise NotImplementedError(f"Unexpected {var_hdates} value(s) == 366 suggesting leap days")
    if np.any(da_sdates == 366):
        raise NotImplementedError(f"Unexpected {var_sdates} value(s) == 366 suggesting leap days")

    da_gslen = (da_hdates - da_sdates) % 365

    da_gslen.attrs["units"] = "days"
    return da_gslen
