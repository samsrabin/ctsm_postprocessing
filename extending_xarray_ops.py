"""
Functions to extend the kinds of operations you can do to an Xarray object
"""

import warnings
from scipy.stats._axis_nan_policy import SmallSampleWarning
from scipy.stats import circmean
import numpy as np
import xarray as xr

DAYS_IN_YEAR = 365


def da_circmean(da, dim=None, **kwargs):
    """
    Compute circular mean of a DataArray along specified dimension(s)

    Parameters
    ----------
    da : xr.DataArray
        Input DataArray
    dim : str, list of str, or None, optional
        Dimension(s) to compute circular mean over. If None, compute over all dimensions.
    **kwargs
        Additional keyword arguments passed to scipy.stats.circmean (e.g., high, low, nan_policy)

    Returns
    -------
    xr.DataArray
        DataArray with circular mean computed along specified dimension(s)
    """

    # Override default circmean nan_policy of "propagate" with the behavior from xarray's mean:
    # "By default, only skips missing values for float dtypes"
    if "nan_policy" not in kwargs and np.issubdtype(da.dtype, np.floating):
        kwargs["nan_policy"] = "omit"

    input_core_dims = [[dim]] if isinstance(dim, str) else [dim] if dim else [da.dims]

    # Determine which dimensions will be excluded from the output
    exclude_dims = set(input_core_dims[0]) if input_core_dims[0] else set()

    # When reducing all dimensions, we need axis=None
    # When reducing multiple dimensions, apply_ufunc moves them to the end,
    # so we need to specify the appropriate axes
    reduce_all_dims = len(exclude_dims) == len(da.dims)
    num_core_dims = len(input_core_dims[0]) if input_core_dims[0] else 0

    # Create a wrapper function that passes the appropriate axis to circmean
    def _circmean_wrapper(data, **kw):
        if reduce_all_dims:
            return circmean(data, axis=None, **kw)

        if num_core_dims == 1:
            # Single dimension reduction
            return circmean(data, axis=-1, **kw)

        # Multiple dimension reduction - they're moved to the end by apply_ufunc
        # So if we're reducing 2 dims, they'll be at positions -2 and -1
        axes = tuple(range(-num_core_dims, 0))
        return circmean(data, axis=axes, **kw)

    # When reducing all dimensions, we don't need vectorize
    # Otherwise, we need vectorize to handle the remaining dimensions
    use_vectorize = not reduce_all_dims

    result = xr.apply_ufunc(
        _circmean_wrapper,
        da,
        input_core_dims=input_core_dims,
        exclude_dims=exclude_dims,
        kwargs=kwargs,
        keep_attrs=True,
        vectorize=use_vectorize,
    )
    return result


def _round_to_nearest_day(value):
    """
    Round a DOY value to the nearest day. The philosophy here is that 1 represents noon on Jan. 1,
    2 represents noon on Jan. 2, ..., and 365 represents noon on Dec. 31. Thus,
    365.5 == 0.5 == midnight on Jan. 1, so either of those inputs should return 1 (Jan. 1).
    """

    # Just because this is weird to think about
    if value < 0:
        raise ValueError("_round_to_nearest_day doesn't support negative inputs")

    if 0 <= value < 0.5 or -0.5 <= value - DAYS_IN_YEAR < 0.5:
        return DAYS_IN_YEAR

    # We want to always round X.5 to X+1. Unfortunately round() and np.round() use "banker’s
    # rounding," meaning that they round X.5 to the nearest even integer.
    return np.mod(np.floor(value + 0.5), DAYS_IN_YEAR)


def da_circmean_doy(da, dim=None, **kwargs):
    """
    Like da_circmean, but wrapped to work with integer day-of-year outputs in the range [1, 365].
    """

    # We hard-code the circular range for this function
    low = 1
    high = DAYS_IN_YEAR + 1  # Yes, high is one more than number of days in year.
    if "low" in kwargs or "high" in kwargs:
        raise TypeError(
            f"Do not specify low or high for da_circmean_doy(); those are set to {low} and {high}. "
            "Use da_circmean() directly instead."
        )

    # This function assumes that all inputs are integers (or close enough)
    all_nearly_integers = np.allclose(da.values, np.round(da.values), equal_nan=True)
    assert all_nearly_integers, "All input values should be whole numbers or NaN"

    # This function assumes that all values are in the range 1-365.
    # -1 is used by CTSM for "no sowing"/"no harvest", so allow that (convert to NaN)
    da = da.where(da != -1.0)
    all_in_1_365 = np.all(((da.values >= 1) & (da.values <= DAYS_IN_YEAR)) | np.isnan(da.values))
    assert all_in_1_365, f"All input values should be in the range 1-{DAYS_IN_YEAR} or NaN"

    # Suppress SmallSampleWarning for this call (caused by all NaNs)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=SmallSampleWarning)
        result = da_circmean(da, dim=dim, low=low, high=high, **kwargs)

    # Round to nearest day
    result = xr.apply_ufunc(_round_to_nearest_day, result, vectorize=True)

    # Check that we ended up with integers (or close enough)
    all_nearly_integers = np.allclose(result.values, np.round(result.values), equal_nan=True)
    print(result.values)
    assert all_nearly_integers, "Not all output values are whole numbers or NaN"

    return result
