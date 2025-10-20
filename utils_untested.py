"""
Potentially useful utilities that are untested, either via unit/system tests in ctsm_postprocessing
or via use in CUPiD

mostly
copied from klindsay, https://github.com/klindsay28/CESM2_coup_carb_cycle_JAMES/blob/master/utils.py
"""

# pylint: disable=wrong-import-position

import importlib
import re
import warnings

with warnings.catch_warnings():
    warnings.filterwarnings(action="ignore", category=DeprecationWarning)
    try:
        import cf_units as cf  # pylint: disable=import-error
    except:  # pylint: disable=bare-except
        pass
    try:
        from cartopy.util import add_cyclic_point  # pylint: disable=import-error
    except:  # pylint: disable=bare-except
        pass
import cftime
import numpy as np
import xarray as xr

from .utils import define_pftlist, make_lon_increasing, is_strictly_increasing, is_each_vegtype


# generate annual means, weighted by days / month
def weighted_annual_mean(array, time_in="time", time_out="time"):
    if isinstance(array[time_in].values[0], cftime.datetime):
        month_length = array[time_in].dt.days_in_month

        # After https://docs.xarray.dev/en/v0.5.1/examples/monthly-means.html
        group = f"{time_in}.year"
        weights = month_length.groupby(group) / month_length.groupby(group).sum()
        np.testing.assert_allclose(weights.groupby(group).sum().values, 1)
        array = (array * weights).groupby(group).sum(dim=time_in, skipna=True)
        if time_out != "year":
            array = array.rename({"year": time_out})

    else:
        mon_day = xr.DataArray(
            np.array([31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]), dims=["month"]
        )
        mon_wgt = mon_day / mon_day.sum()
        array = (
            array.rolling({time_in: 12}, center=False)  # rolling
            .construct("month")  # construct the array
            .isel(
                {time_in: slice(11, None, 12)}
            )  # slice so that the first element is [1..12], second is [13..24]
            .dot(mon_wgt, dims=["month"])
        )
        if time_in != time_out:
            array = array.rename({time_in: time_out})

    return array


def change_units(ds, variable_str, variable_bounds_str, target_unit_str):
    """Applies unit conversion on an xarray DataArray"""
    with warnings.catch_warnings():
        warnings.filterwarnings(action="ignore", category=DeprecationWarning)
        if importlib.util.find_spec("cf_units") is None:
            raise ModuleNotFoundError("change_units() depends on cf_units, which is not available")
    orig_units = cf.Unit(ds[variable_str].attrs["units"])
    target_units = cf.Unit(target_unit_str)
    variable_in_new_units = xr.apply_ufunc(
        orig_units.convert,
        ds[variable_bounds_str],
        target_units,
        output_dtypes=[ds[variable_bounds_str].dtype],
    )
    return variable_in_new_units


def clean_units(units):
    """replace some troublesome unit terms with acceptable replacements"""
    replacements = {
        "kgC": "kg",
        "gC": "g",
        "gC13": "g",
        "gC14": "g",
        "gN": "g",
        "unitless": "1",
        "years": "common_years",
        "yr": "common_year",
        "meq": "mmol",
        "neq": "nmol",
    }
    units_split = re.split("( |\(|\)|\^|\*|/|-[0-9]+|[0-9]+)", units)
    units_split_repl = [
        replacements[token] if token in replacements else token for token in units_split
    ]
    return "".join(units_split_repl)


def copy_fill_settings(da_in, da_out):
    """
    propagate _FillValue and missing_value settings from da_in to da_out
    return da_out
    """
    if "_FillValue" in da_in.encoding:
        da_out.encoding["_FillValue"] = da_in.encoding["_FillValue"]
    else:
        da_out.encoding["_FillValue"] = None
    if "missing_value" in da_in.encoding:
        da_out.attrs["missing_value"] = da_in.encoding["missing_value"]
    return da_out


def dim_cnt_check(ds, varname, dim_cnt):
    """confirm that varname in ds has dim_cnt dimensions"""
    if len(ds[varname].dims) != dim_cnt:
        msg_full = "unexpected dim_cnt=%d, varname=%s" % (len(ds[varname].dims), varname)
        raise ValueError(msg_full)


def time_set_mid(ds, time_name):
    """
    set ds[time_name] to midpoint of ds[time_name].attrs['bounds'], if bounds attribute exists
    type of ds[time_name] is not changed
    ds is returned
    """

    if "bounds" not in ds[time_name].attrs:
        return ds

    # determine units and calendar of unencoded time values
    if ds[time_name].dtype == np.dtype("O"):
        units = "days since 0000-01-01"
        calendar = "noleap"
    else:
        units = ds[time_name].attrs["units"]
        calendar = ds[time_name].attrs["calendar"]

    # construct unencoded midpoint values, assumes bounds dim is 2nd
    tb_name = ds[time_name].attrs["bounds"]
    if ds[tb_name].dtype == np.dtype("O"):
        tb_vals = cftime.date2num(ds[tb_name].values, units=units, calendar=calendar)
    else:
        tb_vals = ds[tb_name].values
    tb_mid = tb_vals.mean(axis=1)

    # set ds[time_name] to tb_mid
    if ds[time_name].dtype == np.dtype("O"):
        ds[time_name] = cftime.num2date(tb_mid, units=units, calendar=calendar)
    else:
        ds[time_name] = tb_mid

    return ds


def time_year_plus_frac(ds, time_name):
    """return time variable, as year plus fraction of year"""

    # this is straightforward if time has units='days since 0000-01-01' and calendar='noleap'
    # so convert specification of time to that representation

    # get time values as an np.ndarray of cftime objects
    if np.dtype(ds[time_name]) == np.dtype("O"):
        tvals_cftime = ds[time_name].values
    else:
        tvals_cftime = cftime.num2date(
            ds[time_name].values, ds[time_name].attrs["units"], ds[time_name].attrs["calendar"]
        )

    # convert cftime objects to representation mentioned above
    tvals_days = cftime.date2num(tvals_cftime, "days since 0000-01-01", calendar="noleap")

    return tvals_days / 365.0


# add cyclic point
def cyclic_dataarray(da, coord="lon"):
    """Add a cyclic coordinate point to a DataArray along a specified
    named coordinate dimension.
    >>> from xray import DataArray
    >>> data = DataArray([[1, 2, 3], [4, 5, 6]],
    ...                      coords={'x': [1, 2], 'y': range(3)},
    ...                      dims=['x', 'y'])
    >>> cd = cyclic_dataarray(data, 'y')
    >>> print cd.data
    array([[1, 2, 3, 1],
           [4, 5, 6, 4]])
    """
    assert isinstance(da, xr.DataArray)
    with warnings.catch_warnings():
        warnings.filterwarnings(action="ignore", category=DeprecationWarning)
        if importlib.util.find_spec("cartopy") is None:
            raise ModuleNotFoundError(
                "cyclic_dataarray() depends on cartopy, which is not available"
            )

    lon_idx = da.dims.index(coord)
    cyclic_data, cyclic_coord = add_cyclic_point(da.values, coord=da.coords[coord], axis=lon_idx)

    # Copy and add the cyclic coordinate and data
    new_coords = dict(da.coords)
    new_coords[coord] = cyclic_coord
    new_values = cyclic_data

    new_da = xr.DataArray(new_values, dims=da.dims, coords=new_coords)

    # Copy the attributes for the re-constructed data and coords
    for att, val in da.attrs.items():
        new_da.attrs[att] = val
    for c in da.coords:
        for att in da.coords[c].attrs:
            new_da.coords[c].attrs[att] = da.coords[c].attrs[att]

    return new_da


# as above, but for a dataset
# doesn't work because dims are locked in a dataset
"""
def cyclic_dataset(ds, coord='lon'):
    assert isinstance(ds, xr.Dataset)
    with warnings.catch_warnings():
        warnings.filterwarnings(action='ignore', category=DeprecationWarning)
        if importlib.util.find_spec('cartopy') is None:
            raise ModuleNotFoundError("cyclic_dataset() depends on cartopy, which is not available")

    lon_idx = ds.dims.index(coord)
    cyclic_data, cyclic_coord = add_cyclic_point(ds.values,
                                                 coord=ds.coords[coord],
                                                 axis=lon_idx)

    # Copy and add the cyclic coordinate and data
    new_coords = dict(ds.coords)
    new_coords[coord] = cyclic_coord
    new_values = cyclic_data

    new_ds = xr.DataSet(new_values, dims=ds.dims, coords=new_coords)

    # Copy the attributes for the re-constructed data and coords
    for att, val in ds.attrs.items():
        new_ds.attrs[att] = val
    for c in ds.coords:
        for att in ds.coords[c].attrs:
            new_ds.coords[c].attrs[att] = ds.coords[c].attrs[att]

    return new_ds
"""


# Get CLM ivt number corresponding to a given name
def ivt_str2int(ivt_str):
    pftlist = define_pftlist()
    if isinstance(ivt_str, str):
        ivt_int = pftlist.index(ivt_str)
    elif isinstance(ivt_str, list) or isinstance(ivt_str, np.ndarray):
        ivt_int = [ivt_str2int(x) for x in ivt_str]
        if isinstance(ivt_str, np.ndarray):
            ivt_int = np.array(ivt_int)
    else:
        raise RuntimeError(
            f"Update ivt_str_to_int() to handle input of type {type(ivt_str)} (if possible)"
        )

    return ivt_int


# Convert a longitude axis that's -180 to 180 around the international date line to one that's 0 to 360 around the prime meridian. If you pass in a Dataset or DataArray, the "lon" coordinates will be changed. Otherwise, it assumes you're passing in numeric data.
def lon_idl2pm(lons_in, fail_silently=False):
    def check_ok(tmp, fail_silently):
        msg = ""

        if np.any(tmp > 180):
            msg = f"Maximum longitude is already > 180 ({np.max(tmp)})"
        elif np.any(tmp < -180):
            msg = f"Minimum longitude is < -180 ({np.min(tmp)})"

        if msg == "":
            return True
        elif fail_silently:
            return False
        else:
            raise ValueError(msg)

    def do_it(tmp):
        tmp = tmp + 360
        tmp = np.mod(tmp, 360)
        return tmp

    if isinstance(lons_in, (xr.DataArray, xr.Dataset)):
        if not check_ok(lons_in.lon.values, fail_silently):
            return lons_in
        lons_out = lons_in
        lons_out = lons_out.assign_coords(lon=do_it(lons_in.lon.values))
        lons_out = make_lon_increasing(lons_out)
    else:
        if not check_ok(lons_in, fail_silently):
            return lons_in
        lons_out = do_it(lons_in)
        if not is_strictly_increasing(lons_out):
            print(
                "WARNING: You passed in numeric longitudes to lon_idl2pm() and these have been"
                " converted, but they're not strictly increasing."
            )
        print(
            "To assign the new longitude coordinates to an Xarray object, use"
            " xarrayobject.assign_coordinates()! (Pass the object directly in to lon_idl2pm() in"
            " order to suppress this message.)"
        )

    return lons_out


# List (strings) of managed crops in CLM.
def define_mgdcrop_list():
    notcrop_list = ["tree", "grass", "shrub", "unmanaged", "not_vegetated"]
    defined_pftlist = define_pftlist()
    is_crop = is_each_vegtype(defined_pftlist, notcrop_list, "notok_contains")
    return [defined_pftlist[i] for i, x in enumerate(is_crop) if x]


# Rename "patch" dimension and any associated variables back to "pft". Uses a dictionary with the names of the dimensions and variables we want to rename. This allows us to do it all at once, which may be more efficient than one-by-one.
def patch2pft(xr_object):
    # Rename "patch" dimension
    patch2pft_dict = {}
    for thisDim in xr_object.dims:
        if thisDim == "patch":
            patch2pft_dict["patch"] = "pft"
            break

    # Rename variables containing "patch"
    if isinstance(xr_object, xr.Dataset):
        pattern = re.compile("patch.*1d")
        matches = [x for x in list(xr_object.keys()) if pattern.search(x) != None]
        if len(matches) > 0:
            for m in matches:
                patch2pft_dict[m] = m.replace("patches", "patchs").replace("patch", "pft")

    # Do the rename
    if len(patch2pft_dict) > 0:
        xr_object = xr_object.rename(patch2pft_dict)

    return xr_object


# Given a DataArray, remove all patches except those planted with managed crops.
def trim_da_to_mgd_crop(thisvar_da, patches1d_itype_veg_str):
    # Handle input DataArray without patch dimension
    if not any(np.array(list(thisvar_da.dims)) == "patch"):
        print(
            "Input DataArray has no patch dimension and therefore trim_to_mgd_crop() has no effect."
        )
        return thisvar_da

    # Throw error if patches1d_itype_veg_str isn't strings
    if isinstance(patches1d_itype_veg_str, xr.DataArray):
        patches1d_itype_veg_str = patches1d_itype_veg_str.values
    if not isinstance(patches1d_itype_veg_str[0], str):
        raise TypeError(
            "Input patches1d_itype_veg_str is not in string form, and therefore trim_to_mgd_crop()"
            " cannot work."
        )

    # Get boolean list of whether each patch is planted with a managed crop
    notcrop_list = ["tree", "grass", "shrub", "unmanaged", "not_vegetated"]
    is_crop = is_each_vegtype(patches1d_itype_veg_str, notcrop_list, "notok_contains")

    # Warn if no managed crops were found, but still return the empty result
    if np.all(np.bitwise_not(is_crop)):
        print("No managed crops found! Returning empty DataArray.")
    return thisvar_da.isel(patch=[i for i, x in enumerate(is_crop) if x])


# Xarray's native resampler is nice, but it will result in ALL variables being resampled
# along the N specified dimension(s). This means that, e.g., variables that were
# supposed to be 1d will be 1+Nd afterwards. This function undoes that. The syntax is the
# same as for Xarray's resampler plus, at the beginning:
#    (1) which object you want to resample, and
#    (2) the function you want to use for downsampling (e.g., "mean").
def resample(ds_in, thefunction, **kwargs):
    # This problem is not applicable to DataArrays, so just use Xarray's resampler.
    if isinstance(ds_in, xr.DataArray):
        da_resampler = ds_in.resample(**kwargs)
        da_out = getattr(da_resampler, thefunction)()
        return da_out

    # Get the original dimensions of each variable
    orig_dims = dict([(x, ds_in[x].dims) for x in ds_in])

    # Do the Xarray resampling
    ds_resampler = ds_in.resample(**kwargs)
    ds_out = getattr(ds_resampler, thefunction)()

    for v in ds_out:
        extra_dims = [d for d in ds_out[v].dims if d not in orig_dims[v]]
        if not extra_dims:
            print(f"Skipping {v}")
            continue

        # Xarray's resampler for now only supports resampling along one
        # dimension. However, I'm going to try and support a future
        # version that supports an arbitrary number of resampled dimensions;
        # this is, of course, untested.

        # For any newly-created dimensions, select the first value. Should
        # Be the same as all other values.
        indexers = dict([(d, 0) for d in extra_dims])

        ds_out[v] = ds_out[v].isel(**indexers, drop=True)

    return ds_out


# Have a DataArray that's just for one time point? Repeat it for many.
def tile_over_time(da_in, years=None):
    if "time" not in da_in.dims:
        raise RuntimeError("Rework tile_over_time() to function with da_in lacking time dimension.")

    # Deal with Datasets
    if isinstance(da_in, xr.Dataset):
        new_attrs = {}
        for x in da_in.attrs:
            if x == "created":
                continue
            else:
                new_attrs[x] = da_in.attrs[x]
        ds_out = xr.Dataset(attrs=new_attrs)
        for v in da_in:
            if "time" in da_in[v].dims and v != "time_bounds":
                ds_out[v] = tile_over_time(da_in[v], years=years)
            else:
                ds_out[v] = da_in[v].copy()
        return ds_out
    elif not isinstance(da_in, xr.DataArray):
        raise RuntimeError(
            f"tile_over_time() only works with xarray Datasets and DataArrays, not {type(da_in)}"
        )

    # Get info about time in input DataArray
    dt0 = da_in.time.values[0]
    dt_type = type(dt0)
    has_year_zero = dt0.has_year_zero

    if type(years) != type(None):
        new_time = np.array(
            [dt_type(x, dt0.month, dt0.day, has_year_zero=has_year_zero) for x in years]
        )
        strftime_fmt = "%Y-%m-%d"
        Ntime = len(years)
    else:
        raise RuntimeError("Rework tile_over_time to work with something other than years")

    # Convert from cftime to int days since, for compatibility with NETCDF3-CLASSIC format.
    new_time_units = "days since " + new_time[0].strftime(strftime_fmt)
    new_time = cftime.date2num(new_time, new_time_units)

    # Set up time DataArray
    new_time_attrs = da_in.time.attrs
    new_time_attrs["units"] = new_time_units
    new_time_attrs["calendar"] = dt0.calendar
    new_time_da = xr.DataArray(
        new_time, dims=["time"], coords={"time": new_time}, attrs=new_time_attrs
    )

    # Get coordinates to be used in new DataArray
    new_coords = {}
    for x in da_in.coords:
        if x == "time":
            new_coords[x] = new_time_da
        else:
            new_coords[x] = da_in.coords[x]

    # Set up broadcasting DataArray
    new_shape = tuple([da_in.shape[i] if x != "time" else Ntime for i, x in enumerate(da_in.dims)])
    bc_da = xr.DataArray(np.ones(new_shape), dims=da_in.dims, coords=new_coords)

    # Get new DataArray
    da_out = (da_in.squeeze() * bc_da).transpose(*da_in.dims)
    da_out = da_out.assign_attrs(da_in.attrs)

    return da_out
