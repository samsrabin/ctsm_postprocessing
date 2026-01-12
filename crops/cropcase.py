"""
Module for handling crop cases in the Community Terrestrial Systems Model (CTSM).

This module defines classes and functions for managing crop cases, including initializing crop
cases, updating crop data, and retrieving crop-specific information.
"""

import os
import sys
import glob
from time import time
from typing import Union
from tempfile import NamedTemporaryFile
from shutil import move

import xarray as xr
import numpy as np

# A type alias for convenience
PathLike = Union[str, bytes, os.PathLike]

try:
    # Attempt relative import if running as part of a package
    from .cftlist import CftList
    from .croplist import CropList
    from . import crop_secondary_variables as c2o
    from . import crop_utils as cu
    from .crop_defaults import N_PFTS, DEFAULT_CFTS_TO_INCLUDE, DEFAULT_CROPS_TO_INCLUDE
    from .extra_area_prod_yield_etc import extra_area_prod_yield_etc
    from .crop_biomass import get_crop_biomass_vars
except ImportError:
    # Fallback to absolute import if running as a script
    sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    from crops.cftlist import CftList
    from crops.croplist import CropList
    import crops.crop_secondary_variables as c2o
    import crops.crop_utils as cu
    from crops.crop_defaults import N_PFTS, DEFAULT_CFTS_TO_INCLUDE, DEFAULT_CROPS_TO_INCLUDE
    from crops.extra_area_prod_yield_etc import extra_area_prod_yield_etc
    from crops.crop_biomass import get_crop_biomass_vars

CFT_DS_FILENAME = "cft_ds.nc"
CFT_DS_CHUNKING = {"cft": 1, "crop": 1}


def _save_cft_ds_to_netcdf(cft_ds: xr.Dataset, file_path: PathLike, verbose: bool):
    """Save cft_ds to a temporary netCDF file, then move it to final destination"""
    if verbose:
        print(f"Saving {file_path}...")

    with NamedTemporaryFile(delete=False) as tf:
        cft_ds.to_netcdf(tf.name)
        move(tf.name, file_path)


def _mf_preproc(ds):
    """Every netCDF file is run through this function after being read"""

    # Check that time axis is what we expect: The variables we care about need to be saved
    # on instantaneous files at the end of every year in order to be meaningful.
    if ds["time"].attrs["long_name"] != "time at end of time step":
        raise ValueError("Variables for cft_ds should be on instantaneous files but aren't")
    t = ds["time"].values[0]
    if not (t.month == t.day == 1) and (t.hour == t.minute == t.second == t.microsecond == 0):
        raise ValueError(f"Expected file timestamp Jan. 1 00:00:00; got {t}")

    # If we care about pfts1d_wtgcell (the fraction of the grid cell taken up by each PFT during the
    # model run; i.e., after applying mergetoclmpft), this is necessary for outputs before
    # ctsm5.3.064. Otherwise, xarray will not recognize that it can vary over time, and we'll just
    # get the first timestep's values.
    if "time" not in ds["pfts1d_wtgcell"].dims:
        ds["pfts1d_wtgcell"] = ds["pfts1d_wtgcell"].expand_dims(dim="time", axis=0)

    # Some variables are only saved in the first file of a run segment. These cause problems for
    # open_mfdataset(), and since we don't really care about them, just drop them.
    vars_to_drop = [x for x in ds if any("lev" in d for d in ds[x].dims)]
    ds = ds.drop_vars(vars_to_drop)

    return ds


def _get_crop_tape(file_dir, name):
    """
    Get the tape (h0i, h2a, etc.) associated with crop outputs
    """

    # The variable we will use to determine which tape has the outputs we need
    test_var = "GRAINC_TO_FOOD_PERHARV"

    # Get the list of tape IDs
    results = os.listdir(file_dir)
    results.sort()
    results = [x for x in results if name in x]
    h_tapes = set(x.split(".")[-3] for x in results)
    for h_tape in h_tapes:
        if not h_tape.startswith("h"):
            raise RuntimeError(f"Failed to parse list of tapes (?); got: {h_tapes}")

    # Figure out which tape has the variable we're checking for
    this_h_tape = None
    for h_tape in h_tapes:
        pattern = os.path.join(file_dir, f"*clm2.{h_tape}.*.nc")
        file_list = glob.glob(pattern)
        if not file_list:
            print(f"_get_crop_tape(): No {h_tape} files found")
            continue
        ds = xr.open_dataset(file_list[0], decode_timedelta=False)
        if test_var in ds:
            this_h_tape = h_tape
            break
    if this_h_tape is None:
        raise KeyError(f"No history tape in {h_tapes} had {test_var}")
    return this_h_tape


def _get_area_p(ds):
    """
    Get area of gridcell that is parent of each pft (patch)
    """
    area_g = []
    area_da = ds["area"] * ds["landfrac"]
    for i, lon in enumerate(ds["grid1d_lon"].values):
        lat = ds["grid1d_lat"].values[i]
        area_g.append(area_da.sel(lat=lat, lon=lon))
    area_g = np.array(area_g)
    area_p = []
    gridcell_vals = list(np.unique(ds["pfts1d_gi"].isel(cft=0).values))
    for i in ds["pfts1d_gi"].isel(cft=0).values:
        area_p.append(area_g[gridcell_vals.index(int(i))])
    area_p = np.array(area_p)
    return area_p


class CropCase:
    # pylint: disable=too-few-public-methods
    """
    Represents a crop case in the Community Terrestrial Systems Model (CTSM).

    Attributes:
        name (str): Name of the crop case.
        crops (list): List of crops included in this crop case.
        ds (xarray.Dataset): Dataset containing crop data.
    """

    def __init__(
        self,
        name,
        file_dir,
        *,
        start_year=None,
        end_year=None,
        verbose=False,
        n_pfts=N_PFTS,
        force_new_cft_ds_file=False,
        force_no_cft_ds_file=False,
        cft_ds_dir=None,
        this_h_tape=None,
        # TODO: Future-proof default: Determine from ds upon initialization.
        cfts_to_include=DEFAULT_CFTS_TO_INCLUDE,
        crops_to_include=DEFAULT_CROPS_TO_INCLUDE,
    ):
        # pylint: disable=too-many-positional-arguments
        """
        Initialize a CropCase instance.

        Parameters:
            name (str): Name of the crop case.
            file_dir (str): Directory containing the crop data files.
            crops_to_include (list): List of crops to include in the crop case.
            start_year (int): The first calendar year whose data should be included. Note that,
                              because the variables we're processing are only meaningful if saved
                              to instantaneous files at the end of a year, calendar year Y gets its
                              data associated with a timestep whose year is Y+1. This class assumes
                              you want to read files starting with timestep start_year-1.
            end_year (int): The last calendar year whose data should be included. See note for
                            start_year above.
            verbose (bool): Whether to print verbose output. Default False.
            n_pfts (int): Number of PFTs. Default N_PFTS.
            force_new_cft_ds_file (bool): Even if cft_ds file exists, read and save a new one. Default False.
            force_no_cft_ds_file (bool): Don't try to read or save cft_ds file.
            cft_ds_dir (str): Where to save the cft_ds file. Default same as file_dir.
        """
        self.verbose = verbose
        self.name = name
        self.file_dir = file_dir
        self.file_list = []
        self.cft_list = None
        self.crop_list = None

        # Check incompatible options
        if force_new_cft_ds_file and force_no_cft_ds_file:
            raise ValueError("force_new_cft_ds_file and force_no_cft_ds_file can't both be True")
        for cft in cfts_to_include:
            if not any(crop in cft for crop in crops_to_include):
                raise KeyError(f"Which crop should {cft} be associated with?")

        # Create CFT dataset file if needed
        if cft_ds_dir is None:
            cft_ds_dir = self.file_dir
        self.cft_ds_file = os.path.join(cft_ds_dir, CFT_DS_FILENAME)
        if force_new_cft_ds_file or force_no_cft_ds_file or not os.path.exists(self.cft_ds_file):
            user_has_write_perms = os.access(cft_ds_dir, os.W_OK)
            save_netcdf = user_has_write_perms and not force_no_cft_ds_file
            if save_netcdf:
                # If we're generating cft_ds.nc, we'll read all years
                start_file_year = None
                end_file_year = None
            else:
                if not user_has_write_perms and not force_no_cft_ds_file:
                    print(f"User can't write in {cft_ds_dir}, so {CFT_DS_FILENAME} won't be saved")
                start_file_year = start_year
                end_file_year = end_year
            msg = f"Making {CFT_DS_FILENAME}"
            if save_netcdf:
                msg = msg.replace("Making", "Making and saving")
            start = time()
            self.cft_ds = self._read_and_process_files(
                cfts_to_include,
                crops_to_include,
                n_pfts,
                start_file_year,
                end_file_year,
                this_h_tape,
            )
            if save_netcdf:
                _save_cft_ds_to_netcdf(self.cft_ds, self.cft_ds_file, self.verbose)
            end = time()
            if self.verbose:
                print(f"{msg} took {int(end - start)} s")

        # Open CFT dataset and slice based on years
        if os.path.exists(self.cft_ds_file) and not force_no_cft_ds_file:
            # Always prefer to read from the file, to ensure consistency of performance
            self.cft_ds = None
            start = time()
            self.cft_ds = xr.open_dataset(
                self.cft_ds_file,
                decode_timedelta=False,
                chunks=CFT_DS_CHUNKING,
                chunked_array_type="dask",
            )

            # Slice based on years, if start_year or end_year requested.
            # A variable saved at the end of the last timestep of a year (and therefore containing
            # data for that year) gets a timestamp with the NEXT year, but we want the user to give
            # the calendar years they actually care about. Thus, here we add 1 to the start and end
            # years the user requested. (See _mf_preproc() for check that time axis is right for
            # this.)
            if any(date is not None for date in [start_year, end_year]):
                start_date = None if start_year is None else f"{start_year + 1}-01-01"
                end_date = None if end_year is None else f"{end_year + 1}-12-31"
                time_slice = slice(start_date, end_date)
                self.cft_ds = self.cft_ds.sel(time=time_slice)

            end = time()
            print(f"Opening cft_ds took {int(end - start)} s")

        # The time axis is weird: Timestep Y-01-01 00:00:00 actually has data for calendar year
        # Y+1. Here, we replace it to be simpler, where the timestep is just an integer year
        # corresponding to the year the data came from.
        self.cft_ds["time"] = xr.DataArray(
            data=np.array([t.year - 1 for t in self.cft_ds["time"].values]),
            dims=["time"],
            attrs={"long_name": "year"}
        )

        # cft_crop is often a groupby() variable, so computing it makes things more efficient.
        # Avoids DeprecationWarning that will become an error in xarray v2025.05.0+
        if hasattr(self.cft_ds["cft_crop"].data, "compute"):
            self.cft_ds["cft_crop"] = self.cft_ds["cft_crop"].compute()

        # At some point I changed "viable"/"valid" harvest variable names to "marketable". cft_ds
        # variables saved before that need to be renamed to match.
        rename_dict = {}
        for v in self.cft_ds:
            if v == "VALID_HARVEST":
                rename_dict[v] = "MARKETABLE_HARVEST"
            elif "VIABLE" in v:
                rename_dict[v] = v.replace("VIABLE", "MARKETABLE")
            elif v.endswith("_harv_area_failed"):
                rename_dict[v] = v.replace("failed", "unmarketable")
        if rename_dict:
            self.cft_ds = self.cft_ds.rename(rename_dict)

    def __eq__(self, other):
        # Check that they're both CropCases
        if not isinstance(other, self.__class__):
            raise TypeError(f"== not supported between {self.__class__} and {type(other)}")

        # Check that all attributes match (excluding methods)
        for attr in [a for a in dir(self) if not a.startswith("__")]:
            # Skip callable attributes (methods)
            if callable(getattr(self, attr)):
                continue
            if not hasattr(other, attr):
                return False
            try:
                value_self = getattr(self, attr)
                value_other = getattr(other, attr)
                if not isinstance(value_other, type(value_self)):
                    return False
                if isinstance(value_self, xr.Dataset) and not value_self.equals(value_other):
                    return False
                if not value_self == value_other:
                    return False
            except:  # pylint: disable=bare-except
                return False
        return True

    def __ne__(self, other):
        return not self == other

    def __repr__(self):
        if self.name is not None:
            return f"CropCase({self.name})"
        if "case_id" in self.cft_ds.attrs and self.cft_ds.attrs["case_id"] is not None:
            return f"CropCase({self.cft_ds.attrs['case_id']})"
        return super().__repr__()

    def _read_and_process_files(
        self, cfts_to_include, crops_to_include, n_pfts, start_year, end_year, this_h_tape
    ):
        """
        Read all history files and create the "CFT dataset"
        """
        self._get_file_list(start_year, end_year, this_h_tape)

        # Read files
        # Adding join="override", compat="override", coords="minimal", doesn't fix the graph size
        # Adding combine="nested", concat_dim="time" doesn't give time axis to only variables we
        # want
        ds = xr.open_mfdataset(
            self.file_list,
            decode_times=True,
            chunks={},
            join="override",
            compat="override",
            coords="minimal",
            # combine="nested", concat_dim="time",
            data_vars="minimal",
            preprocess=_mf_preproc,
            decode_timedelta=False,
        )

        # Get CFT and crop lists
        self._get_cft_and_crop_lists(cfts_to_include, crops_to_include, n_pfts, ds)

        # Process into CFT dataset
        cft_ds = self._get_cft_ds(crops_to_include, ds)

        # Chunk
        cft_ds = cft_ds.chunk(chunks=CFT_DS_CHUNKING)

        return cft_ds

    def _get_file_list(self, start_year, end_year, this_h_tape):
        """
        Get the files to import
        """
        # Get the tape we need to import (h0i, h2a, etc.)
        if this_h_tape is None:
            this_h_tape = _get_crop_tape(self.file_dir, self.name)
        # Get list of all files
        file_pattern = os.path.join(self.file_dir, self.name + ".clm2." + this_h_tape + ".*.nc")
        file_list = np.sort(glob.glob(file_pattern))
        if len(file_list) == 0:
            raise FileNotFoundError("No files found matching pattern: " + file_pattern)

        # Get list of files to actually include
        for filename in file_list:
            ds = xr.open_dataset(filename)

            # A variable saved at the end of the last timestep of a year (and therefore containing
            # data for that year) gets a timestamp with the NEXT year, but we want the user to give
            # the calendar years they actually care about. Thus, here we add 1 to the start and end
            # years the user requested. (See _mf_preproc() for check that time axis is right for
            # this.)
            start_year_ok = start_year is None or (start_year + 1) <= ds.time.values[-1].year
            end_year_ok = end_year is None or ds.time.values[0].year <= (end_year + 1)
            if start_year_ok and end_year_ok:
                self.file_list.append(filename)
        if not self.file_list:
            raise FileNotFoundError(f"No files found with timestamps in {start_year}-{end_year}")

    def _get_cft_and_crop_lists(self, cfts_to_include, crops_to_include, n_pfts, ds):
        """
        Get lists of CFTs and crops included in history
        """
        # Get CFT list
        self.cft_list = CftList(ds, n_pfts, cfts_to_include=cfts_to_include)

        # Get crop list
        self.crop_list = CropList(crops_to_include, self.cft_list, ds)

    def _get_cft_ds(self, crops_to_include, ds):
        """
        Postprocess the history dataset into the "CFT dataset"
        """
        for i, cft in enumerate(self.cft_list):
            this_cft_ds = cu.get_cft_ds(ds, cft)

            if i == 0:
                cft_ds = this_cft_ds.copy()
                n_expected = cft_ds.sizes["pft"]
            else:
                # Check # of gridcells with this PFT
                n_this = this_cft_ds.sizes["pft"]
                if n_this != n_expected:
                    raise RuntimeError(
                        f"Expected {n_expected} gridcells with {cft.name}; found {n_this}"
                    )
                cft_ds = xr.concat(
                    [cft_ds, this_cft_ds],
                    dim="cft",
                    data_vars="minimal",
                    compat="override",
                    join="override",
                    coords="minimal",
                )

        # Get secondary variables
        if self.verbose:
            start = time()
            print("Getting secondary variables")
        for var in ["HDATES", "SDATES_PERHARV"]:
            if var not in cft_ds:
                print(f"{var} not found in Dataset")
                continue
            cft_ds[var] = cft_ds[var].where(cft_ds[var] >= 0)
        cft_ds["HUIFRAC_PERHARV"] = c2o.get_huifrac(cft_ds)
        gslen_perharv = c2o.get_gslen(cft_ds)
        if gslen_perharv is None:
            print("Could not calculate GSLEN_PERHARV")
        else:
            cft_ds["GSLEN_PERHARV"] = gslen_perharv
        if self.verbose:
            end = time()
            print(f"Secondary variables took {int(end - start)} s")

        # Get gridcell land area
        cft_ds.load()
        area_p = _get_area_p(cft_ds)
        cft_ds["pfts1d_landarea"] = xr.DataArray(
            data=area_p,
            coords={"pft": cft_ds["pft"].values},
            dims=["pft"],
        )

        # Get more stuff
        cft_ds = extra_area_prod_yield_etc(crops_to_include, self, cft_ds)
        cft_ds = get_crop_biomass_vars(cft_ds, self.name)

        return cft_ds

    @classmethod
    def _create_empty(cls):
        """
        Create an empty CropCase without going through the normal initialization (i.e., import).
        Used internally by sel() and isel() for creating copies.
        """
        # Create instance without calling __init__
        instance = cls.__new__(cls)
        return instance

    def _copy_other_attributes(self, dest_case_list):
        """
        Copy all CropCase attributes from self to destination CropCase, skipping cft_ds.
        """
        for attr in [a for a in dir(self) if not a.startswith("__")]:
            if attr == "cft_ds":
                continue
            # Skip callable attributes (methods) - they should be inherited from the class
            if callable(getattr(self, attr)):
                continue
            setattr(dest_case_list, attr, getattr(self, attr))
        return dest_case_list

    def sel(self, *args, **kwargs):
        """
        Makes a copy of CropCase with cft_ds having had Dataset.sel() applied with given arguments.
        """

        new_case = self._create_empty()

        # .sel() from cft_ds
        new_case.cft_ds = self.cft_ds.sel(*args, **kwargs)

        # Copy over other attributes
        new_case = self._copy_other_attributes(new_case)

        return new_case

    def isel(self, *args, **kwargs):
        """
        Makes a copy of CropCase with cft_ds having had Dataset.isel() applied with given arguments.
        """
        new_case = self._create_empty()

        # .isel() from cft_ds
        new_case.cft_ds = self.cft_ds.isel(*args, **kwargs)

        # Copy over other attributes
        new_case = self._copy_other_attributes(new_case)

        return new_case
