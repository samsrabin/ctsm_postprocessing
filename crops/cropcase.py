"""
Module for handling crop cases in the Community Terrestrial Systems Model (CTSM).

This module defines classes and functions for managing crop cases, including initializing crop
cases, updating crop data, and retrieving crop-specific information.
"""

import os
import sys
import glob
import re
from time import time
import xarray as xr
import numpy as np

try:
    # Attempt relative import if running as part of a package
    from .cftlist import CftList
    from .croplist import CropList
    from .mark_crops_invalid import mark_crops_invalid
    from . import crop_secondary_variables as c2o
    from . import crop_utils as cu
    from .crop_defaults import N_PFTS
    from ..utils import food_grainc_to_harvested_tons_onecrop, ivt_int2str
    from .extra_area_prod_yield_etc import extra_area_prod_yield_etc
except ImportError:
    # Fallback to absolute import if running as a script
    sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    from crops.cftlist import CftList
    from crops.croplist import CropList
    from crops.mark_crops_invalid import mark_crops_invalid
    import crops.crop_secondary_variables as c2o
    import crops.crop_utils as cu
    from crops.crop_defaults import N_PFTS
    from crops.extra_area_prod_yield_etc import extra_area_prod_yield_etc
    from utils import food_grainc_to_harvested_tons_onecrop, ivt_int2str

CFT_DS_FILENAME = "cft_ds.nc"
CFT_DS_CHUNKING = {"cft": 1, "crop": 1}


def _mf_preproc(ds):
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
        cfts_to_include,
        crops_to_include,
        start_year,
        end_year,
        verbose=False,
        n_pfts=N_PFTS,
        force_new_cft_ds_file=False,
        force_no_cft_ds_file=False,
        cft_ds_dir=None,
        this_h_tape=None,
    ):
        # pylint: disable=too-many-positional-arguments
        """
        Initialize a CropCase instance.

        Parameters:
            name (str): Name of the crop case.
            file_dir (str): Directory containing the crop data files.
            cfts_to_include (list): List of CFTs to include in the crop case.
            crops_to_include (list): List of crops to include in the crop case.
            start_year (int): Start year for the crop data.
            end_year (int): End year for the crop data.
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
                if self.verbose:
                    print(f"Saving {self.cft_ds_file}...")
                self.cft_ds.to_netcdf(self.cft_ds_file)
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
            start_date = f"{start_year}-01-01"
            end_date = f"{end_year}-12-31"
            time_slice = slice(start_date, end_date)
            self.cft_ds = self.cft_ds.sel(time=time_slice)
            end = time()
            print(f"Opening cft_ds took {int(end - start)} s")

        # cft_crop is often a groupby() variable, so computing it makes things more efficient.
        # Avoids DeprecationWarning that will become an error in xarray v2025.05.0+
        if hasattr(self.cft_ds["cft_crop"].data, "compute"):
            self.cft_ds["cft_crop"] = self.cft_ds["cft_crop"].compute()

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
            start_year_ok = start_year is None or start_year <= ds.time.values[-1].year
            end_year_ok = end_year is None or ds.time.values[0].year <= end_year
            if start_year_ok and end_year_ok:
                self.file_list.append(filename)
        if not self.file_list:
            raise FileNotFoundError(f"No files found with timestamps in {start_year}-{end_year}")

    def _get_cft_and_crop_lists(self, cfts_to_include, crops_to_include, n_pfts, ds):
        """
        Get lists of CFTs and crops included in history
        """
        # Get CFT list
        self.cft_list = CftList(ds, n_pfts, cfts_to_include)

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

        # Get yield, marking non-viable harvests as zero and converting to wet matter
        self.get_yield(cft_ds)

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

        return cft_ds

    def get_yield(self, cft_ds):
        """
        Get yield, marking non-viable harvests as zero and converting to wet matter
        """
        if not any("_TO_FOOD_PERHARV" in v for v in cft_ds):
            print("WARNING: Will not calculate yield because crop maturity can't be assessed")
            return

        # Create DataArray with zeroes where harvest is invalid and ones elsewhere
        is_valid_harvest = mark_crops_invalid(cft_ds, min_viable_hui="isimip3")
        cft_ds["VALID_HARVEST"] = is_valid_harvest

        # Mark invalid harvests as zero
        product_list = ["FOOD", "SEED"]
        for p in product_list:
            for v in cft_ds:
                if not re.match(fr"GRAIN[CN]_TO_{p}_PERHARV", v):
                    continue

                # Change, e.g., GRAINC_TO_FOOD_PERHARV to GRAINC_TO_FOOD_PERHARV
                new_var = v.replace("_PERHARV", "_VIABLE_PERHARV")
                da_new = cft_ds[v] * is_valid_harvest
                cft_ds[new_var] = da_new
                cft_ds[new_var].attrs["units"] = cft_ds[v].attrs["units"]
                long_name = f"grain C to {p.lower()} in VIABLE harvested organ per harvest"
                cft_ds[new_var].attrs["long_name"] = long_name

                # Get annual values
                new_var_ann = new_var.replace("PERHARV", "ANN")
                cft_ds[new_var_ann] = cft_ds[new_var].sum(dim="mxharvests")
                cft_ds[new_var_ann].attrs[
                    "long_name"
                ] = long_name.replace("per harvest", "per calendar year")

        # Calculate actual yield (wet matter)
        c_var = "GRAINC_TO_FOOD_VIABLE_PERHARV"
        if c_var in cft_ds:
            wm_arr = np.full_like(cft_ds[c_var].values, np.nan)
            for i, pft_int in enumerate(cft_ds["cft"].values):
                pft_str = ivt_int2str(pft_int)
                if cft_ds[c_var].dims[0] != "cft":
                    raise NotImplementedError(
                        "Below code (wm_arr[i]) assumes cft is the 0th dimension"
                    )
                wm_arr[i] = cft_ds[c_var].sel(cft=pft_int)
                wm_arr[i] = food_grainc_to_harvested_tons_onecrop(wm_arr[i], pft_str)
            cft_ds["YIELD_PERHARV"] = xr.DataArray(
                data=wm_arr,
                coords=cft_ds[c_var].coords,
                dims=cft_ds[c_var].dims,
                attrs={
                    "long_name": "viable wet matter yield (minus losses) per harvest",
                    "units": "g wet matter / m^2",
                },
            )
            cft_ds["YIELD_ANN"] = cft_ds["YIELD_PERHARV"].sum(dim="mxharvests", keep_attrs=True)
            long_name = cft_ds["YIELD_ANN"].attrs["long_name"]
            long_name = long_name.replace("per harvest", "per calendar year")

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
