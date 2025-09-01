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
except ImportError:
    # Fallback to absolute import if running as a script
    sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    from crops.cftlist import CftList
    from crops.croplist import CropList
    from crops.mark_crops_invalid import mark_crops_invalid
    import crops.crop_secondary_variables as c2o
    import crops.crop_utils as cu
    from crops.crop_defaults import N_PFTS
    from utils import food_grainc_to_harvested_tons_onecrop, ivt_int2str


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
    test_var = "HDATES"

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
        ds = xr.open_dataset(file_list[0])
        if test_var in ds:
            this_h_tape = h_tape
            break
    if this_h_tape is None:
        raise KeyError(f"No history tape in {h_tapes} had {test_var}")
    return this_h_tape


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
            verbose (bool): Whether to print verbose output.
            n_pfts (int): Number of PFTs.
        """
        self.verbose = verbose
        self.name = name
        # Get the tape we need to import (h0i, h2a, etc.)
        this_h_tape = _get_crop_tape(file_dir, name)
        # Get list of all files
        file_pattern = os.path.join(file_dir, name + ".clm2." + this_h_tape + ".*.nc")
        file_list = np.sort(glob.glob(file_pattern))
        if len(file_list) == 0:
            raise FileNotFoundError("No files found matching pattern: " + file_pattern)

        # Get list of files to actually include
        self.file_list = []
        for filename in file_list:
            ds = xr.open_dataset(filename)
            if ds.time.values[0].year <= end_year and start_year <= ds.time.values[-1].year:
                self.file_list.append(filename)
        if not self.file_list:
            raise FileNotFoundError(f"No files found with timestamps in {start_year}-{end_year}")

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
        )

        # Get CFT info
        self.cft_list = CftList(ds, n_pfts, cfts_to_include)

        # Get crop list
        self.crop_list = CropList(crops_to_include, self.cft_list, ds)

        # Save CFT dataset
        for i, cft in enumerate(self.cft_list):
            this_cft_ds = cu.get_cft_ds(ds, cft)

            if i == 0:
                self.cft_ds = this_cft_ds.copy()
                n_expected = self.cft_ds.sizes["pft"]
            else:
                # Check # of gridcells with this PFT
                n_this = this_cft_ds.sizes["pft"]
                if n_this != n_expected:
                    raise RuntimeError(
                        f"Expected {n_expected} gridcells with {cft.name}; found {n_this}"
                    )
                self.cft_ds = xr.concat(
                    [self.cft_ds, this_cft_ds],
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
            self.cft_ds[var] = self.cft_ds[var].where(self.cft_ds[var] >= 0)
        self.cft_ds["HUIFRAC_PERHARV"] = c2o.get_huifrac(self.cft_ds)
        self.cft_ds["GSLEN_PERHARV"] = c2o.get_gslen(self.cft_ds)
        if self.verbose:
            end = time()
            print(f"Secondary variables took {int(end - start)} s")

        # Get yield, marking non-viable harvests as zero and converting to wet matter
        self.get_yield()

    def get_yield(self):
        """
        Get yield, marking non-viable harvests as zero and converting to wet matter
        """
        if not any("_TO_FOOD_PERHARV" in v for v in self.cft_ds):
            print("WARNING: Will not calculate yield because crop maturity can't be assessed")
            return

        # Create DataArray with zeroes where harvest is invalid and ones elsewhere
        is_valid_harvest = mark_crops_invalid(self.cft_ds, min_viable_hui="isimip3")

        # Mark invalid harvests as zero
        for v in self.cft_ds:
            if not re.match(r"GRAIN[CN]_TO_FOOD_PERHARV", v):
                continue

            # Change, e.g., GRAINC_TO_FOOD_PERHARV to GRAINC_TO_FOOD_PERHARV
            new_var = v.replace("_PERHARV", "_VIABLE_PERHARV")
            da_new = self.cft_ds[v] * is_valid_harvest
            self.cft_ds[new_var] = da_new
            self.cft_ds[new_var].attrs["long_name"] = "grain C to food in VIABLE harvested organ per harvest"

            # Get annual values
            new_var_ann = new_var.replace("PERHARV", "ANN")
            self.cft_ds[new_var_ann] = self.cft_ds[new_var].sum(dim="mxharvests")
            self.cft_ds[new_var].attrs["long_name"] = "grain C to food in VIABLE harvested organ per calendar year"

        # Calculate actual yield (wet matter)
        c_var = "GRAINC_TO_FOOD_VIABLE_PERHARV"
        if c_var in self.cft_ds:
            wm_arr = np.full_like(self.cft_ds[c_var].values, np.nan)
            for i, pft_int in enumerate(self.cft_ds["cft"].values):
                pft_str = ivt_int2str(pft_int)
                if self.cft_ds[c_var].dims[0] != "cft":
                    raise NotImplementedError("Below code (wm_arr[i]) assumes cft is the 0th dimension")
                wm_arr[i] = self.cft_ds[c_var].sel(cft=pft_int)
                wm_arr[i] = food_grainc_to_harvested_tons_onecrop(wm_arr[i], pft_str)
            self.cft_ds["YIELD_PERHARV"] = xr.DataArray(
                data=wm_arr,
                coords=self.cft_ds[c_var].coords,
                dims=self.cft_ds[c_var].dims,
                attrs={
                    "long_name": "viable wet matter yield (minus losses) per harvest",
                    "units": "g wet matter / m^2"
                },
            )
            self.cft_ds["YIELD_ANN"] = self.cft_ds["YIELD_PERHARV"].sum(dim="mxharvests")
