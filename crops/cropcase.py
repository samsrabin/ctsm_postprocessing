"""
Module for handling crop cases in the Community Terrestrial Systems Model (CTSM).

This module defines classes and functions for managing crop cases, including initializing crop
cases, updating crop data, and retrieving crop-specific information.
"""

import os
import sys
import glob
from time import time
import xarray as xr
import numpy as np

try:
    # Attempt relative import if running as part of a package
    from .cftlist import CftList
    from .croplist import CropList
    from . import crop_secondary_variables as c2o
    from . import crop_utils as cu
    from .crop_defaults import N_PFTS
except ImportError:
    # Fallback to absolute import if running as a script
    sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    from crops.cftlist import CftList
    from crops.croplist import CropList
    from crops.mark_crops_invalid import mark_crops_invalid
    import crops.crop_secondary_variables as c2o
    import crops.crop_utils as cu
    from crops.crop_defaults import N_PFTS


def _mf_preproc(ds):
    ds["pfts1d_wtgcell"] = ds["pfts1d_wtgcell"].expand_dims(dim="time", axis=0)

    # Some variables are only saved in the first file of a run segment. These cause problems for
    # open_mfdataset(), and since we don't really care about them, just drop them.
    vars_to_drop = [x for x in ds if any("lev" in d for d in ds[x].dims)]
    ds = ds.drop_vars(vars_to_drop)

    return ds


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
        clm_file_h,
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
            clm_file_h (str): File header for the crop data files.
            cfts_to_include (list): List of CFTs to include in the crop case.
            crops_to_include (list): List of crops to include in the crop case.
            start_year (int): Start year for the crop data.
            end_year (int): End year for the crop data.
            verbose (bool): Whether to print verbose output.
            n_pfts (int): Number of PFTs.
        """
        self.verbose = verbose
        # Get list of all time series files
        file_pattern = os.path.join(file_dir, name + ".clm2" + clm_file_h + "*.nc")
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
