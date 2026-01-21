"""
A class for holding a list of CropCases and information about them
"""

from __future__ import annotations

import os
import warnings
from typing import Any
from time import time

import xarray as xr

from .cropcase import CropCase
from ..resolutions import identify_resolution

# The variables needed for regridding
REGRID_VARS = ["area", "landfrac", "landmask"]


def _ds_has_regrid_vars(ds: xr.Dataset) -> bool:
    """
    Check if a dataset has all required regridding variables.

    Parameters
    ----------
    ds : xarray.Dataset
        Dataset to check for regridding variables.

    Returns
    -------
    bool
        True if all REGRID_VARS are present in the dataset, False otherwise.
    """
    return all(v in ds.variables for v in REGRID_VARS)


class CropCaseList(list):
    """
    A class for holding a list of CropCases and information about them
    """

    def __init__(
        self,
        *args,
        opts: dict[str, Any],
    ) -> None:
        """
        Initialize a CropCaseList instance.

        Parameters
        ----------
        *args
            Arguments passed to list initialization.
        opts : dict[str, Any]
            Dictionary containing configuration options. Expected keys include:
            - case_name_list: List of case names to import
            - CESM_output_dir: Directory containing CESM output
            - start_year: Starting year for data
            - end_year: Ending year for data
            - verbose: Whether to print verbose output
            - force_new_cft_ds_file: Whether to force creation of new CFT dataset files
            - force_no_cft_ds_file: Whether to avoid using CFT dataset files
        """
        # Initialize as a normal list...
        super().__init__(*args)
        # ...And then add all the extra stuff

        # Define extra variables
        self.names = opts["case_name_list"]

        # Import cases
        self._import_cases(
            opts,
        )

        # Get resolutions and regrid-target Datasets
        self.resolutions = {}
        for case in self:
            res = case.cft_ds.attrs["resolution"]

            # Get minimum Dataset needed for regridding EarthStat to match
            self._save_or_check_regrid_ds(case, res)

    def _save_or_check_regrid_ds(self, case: CropCase, res: str) -> None:
        """
        Save or check regridding dataset for a given resolution.

        Parameters
        ----------
        case : CropCase
            Case containing the dataset to process.
        res : str
            Resolution identifier.
        """
        vars_to_drop = [v for v in case.cft_ds if v not in REGRID_VARS]
        regrid_ds = case.cft_ds.drop_vars(vars_to_drop)

        # Resolution not seen yet: Save regrid_ds
        if res not in self.resolutions.keys():  # pylint: disable=consider-iterating-dictionary
            self.resolutions[res] = regrid_ds

        # Resolution already seen but doesn't have all needed vars, and this one does, replace
        # saved one with this one
        elif _ds_has_regrid_vars(regrid_ds) and not _ds_has_regrid_vars(self.resolutions[res]):
            self.resolutions[res] = regrid_ds.compute()

        # Resolution already seen: Check that it matches what we have saved
        else:
            self._check_regrid_ds(case, res, regrid_ds)

    def _check_regrid_ds(self, case: CropCase, res: str, regrid_ds: xr.Dataset) -> None:
        """
        Check that regridding dataset matches previously saved dataset for this resolution.

        Parameters
        ----------
        case : CropCase
            Case being checked.
        res : str
            Resolution identifier.
        regrid_ds : xarray.Dataset
            Dataset to check against saved dataset.
        """
        saved_vars = set(self.resolutions[res].keys())
        this_vars = set(regrid_ds.keys())

        # Warn if this one is missing variable(s) present in saved one
        if saved_vars != this_vars:
            case_id_var = "case_id"
            if case_id_var in self.resolutions[res].attrs:
                saved_id = self.resolutions[res].attrs[case_id_var]
            else:
                saved_id = "saved regrid_ds"
            warnings.warn(
                f"Res {res}: {case.name} regrid_ds missing some variables present in {saved_id}",
                UserWarning,
            )

        # Warn if any variables don't match what we saved
        for var in saved_vars.intersection(this_vars):
            if not self.resolutions[res][var].equals(regrid_ds[var]):
                warnings.warn(
                    f"Res {res} regrid_ds[{var}]: {case.name}[{var}] doesn't match",
                    UserWarning,
                )

    def _check_attrs_match(self, other: CropCaseList) -> bool:
        """
        Check if all attributes match between this and another CropCaseList.

        Parameters
        ----------
        other : CropCaseList
            CropCaseList to compare with.

        Returns
        -------
        bool
            True if all attributes match, False otherwise.
        """
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
                if not value_self == value_other:
                    return False
            except:  # pylint: disable=bare-except
                return False
        return True

    def __eq__(self, other: object) -> bool:
        """
        Compare two CropCaseList instances for equality.

        Parameters
        ----------
        other : object
            Object to compare with this CropCaseList instance.

        Returns
        -------
        bool
            True if both CropCaseList instances are equal, False otherwise.

        Raises
        ------
        TypeError
            If other is not a CropCaseList instance.
        """
        # Check that they're both CropCaseLists
        if not isinstance(other, self.__class__):
            raise TypeError(f"== not supported between {self.__class__} and {type(other)}")

        # Check that all attributes match
        if not self._check_attrs_match(other):
            return False

        # Check that all cases match
        if len(self) != len(other):
            return False
        for c, case_self in enumerate(self):
            case_other = other[c]
            if case_self != case_other:
                return False
        return True

    def __ne__(self, other: object) -> bool:
        """
        Compare two CropCaseList instances for inequality.

        Parameters
        ----------
        other : object
            Object to compare with this CropCaseList instance.

        Returns
        -------
        bool
            True if CropCaseList instances are not equal, False otherwise.
        """
        return not self == other

    def _import_cases(
        self,
        opts: dict[str, Any],
    ) -> None:
        """
        Import all cases specified in opts.

        Parameters
        ----------
        opts : dict[str, Any]
            Dictionary containing configuration options including case_name_list,
            CESM_output_dir, start_year, end_year, verbose, force_new_cft_ds_file,
            and force_no_cft_ds_file.
        """
        start = time()
        for case_name in self.names:
            print(f"Importing {case_name}...")
            case_output_dir = os.path.join(
                opts["CESM_output_dir"],
                case_name,
                "lnd",
                "hist",
            )
            self.append(
                CropCase(
                    case_name,
                    case_output_dir,
                    start_year=opts["start_year"],
                    end_year=opts["end_year"],
                    verbose=opts["verbose"],
                    force_new_cft_ds_file=opts["force_new_cft_ds_file"],
                    force_no_cft_ds_file=opts["force_no_cft_ds_file"],
                ),
            )

            # Get resolution
            self[-1].cft_ds.attrs["resolution"] = identify_resolution(
                self[-1].cft_ds,
            ).name

        print("Done.")
        if opts["verbose"]:
            end = time()
            print(f"Importing took {int(end - start)} s")

    @classmethod
    def _create_empty(cls) -> CropCaseList:
        """
        Create an empty CropCaseList without going through the normal initialization (i.e., import).
        Used internally by sel() and isel() for creating copies.

        Returns
        -------
        CropCaseList
            Empty CropCaseList instance.
        """
        # Create instance without calling __init__
        instance = cls.__new__(cls)
        # Initialize as empty list
        list.__init__(instance)
        return instance

    def _copy_attributes(self, dest_case_list: CropCaseList) -> CropCaseList:
        """
        Copy all CropCaseList attributes from self to destination CropCaseList.

        Parameters
        ----------
        dest_case_list : CropCaseList
            Destination CropCaseList to copy attributes to.

        Returns
        -------
        CropCaseList
            The destination CropCaseList with copied attributes.
        """
        for attr in [a for a in dir(self) if not a.startswith("__")]:
            # Skip callable attributes (methods) - they should be inherited from the class
            if callable(getattr(self, attr)):
                continue
            setattr(dest_case_list, attr, getattr(self, attr))
        return dest_case_list

    def sel(self, *args, **kwargs) -> CropCaseList:
        """
        Makes a copy of this CropCaseList, applying CropCase.sel() with the given arguments.

        Parameters
        ----------
        *args
            Positional arguments passed to CropCase.sel().
        **kwargs
            Keyword arguments passed to CropCase.sel().

        Returns
        -------
        CropCaseList
            New CropCaseList with sel() applied to each case.
        """
        new_case_list = self._create_empty()

        # .sel() each CropCase in list
        for case in self:
            new_case_list.append(case.sel(*args, **kwargs))

        # Copy over other attributes
        new_case_list = self._copy_attributes(new_case_list)
        return new_case_list

    def isel(self, *args, **kwargs) -> CropCaseList:
        """
        Makes a copy of this CropCaseList, applying CropCase.isel() with the given arguments.

        Parameters
        ----------
        *args
            Positional arguments passed to CropCase.isel().
        **kwargs
            Keyword arguments passed to CropCase.isel().

        Returns
        -------
        CropCaseList
            New CropCaseList with isel() applied to each case.
        """
        new_case_list = self._create_empty()

        # .isel() each CropCase in list
        for case in self:
            new_case_list.append(case.isel(*args, **kwargs))

        # Copy over other attributes
        new_case_list = self._copy_attributes(new_case_list)
        return new_case_list
