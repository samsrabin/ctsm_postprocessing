"""
A class for holding a list of CropCases and information about them
"""

from __future__ import annotations

import copy
import os
import sys
import warnings
from time import time

try:
    # Attempt relative import if running as part of a package
    from .cropcase import CropCase
    from ..resolutions import identify_resolution
except ImportError:
    # Fallback to absolute import if running as a script
    sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    from crops.cropcase import CropCase
    from resolutions import identify_resolution

# The variables needed for regridding
REGRID_VARS = ["area", "landfrac", "landmask"]


def _ds_has_regrid_vars(ds):
    return all(v in ds.variables for v in REGRID_VARS)


class CropCaseList(list):
    """
    A class for holding a list of CropCases and information about them
    """

    def __init__(
        self,
        *args,
        opts,
    ):
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

    def _save_or_check_regrid_ds(self, case, res):
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

    def _check_regrid_ds(self, case, res, regrid_ds):
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

    def _check_attrs_match(self, other):
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

    def __eq__(self, other):
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

    def __ne__(self, other):
        return not self == other

    def _import_cases(
        self,
        opts,
    ):
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
                    opts["cfts_to_include"],
                    opts["crops_to_include"],
                    opts["start_year"],
                    opts["end_year"],
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
    def _create_empty(cls):
        """
        Create an empty CropCaseList without going through the normal initialization (i.e., import).
        Used internally by sel() and isel() for creating copies.
        """
        # Create instance without calling __init__
        instance = cls.__new__(cls)
        # Initialize as empty list
        list.__init__(instance)
        return instance

    def _copy_attributes(self, dest_case_list):
        """
        Copy all CropCaseList attributes from self to destination CropCaseList.
        """
        for attr in [a for a in dir(self) if not a.startswith("__")]:
            # Skip callable attributes (methods) - they should be inherited from the class
            if callable(getattr(self, attr)):
                continue
            setattr(dest_case_list, attr, getattr(self, attr))
        return dest_case_list

    def sel(self, *args, **kwargs):
        """
        Makes a copy of this CropCaseList, applying CropCase.sel() with the given arguments.
        """
        new_case_list = self._create_empty()

        # .sel() each CropCase in list
        for case in self:
            new_case_list.append(case.sel(*args, **kwargs))

        # Copy over other attributes
        new_case_list = self._copy_attributes(new_case_list)
        return new_case_list

    def isel(self, *args, **kwargs):
        """
        Makes a copy of this CropCaseList, applying CropCase.isel() with the given arguments.
        """
        new_case_list = self._create_empty()

        # .isel() each CropCase in list
        for case in self:
            new_case_list.append(case.isel(*args, **kwargs))

        # Copy over other attributes
        new_case_list = self._copy_attributes(new_case_list)
        return new_case_list
