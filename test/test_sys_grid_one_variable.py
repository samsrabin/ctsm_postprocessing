"""
Module to system-test grid_one_variable()
"""

import sys
import os
import glob
import shutil
import unittest
import tempfile
import numpy as np
import xarray as xr

try:
    # Attempt relative import if running as part of a package
    from ..cropcase import CropCase
    from ..utils import grid_one_variable, import_ds
except ImportError:
    # Fallback to absolute import if running as a script
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from crops.cropcase import CropCase
    from utils import grid_one_variable, import_ds

cfts_to_include = [
    "temperate_corn",
    "tropical_corn",
    "cotton",
    "rice",
    "temperate_soybean",
    "tropical_soybean",
    "sugarcane",
    "spring_wheat",
    "irrigated_temperate_corn",
    "irrigated_tropical_corn",
    "irrigated_cotton",
    "irrigated_rice",
    "irrigated_temperate_soybean",
    "irrigated_tropical_soybean",
    "irrigated_sugarcane",
    "irrigated_spring_wheat",
]

crops_to_include = [
    "corn",
    "cotton",
    "rice",
    "soybean",
    "sugarcane",
    "wheat",
]

START_YEAR = 1988
END_YEAR = 1990


class TestSysGridOneVariable(unittest.TestCase):
    """
    Class for testing grid_one_variable
    """

    def setUp(self):
        self.in_dir =  os.path.join(os.path.dirname(__file__), "testdata")
        self._tempdir = tempfile.mkdtemp()

    def tearDown(self):
        """
        Remove temporary directory
        """
        shutil.rmtree(self._tempdir, ignore_errors=True)

    def get_cft_ds(self):
        this_case = CropCase(
            name="crujra_matreqs",
            file_dir=self.in_dir,
            start_year=START_YEAR,
            end_year=END_YEAR,
            cfts_to_include=cfts_to_include,
            crops_to_include=crops_to_include,
            cft_ds_dir=self._tempdir,
            this_h_tape="h0i",
            force_no_cft_ds_file=True,
        )
        return this_case.cft_ds

    def test_raw(self):
        """
        Make sure that grid_one_variable works with raw CLM outputs
        """
        file_list = glob.glob(os.path.join(self.in_dir, "*nc"))
        file_list.sort()
        file = file_list[0]
        ds = import_ds(file)
        result = grid_one_variable(ds, "HUI_PERHARV")
        for dim in ["time", "mxharvests", "ivt_str", "lat", "lon"]:
            self.assertIn(dim, result.dims)

    #### TODO: FIX
    # def test_cftds(self):
    #     """
    #     Make sure that grid_one_variable works with cft_ds
    #     """
    #     cft_ds = self.get_cft_ds()
    #     for v in cft_ds:
    #         print(f"{v}: {cft_ds[v].dims}")
    #     self.assertTrue(False)
    #     grid_one_variable(cft_ds, "GRAINC_TO_FOOD_ANN")
