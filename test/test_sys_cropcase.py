"""
Module to unit-test cropcase.py
"""

import sys
import os
import unittest
import numpy as np

try:
    # Attempt relative import if running as part of a package
    from ..cropcase import CropCase
except ImportError:
    # Fallback to absolute import if running as a script
    sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    from crops.cropcase import CropCase

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

START_YEAR = 2023
END_YEAR = 2024


class TestSysCropCase(unittest.TestCase):
    """
    Class for testing CropCase
    """

    def test_setup_cropcase(self):
        """
        Make sure that CropCase does not error when importing test data
        """
        name = "ctsm53019_f09_BNF_hist"
        file_dir = os.path.join(os.path.dirname(__file__), "testdata")
        this_case = CropCase(
            name=name,
            file_dir=file_dir,
            start_year=START_YEAR,
            end_year=END_YEAR,
            cfts_to_include=cfts_to_include,
            crops_to_include=crops_to_include,
        )
        self.assertListEqual(
            [x.name for x in this_case.crop_list],
            ["corn", "cotton", "rice", "soybean", "sugarcane", "wheat"],
        )

        # Ensure that derived variables are present.
        self.assertTrue("GRAINC_TO_FOOD_VIABLE_PERHARV" in this_case.cft_ds)
        self.assertTrue("YIELD_PERHARV" in this_case.cft_ds)
        self.assertTrue("YIELD_ANN" in this_case.cft_ds)

        # Ensure that not all yield values are zero
        self.assertTrue(np.any(this_case.cft_ds["GRAINC_TO_FOOD_VIABLE_PERHARV"] > 0))
        self.assertTrue(np.any(this_case.cft_ds["YIELD_PERHARV"] > 0))
        self.assertTrue(np.any(this_case.cft_ds["YIELD_ANN"] > 0))

        # Ensure that NaN values are handled correctly.
        # First, ensure that there are actually some NaN values that will be tested.
        self.assertTrue(np.any(np.isnan(this_case.cft_ds["GRAINC_TO_FOOD_PERHARV"])))
        self.assertTrue(np.any(np.isnan(this_case.cft_ds["YIELD_PERHARV"])))
        # Now test that YIELD_ANN, which is just YIELD_PERHARV summed over the mxharvests dimension,
        # doesn't have any NaN values.
        self.assertFalse(np.any(np.isnan(this_case.cft_ds["YIELD_ANN"])))
