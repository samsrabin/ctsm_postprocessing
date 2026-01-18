"""
Module to unit-test mark_crops_invalid.py
"""

import sys
import os
import unittest
import numpy as np
import xarray as xr

# pylint: disable=wrong-import-position
x = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(x)
import resolutions as res

# pylint: disable=protected-access
# pylint: disable=too-many-public-methods

def setup_coordinate(coord_n, coord_min, coord_max, name):
    """
    Set up a coordinate variable
    """
    coord_ar = np.full((coord_n), fill_value=coord_min, dtype=float)
    coord_ar[-1] = float(coord_max)
    coord = xr.DataArray(
        data=coord_ar,
        dims=[name],
    )
    return coord

class TestResolutions(unittest.TestCase):
    """
    Class to unit-test resolutions.py
    """

    def test_f09_exact(self):
        """
        Test that f09 is correctly identified when exact values are given
        """
        # Set up a minimal xarray Dataset that should match Resolution f09 exactly
        lon = setup_coordinate(288, 0, 358.75, "lon")
        lat = setup_coordinate(192, -90, 90, "lat")
        ds = xr.Dataset(
            coords={
                "lon": lon,
                "lat": lat,
            }
        )
        ds_res = res.identify_resolution(ds)
        self.assertEqual(ds_res.name, "f09")

    def test_f09_close(self):
        """
        Test that f09 is correctly identified when close values are given
        """
        # Set up a minimal xarray Dataset that should almost match Resolution f09
        lon = setup_coordinate(288, 0, 358.7500000001, "lon")
        lat = setup_coordinate(192, -90.0002, 90, "lat")
        ds = xr.Dataset(
            coords={
                "lon": lon,
                "lat": lat,
            }
        )
        ds_res = res.identify_resolution(ds)
        self.assertEqual(ds_res.name, "f09")

    def test_f09_not_close_enough(self):
        """
        Test that no match is found when a close-but-not-close-enough resolution is given
        """
        # Set up a minimal xarray Dataset that should almost match Resolution f09 but not within
        # precision
        lon = setup_coordinate(288, 0, 358.7500000001, "lon")
        lat = setup_coordinate(192, -90.002, 90, "lat")
        ds = xr.Dataset(
            coords={
                "lon": lon,
                "lat": lat,
            }
        )
        with self.assertRaisesRegex(KeyError, "Unidentified resolution"):
            res.identify_resolution(ds)
