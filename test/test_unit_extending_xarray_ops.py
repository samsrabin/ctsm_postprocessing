"""
Module to unit-test extending_xarray_ops.py
"""

import sys
import os
import unittest
import warnings
from scipy.stats._axis_nan_policy import SmallSampleWarning
from scipy.stats import circmean
import numpy as np
import xarray as xr

try:
    # Attempt relative import if running as part of a package
    from ..extending_xarray_ops import da_circmean, da_circmean_doy, _round_to_nearest_day
except ImportError:
    # Fallback to absolute import if running as a script
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from extending_xarray_ops import da_circmean, da_circmean_doy, _round_to_nearest_day


def da_equals_round(da0, da1, tol_decimals=14):
    """
    Check whether two DataArrays are equal within some tolerance
    """
    da0 = da0.round(decimals=tol_decimals)
    da1 = da1.round(decimals=tol_decimals)
    result = da0.equals(da1)
    if not result:
        print(f"da0: {da0}")
        print(f"da1: {da1}")
        print(f"diff: {da1 - da0}")
    return result


class TestRoundToNearestDay(unittest.TestCase):
    """
    Class for testing _round_to_nearest_day()
    """

    def test_roundday_1(self):
        """Test that _round_to_nearest_day() returns 1 if given 1"""
        self.assertEqual(_round_to_nearest_day(1), 1)

    def test_roundday_365(self):
        """Test that _round_to_nearest_day() returns 365 if given 365"""
        self.assertEqual(_round_to_nearest_day(365), 365)

    def test_roundday_0(self):
        """Test that _round_to_nearest_day() returns 365 if given 0"""
        self.assertEqual(_round_to_nearest_day(0), 365)

    def test_roundday_neg_errors(self):
        """Test that _round_to_nearest_day() errors if given a negative"""
        with self.assertRaises(ValueError):
            _round_to_nearest_day(-1)

    def test_roundday_0p75(self):
        """Test that _round_to_nearest_day() returns 1 if given 0.75"""
        self.assertEqual(_round_to_nearest_day(0.75), 1)

    def test_roundday_0p5(self):
        """Test that _round_to_nearest_day() returns 1 if given 0.5"""
        self.assertEqual(_round_to_nearest_day(0.5), 1)

    def test_roundday_365p5(self):
        """Test that _round_to_nearest_day() returns 1 if given 365.5"""
        self.assertEqual(_round_to_nearest_day(365.5), 1)

    def test_roundday_3p5(self):
        """Test that _round_to_nearest_day() returns 4 if given 3.5"""
        self.assertEqual(_round_to_nearest_day(3.5), 4)

    def test_roundday_363p5(self):
        """Test that _round_to_nearest_day() returns 364 if given 363.5"""
        self.assertEqual(_round_to_nearest_day(363.5), 364)

    def test_roundday_364p5(self):
        """Test that _round_to_nearest_day() returns 365 if given 364.5"""
        self.assertEqual(_round_to_nearest_day(364.5), 365)

    def test_roundday_365p1(self):
        """Test that _round_to_nearest_day() returns 365 if given 365.1"""
        self.assertEqual(_round_to_nearest_day(365.1), 365)

    def test_roundday_180p1(self):
        """Test that _round_to_nearest_day() returns 180 if given 180.1"""
        self.assertEqual(_round_to_nearest_day(180.1), 180)

    def test_roundday_nan(self):
        """Test that _round_to_nearest_day() returns NaN if given NaN"""
        self.assertTrue(np.isnan(_round_to_nearest_day(np.nan)))


class TestCircMean(unittest.TestCase):
    """
    Class for testing da_circmean
    """

    def test_da_circmean_basic_0_360(self):
        """Basic test of da_circmean using default 0-2pi range"""
        da = xr.DataArray(data=np.pi * np.array([1.9, 2.1]))
        result = da_circmean(da)
        self.assertAlmostEqual(result.values, 2 * np.pi)

    def test_da_circmean_allnan(self):
        """
        Basic test of da_circmean on an all-nan vector. If it doesn't raise a warning, the warning
        suppression code can be removed from da_circmean_doy().
        """
        da = xr.DataArray(data=np.array([np.nan, np.nan]))
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = da_circmean(da)
            # Check that SmallSampleWarning was raised
            self.assertTrue(any(issubclass(warning.category, SmallSampleWarning) for warning in w))
        self.assertTrue(np.isnan(result.values))

    def test_circmean_365day(self):
        """Test of scipy's circmean using range for days of a 365-day year"""
        values = np.array([1, 364])
        self.assertEqual(circmean(values, low=1, high=366), 365)

    def test_da_circmean_365day(self):
        """Test of da_circmean using range for days of a 365-day year"""
        da = xr.DataArray(data=np.array([1, 364]))
        result = da_circmean(da, low=1, high=366)
        self.assertEqual(result.values, 365)

    def test_da_circmean_alldims(self):
        """Test of da_circmean on a 2-d var across all dims"""
        da = xr.DataArray(
            data=np.array([[1, 3], [1, 3]]),
            dims=["x", "y"],
        )
        result = da_circmean(da)
        expected = xr.DataArray(data=np.array(2))
        self.assertTrue(da_equals_round(result, expected))

    def test_xarray_mean_1dim_of2(self):
        """
        Test xarray's mean on 2-d var across 1 dim, just to make sure the var is set up right
        """
        da = xr.DataArray(
            data=np.array([[1, 3], [1, 3]]),
            dims=["x", "y"],
        )
        result = da.mean(dim="y")
        expected = xr.DataArray(data=np.array([2, 2]), dims=["x"])
        self.assertTrue(result.equals(expected))

    def test_da_circmean_1dimy_of2(self):
        """
        Test da_circmean on 2-d var across 1 dim, using same da/operation as test_xarray_mean_1dim
        """
        da = xr.DataArray(
            data=np.array([[1, 3], [1, 3]]),
            dims=["x", "y"],
        )
        result = da_circmean(da, dim="y")
        expected = xr.DataArray(data=np.array([2, 2]), dims=["x"])
        self.assertTrue(da_equals_round(result, expected))

    def test_da_circmean_1dimx_of2(self):
        """
        Test da_circmean on 2-d var across 1 dim, using x dimension instead
        """
        da = xr.DataArray(
            data=np.array([[1, 3], [1, 3]]),
            dims=["x", "y"],
        )
        result = da_circmean(da, dim="x")
        expected = xr.DataArray(data=np.array([1, 3]), dims=["y"])
        self.assertTrue(da_equals_round(result, expected))

    def test_da_circmean_1st2_of3(self):
        """
        Test da_circmean on 3-d var across first 2 dimensions
        """
        da = xr.DataArray(
            data=np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]]),
            dims=["x", "y", "z"],
        )
        result = da_circmean(da, dim=["x", "y"], high=10)
        expected = xr.DataArray(data=np.array([4, 5]), dims=["z"])
        self.assertTrue(da_equals_round(result, expected))

    def test_da_circmean_2nd2_of3(self):
        """
        Test da_circmean on 3-d var across second 2 dimensions
        """
        da = xr.DataArray(
            data=np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]]),
            dims=["x", "y", "z"],
        )
        result = da_circmean(da, dim=["y", "z"], high=10)
        expected = xr.DataArray(data=np.array([2.5, 6.5]), dims=["x"])
        self.assertTrue(da_equals_round(result, expected))


class TestCircMeanDoy(unittest.TestCase):
    """
    Class for testing da_circmean_doy
    """

    def test_da_circmean_doy(self):
        """Test of da_circmean_doy"""
        da = xr.DataArray(data=np.array([1.0, 364.0]))
        result = da_circmean_doy(da)
        self.assertEqual(result.values, 365)

    def test_da_circmean_doy_nan(self):
        """Test of da_circmean_doy with a NaN"""
        da = xr.DataArray(data=np.array([1.0, 364.0, np.nan]))
        result = da_circmean_doy(da)
        self.assertEqual(result.values, 365)

    def test_da_circmean_doy_neg1(self):
        """Test of da_circmean_doy with a -1, which should get converted to NaN"""
        da = xr.DataArray(data=np.array([1.0, 364.0, -1.0]))
        result = da_circmean_doy(da)
        self.assertEqual(result.values, 365)

    def test_da_circmean_doy_nan_propagate(self):
        """Test of da_circmean_doy with a propagating NaN"""
        da = xr.DataArray(data=np.array([1.0, 364.0, np.nan]))
        result = da_circmean_doy(da, nan_policy="propagate")
        self.assertTrue(np.isnan(result.values))

    def test_da_circmean_doy_low_errors(self):
        """Test that da_circmean_doy errors if given kwarg low"""
        with self.assertRaises(TypeError):
            da_circmean_doy(xr.DataArray(), low=1)

    def test_da_circmean_doy_high_errors(self):
        """Test that da_circmean_doy errors if given kwarg high"""
        with self.assertRaises(TypeError):
            da_circmean_doy(xr.DataArray(), high=1)

    def test_da_circmean_doy_toolow_errors(self):
        """Test that da_circmean_doy errors if given value < 1"""
        da = xr.DataArray(data=np.array([0, 365]))
        with self.assertRaisesRegex(
            AssertionError, "All input values should be in the range 1-365 or NaN"
        ):
            da_circmean_doy(da)

    def test_da_circmean_doy_toohigh_errors(self):
        """Test that da_circmean_doy errors if given value > 365"""
        da = xr.DataArray(data=np.array([1, 366]))
        with self.assertRaisesRegex(
            AssertionError, "All input values should be in the range 1-365 or NaN"
        ):
            da_circmean_doy(da)

    def test_da_circmean_doy_notint_errors(self):
        """Test that da_circmean_doy errors if given values that aren't close to being integers"""
        da = xr.DataArray(data=np.array([1, 364.1]))
        with self.assertRaisesRegex(
            AssertionError, "All input values should be whole numbers or NaN"
        ):
            da_circmean_doy(da)

    def test_da_circmean_doy_noon_jan1(self):
        """Test of da_circmean_doy where mean is 0.5: Give Jan. 1"""
        da = xr.DataArray(data=np.array([1, 365]))
        result = da_circmean_doy(da)
        self.assertEqual(result.values, 1)

    def test_da_circmean_doy_afternoon_dec31(self):
        """Test of da_circmean_doy where mean is afternoon of Dec. 31"""
        da = xr.DataArray(data=np.array([1, 365, 365]))
        result = da_circmean_doy(da)
        self.assertEqual(result.values, 365)

    def test_da_circmean_doy_1dim_of2(self):
        """Test of da_circmean_doy on just one dimension of 2"""
        da = xr.DataArray(
            data=np.array([[1, 364], [1, 364]]),
            dims=["x", "time"],
        )
        result = da_circmean_doy(da, dim="time")
        expected = xr.DataArray(data=np.array([365, 365]), dims=["x"])
        self.assertTrue(result.equals(expected))

    def test_da_circmean_doy_1dim_of2_allnan(self):
        """Test of da_circmean_doy on just one dimension of 2 with all NaNs; expect no warnings"""
        da = xr.DataArray(
            data=np.array([[np.nan, np.nan], [np.nan, np.nan]]),
            dims=["x", "time"],
        )
        result = da_circmean_doy(da, dim="time")
        expected = xr.DataArray(data=np.array([np.nan, np.nan]), dims=["x"])
        self.assertTrue(result.equals(expected))

    def test_da_circmean_doy_alldims(self):
        """Test of da_circmean_doy across a 2-d array"""
        da = xr.DataArray(
            data=np.array([[1, 364], [1, 364]]),
            dims=["x", "time"],
        )

        with warnings.catch_warnings(record=True) as w:
            # Ensure all warnings are treated as errors within this context
            warnings.simplefilter("error")
            try:
                result = da_circmean_doy(da)
            except Warning as e:
                self.fail(f"Unexpected warning raised: {e}")
            self.assertEqual(len(w), 0, "No warnings should have been recorded.")

        expected = xr.DataArray(data=np.array(365))
        self.assertTrue(result.equals(expected))
