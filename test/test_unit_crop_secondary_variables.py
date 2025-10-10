"""
Module to unit-test crop_secondary_variables.py
"""

import sys
import os
import unittest
import numpy as np
import xarray as xr
import cftime

try:
    # Attempt relative import if running as part of a package
    from ..crops import crop_secondary_variables as c2o
except ImportError:
    # Fallback to absolute import if running as a script
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from crops import crop_secondary_variables as c2o

# pylint: disable=protected-access
# pylint: disable=too-many-public-methods


class TestUnitCropSecondaryVariables(unittest.TestCase):
    """
    Class to unit-test crop_secondary_variables.py
    """

    def test_handle_huifrac_where_gddharv_0(self):
        """
        Test that _handle_huifrac_where_gddharv_notpos() replaces zero values as expected
        """
        huifrac_in = np.array([np.nan, 1, 0.5, 0.2])
        huifrac_target = np.array([1, 1, 0.5, 0.2])
        gddharv_in = np.array([0, 1987, 2012, 2016.4])
        da_huifrac_in = xr.DataArray(data=huifrac_in)
        da_gddharv_in = xr.DataArray(data=gddharv_in)
        huifrac_out = c2o._handle_huifrac_where_gddharv_notpos(da_huifrac_in, da_gddharv_in)
        self.assertTrue(np.array_equal(huifrac_out, huifrac_target))

    def test_handle_huifrac_where_gddharv_negative(self):
        """
        Test that _handle_huifrac_where_gddharv_notpos() errors on negative values as expected
        """
        huifrac_in = np.array([np.nan, 1, 0.5, 0.2])
        gddharv_in = np.array([0, -1987, 2012, 2016.4])
        da_huifrac_in = xr.DataArray(data=huifrac_in)
        da_gddharv_in = xr.DataArray(data=gddharv_in)
        with self.assertRaises(NotImplementedError):
            c2o._handle_huifrac_where_gddharv_notpos(da_huifrac_in, da_gddharv_in)

    def test_handle_huifrac_where_gddharv_negative_but_nan_huifrac(self):
        """
        Test that _handle_huifrac_where_gddharv_notpos() errors on negative values as expected
        """
        huifrac_in = np.array([np.nan, np.nan, 0.5, 0.2])
        gddharv_in = np.array([0, -1987, 2012, 2016.4])
        da_huifrac_in = xr.DataArray(data=huifrac_in)
        da_gddharv_in = xr.DataArray(data=gddharv_in)
        huifrac_out = c2o._handle_huifrac_where_gddharv_notpos(da_huifrac_in, da_gddharv_in)
        huifrac_target = np.array([1, np.nan, 0.5, 0.2])
        self.assertTrue(np.array_equal(huifrac_out, huifrac_target, equal_nan=True))

    def test_get_huifrac(self):
        """
        Test get_huifrac()
        """
        hui_da = xr.DataArray(
            data=np.array([[1, 2, 3, 4], [5, 6, 7, 8]]),
        )
        gddharv_da = xr.DataArray(
            data=np.array([[1, 4, 12, 4], [25, 6, 7, 8]]),
        )
        target_da = xr.DataArray(
            data=np.array([[1, 0.5, 0.25, 1], [0.2, 1, 1, 1]]),
            attrs={"units": "Fraction of required"},
        )
        hui_var = c2o.DEFAULT_VAR_DICT["hui_var"]
        gddharv_var = c2o.DEFAULT_VAR_DICT["gddharv_var"]
        ds = xr.Dataset(
            data_vars={
                hui_var: hui_da,
                gddharv_var: gddharv_da,
            }
        )

        da_out = c2o.get_huifrac(ds)
        self.assertTrue(da_out.equals(target_da))

    def test_get_huifrac_bothneg(self):
        """
        Test get_huifrac() with HUI and GDDHARV both negative
        """
        hui_da = xr.DataArray(
            data=np.array([[1, 2, -3, 4], [5, 6, 7, 8]]),
        )
        gddharv_da = xr.DataArray(
            data=np.array([[1, 4, -12, 4], [25, 6, 7, 8]]),
        )
        target_da = xr.DataArray(
            data=np.array([[1, 0.5, np.nan, 1], [0.2, 1, 1, 1]]),
            attrs={"units": "Fraction of required"},
        )
        hui_var = c2o.DEFAULT_VAR_DICT["hui_var"]
        gddharv_var = c2o.DEFAULT_VAR_DICT["gddharv_var"]
        ds = xr.Dataset(
            data_vars={
                hui_var: hui_da,
                gddharv_var: gddharv_da,
            }
        )

        da_out = c2o.get_huifrac(ds)
        self.assertTrue(da_out.equals(target_da))

    def test_get_huifrac_customnames(self):
        """
        Test get_huifrac() with custom variable names
        """
        hui_da = xr.DataArray(
            data=np.array([[1, 2, 3, 4], [5, 6, 7, 8]]),
        )
        gddharv_da = xr.DataArray(
            data=np.array([[1, 4, 12, 4], [25, 6, 7, 8]]),
        )
        target_da = xr.DataArray(
            data=np.array([[1, 0.5, 0.25, 1], [0.2, 1, 1, 1]]),
            attrs={"units": "Fraction of required"},
        )
        hui_var = "hui_custom_name"
        gddharv_var = "gddharv_custom_name"
        var_dict = {
            "hui_var": hui_var,
            "gddharv_var": gddharv_var,
        }
        ds = xr.Dataset(
            data_vars={
                hui_var: hui_da,
                gddharv_var: gddharv_da,
            }
        )

        da_out = c2o.get_huifrac(ds, var_dict=var_dict)
        self.assertTrue(da_out.equals(target_da))

    def test_get_huifrac_gddharv0(self):
        """
        Test that get_huifrac() correctly handles GDDHARV 0
        """
        hui_da = xr.DataArray(
            data=np.array([[1, 2, 3, 4], [5, 6, 7, 8]]),
        )
        gddharv_da = xr.DataArray(
            data=np.array([[1, 4, 0, 4], [25, 6, 7, 8]]),
        )
        target_da = xr.DataArray(
            data=np.array([[1, 0.5, 1, 1], [0.2, 1, 1, 1]]), attrs={"units": "Fraction of required"}
        )
        hui_var = c2o.DEFAULT_VAR_DICT["hui_var"]
        gddharv_var = c2o.DEFAULT_VAR_DICT["gddharv_var"]
        ds = xr.Dataset(
            data_vars={
                hui_var: hui_da,
                gddharv_var: gddharv_da,
            }
        )

        da_out = c2o.get_huifrac(ds)
        self.assertTrue(da_out.equals(target_da))

    def test_get_huifrac_preserves_metadata(self):
        """
        Test that get_huifrac() preserves metadata
        """
        pft = np.array([1, 2, 3, 4])
        pft_da = xr.DataArray(
            data=pft,
            dims=["pft"],
        )
        time = np.array([1, 2])
        time_da = xr.DataArray(
            data=time,
            dims=["time"],
        )
        hui_da = xr.DataArray(
            data=np.array([[1, 2, 3, 4], [5, 6, 7, 8]]),
            dims=["time", "pft"],
            coords=[time_da, pft_da],
        )
        gddharv_da = xr.DataArray(
            data=np.array([[1, 4, 12, 4], [25, 6, 7, 8]]),
            dims=["time", "pft"],
            coords=[time_da, pft_da],
        )
        target_da = xr.DataArray(
            data=np.array([[1, 0.5, 0.25, 1], [0.2, 1, 1, 1]]),
            attrs={"units": "Fraction of required"},
            dims=["time", "pft"],
            coords=[time_da, pft_da],
        )
        hui_var = c2o.DEFAULT_VAR_DICT["hui_var"]
        gddharv_var = c2o.DEFAULT_VAR_DICT["gddharv_var"]
        ds = xr.Dataset(
            data_vars={
                hui_var: hui_da,
                gddharv_var: gddharv_da,
            }
        )

        da_out = c2o.get_huifrac(ds)
        self.assertTrue(da_out.equals(target_da))

    def test_calendar_has_leapdays_no(self):
        """
        Test that _calendar_has_leapdays() returns False if calendar has no leap days
        """
        da = xr.DataArray(
            data=np.array([cftime.DatetimeNoLeap(1, 1, 1)]),
            dims=["time"],
        )
        self.assertFalse(c2o._calendar_has_leapdays(da))

    def test_calendar_has_leapdays_yes(self):
        """
        Test that _calendar_has_leapdays() returns True if calendar has leap days
        """
        da = xr.DataArray(
            data=[cftime.DatetimeGregorian(1987, 1, 1)],
            dims=["time"],
        )
        self.assertTrue(c2o._calendar_has_leapdays(da))

    def test_calendar_has_leapdays_false_notime(self):
        """
        Test that _calendar_has_leapdays() returns False if time not in dims
        """
        da = xr.DataArray(
            data=[cftime.DatetimeGregorian(1987, 1, 1)],
        )
        self.assertFalse(c2o._calendar_has_leapdays(da))

    def test_calendar_has_leapdays_false_emptytime(self):
        """
        Test that _calendar_has_leapdays() returns False if time is empty
        """
        da = xr.DataArray(
            data=[],
            dims=["time"],
        )
        self.assertFalse(c2o._calendar_has_leapdays(da))

    def test_calendar_has_leapdays_false_numpy(self):
        """
        Test that _calendar_has_leapdays() returns False if time is a plain numpy type
        """
        da = xr.DataArray(
            data=[1987],
            dims=["time"],
        )
        self.assertFalse(c2o._calendar_has_leapdays(da))

    def test_get_gslen_within1year(self):
        """
        Check that get_gslen() correctly calculates growing season length when both dates are in the
        same calendar year
        """
        sdates_da = xr.DataArray(data=np.array([15]))
        hdates_da = xr.DataArray(data=np.array([17]))
        ds = xr.Dataset(
            data_vars={
                "HDATES": hdates_da,
                "SDATES_PERHARV": sdates_da,
            }
        )
        target_da = xr.DataArray(data=np.array([2]))

        da_out = c2o.get_gslen(ds)
        self.assertTrue(da_out.equals(target_da))

    def test_get_gslen_differentyears(self):
        """
        Check that get_gslen() correctly calculates growing season length when dates are in
        different calendar years
        """
        sdates_da = xr.DataArray(data=np.array([364]))
        hdates_da = xr.DataArray(data=np.array([1]))
        ds = xr.Dataset(
            data_vars={
                "HDATES": hdates_da,
                "SDATES_PERHARV": sdates_da,
            }
        )
        target_da = xr.DataArray(data=np.array([2]))

        da_out = c2o.get_gslen(ds)
        self.assertTrue(da_out.equals(target_da))

    def test_get_gslen_neg_hdates(self):
        """
        Check that get_gslen() correctly errors if harvest date negative
        """
        sdates_da = xr.DataArray(data=np.array([364]))
        hdates_da = xr.DataArray(data=np.array([-1]))
        ds = xr.Dataset(
            data_vars={
                "HDATES": hdates_da,
                "SDATES_PERHARV": sdates_da,
            }
        )

        with self.assertRaises(ValueError):
            c2o.get_gslen(ds)

    def test_get_gslen_neg_sdates(self):
        """
        Check that get_gslen() correctly errors if sowing date negative
        """
        sdates_da = xr.DataArray(data=np.array([-364]))
        hdates_da = xr.DataArray(data=np.array([1]))
        ds = xr.Dataset(
            data_vars={
                "HDATES": hdates_da,
                "SDATES_PERHARV": sdates_da,
            }
        )

        with self.assertRaises(ValueError):
            c2o.get_gslen(ds)

    def test_get_gslen_0_hdates(self):
        """
        Check that get_gslen() correctly errors if harvest date zero
        """
        sdates_da = xr.DataArray(data=np.array([364]))
        hdates_da = xr.DataArray(data=np.array([0]))
        ds = xr.Dataset(
            data_vars={
                "HDATES": hdates_da,
                "SDATES_PERHARV": sdates_da,
            }
        )

        with self.assertRaisesRegex(ValueError, "HDATE.*< 1"):
            c2o.get_gslen(ds)

    def test_get_gslen_0_sdates(self):
        """
        Check that get_gslen() correctly errors if sowing date zero
        """
        sdates_da = xr.DataArray(data=np.array([0]))
        hdates_da = xr.DataArray(data=np.array([1]))
        ds = xr.Dataset(
            data_vars={
                "HDATES": hdates_da,
                "SDATES_PERHARV": sdates_da,
            }
        )

        with self.assertRaisesRegex(ValueError, "SDATE.*< 1"):
            c2o.get_gslen(ds)

    def test_get_gslen_leapdays(self):
        """
        Check that get_gslen() correctly errors if calendar suggests leap days
        """
        time = xr.DataArray(
            data=[cftime.DatetimeGregorian(1987, 7, 24)],
            dims=["time"],
        )
        sdates_da = xr.DataArray(
            data=np.array([366]),
            dims=["time"],
            coords={"time": time},
        )
        hdates_da = xr.DataArray(
            data=np.array([1]),
            dims=["time"],
            coords={"time": time},
        )
        ds = xr.Dataset(
            data_vars={
                "HDATES": hdates_da,
                "SDATES_PERHARV": sdates_da,
            }
        )

        with self.assertRaisesRegex(NotImplementedError, "Unexpected calendar with leap days"):
            c2o.get_gslen(ds)

    def test_get_gslen_366_hdates(self):
        """
        Check that get_gslen() correctly errors if harvest date suggests leap year
        """
        sdates_da = xr.DataArray(data=np.array([364]))
        hdates_da = xr.DataArray(data=np.array([366]))
        ds = xr.Dataset(
            data_vars={
                "HDATES": hdates_da,
                "SDATES_PERHARV": sdates_da,
            }
        )

        with self.assertRaisesRegex(NotImplementedError, "HDATE.*== 366"):
            c2o.get_gslen(ds)

    def test_get_gslen_366_sdates(self):
        """
        Check that get_gslen() correctly errors if sowing date suggests leap year
        """
        sdates_da = xr.DataArray(data=np.array([366]))
        hdates_da = xr.DataArray(data=np.array([1]))
        ds = xr.Dataset(
            data_vars={
                "HDATES": hdates_da,
                "SDATES_PERHARV": sdates_da,
            }
        )

        with self.assertRaisesRegex(NotImplementedError, "SDATE.*== 366"):
            c2o.get_gslen(ds)

    def test_get_gslen_367_hdates(self):
        """
        Check that get_gslen() correctly errors if harvest date too high
        """
        sdates_da = xr.DataArray(data=np.array([364]))
        hdates_da = xr.DataArray(data=np.array([367]))
        ds = xr.Dataset(
            data_vars={
                "HDATES": hdates_da,
                "SDATES_PERHARV": sdates_da,
            }
        )

        with self.assertRaisesRegex(ValueError, "HDATE.*> 366"):
            c2o.get_gslen(ds)

    def test_get_gslen_367_sdates(self):
        """
        Check that get_gslen() correctly errors if sowing date too high
        """
        sdates_da = xr.DataArray(data=np.array([367]))
        hdates_da = xr.DataArray(data=np.array([1]))
        ds = xr.Dataset(
            data_vars={
                "HDATES": hdates_da,
                "SDATES_PERHARV": sdates_da,
            }
        )

        with self.assertRaisesRegex(ValueError, "SDATE.*> 366"):
            c2o.get_gslen(ds)

    def test_get_gslen_nan_mismatch(self):
        """
        Check that get_gslen() correctly errors if NaNs don't match between sowing and harvest dates
        """
        sdates_da = xr.DataArray(data=np.array([15, np.nan]))
        hdates_da = xr.DataArray(data=np.array([np.nan, 17]))
        ds = xr.Dataset(
            data_vars={
                "HDATES": hdates_da,
                "SDATES_PERHARV": sdates_da,
            }
        )

        with self.assertRaisesRegex(ValueError, "NaN mismatch"):
            c2o.get_gslen(ds)

    def test_get_gslen_preserves_metadata(self):
        """
        Test that get_gslen() preserves metadata
        """
        pft = np.array([1, 2, 3, 4])
        pft_da = xr.DataArray(
            data=pft,
            dims=["pft"],
        )
        time = np.array([1, 2])
        time_da = xr.DataArray(
            data=time,
            dims=["time"],
        )
        sdate_da = xr.DataArray(
            data=np.array([[1, 2, 3, 4], [361, 362, 363, 364]]),
            dims=["time", "pft"],
            coords=[time_da, pft_da],
        )
        hdate_da = xr.DataArray(
            data=np.array([[5, 6, 7, 8], [1, 2, 3, 4]]),
            dims=["time", "pft"],
            coords=[time_da, pft_da],
        )
        target_da = xr.DataArray(
            data=np.array([[4, 4, 4, 4], [5, 5, 5, 5]]),
            attrs={"units": "days"},
            dims=["time", "pft"],
            coords=[time_da, pft_da],
        )
        ds = xr.Dataset(
            data_vars={
                "HDATES": hdate_da,
                "SDATES_PERHARV": sdate_da,
            }
        )

        da_out = c2o.get_gslen(ds)
        self.assertTrue(da_out.equals(target_da))
