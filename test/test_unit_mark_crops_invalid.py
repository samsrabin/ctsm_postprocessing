"""
Module to unit-test mark_crops_invalid.py
"""

import sys
import os
import unittest
import numpy as np
import xarray as xr

try:
    # Attempt relative import if running as part of a package
    from ..crops import mark_crops_invalid as mci
except ImportError:
    # Fallback to absolute import if running as a script
    x = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    print(x)
    sys.path.append(x)
    from crops import mark_crops_invalid as mci

# pylint: disable=protected-access
# pylint: disable=too-many-public-methods


class TestUnitMarkCropsInvalid(unittest.TestCase):
    """
    Class to unit-test mark_crops_invalid.py
    """

    def setUp(self):
        self.huifrac_var = mci.DEFAULT_VAR_DICT["huifrac_var"]
        self.gddharv_var = mci.DEFAULT_VAR_DICT["gddharv_var"]
        self.gslen_var = mci.DEFAULT_VAR_DICT["gslen_var"]

    def test_pft_or_patch_patch(self):
        """
        Test that _pft_or_patch() works for patch-dimensioned Dataset
        """
        da = xr.DataArray(data=[[1, 2], [3, 4]], dims=["patch", "time"])
        ds = xr.Dataset(data_vars={"test_var": da})
        pftpatch_dimname = mci._pft_or_patch(ds)
        self.assertEqual(pftpatch_dimname, "patch")

    def test_pft_or_patch_pft(self):
        """
        Test that _pft_or_patch() works for pft-dimensioned Dataset
        """
        da = xr.DataArray(data=[[1, 2], [3, 4]], dims=["pft", "time"])
        ds = xr.Dataset(data_vars={"test_var": da})
        pftpatch_dimname = mci._pft_or_patch(ds)
        self.assertEqual(pftpatch_dimname, "pft")

    def test_pft_or_patch_error_neither(self):
        """
        Test that _pft_or_patch() errors if neither patch nor pft is found
        """
        da = xr.DataArray(data=[[1, 2], [3, 4]], dims=["grid", "time"])
        ds = xr.Dataset(data_vars={"test_var": da})
        with self.assertRaises(KeyError):
            mci._pft_or_patch(ds)

    def test_pft_or_patch_error_both(self):
        """
        Test that _pft_or_patch() errors if both patch and pft are found
        """
        da = xr.DataArray(data=[[1, 2], [3, 4]], dims=["pft", "patch"])
        ds = xr.Dataset(data_vars={"test_var": da})
        with self.assertRaises(NotImplementedError):
            mci._pft_or_patch(ds)

    def test_get_itype_veg_str_varname_notimplemented(self):
        """
        Test that _get_itype_veg_str_varname() errors correctly when given an invalid dimension name
        """
        with self.assertRaises(NotImplementedError):
            mci._get_itype_veg_str_varname("invalid dimension name")

    def test_get_isimip3_min_hui_patch0th(self):
        """
        Test that _get_isimip3_min_hui() works as expected when patch is on 0th dimension
        """
        n_patch = 4
        n_time = 2
        shape = (n_patch, n_time)
        huifrac_in = np.empty(shape)
        huifrac_in_da = xr.DataArray(data=huifrac_in, dims=["patch", "time"])
        vegstr = ["corn", "wheat", "soy", "rice"]
        vegstr_da = xr.DataArray(data=vegstr, dims=["patch"])
        ds = xr.Dataset(
            data_vars={
                self.huifrac_var: huifrac_in_da,
                "patches1d_itype_veg_str": vegstr_da,
            }
        )

        # Check that you've set things up right
        self.assertTupleEqual(huifrac_in.shape, shape)
        self.assertTrue("patch" in ds.dims)
        self.assertTrue("time" in ds.dims)

        # Expect 0.8 where corn, 0.9 elsewhere
        target = np.array([[0.8, 0.8], [0.9, 0.9], [0.9, 0.9], [0.9, 0.9]])

        result = mci._get_isimip3_min_hui(ds, self.huifrac_var)
        self.assertTrue(np.array_equal(result, target))

    def test_get_isimip3_min_hui_patchinwrongplace(self):
        """
        Test that _get_isimip3_min_hui() errors as expected when patch is on neither 0th nor last
        dimension
        """
        n_patch = 4
        n_harvest = 2
        n_time = 2
        shape = (n_harvest, n_patch, n_time)
        huifrac_in = np.empty(shape)
        huifrac_in_da = xr.DataArray(data=huifrac_in, dims=["harvest", "patch", "time"])
        vegstr = ["corn", "wheat", "soy", "rice"]
        vegstr_da = xr.DataArray(data=vegstr, dims=["patch"])
        ds = xr.Dataset(
            data_vars={
                self.huifrac_var: huifrac_in_da,
                "patches1d_itype_veg_str": vegstr_da,
            }
        )

        # Check that you've set things up right
        self.assertTupleEqual(huifrac_in.shape, shape)
        self.assertTrue("harvest" in ds.dims)
        self.assertTrue("patch" in ds.dims)
        self.assertTrue("time" in ds.dims)

        with self.assertRaises(NotImplementedError):
            mci._get_isimip3_min_hui(ds, self.huifrac_var)

    def setup_minviablehui_ds_pftlast(self):
        """
        Set up Dataset and target for minimum viable HUI testing with pft in last dimension
        """
        n_pft = 4
        n_time = 2
        shape = (n_time, n_pft)
        huifrac_in = np.array([[0.1, 0.9, 0.2, 0.8], [0.3, 0.7, 0.4, 0.6]])
        huifrac_coords = {
            "time": np.arange(n_time),
            "pft": np.arange(n_pft),
        }
        huifrac_in_da = xr.DataArray(
            data=huifrac_in,
            dims=["time", "pft"],
            coords=huifrac_coords,
            attrs={"test_attribute": 15},
        )
        vegstr = ["corn", "wheat", "soy", "rice"]
        vegstr_da = xr.DataArray(data=vegstr, dims=["pft"], coords={"pft": np.arange(n_pft)})
        ds = xr.Dataset(
            data_vars={
                self.huifrac_var: huifrac_in_da,
                "pfts1d_itype_veg_str": vegstr_da,
            }
        )

        # Check that you've set things up right
        self.assertTupleEqual(huifrac_in.shape, shape)
        self.assertTrue("pft" in ds.dims)
        self.assertTrue("time" in ds.dims)

        # Expect 0.8 where corn, 0.9 elsewhere
        target = np.array([[0.8, 0.9, 0.9, 0.9], [0.8, 0.9, 0.9, 0.9]])

        return ds, target

    def test_get_isimip3_min_hui_pftlast(self):
        """
        Test that _get_isimip3_min_hui() works as expected when pft is on last dimension
        """
        ds, target = self.setup_minviablehui_ds_pftlast()

        result = mci._get_isimip3_min_hui(ds, self.huifrac_var)
        self.assertTrue(np.array_equal(result, target))

    def test_get_isimip3_min_hui_soy(self):
        """
        Test that _get_isimip3_min_hui() works as expected when this_pft is soy. Because we want
        most of that function to get short-circuited, do not add a pft or patch dimension. This way,
        if the short-circuit doesn't happen, it will error.
        """
        n_patch = 4
        n_time = 2
        shape = (n_patch, n_time)
        huifrac_in = np.empty(shape)
        huifrac_in_da = xr.DataArray(data=huifrac_in, dims=["grid", "time"])
        ds = xr.Dataset(
            data_vars={
                self.huifrac_var: huifrac_in_da,
            }
        )

        # Check that you've set things up right
        self.assertTupleEqual(huifrac_in.shape, shape)
        self.assertFalse("patch" in ds.dims)
        self.assertFalse("pft" in ds.dims)

        result = mci._get_isimip3_min_hui(ds, self.huifrac_var, this_pft="soy")
        # Expect 0.9 everywhere because not corn
        target = np.array([[0.9, 0.9], [0.9, 0.9], [0.9, 0.9], [0.9, 0.9]])
        self.assertTrue(np.array_equal(result, target))

    def test_get_isimip3_min_hui_corn(self):
        """
        Test that _get_isimip3_min_hui() works as expected when this_pft is corn. Because we want
        most of that function to get short-circuited, do not add a pft or patch dimension. This way,
        if the short-circuit doesn't happen, it will error.
        """
        n_patch = 4
        n_time = 2
        shape = (n_patch, n_time)
        huifrac_in = np.empty(shape)
        huifrac_in_da = xr.DataArray(data=huifrac_in, dims=["grid", "time"])
        ds = xr.Dataset(
            data_vars={
                self.huifrac_var: huifrac_in_da,
            }
        )

        # Check that you've set things up right
        self.assertTupleEqual(huifrac_in.shape, shape)
        self.assertFalse("patch" in ds.dims)
        self.assertFalse("pft" in ds.dims)

        result = mci._get_isimip3_min_hui(ds, self.huifrac_var, this_pft="corn")
        # Expect 0.8 everywhere because corn
        target = np.array([[0.8, 0.8], [0.8, 0.8], [0.8, 0.8], [0.8, 0.8]])
        self.assertTrue(np.array_equal(result, target))

    def test_get_isimip3_min_hui_tropicalcorn(self):
        """
        Test that _get_isimip3_min_hui() works as expected when this_pft is tropical_corn
        """
        n_patch = 4
        n_time = 2
        shape = (n_patch, n_time)
        huifrac_in = np.empty(shape)
        huifrac_in_da = xr.DataArray(data=huifrac_in, dims=["grid", "time"])
        ds = xr.Dataset(
            data_vars={
                self.huifrac_var: huifrac_in_da,
            }
        )

        # Check that you've set things up right
        self.assertTupleEqual(huifrac_in.shape, shape)
        self.assertFalse("patch" in ds.dims)
        self.assertFalse("pft" in ds.dims)

        result = mci._get_isimip3_min_hui(ds, self.huifrac_var, this_pft="tropical_corn")
        # Expect 0.8 everywhere because corn
        target = np.array([[0.8, 0.8], [0.8, 0.8], [0.8, 0.8], [0.8, 0.8]])
        self.assertTrue(np.array_equal(result, target))

    def test_get_isimip3_min_hui_badthispft(self):
        """
        Test that _get_isimip3_min_hui() errors as expected when given a non-string this_pft
        """
        dummy_ds = xr.Dataset()
        with self.assertRaises(TypeError):
            mci._get_isimip3_min_hui(dummy_ds, self.huifrac_var, this_pft=15)
        with self.assertRaises(TypeError):
            mci._get_isimip3_min_hui(dummy_ds, self.huifrac_var, this_pft=["corn"])

    def test_get_min_viable_hui_number(self):
        """
        Test that _get_min_viable_hui() returns min_viable_hui if it's a number
        """
        dummy_ds = xr.Dataset()
        min_viable_hui = 0.87
        self.assertEqual(
            mci._get_min_viable_hui(dummy_ds, min_viable_hui, self.huifrac_var), min_viable_hui
        )

    def test_get_min_viable_hui_isimip3(self):
        """
        Test that _get_min_viable_hui() returns min_viable_hui if it's "isimip3"
        """
        ds, target = self.setup_minviablehui_ds_pftlast()
        self.assertTrue(
            np.array_equal(mci._get_min_viable_hui(ds, "isimip3", self.huifrac_var), target)
        )

    def test_get_min_viable_hui_ggcmi3(self):
        """
        Test that _get_min_viable_hui() returns min_viable_hui if it's "ggcmi3"
        """
        ds, target = self.setup_minviablehui_ds_pftlast()
        self.assertTrue(
            np.array_equal(mci._get_min_viable_hui(ds, "ggcmi3", self.huifrac_var), target)
        )

    def test_get_min_viable_hui_error(self):
        """
        Test that _get_min_viable_hui() errors for invalid min_viable_hui
        """
        dummy_ds = xr.Dataset()
        with self.assertRaises(NotImplementedError):
            mci._get_min_viable_hui(dummy_ds, "abc123", self.huifrac_var)

    def test_mark_invalid_hui_too_low(self):
        """
        Test mark_invalid_hui_too_low() with default invalid_value
        """
        ds, _ = self.setup_minviablehui_ds_pftlast()
        da_in = xr.DataArray(
            data=np.array([[1, 2, 3, 4], [5, 6, 7, 8]]),
            dims=ds[self.huifrac_var].dims,
            coords=ds[self.huifrac_var].coords,
            attrs=ds[self.huifrac_var].attrs,
        )

        min_viable_hui = 0.7

        da_out = mci.mark_invalid_hui_too_low(da_in, ds[self.huifrac_var], min_viable_hui)
        target = np.array([[0, 2, 0, 4], [0, 6, 0, 0]])

        self.assertTrue(np.array_equal(da_out.values, target))

    def test_mark_invalid_hui_too_low_nan(self):
        """
        Test mark_invalid_hui_too_low() with invalid_value=np.nan
        """
        ds, _ = self.setup_minviablehui_ds_pftlast()
        da_in = xr.DataArray(
            data=np.array([[1, 2, 3, 4], [5, 6, 7, 8]], dtype=np.float64),
            dims=ds[self.huifrac_var].dims,
            coords=ds[self.huifrac_var].coords,
            attrs=ds[self.huifrac_var].attrs,
        )

        min_viable_hui = 0.7
        invalid_value = np.nan

        da_out = mci.mark_invalid_hui_too_low(
            da_in, ds[self.huifrac_var], min_viable_hui, invalid_value=invalid_value
        )
        target = np.array([[np.nan, 2, np.nan, 4], [np.nan, 6, np.nan, np.nan]])

        self.assertTrue(np.array_equal(da_out.values, target, equal_nan=True))

    def test_mark_invalid_season_too_long(self):
        """
        Test mark_invalid_season_too_long() with default invalid_value
        """
        ds, _ = self.setup_minviablehui_ds_pftlast()
        da_in = xr.DataArray(
            data=np.array([[1, 2, 3, 4], [5, 6, 7, 8]]),
            dims=ds[self.huifrac_var].dims,
            coords=ds[self.huifrac_var].coords,
            attrs=ds[self.huifrac_var].attrs,
        )
        ds[self.gslen_var] = da_in.copy()

        mxmats = {
            "corn": 3,
            "wheat": 6,
            "soy": 15,
            "rice": 1,
        }

        da_out = mci.mark_invalid_season_too_long(ds, da_in, mxmats, self.gslen_var)
        target = np.array([[1, 2, 3, 0], [0, 6, 7, 0]])

        self.assertTrue(np.array_equal(da_out.values, target))

    def test_mark_invalid_season_too_long_onepft(self):
        """
        Test mark_invalid_season_too_long() with one specified PFT
        """
        ds, _ = self.setup_minviablehui_ds_pftlast()
        da_in = xr.DataArray(
            data=np.array([[1, 2, 3, 4], [5, 6, 7, 8]]),
            dims=ds[self.huifrac_var].dims,
            coords=ds[self.huifrac_var].coords,
            attrs=ds[self.huifrac_var].attrs,
        )
        ds[self.gslen_var] = da_in.copy()

        mxmats = {
            "corn": 3,
            "wheat": 6,
            "soy": 15,
            "rice": 1,
        }

        da_out = mci.mark_invalid_season_too_long(
            ds, da_in, mxmats, self.gslen_var, this_pft="corn"
        )
        target = np.array([[1, 2, 3, 0], [0, 0, 0, 0]])

        self.assertTrue(np.array_equal(da_out.values, target))

    def test_mark_invalid_season_too_long_neg1(self):
        """
        Test mark_invalid_season_too_long() with invalid_value=-1
        """
        ds, _ = self.setup_minviablehui_ds_pftlast()
        da_in = xr.DataArray(
            data=np.array([[1, 2, 3, 4], [5, 6, 7, 8]]),
            dims=ds[self.huifrac_var].dims,
            coords=ds[self.huifrac_var].coords,
            attrs=ds[self.huifrac_var].attrs,
        )
        ds[self.gslen_var] = da_in.copy()

        mxmats = {
            "corn": 3,
            "wheat": 6,
            "soy": 15,
            "rice": 1,
        }

        da_out = mci.mark_invalid_season_too_long(
            ds, da_in, mxmats, self.gslen_var, invalid_value=-1
        )
        target = np.array([[1, 2, 3, -1], [-1, 6, 7, -1]])

        self.assertTrue(np.array_equal(da_out.values, target))

    def test_mark_crops_invalid_just_seasonlength(self):
        """
        Test mark_crops_invalid() when not setting minimum viable HUI
        """
        ds, _ = self.setup_minviablehui_ds_pftlast()
        da_in = xr.DataArray(
            data=np.array([[1, 2, 3, 4], [5, 6, 7, 8]]),
            dims=ds[self.huifrac_var].dims,
            coords=ds[self.huifrac_var].coords,
            attrs=ds[self.huifrac_var].attrs,
        )
        ds[self.gslen_var] = da_in.copy()
        ds["test_var"] = da_in.copy()

        mxmats = {
            "corn": 3,
            "wheat": 6,
            "soy": 15,
            "rice": 1,
        }

        da_out = mci.mark_crops_invalid(ds, "test_var", min_viable_hui=None, mxmats=mxmats)
        target = np.array([[1, 2, 3, 0], [0, 6, 7, 0]])

        self.assertTrue(np.array_equal(da_out.values, target))
        self.assertNotIn("min_viable_hui", da_out.attrs)
        self.assertTrue(da_out.attrs["mxmat_limited"])

    def setup_time_grid_ds(self):
        """
        Set up a fake Dataset with dimensions time x grid
        """
        n_grid = 4
        n_time = 2
        huifrac_in = np.array([[0.1, 0.9, 0.2, 0.8], [0.3, 0.7, 0.4, 0.6]])
        huifrac_coords = {
            "time": np.arange(n_time),
            "grid": np.arange(n_grid),
        }
        huifrac_in_da = xr.DataArray(
            data=huifrac_in,
            dims=["time", "grid"],
            coords=huifrac_coords,
            attrs={"test_attribute": 15},
        )
        ds = xr.Dataset(
            data_vars={
                self.huifrac_var: huifrac_in_da,
            }
        )

        return ds

    def test_mark_crops_invalid_just_seasonlength_corn(self):
        """
        Test mark_crops_invalid() when not setting minimum viable HUI, corn only
        """
        ds = self.setup_time_grid_ds()

        da_in = xr.DataArray(
            data=np.array([[1, 2, 3, 4], [5, 6, 7, 8]]),
            dims=ds[self.huifrac_var].dims,
            coords=ds[self.huifrac_var].coords,
            attrs=ds[self.huifrac_var].attrs,
        )
        ds[self.gslen_var] = da_in.copy()
        ds["test_var"] = da_in.copy()

        mxmats = {
            "corn": 3,
            "wheat": 6,
            "soy": 15,
            "rice": 1,
        }

        da_out = mci.mark_crops_invalid(
            ds, "test_var", min_viable_hui=None, mxmats=mxmats, this_pft="corn"
        )
        target = np.array([[1, 2, 3, 0], [0, 0, 0, 0]])
        self.assertTrue(np.array_equal(da_out.values, target))
        self.assertNotIn("min_viable_hui", da_out.attrs)
        self.assertTrue(da_out.attrs["mxmat_limited"])

        # Test again, just getting the mask
        da_out = mci.mark_crops_invalid(
            ds, min_viable_hui=None, mxmats=mxmats, this_pft="corn"
        )
        target = np.array([[1, 1, 1, 0], [0, 0, 0, 0]])
        self.assertTrue(np.array_equal(da_out.values, target))
        self.assertNotIn("min_viable_hui", da_out.attrs)
        self.assertTrue(da_out.attrs["mxmat_limited"])

    def test_mark_crops_invalid_just_minviablehui(self):
        """
        Test mark_crops_invalid() when not setting max season length
        """
        ds, _ = self.setup_minviablehui_ds_pftlast()
        da_in = xr.DataArray(
            data=np.array([[1, 2, 3, 4], [5, 6, 7, 8]]),
            dims=ds[self.huifrac_var].dims,
            coords=ds[self.huifrac_var].coords,
            attrs=ds[self.huifrac_var].attrs,
        )
        ds["test_var"] = da_in.copy()

        min_viable_hui = 0.7
        ds[self.gddharv_var] = xr.full_like(da_in, fill_value=1)  # Just need nonzero

        da_out = mci.mark_crops_invalid(ds, "test_var", min_viable_hui=min_viable_hui, mxmats=None)
        target = np.array([[0, 2, 0, 4], [0, 6, 0, 0]])
        self.assertTrue(np.array_equal(da_out.values, target))
        self.assertEqual(da_out.attrs["min_viable_hui"], min_viable_hui)
        self.assertNotIn("mxmat_limited", da_out.attrs)

        # Test again, just getting the mask
        da_out = mci.mark_crops_invalid(ds, min_viable_hui=min_viable_hui, mxmats=None)
        target = np.array([[0, 1, 0, 1], [0, 1, 0, 0]])
        self.assertTrue(np.array_equal(da_out.values, target))
        self.assertEqual(da_out.attrs["min_viable_hui"], min_viable_hui)
        self.assertNotIn("mxmat_limited", da_out.attrs)

    def test_mark_crops_invalid_just_minviablehui_isimip3(self):
        """
        Test mark_crops_invalid() when not setting max season length and saying
        min_viable_hui="isimip3"
        """
        ds, _ = self.setup_minviablehui_ds_pftlast()
        da_in = xr.DataArray(
            data=np.array([[1, 2, 3, 4], [5, 6, 7, 8]]),
            dims=ds[self.huifrac_var].dims,
            coords=ds[self.huifrac_var].coords,
            attrs=ds[self.huifrac_var].attrs,
        )
        ds["test_var"] = da_in.copy()

        min_viable_hui = "isimip3"
        ds[self.gddharv_var] = xr.full_like(da_in, fill_value=1)  # Just need nonzero

        da_out = mci.mark_crops_invalid(ds, "test_var", min_viable_hui=min_viable_hui, mxmats=None)
        target = np.array([[0, 2, 0, 0], [0, 0, 0, 0]])
        self.assertTrue(np.array_equal(da_out.values, target))
        self.assertEqual(da_out.attrs["min_viable_hui"], min_viable_hui)
        self.assertNotIn("mxmat_limited", da_out.attrs)

    def test_mark_crops_invalid_just_minviablehui_isimip3_corn(self):
        """
        Test mark_crops_invalid() when not setting max season length and saying
        min_viable_hui="isimip3", corn only
        """
        ds = self.setup_time_grid_ds()

        da_in = xr.DataArray(
            data=np.array([[1, 2, 3, 4], [5, 6, 7, 8]]),
            dims=ds[self.huifrac_var].dims,
            coords=ds[self.huifrac_var].coords,
            attrs=ds[self.huifrac_var].attrs,
        )
        ds["test_var"] = da_in.copy()

        min_viable_hui = "isimip3"
        ds[self.gddharv_var] = xr.full_like(da_in, fill_value=1)  # Just need nonzero

        da_out = mci.mark_crops_invalid(
            ds, "test_var", min_viable_hui=min_viable_hui, mxmats=None, this_pft="corn"
        )
        target = np.array([[0, 2, 0, 4], [0, 0, 0, 0]])
        self.assertTrue(np.array_equal(da_out.values, target))
        self.assertEqual(da_out.attrs["min_viable_hui"], min_viable_hui)
        self.assertNotIn("mxmat_limited", da_out.attrs)

    def test_mark_crops_invalid_both(self):
        """
        Test mark_crops_invalid() when setting both max season length and min viable HUI
        """
        ds, _ = self.setup_minviablehui_ds_pftlast()
        da_in = xr.DataArray(
            data=np.array([[1, 2, 3, 4], [5, 6, 7, 8]]),
            dims=ds[self.huifrac_var].dims,
            coords=ds[self.huifrac_var].coords,
            attrs=ds[self.huifrac_var].attrs,
        )
        ds[self.gslen_var] = da_in.copy()
        ds["test_var"] = da_in.copy()

        mxmats = {
            "corn": 3,
            "wheat": 6,
            "soy": 15,
            "rice": 1,
        }

        min_viable_hui = 0.7
        ds[self.gddharv_var] = xr.full_like(da_in, fill_value=1)  # Just need nonzero

        da_out = mci.mark_crops_invalid(
            ds, "test_var", min_viable_hui=min_viable_hui, mxmats=mxmats
        )
        target = np.array([[0, 2, 0, 0], [0, 6, 0, 0]])
        self.assertTrue(np.array_equal(da_out.values, target))
        self.assertEqual(da_out.attrs["min_viable_hui"], min_viable_hui)
        self.assertTrue(da_out.attrs["mxmat_limited"])

        # Test again, just getting the mask
        da_out = mci.mark_crops_invalid(
            ds, min_viable_hui=min_viable_hui, mxmats=mxmats
        )
        target = np.array([[0, 1, 0, 0], [0, 1, 0, 0]])
        self.assertTrue(np.array_equal(da_out.values, target))
        self.assertEqual(da_out.attrs["min_viable_hui"], min_viable_hui)
        self.assertTrue(da_out.attrs["mxmat_limited"])

    def test_mark_crops_invalid_neither(self):
        """
        Test mark_crops_invalid() when setting neither max season length nor min viable HUI
        """
        target = [[1, 2, 3, 4], [5, 6, 7, 8]]
        da_in = xr.DataArray(
            data=np.array(target),
        )
        ds = xr.Dataset(data_vars={"test_var": da_in})

        da_out = mci.mark_crops_invalid(ds, "test_var", min_viable_hui=None, mxmats=None)

        self.assertTrue(np.array_equal(da_out.values, target))
        self.assertNotIn("min_viable_hui", da_out.attrs)
        self.assertNotIn("mxmat_limited", da_out.attrs)
