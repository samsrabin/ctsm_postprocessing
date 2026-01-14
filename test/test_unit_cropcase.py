"""
Module to unit-test functions in cropcase.py
"""

# pylint: disable=redefined-outer-name
# Note: redefined-outer-name is disabled because pytest fixtures are used as test function parameters

import sys
import os
import io
from contextlib import redirect_stdout
import numpy as np
import xarray as xr
import pytest

try:
    # Attempt relative import if running as part of a package
    from ..crops.cropcase import _save_cft_ds_to_netcdf, CropCase, CFT_DS_FILENAME
except ImportError:
    # Fallback to absolute import if running as a script
    # Add both the parent directory (for crops module) and grandparent (for test module)
    parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    grandparent_dir = os.path.dirname(parent_dir)
    sys.path.insert(0, parent_dir)
    sys.path.insert(0, grandparent_dir)
    from crops.cropcase import _save_cft_ds_to_netcdf, CropCase, CFT_DS_FILENAME

# pylint: disable=too-many-public-methods
# pylint: disable=too-few-public-methods
# pylint: disable=protected-access


@pytest.fixture(name="test_ds", scope="function")
def fixture_test_ds() -> xr.Dataset:
    """Create an xarray Dataset object (fresh for each test)."""
    da0 = xr.DataArray(data=np.array([1, 2, 3, 4]), attrs={"units": "kg"})
    da1 = xr.DataArray(data=np.array([5, 6, 7, 8]), attrs={"units": "kg"})
    ds = xr.Dataset(
        data_vars={
            "var0": da0,
            "var1": da1,
        },
    )
    return ds


class TestSaveCftDsToNetcdf:
    """Test the _save_cft_ds_to_netcdf() function"""

    def test_basic(self, tmp_path, test_ds):
        """Test basic functionality"""
        file_path = tmp_path / "ds.nc"
        assert not os.path.exists(file_path)
        _save_cft_ds_to_netcdf(test_ds, file_path, False)
        assert os.path.exists(file_path)

    def test_verbose(self, tmp_path, test_ds):
        """Test verbose functionality"""
        file_path = tmp_path / "ds.nc"
        f = io.StringIO()
        with redirect_stdout(f):
            _save_cft_ds_to_netcdf(test_ds, file_path, True)
        assert f"Saving {file_path}..." in f.getvalue()


class TestGetCftDsFilepath:
    """Test the CropCase._get_cft_ds_filepath() method"""

    def test_with_none_uses_file_dir(self):
        """Test that cft_ds_dir=None uses self.file_dir"""
        crop_case = CropCase._create_empty()
        file_dir = os.path.join("some", "test", "directory")
        crop_case.file_dir = file_dir
        crop_case.cft_ds_dir = None

        crop_case._get_cft_ds_filepath()

        assert crop_case.cft_ds_dir == file_dir
        assert crop_case.cft_ds_file == os.path.join(file_dir, CFT_DS_FILENAME)

    def test_with_custom_dir(self):
        """Test that a custom cft_ds_dir is preserved"""
        crop_case = CropCase._create_empty()
        crop_case.file_dir = os.path.join("some", "test", "directory")
        custom_dir = os.path.join("custom", "output", "path")
        crop_case.cft_ds_dir = custom_dir

        crop_case._get_cft_ds_filepath()

        assert crop_case.cft_ds_dir == custom_dir
        assert crop_case.cft_ds_file == os.path.join(custom_dir, CFT_DS_FILENAME)

    def test_cft_ds_dir_set_when_none(self):
        """Test that cft_ds_dir is set to file_dir when it's None"""
        crop_case = CropCase._create_empty()
        crop_case.file_dir = os.path.join("my", "file", "dir")
        crop_case.cft_ds_dir = None

        crop_case._get_cft_ds_filepath()

        assert crop_case.cft_ds_dir == crop_case.file_dir

    def test_cft_ds_dir_preserved_when_not_none(self):
        """Test that cft_ds_dir is preserved when it's not None"""
        crop_case = CropCase._create_empty()
        crop_case.file_dir = os.path.join("original", "dir")
        custom_dir = os.path.join("different", "dir")
        crop_case.cft_ds_dir = custom_dir

        crop_case._get_cft_ds_filepath()

        assert crop_case.cft_ds_dir == custom_dir
