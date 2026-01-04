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
    from ..cropcase import _save_cft_ds_to_netcdf
except ImportError:
    # Fallback to absolute import if running as a script
    # Add both the parent directory (for crops module) and grandparent (for test module)
    parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    grandparent_dir = os.path.dirname(parent_dir)
    sys.path.insert(0, parent_dir)
    sys.path.insert(0, grandparent_dir)
    from crops.cropcase import _save_cft_ds_to_netcdf

# pylint: disable=too-many-public-methods
# pylint: disable=too-few-public-methods


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
