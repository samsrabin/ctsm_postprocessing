"""
Module to unit-test functions in cropcase.py
"""

# pylint: disable=redefined-outer-name
# Note: redefined-outer-name is disabled because pytest fixtures are used as test function parameters

import sys
import os
import io
from contextlib import redirect_stdout
from unittest.mock import patch, MagicMock
from tempfile import TemporaryDirectory
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

    @pytest.fixture(name="base_crop_case")
    def fixture_base_crop_case(self):
        """Create a basic CropCase instance with common defaults"""
        crop_case = CropCase._create_empty()
        crop_case._force_no_cft_ds_file = False
        crop_case._force_new_cft_ds_file = False
        crop_case.name = "some_case"
        crop_case._cft_ds_file_scratch = None
        crop_case._cft_ds_dir = None
        return crop_case

    @patch.dict(os.environ, {"SCRATCH": ""}, clear=False)
    def test_with_none_uses_file_dir(self, base_crop_case):
        """Test that cft_ds_dir=None uses self._file_dir"""
        file_dir = os.path.join("some", "test", "directory")
        base_crop_case._file_dir = file_dir

        base_crop_case._get_cft_ds_filepath()

        assert base_crop_case._cft_ds_dir == file_dir
        assert base_crop_case._cft_ds_file == os.path.join(file_dir, CFT_DS_FILENAME)

    def test_with_custom_dir(self, base_crop_case, tmp_path):
        """Test that a custom cft_ds_dir is preserved"""
        base_crop_case._file_dir = os.path.join("some", "test", "directory")
        custom_dir = str(tmp_path)
        base_crop_case._cft_ds_dir = custom_dir

        base_crop_case._get_cft_ds_filepath()

        assert base_crop_case._cft_ds_dir == custom_dir
        assert base_crop_case._cft_ds_file == os.path.join(custom_dir, CFT_DS_FILENAME)

    def test_cft_ds_dir_set_when_none(self, base_crop_case):
        """Test that cft_ds_dir is set to file_dir when it's None"""
        base_crop_case._file_dir = os.path.join("my", "file", "dir")

        base_crop_case._get_cft_ds_filepath()

        assert base_crop_case._cft_ds_dir == base_crop_case._file_dir

    def test_cft_ds_dir_preserved_when_not_none(self, base_crop_case, tmp_path):
        """Test that cft_ds_dir is preserved when it's not None"""
        base_crop_case._file_dir = os.path.join("original", "dir")
        custom_dir = str(tmp_path)
        base_crop_case._cft_ds_dir = custom_dir

        base_crop_case._get_cft_ds_filepath()

        assert base_crop_case._cft_ds_dir == custom_dir

    def test_returns_save_netcdf_when_writable(self, base_crop_case, tmp_path):
        """Test that sets save_netcdf when directory is writable"""
        base_crop_case._file_dir = str(tmp_path)

        base_crop_case._get_cft_ds_filepath()

        assert base_crop_case._save_netcdf is True

    def test_returns_no_save_when_force_no_file(self, base_crop_case, tmp_path):
        """Test that save_netcdf is False when force_no_cft_ds_file is True"""
        base_crop_case._file_dir = str(tmp_path)
        base_crop_case._force_no_cft_ds_file = True

        base_crop_case._get_cft_ds_filepath()

        assert base_crop_case._save_netcdf is False

    @patch("os.access", return_value=False)
    @patch("builtins.print")
    def test_returns_no_perms_when_not_writable(self, mock_print, _mock_access, base_crop_case, tmp_path):
        """Test that sets save_netcdf to False when directory is not writable"""
        # Use an existing directory
        existing_dir = str(tmp_path / "existing_dir")
        os.makedirs(existing_dir, exist_ok=True)
        base_crop_case._file_dir = existing_dir

        base_crop_case._get_cft_ds_filepath()

        assert base_crop_case._save_netcdf is False
        # Should print a warning
        mock_print.assert_called()
        assert "can't write" in str(mock_print.call_args)

    def test_read_history_files_true_when_file_not_exists(self, base_crop_case, tmp_path):
        """Test that read_history_files is True when cft_ds file doesn't exist"""
        base_crop_case._file_dir = str(tmp_path)

        base_crop_case._get_cft_ds_filepath()

        assert base_crop_case.read_history_files is True

    def test_read_history_files_false_when_file_exists(self, base_crop_case, tmp_path):
        """Test that read_history_files is False when cft_ds file exists"""
        base_crop_case._file_dir = str(tmp_path)

        # Create the file
        cft_ds_file = tmp_path / CFT_DS_FILENAME
        cft_ds_file.touch()

        base_crop_case._get_cft_ds_filepath()

        assert base_crop_case.read_history_files is False

    def test_read_history_files_true_when_force_new_file(self, base_crop_case, tmp_path):
        """Test that read_history_files is True when force_new_cft_ds_file is True"""
        base_crop_case._file_dir = str(tmp_path)
        base_crop_case._force_new_cft_ds_file = True

        # Create the file (should still read because of force flag)
        cft_ds_file = tmp_path / CFT_DS_FILENAME
        cft_ds_file.touch()

        base_crop_case._get_cft_ds_filepath()

        assert base_crop_case.read_history_files is True

    def test_read_history_files_true_when_force_no_file(self, base_crop_case, tmp_path):
        """Test that read_history_files is True when force_no_cft_ds_file is True"""
        base_crop_case._file_dir = str(tmp_path)
        base_crop_case._force_no_cft_ds_file = True

        # Create the file (should still read because of force flag)
        cft_ds_file = tmp_path / CFT_DS_FILENAME
        cft_ds_file.touch()

        base_crop_case._get_cft_ds_filepath()

        assert base_crop_case.read_history_files is True

    def test_creates_cft_ds_dir_if_not_exists(self, base_crop_case, tmp_path):
        """Test that cft_ds_dir is created if it doesn't exist"""
        base_crop_case._file_dir = str(tmp_path)
        # Set cft_ds_dir to a non-existent subdirectory
        nonexistent_dir = tmp_path / "subdir" / "nested"
        base_crop_case._cft_ds_dir = str(nonexistent_dir)

        # Directory should not exist yet
        assert not os.path.exists(nonexistent_dir)

        base_crop_case._get_cft_ds_filepath()

        # Directory should now exist
        assert os.path.exists(nonexistent_dir)
        assert os.path.isdir(nonexistent_dir)

    @patch("os.makedirs", side_effect=PermissionError("Permission denied"))
    @patch("builtins.print")
    def test_handles_permission_error_on_makedirs(self, mock_print, _mock_makedirs, base_crop_case):
        """Test that PermissionError from os.makedirs is handled correctly"""
        base_crop_case._file_dir = os.path.join("some", "dir")
        base_crop_case._cft_ds_dir = os.path.join("nonexistent", "dir")

        base_crop_case._get_cft_ds_filepath()

        # save_netcdf should be False due to permission error
        assert base_crop_case._save_netcdf is False
        # Should print a warning
        mock_print.assert_called()
        assert "can't write" in str(mock_print.call_args)

    @patch.dict(os.environ, {"SCRATCH": ""}, clear=False)  # Ensure SCRATCH env var is empty
    def test_no_scratch_fallback_when_scratch_empty(self, base_crop_case, tmp_path):
        """Test that scratch fallback doesn't happen when SCRATCH is empty"""
        base_crop_case._file_dir = str(tmp_path)
        base_crop_case.name = "test_case"

        # Don't create the file in the primary location

        base_crop_case._get_cft_ds_filepath()

        # Should still use the primary location, not scratch
        expected_file = os.path.join(str(tmp_path), CFT_DS_FILENAME)
        assert base_crop_case._cft_ds_file == expected_file
        assert base_crop_case._cft_ds_file_scratch is None

    @patch("builtins.print")
    def test_uses_scratch_when_file_exists_there(self, mock_print, base_crop_case, tmp_path):
        """Test that cft_ds_file uses scratch location when file exists there but not in primary"""
        with TemporaryDirectory() as scratch_dir:
            base_crop_case._file_dir = str(tmp_path)
            base_crop_case._cft_ds_dir = None
            base_crop_case.name = "test_case"

            # Create the scratch file
            scratch_file_path = os.path.join(
                scratch_dir, "clm_crop_case_cft_ds_files", "test_case", CFT_DS_FILENAME
            )
            os.makedirs(os.path.dirname(scratch_file_path), exist_ok=True)
            with open(scratch_file_path, 'w', encoding='utf-8') as f:
                f.write("test")

            with patch.dict(os.environ, {"SCRATCH": scratch_dir}, clear=False):
                base_crop_case._get_cft_ds_filepath()

            # Should use the scratch file
            assert base_crop_case._cft_ds_file == scratch_file_path
            # cft_ds_dir should be updated to the scratch directory
            assert base_crop_case._cft_ds_dir == os.path.dirname(scratch_file_path)
            # Should print a message about reading from scratch
            mock_print.assert_called()
            assert "Reading cft_ds from $SCRATCH" in str(mock_print.call_args)

    # @patch.dict decorator ensures there is a SCRATCH var defined in the environment
    @patch.dict(os.environ, {"SCRATCH": os.path.join("mock", "scratch")}, clear=False)
    def test_no_scratch_fallback_when_file_not_in_scratch(self, base_crop_case, tmp_path):
        """Test that primary location is used when scratch file doesn't exist"""
        base_crop_case._file_dir = str(tmp_path)
        base_crop_case.name = "test_case"

        # Don't create the file anywhere

        base_crop_case._get_cft_ds_filepath()

        # Should use the primary location since scratch file doesn't exist
        expected_file = os.path.join(str(tmp_path), CFT_DS_FILENAME)
        assert base_crop_case._cft_ds_file == expected_file


class TestCreateCftDsFile:
    """Test the CropCase._create_cft_ds_file() method"""

    @pytest.fixture(name="mock_crop_case")
    def fixture_mock_crop_case(self, tmp_path, test_ds):
        """Create a CropCase instance with mocked methods"""
        crop_case = CropCase._create_empty()
        crop_case._cft_ds_dir = str(tmp_path)
        crop_case._cft_ds_file = os.path.join(str(tmp_path), CFT_DS_FILENAME)
        crop_case._force_new_cft_ds_file = False
        crop_case._force_no_cft_ds_file = False
        crop_case._n_pfts = 78
        crop_case._verbose = False

        # Mock _read_and_process_files to
