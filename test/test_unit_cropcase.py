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
        crop_case.force_no_cft_ds_file = False
        crop_case.force_new_cft_ds_file = False

        crop_case._get_cft_ds_filepath()

        assert crop_case.cft_ds_dir == file_dir
        assert crop_case.cft_ds_file == os.path.join(file_dir, CFT_DS_FILENAME)

    def test_with_custom_dir(self, tmp_path):
        """Test that a custom cft_ds_dir is preserved"""
        crop_case = CropCase._create_empty()
        crop_case.file_dir = os.path.join("some", "test", "directory")
        custom_dir = str(tmp_path)
        crop_case.cft_ds_dir = custom_dir
        crop_case.force_no_cft_ds_file = False
        crop_case.force_new_cft_ds_file = False

        crop_case._get_cft_ds_filepath()

        assert crop_case.cft_ds_dir == custom_dir
        assert crop_case.cft_ds_file == os.path.join(custom_dir, CFT_DS_FILENAME)

    def test_cft_ds_dir_set_when_none(self):
        """Test that cft_ds_dir is set to file_dir when it's None"""
        crop_case = CropCase._create_empty()
        crop_case.file_dir = os.path.join("my", "file", "dir")
        crop_case.cft_ds_dir = None
        crop_case.force_no_cft_ds_file = False
        crop_case.force_new_cft_ds_file = False

        crop_case._get_cft_ds_filepath()

        assert crop_case.cft_ds_dir == crop_case.file_dir

    def test_cft_ds_dir_preserved_when_not_none(self, tmp_path):
        """Test that cft_ds_dir is preserved when it's not None"""
        crop_case = CropCase._create_empty()
        crop_case.file_dir = os.path.join("original", "dir")
        custom_dir = str(tmp_path)
        crop_case.cft_ds_dir = custom_dir
        crop_case.force_no_cft_ds_file = False
        crop_case.force_new_cft_ds_file = False

        crop_case._get_cft_ds_filepath()

        assert crop_case.cft_ds_dir == custom_dir

    def test_returns_save_netcdf_when_writable(self, tmp_path):
        """Test that returns correct values when directory is writable"""
        crop_case = CropCase._create_empty()
        crop_case.file_dir = str(tmp_path)
        crop_case.cft_ds_dir = None
        crop_case.force_no_cft_ds_file = False
        crop_case.force_new_cft_ds_file = False

        _, save_netcdf = crop_case._get_cft_ds_filepath()

        assert save_netcdf is True

    def test_returns_no_save_when_force_no_file(self, tmp_path):
        """Test that save_netcdf is False when force_no_cft_ds_file is True"""
        crop_case = CropCase._create_empty()
        crop_case.file_dir = str(tmp_path)
        crop_case.cft_ds_dir = None
        crop_case.force_no_cft_ds_file = True
        crop_case.force_new_cft_ds_file = False

        _, save_netcdf = crop_case._get_cft_ds_filepath()

        assert save_netcdf is False

    @patch("os.access", return_value=False)
    @patch("builtins.print")
    def test_returns_no_perms_when_not_writable(self, mock_print, _mock_access):
        """Test that returns correct values when directory is not writable"""
        crop_case = CropCase._create_empty()
        crop_case.file_dir = os.path.join("some", "dir")
        crop_case.cft_ds_dir = None
        crop_case.force_no_cft_ds_file = False
        crop_case.force_new_cft_ds_file = False

        _, save_netcdf = crop_case._get_cft_ds_filepath()

        assert save_netcdf is False
        # Should print a warning
        mock_print.assert_called()
        assert "can't write" in str(mock_print.call_args)

    def test_read_history_files_true_when_file_not_exists(self, tmp_path):
        """Test that read_history_files is True when cft_ds file doesn't exist"""
        crop_case = CropCase._create_empty()
        crop_case.file_dir = str(tmp_path)
        crop_case.cft_ds_dir = None
        crop_case.force_no_cft_ds_file = False
        crop_case.force_new_cft_ds_file = False

        read_history_files, _ = crop_case._get_cft_ds_filepath()

        assert read_history_files is True

    def test_read_history_files_false_when_file_exists(self, tmp_path):
        """Test that read_history_files is False when cft_ds file exists"""
        crop_case = CropCase._create_empty()
        crop_case.file_dir = str(tmp_path)
        crop_case.cft_ds_dir = None
        crop_case.force_no_cft_ds_file = False
        crop_case.force_new_cft_ds_file = False

        # Create the file
        cft_ds_file = tmp_path / CFT_DS_FILENAME
        cft_ds_file.touch()

        read_history_files, _ = crop_case._get_cft_ds_filepath()

        assert read_history_files is False

    def test_read_history_files_true_when_force_new_file(self, tmp_path):
        """Test that read_history_files is True when force_new_cft_ds_file is True"""
        crop_case = CropCase._create_empty()
        crop_case.file_dir = str(tmp_path)
        crop_case.cft_ds_dir = None
        crop_case.force_no_cft_ds_file = False
        crop_case.force_new_cft_ds_file = True

        # Create the file (should still read because of force flag)
        cft_ds_file = tmp_path / CFT_DS_FILENAME
        cft_ds_file.touch()

        read_history_files, _ = crop_case._get_cft_ds_filepath()

        assert read_history_files is True

    def test_read_history_files_true_when_force_no_file(self, tmp_path):
        """Test that read_history_files is True when force_no_cft_ds_file is True"""
        crop_case = CropCase._create_empty()
        crop_case.file_dir = str(tmp_path)
        crop_case.cft_ds_dir = None
        crop_case.force_no_cft_ds_file = True
        crop_case.force_new_cft_ds_file = False

        # Create the file (should still read because of force flag)
        cft_ds_file = tmp_path / CFT_DS_FILENAME
        cft_ds_file.touch()

        read_history_files, _ = crop_case._get_cft_ds_filepath()

        assert read_history_files is True


class TestCreateCftDsFile:
    """Test the CropCase._create_cft_ds_file() method"""

    @pytest.fixture(name="mock_crop_case")
    def fixture_mock_crop_case(self, tmp_path, test_ds):
        """Create a CropCase instance with mocked methods"""
        crop_case = CropCase._create_empty()
        crop_case.cft_ds_dir = str(tmp_path)
        crop_case.cft_ds_file = os.path.join(str(tmp_path), CFT_DS_FILENAME)
        crop_case.force_new_cft_ds_file = False
        crop_case.force_no_cft_ds_file = False
        crop_case.n_pfts = 78
        crop_case.verbose = False

        # Mock _read_and_process_files to return a test dataset
        crop_case._read_and_process_files = MagicMock(return_value=test_ds)

        return crop_case

    @patch("ctsm_postprocessing.crops.cropcase._save_cft_ds_to_netcdf")
    def test_force_new_file_recreates_existing(self, mock_save, mock_crop_case, tmp_path):
        """Test that force_new_cft_ds_file recreates even if file exists"""
        # Create the file
        cft_ds_file = tmp_path / CFT_DS_FILENAME
        cft_ds_file.touch()

        mock_crop_case.force_new_cft_ds_file = True

        mock_crop_case._create_cft_ds_file(
            start_year=2000, end_year=2010, read_history_files=True, save_netcdf=True
        )

        # _read_and_process_files should be called
        mock_crop_case._read_and_process_files.assert_called_once()
        # Should save the file
        mock_save.assert_called_once()

    @patch("ctsm_postprocessing.crops.cropcase._save_cft_ds_to_netcdf")
    def test_file_not_exists_creates_and_saves(self, mock_save, mock_crop_case, test_ds):
        """Test that if file doesn't exist, it's created and saved"""
        mock_crop_case._create_cft_ds_file(
            start_year=2000, end_year=2010, read_history_files=True, save_netcdf=True
        )

        # _read_and_process_files should be called with None years (to read all)
        mock_crop_case._read_and_process_files.assert_called_once_with(
            mock_crop_case.n_pfts, None, None
        )
        # Should save the file
        mock_save.assert_called_once_with(test_ds, mock_crop_case.cft_ds_file, False)
        # cft_ds should be set
        assert mock_crop_case.cft_ds is test_ds

    @patch("ctsm_postprocessing.crops.cropcase._save_cft_ds_to_netcdf")
    def test_force_no_file_reads_but_doesnt_save(self, mock_save, mock_crop_case):
        """Test that force_no_cft_ds_file reads data but doesn't save"""
        mock_crop_case.force_no_cft_ds_file = True

        mock_crop_case._create_cft_ds_file(
            start_year=2000, end_year=2010, read_history_files=True, save_netcdf=False
        )

        # _read_and_process_files should be called with the specified years
        mock_crop_case._read_and_process_files.assert_called_once_with(
            mock_crop_case.n_pfts, 2000, 2010
        )
        # Should NOT save the file
        mock_save.assert_not_called()
        # cft_ds should still be set
        assert hasattr(mock_crop_case, "cft_ds")

    @patch("ctsm_postprocessing.crops.cropcase._save_cft_ds_to_netcdf")
    def test_no_write_permission_reads_but_doesnt_save(self, mock_save, mock_crop_case):
        """Test that without write permissions, data is read but not saved"""
        mock_crop_case._create_cft_ds_file(
            start_year=2000, end_year=2010, read_history_files=True, save_netcdf=False
        )

        # _read_and_process_files should be called with the specified years
        mock_crop_case._read_and_process_files.assert_called_once_with(
            mock_crop_case.n_pfts, 2000, 2010
        )
        # Should NOT save the file
        mock_save.assert_not_called()

    @patch("ctsm_postprocessing.crops.cropcase._save_cft_ds_to_netcdf")
    def test_verbose_output(self, _mock_save, mock_crop_case):
        """Test that verbose mode produces output"""
        # pylint: disable=unused-argument
        mock_crop_case.verbose = True

        f = io.StringIO()
        with redirect_stdout(f):
            mock_crop_case._create_cft_ds_file(
                start_year=2000, end_year=2010, read_history_files=True, save_netcdf=True
            )

        output = f.getvalue()
        assert "took" in output
        assert "s" in output

    @patch("ctsm_postprocessing.crops.cropcase._save_cft_ds_to_netcdf")
    def test_save_netcdf_reads_all_years(self, _mock_save, mock_crop_case):
        """Test that when saving netCDF, all years are read (None, None)"""
        # pylint: disable=unused-argument
        mock_crop_case._create_cft_ds_file(
            start_year=2000, end_year=2010, read_history_files=True, save_netcdf=True
        )

        # Should read all years when saving
        mock_crop_case._read_and_process_files.assert_called_once_with(
            mock_crop_case.n_pfts, None, None
        )

    @patch("ctsm_postprocessing.crops.cropcase._save_cft_ds_to_netcdf")
    def test_no_save_reads_specified_years(self, mock_save, mock_crop_case):
        """Test that when not saving, specified years are read"""
        mock_crop_case.force_no_cft_ds_file = True

        mock_crop_case._create_cft_ds_file(
            start_year=1995, end_year=2005, read_history_files=True, save_netcdf=False
        )

        # Should read only specified years when not saving
        mock_crop_case._read_and_process_files.assert_called_once_with(
            mock_crop_case.n_pfts, 1995, 2005
        )
        mock_save.assert_not_called()

    @patch("ctsm_postprocessing.crops.cropcase._save_cft_ds_to_netcdf")
    def test_message_changes_when_saving(self, _mock_save, mock_crop_case):
        """Test that the message indicates 'Making and saving' when saving"""
        # pylint: disable=unused-argument
        mock_crop_case.verbose = True

        f = io.StringIO()
        with redirect_stdout(f):
            mock_crop_case._create_cft_ds_file(
                start_year=2000, end_year=2010, read_history_files=True, save_netcdf=True
            )

        output = f.getvalue()
        assert "Making and saving" in output
        assert CFT_DS_FILENAME in output

    @patch("ctsm_postprocessing.crops.cropcase._save_cft_ds_to_netcdf")
    def test_read_history_files_false_skips_processing(self, mock_crop_case):
        """Test that when read_history_files=False, no processing occurs"""
        mock_crop_case._create_cft_ds_file(
            start_year=2000, end_year=2010, read_history_files=False, save_netcdf=True
        )

        # _read_and_process_files should not be called when read_history_files=False
        mock_crop_case._read_and_process_files.assert_not_called()
