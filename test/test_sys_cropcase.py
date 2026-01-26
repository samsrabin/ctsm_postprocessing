"""
Module to system-test cropcase.py
"""

# pylint: disable=redefined-outer-name
# Note: redefined-outer-name is disabled because pytest fixtures are used as test function parameters

import sys
import os
from shutil import copyfile
import copy
from unittest.mock import patch
import numpy as np
import xarray as xr
import pytest

from ..crops.cropcase import CropCase, CFT_DS_FILENAME
from ..crops.crop_defaults import DEFAULT_CROPS_TO_INCLUDE
from .defaults import START_YEAR, END_YEAR, CASE_NAME, FILE_DIR
from ..crops.extra_area_prod_yield_etc import MATURITY_LEVELS

# The first and last years we have test files for. Note that these refer to the years whose data is
# included; add +1 to get years in filenames and netCDF timestamps.
FIRST_AVAILABLE_YEAR = 1986
LAST_AVAILABLE_YEAR = 1990


def import_default_cropcase(cft_ds_dir):
    """
    Function to import our default CropCase
    """
    return CropCase(
        name=CASE_NAME,
        file_dir=FILE_DIR,
        start_year=START_YEAR,
        end_year=END_YEAR,
        cft_ds_dir=cft_ds_dir,
    )


@pytest.fixture(scope="session")
def cropcase_base(tmp_path_factory):
    """
    Session-scoped fixture to create a CropCase instance once for all tests.
    This is created only once per test session to speed up testing.
    """
    temp_dir = str(tmp_path_factory.mktemp("cropcase_data"))
    return import_default_cropcase(temp_dir)


@pytest.fixture
def cropcase(cropcase_base):
    """
    Fixture to provide a deep copy of the base CropCase instance for testing.
    Each test gets its own copy to ensure test isolation.
    """
    return copy.deepcopy(cropcase_base)


def check_crujra_matreqs_case_shared(this_case):
    """
    Function, shared with test_sys_crop_case_list, to check the crujra_matreqs case
    """
    assert [x.name for x in this_case.crop_list] == DEFAULT_CROPS_TO_INCLUDE

    # Check that cft-to-crop is working right
    assert list(this_case.cft_ds["cft_crop"].values) == [
        "corn",
        "corn",
        "wheat",
        "wheat",
        "soybean",
        "soybean",
        "cotton",
        "cotton",
        "rice",
        "rice",
        "sugarcane",
        "sugarcane",
        "corn",
        "corn",
        "soybean",
        "soybean",
    ]

    # Ensure that derived variables are present.
    for m in MATURITY_LEVELS:
        derived_yield_var_list = [
            f"GRAINC_TO_FOOD_{m}_PERHARV",
            f"YIELD_{m}_PERHARV",
            f"YIELD_{m}_ANN",
        ]
        for var in derived_yield_var_list:
            msg = f"{var} missing from cft_ds"
            assert var in this_case.cft_ds, msg

        # Ensure that not all yield values are NaN
        for var in derived_yield_var_list:
            msg = f"{var} is all NaN"
            assert not this_case.cft_ds[var].isnull().all(), msg

        # Ensure that not all yield values are zero
        for var in derived_yield_var_list:
            msg = f"{var} is all zero"
            assert np.any(this_case.cft_ds[var] > 0), msg

        # Ensure that all yield values have units
        for var in derived_yield_var_list:
            msg = f"{var} is missing units"
            assert "units" in this_case.cft_ds[var].attrs, msg

        # Ensure that NaN values are handled correctly.
        # First, ensure that there are actually some NaN values that will be tested.
        assert np.any(np.isnan(this_case.cft_ds["GRAINC_TO_FOOD_PERHARV"]))
        assert np.any(np.isnan(this_case.cft_ds[f"GRAINC_TO_FOOD_{m}_PERHARV"]))
        assert np.any(np.isnan(this_case.cft_ds[f"YIELD_{m}_PERHARV"]))
        # Now test that YIELD_ANN, which is just YIELD_PERHARV summed over the mxharvests dimension,
        # doesn't have any NaN values.
        assert not np.any(np.isnan(this_case.cft_ds[f"YIELD_{m}_ANN"]))

    # Ensure that these derived variables have the right dims
    assert this_case.cft_ds["crop_cft_area"].dims == ("pft", "cft", "time")
    assert this_case.cft_ds["crop_area"].dims == ("pft", "crop", "time")

    # Check that time axis is what we expect
    expected_time_da = xr.DataArray(
        data=np.array(np.arange(START_YEAR, END_YEAR + 1)),
        dims=["time"],
        attrs={"long_name": "year"},
    )
    expected_time_da = expected_time_da.assign_coords({"time": expected_time_da})
    assert this_case.cft_ds["time"].equals(expected_time_da)

    # Ensure that values of some derived variables are correct
    assert this_case.cft_ds["crop_cft_area"].mean().values == pytest.approx(375510026.3004716)
    assert this_case.cft_ds["crop_cft_prod_marketable"].mean().values == pytest.approx(
        197061665729.85678
    )
    assert this_case.cft_ds["crop_cft_yield_marketable"].mean().values == pytest.approx(
        605.3616856892903
    )
    assert this_case.cft_ds["crop_area"].mean().values == pytest.approx(1001360070.1345912)
    assert this_case.cft_ds["crop_prod_marketable"].mean().values == pytest.approx(
        525497775279.61804
    )
    assert this_case.cft_ds["crop_yield_marketable"].mean().values == pytest.approx(
        569.6446054172038
    )


def test_setup_cropcase(cropcase):
    """
    Make sure that CropCase does not error when importing test data
    """
    this_case = cropcase

    # Perform a bunch of checks
    check_crujra_matreqs_case_shared(this_case)

    # Ensure that saved file has all 5 years even though we only asked for 3
    ds = xr.open_dataset(this_case._cft_ds_file, decode_timedelta=False)
    assert END_YEAR - START_YEAR + 1 < 5
    assert ds.sizes["time"] == 5

    # Check that equality works when called on a deep copy of itself...
    this_case_copy = copy.deepcopy(this_case)
    assert this_case == this_case_copy
    # ... but not after changing something
    this_case_copy.name = "82nr924nd"
    assert this_case != this_case_copy


def test_reimport_cropcase(cropcase):
    """
    Make sure that CropCase works when importing a saved cft_ds.nc file
    """

    # Make sure everything is set up how we expect:
    # Directory should exist to hold cft_ds.nc
    cft_ds_dir = cropcase._cft_ds_dir
    assert os.path.exists(cft_ds_dir) and os.path.isdir(cft_ds_dir)
    # cft_ds.nc should be in that directory
    cft_ds_file = cropcase._cft_ds_file
    assert os.path.dirname(cft_ds_file) == cft_ds_dir

    # Make copy of cft_ds.nc
    new_subdir = os.path.join(cft_ds_dir, "some_subdir")
    os.makedirs(new_subdir)
    new_file = os.path.join(new_subdir, os.path.basename(cft_ds_file))
    copyfile(cft_ds_file, new_file)
    assert os.path.exists(new_file)

    # Reimport, making sure we didn't re-save the file
    with patch("ctsm_postprocessing.crops.cropcase._save_cft_ds_to_netcdf") as mock_save:
        this_case_2 = import_default_cropcase(new_subdir)
        # Verify that _save_cft_ds_to_netcdf was not called
        mock_save.assert_not_called()
    assert this_case_2._save_netcdf is False

    # Make sure we got CFT and crop lists
    assert this_case_2._cft_list is not None
    assert this_case_2.crop_list is not None

    # Perform a bunch of checks
    check_crujra_matreqs_case_shared(this_case_2)


def test_setup_cropcase_noperms(tmp_path):
    """
    Make sure that CropCase doesn't try to save file if user doesn't have write perms
    """
    temp_dir = str(tmp_path)

    # Disable user write bit
    os.chmod(temp_dir, 0o444)

    this_case = CropCase(
        name=CASE_NAME,
        file_dir=FILE_DIR,
        start_year=START_YEAR,
        end_year=END_YEAR,
        cft_ds_dir=temp_dir,
    )
    assert [x.name for x in this_case.crop_list] == DEFAULT_CROPS_TO_INCLUDE

    for m in MATURITY_LEVELS:
        # Ensure that derived variables are present.
        assert f"GRAINC_TO_FOOD_{m}_PERHARV" in this_case.cft_ds
        assert f"YIELD_{m}_PERHARV" in this_case.cft_ds
        assert f"YIELD_{m}_ANN" in this_case.cft_ds

        # Ensure that not all yield values are zero
        assert np.any(this_case.cft_ds[f"GRAINC_TO_FOOD_{m}_PERHARV"] > 0)
        assert np.any(this_case.cft_ds[f"YIELD_{m}_PERHARV"] > 0)
        assert np.any(this_case.cft_ds[f"YIELD_{m}_ANN"] > 0)

        # Ensure that NaN values are handled correctly.
        # First, ensure that there are actually some NaN values that will be tested.
        assert np.any(np.isnan(this_case.cft_ds["GRAINC_TO_FOOD_PERHARV"]))
        assert np.any(np.isnan(this_case.cft_ds[f"GRAINC_TO_FOOD_{m}_PERHARV"]))
        assert np.any(np.isnan(this_case.cft_ds[f"YIELD_{m}_PERHARV"]))
        # Now test that YIELD_ANN, which is just YIELD_PERHARV summed over the mxharvests dimension,
        # doesn't have any NaN values.
        assert not np.any(np.isnan(this_case.cft_ds[f"YIELD_{m}_ANN"]))

    # Check that we only have 3 years in the dataset
    assert this_case.cft_ds.sizes["time"] == 3

    # Set user write bit
    os.chmod(temp_dir, 0o644)


def test_setup_cropcase_nofile(tmp_path):
    """
    Make sure that CropCase doesn't try to save file if user DOES have write perms but says
    force_no_cft_ds_file=True
    """
    temp_dir = str(tmp_path)
    this_case = CropCase(
        name=CASE_NAME,
        file_dir=FILE_DIR,
        start_year=START_YEAR,
        end_year=END_YEAR,
        cft_ds_dir=temp_dir,
        force_no_cft_ds_file=True,
    )
    assert [x.name for x in this_case.crop_list] == DEFAULT_CROPS_TO_INCLUDE

    # Ensure that at least one derived variable is present.
    assert "GRAINC_TO_FOOD_MARKETABLE_PERHARV" in this_case.cft_ds

    # Ensure file wasn't saved
    assert not os.path.exists(os.path.join(temp_dir, CFT_DS_FILENAME))


def test_setup_cropcase_allyears(tmp_path):
    """
    Make sure that CropCase includes all years if start_year and end_year not specified
    """
    temp_dir = str(tmp_path)
    this_case = CropCase(
        name=CASE_NAME,
        file_dir=FILE_DIR,
        cft_ds_dir=temp_dir,
        force_no_cft_ds_file=True,
    )

    # Check that time axis is what we expect
    expected_time_da = xr.DataArray(
        data=np.array(np.arange(FIRST_AVAILABLE_YEAR, LAST_AVAILABLE_YEAR + 1)),
        dims=["time"],
        attrs={"long_name": "year"},
    )
    expected_time_da = expected_time_da.assign_coords({"time": expected_time_da})
    assert this_case.cft_ds["time"].equals(expected_time_da)


def test_setup_cropcase_just_startyear(tmp_path):
    """
    Make sure that CropCase is right if only start_year is specified
    """
    temp_dir = str(tmp_path)
    this_case = CropCase(
        name=CASE_NAME,
        file_dir=FILE_DIR,
        start_year=START_YEAR,
        cft_ds_dir=temp_dir,
        force_no_cft_ds_file=True,
    )

    # Check that time axis is what we expect
    expected_time_da = xr.DataArray(
        data=np.array(np.arange(START_YEAR, LAST_AVAILABLE_YEAR + 1)),
        dims=["time"],
        attrs={"long_name": "year"},
    )
    expected_time_da = expected_time_da.assign_coords({"time": expected_time_da})
    assert this_case.cft_ds["time"].equals(expected_time_da)


def test_setup_cropcase_just_endyear(tmp_path):
    """
    Make sure that CropCase is right if only end_year is specified
    """
    temp_dir = str(tmp_path)
    this_case = CropCase(
        name=CASE_NAME,
        file_dir=FILE_DIR,
        end_year=END_YEAR,
        cft_ds_dir=temp_dir,
        force_no_cft_ds_file=True,
    )

    # Check that time axis is what we expect
    expected_time_da = xr.DataArray(
        data=np.array(np.arange(FIRST_AVAILABLE_YEAR, END_YEAR + 1)),
        dims=["time"],
        attrs={"long_name": "year"},
    )
    expected_time_da = expected_time_da.assign_coords({"time": expected_time_da})
    assert this_case.cft_ds["time"].equals(expected_time_da)


def test_setup_cropcase_error_if_no_crop_for_cft(tmp_path):
    """Make sure error is thrown if a CFT has no crop"""
    crops_to_include = [DEFAULT_CROPS_TO_INCLUDE[0]]
    assert crops_to_include != DEFAULT_CROPS_TO_INCLUDE

    msg = f"Which crop should {DEFAULT_CROPS_TO_INCLUDE[1]} be associated with?"
    with pytest.raises(KeyError, match=msg):
        CropCase(
            name=CASE_NAME,
            file_dir=FILE_DIR,
            start_year=START_YEAR,
            end_year=END_YEAR,
            crops_to_include=crops_to_include,
            cft_ds_dir=str(tmp_path),
        )


def test_setup_cropcase_error_if_newfile_and_nofile(tmp_path):
    """
    Make sure that CropCase raises error if both force_new_cft_ds_file and force_no_cft_ds_file
    are true
    """
    temp_dir = str(tmp_path)
    with pytest.raises(ValueError):
        CropCase(
            name=CASE_NAME,
            file_dir=FILE_DIR,
            start_year=START_YEAR,
            end_year=END_YEAR,
            crops_to_include=DEFAULT_CROPS_TO_INCLUDE,
            cft_ds_dir=temp_dir,
            force_new_cft_ds_file=True,
            force_no_cft_ds_file=True,
        )


def test_cropcase_equality(cropcase):
    """
    Basic checks of CropCase.__eq__ and __ne__
    """
    this_case = cropcase

    # Check that changing cft_ds causes inequality
    copy_case = copy.deepcopy(this_case)
    assert this_case == copy_case
    copy_case.cft_ds = copy_case.cft_ds.isel(crop=0)
    assert this_case != copy_case

    # Check that changing an attribute causes inequality
    copy_case = copy.deepcopy(this_case)
    copy_case.name = "wiefnweurej"
    assert this_case != copy_case


def test_cropcase_sel_nothing(cropcase):
    """
    Make sure that CropCase.sel() with no (kw)args returns an exact copy
    """
    this_case = cropcase
    assert this_case == this_case.sel()


def test_cropcase_sel_cotton(cropcase):
    """
    Test CropCase.sel() with a selection
    """
    this_case = cropcase
    this_dim = "crop"
    sel_crop = "cotton"
    this_case_sel = this_case.sel({this_dim: sel_crop})

    # Check that sel() got rid of crop dimension
    assert this_dim in this_case.cft_ds.dims
    assert this_dim not in this_case_sel.cft_ds.dims

    # Check that sel() got rid of all but one crop
    assert this_case.cft_ds.sizes[this_dim] > 1
    assert this_case_sel.cft_ds[this_dim].values == sel_crop

    # Check that cft_ds objects differ
    assert not this_case.cft_ds.equals(this_case_sel.cft_ds)

    # Check == and != on result
    assert not this_case == this_case_sel
    assert this_case != this_case_sel


def test_cropcase_isel_nothing(cropcase):
    """
    Make sure that CropCase.isel() with no (kw)args returns an exact copy
    """
    this_case = cropcase
    assert this_case == this_case.isel()


def test_cropcaselist_isel_one_timestep(cropcase):
    """
    Test CropCaseList.isel() with a selection
    """
    this_case = cropcase
    this_dim = "time"
    isel_timestep = 2
    this_case_isel = this_case.isel({this_dim: isel_timestep})

    # Check that sel() got rid of time dimension
    assert this_dim in this_case.cft_ds.dims
    assert this_dim not in this_case_isel.cft_ds.dims

    # Check that sel() got rid of all but one timestep
    assert this_case.cft_ds.sizes[this_dim] > 1
    assert this_case_isel.cft_ds[this_dim].size == 1

    # Check that cft_ds objects differ
    assert not this_case.cft_ds.equals(this_case_isel.cft_ds)

    # Check == and != on result
    assert not this_case == this_case_isel
    assert this_case != this_case_isel
