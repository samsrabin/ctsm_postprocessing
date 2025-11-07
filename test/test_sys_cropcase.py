"""
Module to unit-test cropcase.py
"""
# pylint: disable=redefined-outer-name
# Note: redefined-outer-name is disabled because pytest fixtures are used as test function parameters

import sys
import os
import numpy as np
import xarray as xr
import pytest

try:
    # Attempt relative import if running as part of a package
    from ..cropcase import CropCase, CFT_DS_FILENAME
    from ..crop_defaults import DEFAULT_CFTS_TO_INCLUDE, DEFAULT_CROPS_TO_INCLUDE
except ImportError:
    # Fallback to absolute import if running as a script
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from crops.cropcase import CropCase, CFT_DS_FILENAME
    from crops.crop_defaults import DEFAULT_CFTS_TO_INCLUDE, DEFAULT_CROPS_TO_INCLUDE

START_YEAR = 1988
END_YEAR = 1990


def test_setup_cropcase(tmp_path):
    """
    Make sure that CropCase does not error when importing test data
    """
    temp_dir = str(tmp_path)
    name = "crujra_matreqs"
    file_dir = os.path.join(os.path.dirname(__file__), "testdata")
    this_case = CropCase(
        name=name,
        file_dir=file_dir,
        start_year=START_YEAR,
        end_year=END_YEAR,
        cfts_to_include=DEFAULT_CFTS_TO_INCLUDE,
        crops_to_include=DEFAULT_CROPS_TO_INCLUDE,
        cft_ds_dir=temp_dir,
    )
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
    assert "GRAINC_TO_FOOD_VIABLE_PERHARV" in this_case.cft_ds
    assert "YIELD_PERHARV" in this_case.cft_ds
    assert "YIELD_ANN" in this_case.cft_ds

    # Ensure that not all yield values are zero
    assert np.any(this_case.cft_ds["GRAINC_TO_FOOD_VIABLE_PERHARV"] > 0)
    assert np.any(this_case.cft_ds["YIELD_PERHARV"] > 0)
    assert np.any(this_case.cft_ds["YIELD_ANN"] > 0)

    # Ensure that these derived variables have the right dims
    assert this_case.cft_ds["crop_cft_area"].dims == ("pft", "cft", "time")
    assert this_case.cft_ds["crop_area"].dims == ("pft", "crop", "time")

    # Ensure that NaN values are handled correctly.
    # First, ensure that there are actually some NaN values that will be tested.
    assert np.any(np.isnan(this_case.cft_ds["GRAINC_TO_FOOD_PERHARV"]))
    assert np.any(np.isnan(this_case.cft_ds["YIELD_PERHARV"]))
    # Now test that YIELD_ANN, which is just YIELD_PERHARV summed over the mxharvests dimension,
    # doesn't have any NaN values.
    assert not np.any(np.isnan(this_case.cft_ds["YIELD_ANN"]))

    # Ensure that saved file has all 5 years even though we only asked for 3
    ds = xr.open_dataset(os.path.join(temp_dir, CFT_DS_FILENAME))
    assert ds.sizes["time"] == 5

    # Ensure that values of some derived variables are correct
    assert this_case.cft_ds["crop_cft_area"].mean().values == pytest.approx(379009483.9427163)
    assert this_case.cft_ds["crop_cft_prod"].mean().values == pytest.approx(198672315418.34564)
    assert this_case.cft_ds["crop_cft_yield"].mean().values == pytest.approx(602.2368436177571)
    assert this_case.cft_ds["crop_area"].mean().values == pytest.approx(1010691957.180577)
    assert this_case.cft_ds["crop_prod"].mean().values == pytest.approx(529792841115.5884)
    assert this_case.cft_ds["crop_yield"].mean().values == pytest.approx(568.3093914610291)


def test_setup_cropcase_noperms(tmp_path):
    """
    Make sure that CropCase doesn't try to save file if user doesn't have write perms
    """
    temp_dir = str(tmp_path)
    name = "crujra_matreqs"

    # Disable user write bit
    os.chmod(temp_dir, 0o444)

    file_dir = os.path.join(os.path.dirname(__file__), "testdata")
    this_case = CropCase(
        name=name,
        file_dir=file_dir,
        start_year=START_YEAR,
        end_year=END_YEAR,
        cfts_to_include=DEFAULT_CFTS_TO_INCLUDE,
        crops_to_include=DEFAULT_CROPS_TO_INCLUDE,
        cft_ds_dir=temp_dir,
    )
    assert [x.name for x in this_case.crop_list] == DEFAULT_CROPS_TO_INCLUDE

    # Ensure that derived variables are present.
    assert "GRAINC_TO_FOOD_VIABLE_PERHARV" in this_case.cft_ds
    assert "YIELD_PERHARV" in this_case.cft_ds
    assert "YIELD_ANN" in this_case.cft_ds

    # Ensure that not all yield values are zero
    assert np.any(this_case.cft_ds["GRAINC_TO_FOOD_VIABLE_PERHARV"] > 0)
    assert np.any(this_case.cft_ds["YIELD_PERHARV"] > 0)
    assert np.any(this_case.cft_ds["YIELD_ANN"] > 0)

    # Ensure that NaN values are handled correctly.
    # First, ensure that there are actually some NaN values that will be tested.
    assert np.any(np.isnan(this_case.cft_ds["GRAINC_TO_FOOD_PERHARV"]))
    assert np.any(np.isnan(this_case.cft_ds["YIELD_PERHARV"]))
    # Now test that YIELD_ANN, which is just YIELD_PERHARV summed over the mxharvests dimension,
    # doesn't have any NaN values.
    assert not np.any(np.isnan(this_case.cft_ds["YIELD_ANN"]))

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
    name = "crujra_matreqs"
    file_dir = os.path.join(os.path.dirname(__file__), "testdata")
    this_case = CropCase(
        name=name,
        file_dir=file_dir,
        start_year=START_YEAR,
        end_year=END_YEAR,
        cfts_to_include=DEFAULT_CFTS_TO_INCLUDE,
        crops_to_include=DEFAULT_CROPS_TO_INCLUDE,
        cft_ds_dir=temp_dir,
        force_no_cft_ds_file=True,
    )
    assert [x.name for x in this_case.crop_list] == DEFAULT_CROPS_TO_INCLUDE

    # Ensure that at least one derived variable is present.
    assert "GRAINC_TO_FOOD_VIABLE_PERHARV" in this_case.cft_ds

    # Ensure file wasn't saved
    assert not os.path.exists(os.path.join(temp_dir, CFT_DS_FILENAME))


def test_setup_cropcase_error_if_newfile_and_nofile(tmp_path):
    """
    Make sure that CropCase raises error if both force_new_cft_ds_file and force_no_cft_ds_file
    are true
    """
    temp_dir = str(tmp_path)
    name = "crujra_matreqs"
    file_dir = os.path.join(os.path.dirname(__file__), "testdata")
    with pytest.raises(ValueError):
        CropCase(
            name=name,
            file_dir=file_dir,
            start_year=START_YEAR,
            end_year=END_YEAR,
            cfts_to_include=DEFAULT_CFTS_TO_INCLUDE,
            crops_to_include=DEFAULT_CROPS_TO_INCLUDE,
            cft_ds_dir=temp_dir,
            force_new_cft_ds_file=True,
            force_no_cft_ds_file=True,
        )
