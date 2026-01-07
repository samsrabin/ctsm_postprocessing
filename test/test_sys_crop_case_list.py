"""
Module to unit-test CropCaseList
"""

# pylint: disable=redefined-outer-name
# Note: redefined-outer-name is disabled because pytest fixtures are used as test function parameters

import sys
import os
import copy
import pytest

try:
    # Attempt relative import if running as part of a package
    from ..crop_case_list import CropCaseList
    from ..crop_defaults import DEFAULT_CROPS_TO_INCLUDE
    from .defaults import START_YEAR, END_YEAR, CESM_OUTPUT_DIR
    from .test_sys_cropcase import check_crujra_matreqs_case_shared
except ImportError:
    # Add both the parent directory (for crops module) and grandparent (for test module)
    parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    grandparent_dir = os.path.dirname(parent_dir)
    sys.path.insert(0, parent_dir)
    sys.path.insert(0, grandparent_dir)
    from crops.crop_case_list import CropCaseList
    from crops.crop_defaults import DEFAULT_CROPS_TO_INCLUDE
    from test.defaults import START_YEAR, END_YEAR, CESM_OUTPUT_DIR
    from test.test_sys_cropcase import check_crujra_matreqs_case_shared

CASE_NAME_LIST = ["crujra_matreqs", "crujra_matreqs_copy"]
DEFAULT_OPTS = {}
DEFAULT_OPTS["case_name_list"] = CASE_NAME_LIST
DEFAULT_OPTS["CESM_output_dir"] = CESM_OUTPUT_DIR
DEFAULT_OPTS["crops_to_include"] = DEFAULT_CROPS_TO_INCLUDE
DEFAULT_OPTS["start_year"] = START_YEAR
DEFAULT_OPTS["end_year"] = END_YEAR
DEFAULT_OPTS["verbose"] = False
DEFAULT_OPTS["force_new_cft_ds_file"] = False
DEFAULT_OPTS["force_no_cft_ds_file"] = (
    True  # Don't set this to False unless you have a temp dir setup for cft_ds to be saved to
)


@pytest.fixture(scope="session")
def cropcaselist_base():
    """
    Session-scoped fixture to create a CropCaseList instance once for all tests.
    This is created only once per test session to speed up testing.
    """
    return CropCaseList(opts=DEFAULT_OPTS)


@pytest.fixture
def cropcaselist(cropcaselist_base):
    """
    Fixture to provide a deep copy of the base CropCaseList instance for testing.
    Each test gets its own copy to ensure test isolation.
    """
    return copy.deepcopy(cropcaselist_base)


def test_setup_cropcaselist(cropcaselist):
    """
    Make sure that CropCaseList does not error when importing test data
    """
    this_case_list = cropcaselist

    # Perform a bunch of checks
    check_crujra_matreqs_case_shared(this_case_list[0])


def test_cropcaselist_equality(cropcaselist):
    """
    Basic checks of CropCaseList.__eq__() and __ne__()
    """
    this_case_list = cropcaselist

    # Check that equality works when called on a deep copy of itself...
    this_case_list_copy = copy.deepcopy(this_case_list)
    assert this_case_list == this_case_list_copy
    # ... but not after changing something
    this_case_list_copy.names = ["82nr924nd", "jif8eh598h"]
    assert this_case_list != this_case_list_copy


def test_cropcaselist_sel_nothing(cropcaselist):
    """
    Make sure that CropCaseList.sel() with no (kw)args returns an exact copy
    """
    this_case_list = cropcaselist
    assert this_case_list == this_case_list.sel()


def test_cropcaselist_sel_cotton(cropcaselist):
    """
    Test CropCaseList.sel() with a selection
    """
    this_case_list = cropcaselist
    this_dim = "crop"
    sel_crop = "cotton"
    this_case_list_sel = this_case_list.sel({this_dim: sel_crop})

    # Check that sel() got rid of crop dimension
    for case in this_case_list:
        assert this_dim in case.cft_ds.dims
    for case_sel in this_case_list_sel:
        assert this_dim not in case_sel.cft_ds.dims

    # Check that sel() got rid of all but one crop
    for case in this_case_list:
        assert case.cft_ds.sizes[this_dim] > 1
    for case_sel in this_case_list_sel:
        assert case_sel.cft_ds[this_dim].values == sel_crop

    # Check that cft_ds objects differ
    for c, case in enumerate(this_case_list):
        case_sel = this_case_list_sel[c]
        assert not case.cft_ds.equals(case_sel.cft_ds)

    # Check == and != on result
    assert not this_case_list == this_case_list_sel
    assert this_case_list != this_case_list_sel


def test_cropcaselist_isel_nothing(cropcaselist):
    """
    Make sure that CropCaseList.isel() with no (kw)args returns an exact copy
    """
    this_case_list = cropcaselist
    assert this_case_list == this_case_list.isel()


def test_cropcaselist_isel_one_timestep(cropcaselist):
    """
    Test CropCaseList.isel() with a selection
    """
    this_case_list = cropcaselist
    this_dim = "time"
    isel_timestep = 2
    this_case_list_isel = this_case_list.isel({this_dim: isel_timestep})

    # Check that sel() got rid of time dimension
    for case in this_case_list:
        assert this_dim in case.cft_ds.dims
    for case_isel in this_case_list_isel:
        assert this_dim not in case_isel.cft_ds.dims

    # Check that sel() got rid of all but one timestep
    for case in this_case_list:
        assert case.cft_ds.sizes[this_dim] > 1
    for case_isel in this_case_list_isel:
        assert case_isel.cft_ds[this_dim].size == 1

    # Check that cft_ds objects differ
    for c, case in enumerate(this_case_list):
        case_isel = this_case_list_isel[c]
        assert not case.cft_ds.equals(case_isel.cft_ds)

    # Check == and != on result
    assert not this_case_list == this_case_list_isel
    assert this_case_list != this_case_list_isel
