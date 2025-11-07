"""
Module to unit-test CropCaseList
"""

# pylint: disable=redefined-outer-name
# Note: redefined-outer-name is disabled because pytest fixtures are used as test function parameters

import sys
import os

try:
    # Attempt relative import if running as part of a package
    from ..crop_case_list import CropCaseList
    from ..crop_defaults import DEFAULT_CFTS_TO_INCLUDE, DEFAULT_CROPS_TO_INCLUDE
    from .defaults import START_YEAR, END_YEAR, CESM_OUTPUT_DIR
    from .test_sys_cropcase import check_crujra_matreqs_case_shared
except ImportError:
    # Add both the parent directory (for crops module) and grandparent (for test module)
    parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    grandparent_dir = os.path.dirname(parent_dir)
    sys.path.insert(0, parent_dir)
    sys.path.insert(0, grandparent_dir)
    from crops.crop_case_list import CropCaseList
    from crops.crop_defaults import DEFAULT_CFTS_TO_INCLUDE, DEFAULT_CROPS_TO_INCLUDE
    from test.defaults import START_YEAR, END_YEAR, CESM_OUTPUT_DIR
    from test.test_sys_cropcase import check_crujra_matreqs_case_shared

CASE_NAME_LIST = ["crujra_matreqs", "crujra_matreqs_copy"]
DEFAULT_OPTS = {}
DEFAULT_OPTS["case_name_list"] = CASE_NAME_LIST
DEFAULT_OPTS["CESM_output_dir"] = CESM_OUTPUT_DIR
DEFAULT_OPTS["cfts_to_include"] = DEFAULT_CFTS_TO_INCLUDE
DEFAULT_OPTS["crops_to_include"] = DEFAULT_CROPS_TO_INCLUDE
DEFAULT_OPTS["start_year"] = START_YEAR
DEFAULT_OPTS["end_year"] = END_YEAR
DEFAULT_OPTS["verbose"] = False
DEFAULT_OPTS["force_new_cft_ds_file"] = False
DEFAULT_OPTS["force_no_cft_ds_file"] = (
    True  # Don't set this to False unless you have a temp dir setup for cft_ds to be saved to
)


def test_setup_cropcaselist():
    """
    Make sure that CropCaseList does not error when importing test data
    """

    # Define options
    opts = DEFAULT_OPTS

    this_case_list = CropCaseList(opts=opts)

    # Perform a bunch of checks
    check_crujra_matreqs_case_shared(this_case_list[0])
