"""
Defaults for use in tests
"""
import os

START_YEAR = 1988
END_YEAR = 1990

CASE_NAME = "crujra_matreqs"
CESM_OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "testdata", "CESM_output_dir")
FILE_DIR = os.path.join(CESM_OUTPUT_DIR, CASE_NAME, "lnd", "hist")
