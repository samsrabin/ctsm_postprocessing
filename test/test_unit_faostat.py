"""
Module to unit-test faostat.py
"""

import sys
import os
import unittest
import numpy as np
import pandas as pd

try:
    # Attempt relative import if running as part of a package
    from ..crops import faostat
except ImportError:
    # Fallback to absolute import if running as a script
    sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    from crops import faostat

# pylint: disable=protected-access
# pylint: disable=too-many-public-methods


class TestUnitFaostat(unittest.TestCase):
    """
    Class to unit-test faostat.py
    """

    def setUp(self):
        self.file_path = os.path.join(os.path.dirname(__file__), "testdata", "test_faostat.csv")
        self.faostat = None

    def test_restrict_years_unchanged(self):
        """
        Test that restrict_years with no y1/yN given just returns the same DataFrame
        """
        df = pd.read_csv(self.file_path)
        df_orig = df.copy()
        df = faostat.restrict_years(df)
        self.assertTrue(df.equals(df_orig))

    def test_restrict_years_y1(self):
        """
        Test that restrict_years with just y1 works
        """
        df = pd.read_csv(self.file_path)
        df = faostat.restrict_years(df, y1=1987)
        self.assertTrue(np.array_equal(df["Year"].unique(), [1987, 1988, 1989]))

    def test_restrict_years_y1_keyerror(self):
        """
        Test that restrict_years with none found given y1 throws KeyError
        """
        df = pd.read_csv(self.file_path)
        with self.assertRaisesRegex(KeyError, "found >="):
            faostat.restrict_years(df, y1=2025)

    def test_restrict_years_yn(self):
        """
        Test that restrict_years with just yN works
        """
        df = pd.read_csv(self.file_path)
        df = faostat.restrict_years(df, yN=1987)
        self.assertTrue(np.array_equal(df["Year"].unique(), [1986, 1987]))

    def test_restrict_years_yn_keyerror(self):
        """
        Test that restrict_years with none found given yN throws KeyError
        """
        df = pd.read_csv(self.file_path)
        with self.assertRaisesRegex(KeyError, "found <="):
            faostat.restrict_years(df, yN=1955)

    def test_restrict_years_y1yn(self):
        """
        Test that restrict_years with y1-yN works
        """
        df = pd.read_csv(self.file_path)
        df = faostat.restrict_years(df, y1=1987, yN=1988)
        self.assertTrue(np.array_equal(df["Year"].unique(), [1987, 1988]))

    def test_restrict_years_y1yn_keyerror(self):
        """
        Test that restrict_years with none found given y1-yN throws KeyError
        """
        df = pd.read_csv(self.file_path)
        with self.assertRaisesRegex(KeyError, "found in"):
            faostat.restrict_years(df, y1=2024, yN=2025)

    def test_restrict_years_yny1_notimplementederror(self):
        """
        Test that restrict_years with none found given y1-yN with y1>yN throws NotImplementedError
        """
        df = pd.read_csv(self.file_path)
        with self.assertRaisesRegex(NotImplementedError, "y1 > yN"):
            faostat.restrict_years(df, y1=1988, yN=1987)

    def _basic_init(self):
        self.faostat = faostat.FaostatProductionCropsLivestock(
            self.file_path,
        )

    def test_class_init(self):
        """
        Test the most basic initialization of a FaostatProductionCropsLivestock instance: Just
        giving file path
        """

        # Import
        self._basic_init()

        # Check that maizes were combined
        self.assertEqual(
            len([x for x in self.faostat.data["Crop"].unique() if "Maize" in x]),
            1,
        )
        is_usa_prod_1987 = (
            (self.faostat.data["Area"] == "USA")
            & (self.faostat.data["Year"] == 1987)
            & (self.faostat.data["Element"] == "Production")
            & (self.faostat.data["Crop"] == "Maize")
        )
        val = self.faostat.data.Value[is_usa_prod_1987]
        self.assertEqual(len(val), 1)
        self.assertEqual(list(val)[0], 9.5 + 10.5)

        # Check that overall "China" was removed
        self.assertFalse(any(self.faostat.data["Area"] == "China"))

    def test_class_init_y1yn(self):
        """
        Test initializing a FaostatProductionCropsLivestock dataset with y1-yN
        """
        self.faostat = faostat.FaostatProductionCropsLivestock(
            self.file_path,
            y1=1987,
            yN=1988,
        )
        self.assertTrue(np.array_equal(self.faostat.data["Year"].unique(), [1987, 1988]))

    def test_class_get_element(self):
        """
        Test FaostatProductionCropsLivestock.get_element()
        """

        # Import
        self._basic_init()

        # Get element
        df = self.faostat.get_element(
            "Production",
        )

        # Test that all rows in Element are "Production"
        self.assertTrue(np.array_equal(df["Element"].unique(), ["Production"]))

        # Test that the correct columns were set as indices in the MultiIndex
        self.assertTrue(np.array_equal(df.index.names, ["Crop", "Year", "Area"]))

    def test_class_get_element_keyerror(self):
        """
        Test FaostatProductionCropsLivestock.get_element() KeyError for invalid Element
        """

        # Import
        self._basic_init()

        # Get element
        with self.assertRaisesRegex(KeyError, "No FAOSTAT element found matching"):
            self.faostat.get_element(
                "Produsdffdrection",
            )

    def test_class_get_element_years(self):
        """
        Test FaostatProductionCropsLivestock.get_element() with y1-yN
        """

        # Import
        self._basic_init()

        # Get element
        df = self.faostat.get_element(
            "Production",
            y1=1987,
            yN=1988,
        )

        # Test
        self.assertTrue(np.array_equal(df.index.unique(level="Year"), [1987, 1988]))

    def test_class_get_element_crops(self):
        """
        Test FaostatProductionCropsLivestock.get_element() with a crop dict specified
        """

        # Import
        self._basic_init()

        # Get element
        fao_to_clm_dict = {"Maize": "corn", "Seed cotton, unginned": "cotton"}
        df = self.faostat.get_element(
            "Production",
            fao_to_clm_dict=fao_to_clm_dict,
        )

        # Test
        self.assertEqual(set(df.index.unique(level="Crop")), set(["corn", "cotton"]))

    def test_class_get_element_crops_keyerror(self):
        """
        Test that FaostatProductionCropsLivestock.get_element() gives a KeyError if an expected crop
        is missing
        """

        # Import
        self._basic_init()

        # Get element
        fao_to_clm_dict = {"Maize": "corn", "abcdef": "cotton"}
        with self.assertRaises(KeyError):
            self.faostat.get_element(
                "Production",
                fao_to_clm_dict=fao_to_clm_dict,
            )
