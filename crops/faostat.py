"""
Import and work with FAOSTAT crop data.

Specifically, this lets you work with a "normalized" CSV file from the "Production > Crops and
livestock products" database.
"""

# pylint: disable=too-few-public-methods

import pandas as pd
import numpy as np

pd.options.mode.copy_on_write = True


def extract_clm_crops(data, fao_to_clm_dict):
    """
    Extract just the crops we care about for CLM, renamed to match CLM names
    """
    # Extract
    fao_crops = data.Crop.unique()
    fao_crops_from_dict = list(fao_to_clm_dict)
    missing_crops = [crop for crop in fao_crops_from_dict if crop not in fao_crops]
    if missing_crops:
        raise KeyError(f"Crops missing from FAOSTAT: {'; '.join(missing_crops)}")
    data = data[data["Crop"].isin(fao_crops_from_dict)]
    if len(data.Crop.unique()) != len(fao_to_clm_dict):
        raise RuntimeError("Unexpected # crops in FAOSTAT after extracting crops of interest")

    # Rename to match CLM
    data["Crop"] = data["Crop"].replace(fao_to_clm_dict)

    return data


def restrict_years(data, *, y1=None, yN=None):
    """
    Restrict an FAOSTAT DataFrame based on start and/or end year of interest
    """
    if y1 == yN == None:
        return data
    fao_year_range_str = f"{min(data.Year)}-{max(data.Year)}"
    if y1 is not None and yN is not None:
        if y1 > yN:
            raise NotImplementedError("y1 > yN")
        data = data.query(f"Year >= {y1} & Year <= {yN}")
        if data.empty:
            raise KeyError(f"No FAOSTAT years found in {y1}-{yN}; range {fao_year_range_str}")
    elif y1 is not None:
        data = data.query(f"Year >= {y1}")
        if data.empty:
            raise KeyError(f"No FAOSTAT years found >= {y1}; range {fao_year_range_str}")
    elif yN is not None:
        data = data.query(f"Year <= {yN}")
        if data.empty:
            raise KeyError(f"No FAOSTAT years found <= {yN}; range {fao_year_range_str}")
    return data


class FaostatProductionCropsLivestock:
    """
    Class for reading, storing, and working with FAOSTAT data
    """

    def __init__(self, file_path, *, low_memory=False, y1=None, yN=None):

        # Import
        self.file_path = file_path
        self.data = pd.read_csv(
            self.file_path,
            low_memory=low_memory,
        )

        # Because it's easy to confuse Item vs. Element
        self.data = self.data.rename(columns={"Item": "Crop"}, errors="raise")

        # Remove unneeded years, if year range given
        if y1 is not None or yN is not None:
            self.data = restrict_years(self.data, y1=y1, yN=yN)

        # Combine "Maize" and "Maize, green"
        self.data.Crop = self.data.Crop.replace("Maize.*", "Maize", regex=True)
        groupby_vars = ["Crop", "Year", "Element", "Area", "Unit"]
        self.data = self.data.groupby(by=groupby_vars, as_index=False).agg("sum")

        # Filter out "China," which includes all Chinas.
        # Doing this avoids potential double-counting.
        if "China" in self.data.Area.values:
            self.data = self.data.query('Area != "China"')

    def get_element(self, element, *, fao_to_clm_dict=None, y1=None, yN=None):
        """Extract an element of the FAOSTAT DataFrame, optionally restricting to certain crops/yrs.

        Args:
            element (str): Element to extract. E.g.: "Production", "Area harvested"
            fao_to_clm_dict (dict, optional): If you want to extract only the rows relevant to CLM,
                include this dictionary. The keys should be FAO crop names and the values should be
                what you want the FAO crops renamed to. E.g.:
                    {'Maize': 'corn', 'Rice': 'rice', 'Seed cotton, unginned': 'cotton'}
            y1 (int, optional): Minimum year to include in extracted DataFrame.
            yN (int, optional): Maximum year to include in extracted DataFrame.

        Raises:
            KeyError: If element not present in FAOSTAT data.Element

        Returns:
            Pandas DataFrame: A subset of self.df with just the element of interest, restricted to
                the crops and years of interest if relevant inputs provided. The MultiIndex will
                also be set to something useful.
        """
        # pylint: disable=too-many-arguments

        # Extract element of interest
        fao_this = self.data.copy().query(f"Element == '{element}'")
        if fao_this.empty:
            raise KeyError(f"No FAOSTAT element found matching {element}")

        # Drop all but columns of interest
        fao_this = fao_this[["Crop", "Year", "Element", "Area", "Unit", "Value"]]

        # Extract crops of interest
        if fao_to_clm_dict is not None:
            fao_this = extract_clm_crops(fao_this, fao_to_clm_dict)

        # Remove unneeded years, if year range given
        if y1 is not None or yN is not None:
            fao_this = restrict_years(fao_this, y1=y1, yN=yN)

        # Set MultiIndex to make subsequent operations easier
        fao_this = fao_this.set_index(["Crop", "Year", "Area"])

        return fao_this


    def get_clm_yield_prod_area_dict(self, fao_to_clm_dict: dict) -> dict:
        """Get yield, production, and area in terms of CLM crops

        Args:
            fao_to_clm_dict (dict): Keys should be FAO crop names and the values should be what
                you want the FAO crops renamed to. E.g.:
                    {'Maize': 'corn', 'Rice': 'rice', 'Seed cotton, unginned': 'cotton'}

        Raises:
            RuntimeError: On failure to align production and area indices

        Returns:
            dict: A dictionary with keys yield, prod, and area. Values are Pandas DataFrames.
        """

        fao_prod = self.get_element("Production", fao_to_clm_dict=fao_to_clm_dict)
        fao_area = self.get_element("Area harvested", fao_to_clm_dict=fao_to_clm_dict)

        # Only include where both production and area data are present
        def drop_a_where_not_in_b(a, b):
            return a.drop(list(a.index.difference(b.index)))

        fao_prod = drop_a_where_not_in_b(fao_prod, fao_area)
        fao_area = drop_a_where_not_in_b(fao_area, fao_prod)
        if not fao_prod.index.equals(fao_area.index):
            raise RuntimeError("Mismatch of prod and area indices after trying to align them")

        # Don't allow production where no area
        is_bad = (fao_prod["Value"] > 0) & (fao_area["Value"] == 0)
        fao_prod = fao_prod[~is_bad]
        fao_area = fao_area[~is_bad]
        if not fao_prod.index.equals(fao_area.index):
            raise RuntimeError(
                "Mismatch of prod and area indices after disallowing production where no area"
            )

        # Get yield
        fao_yield = fao_prod.copy()
        fao_yield["Element"] = "Yield"
        fao_yield["Unit"] = "/".join([fao_prod["Unit"].iloc[0], fao_area["Unit"].iloc[0]])
        fao_yield["Value"] = fao_prod["Value"] / fao_area["Value"]

        # Get dict
        fao_dict = {}
        fao_dict["yield"] = fao_yield
        fao_dict["prod"] = fao_prod
        fao_dict["area"] = fao_area

        return fao_dict
