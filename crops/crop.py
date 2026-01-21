"""
Module for handling crops in the Community Terrestrial Systems Model (CTSM).

This module defines the Crop class, which represents a crop consisting of multiple Crop Functional
Types (CFTs), and provides methods for retrieving indices corresponding to the crop within a
dataset.
"""

import numpy as np
import xarray as xr

from .cft import Cft


class Crop:
    # pylint: disable=too-few-public-methods
    """
    Represents a crop consisting of multiple Crop Functional Types (CFTs).

    Attributes:
        name (str): Name of the crop.
        cft_list (list[Cft]): List of CFTs included in this crop.
        pft_nums (list[int]): List of PFT numbers corresponding to the CFTs.
    """

    def __init__(self, name: str, cft_list: list[Cft], ds: xr.Dataset) -> None:  # pylint: disable=unused-argument
        """
        Initialize a Crop instance.

        Parameters:
            name (str): Name of the crop.
            cft_list (list[Cft]): List of CFTs to include in this crop. Only CFTs whose names
                                  contain the crop name will be included.
            ds (xarray.Dataset): Dataset containing crop data. Currently unused but kept for API
                                 consistency.
        """
        self.name = name

        # Get CFTs included in this crop
        self.cft_list: list[Cft] = []
        for cft in cft_list:
            if self.name not in cft.name:
                continue
            self.cft_list.append(cft)

        # Get information for all CFTs in this crop
        self.pft_nums: list[int] = []
        for cft in self.cft_list:
            self.pft_nums.append(cft.pft_num)

    def __eq__(self, other: object) -> bool:
        """
        Compare two Crop instances for equality.

        Parameters:
            other (object): Object to compare with this Crop instance.

        Returns:
            bool: True if all attributes match, False otherwise.

        Raises:
            TypeError: If other is not a Crop instance.
        """
        # Check that they're both Crops
        if not isinstance(other, self.__class__):
            raise TypeError(f"== not supported between {self.__class__} and {type(other)}")

        # Check that all attributes match (excluding methods)
        for attr in [a for a in dir(self) if not a.startswith("__")]:
            # Skip callable attributes (methods)
            if callable(getattr(self, attr)):
                continue
            if not hasattr(other, attr):
                return False
            try:
                value_self = getattr(self, attr)
                value_other = getattr(other, attr)
                if not isinstance(value_other, type(value_self)):
                    return False
                if isinstance(value_self, np.ndarray):
                    do_match = np.array_equal(value_self, value_other, equal_nan=True)
                else:
                    do_match = value_self == value_other or (
                        value_self is None and value_other is None
                    )
                if not do_match:
                    return False
            except:  # pylint: disable=bare-except
                return False
        return True

    def __str__(self) -> str:
        """
        Return a string representation of the Crop instance.

        Returns:
            str: String showing the crop name followed by a comma-separated list of CFT names
                 and their PFT numbers in parentheses.
        """
        return f"{self.name}: {', '.join(f'{x.name} ({x.pft_num})' for x in self.cft_list)}"
