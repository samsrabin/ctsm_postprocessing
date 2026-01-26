"""
Module for handling lists of crops in the Community Terrestrial Systems Model (CTSM).

This module defines the CropList class, which represents a list of crops, and provides methods for
accessing crops by index or name.
"""

import numpy as np

from .cftlist import CftList
from .crop import Crop


class CropList:
    """
    Represents a list of crops in the Community Terrestrial Systems Model (CTSM).

    Attributes:
        crop_list (list[Crop]): List of Crop objects.
    """

    def __init__(self, crops_to_include: list[str], cft_list: CftList) -> None:
        """
        Initialize a CropList instance.

        Parameters:
            crops_to_include (list[str]): List of crop names to include.
            cft_list (CftList): CftList instance containing CFTs to include in the crops.

        Raises:
            ValueError: If crops_to_include contains duplicate crop names.
            RuntimeError: If no crops from crops_to_include are found in cft_list.
        """
        if len(crops_to_include) != len(np.unique(crops_to_include)):
            raise ValueError("Duplicate crop(s) found in crops_to_include")
        self.crop_list: list[Crop] = [Crop(x, cft_list) for x in crops_to_include]
        if not self.crop_list:
            raise RuntimeError("No crops_to_include found in cft_list")

    def __eq__(self, other: object) -> bool:
        """
        Compare two CropList instances for equality.

        Parameters:
            other (object): Object to compare with this CropList instance.

        Returns:
            bool: True if both CropList instances have equal crop_list attributes, False otherwise.

        Raises:
            TypeError: If other is not a CropList instance.
        """
        # Check that they're both CropLists
        if not isinstance(other, self.__class__):
            raise TypeError(f"== not supported between {self.__class__} and {type(other)}")
        result = self.crop_list == other.crop_list
        return result

    def __getitem__(self, index: int | str) -> Crop:
        """
        Get a Crop by index or name.

        Parameters:
            index (int | str): Integer index or string name of the crop to retrieve.

        Returns:
            Crop: The Crop at the specified index or with the specified name.

        Raises:
            KeyError: If a string index is provided but no crop with that name is found.
            RuntimeError: If crop_list is empty when searching by name.
        """
        if isinstance(index, str):
            found = False
            i = None
            for i, crop in enumerate(self.crop_list):
                found = crop.name == index
                if found:
                    break
            if not found:
                raise KeyError(f"No crop found matching '{index}'")
            if i is None:
                raise RuntimeError("crop_list is empty")
            return self.crop_list[i]
        return self.crop_list[index]

    def __str__(self) -> str:
        """
        Return a string representation of the CropList instance.

        Returns:
            str: Multi-line string with each Crop's string representation on a separate line.
        """
        results = []
        for crop in self.crop_list:
            results.append(str(crop))
        return "\n".join(results)
