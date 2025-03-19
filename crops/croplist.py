"""
Module for handling lists of crops in the Community Terrestrial Systems Model (CTSM).

This module defines the CropList class, which represents a list of crops, and provides methods for
accessing crops by index or name.
"""

import os
import sys
import numpy as np

try:
    # Attempt relative import if running as part of a package
    from .crop import Crop
except ImportError:
    # Fallback to absolute import if running as a script
    sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    from crops.crop import Crop


class CropList:
    """
    Represents a list of crops in the Community Terrestrial Systems Model (CTSM).

    Attributes:
        crop_list (list): List of Crop objects.
    """

    def __init__(self, crops_to_include, cft_list, ds):
        """
        Initialize a CropList instance.

        Parameters:
            crops_to_include (list): List of crop names to include.
            cft_list (list): List of CFTs to include in the crops.
            ds (xarray.Dataset): Dataset containing crop data.
        """
        if len(crops_to_include) != len(np.unique(crops_to_include)):
            raise ValueError("Duplicate crop(s) found in crops_to_include")
        self.crop_list = [Crop(x, cft_list, ds) for x in crops_to_include]
        if not self.crop_list:
            raise RuntimeError("No crops_to_include found in cft_list")

    def __getitem__(self, index):
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

    def __str__(self):
        results = []
        for crop in self.crop_list:
            results.append(str(crop))
        return "\n".join(results)
