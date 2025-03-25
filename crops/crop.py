"""
Module for handling crops in the Community Terrestrial Systems Model (CTSM).

This module defines the Crop class, which represents a crop consisting of multiple Crop Functional
Types (CFTs), and provides methods for retrieving indices corresponding to the crop within a
dataset.
"""

import numpy as np


class Crop:
    # pylint: disable=too-few-public-methods
    """
    Represents a crop consisting of multiple Crop Functional Types (CFTs).

    Attributes:
        name (str): Name of the crop.
        cft_list (list): List of CFTs included in this crop.
        cft_names (list): List of names of the CFTs.
        pft_nums (list): List of PFT numbers corresponding to the CFTs.
        pft_inds (list): List of PFT indices corresponding to the CFTs.
        where (numpy.ndarray): Indices on the pft dimension corresponding to this crop.
    """

    def __init__(self, name, cft_list, ds):
        """
        Initialize a Crop instance.

        Parameters:
            name (str): Name of the crop.
            cft_list (list): List of CFTs to include in this crop.
            ds (xarray.Dataset): Dataset containing crop data.
        """
        self.name = name

        # Get CFTs included in this crop
        self.cft_list = []
        for cft in cft_list:
            if self.name not in cft.name:
                continue
            self.cft_list.append(cft)

        # Get information for all CFTs in this crop
        self.cft_names = []
        self.pft_nums = []
        self.pft_inds = []
        self.where = np.array([], dtype=np.int64)
        for cft in self.cft_list:
            self.cft_names.append(cft.name)
            self.pft_nums.append(cft.pft_num)
            self.pft_inds.append(cft.pft_ind)
            self.where = np.append(self.where, cft.get_where(ds))
        self.where = np.sort(self.where)

    def __str__(self):
        return f"{self.name}: {', '.join(f'{x.name} ({x.pft_num})' for x in self.cft_list)}"
