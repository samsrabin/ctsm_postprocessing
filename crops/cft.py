"""
Module for handling Crop Functional Types (CFTs) in the Community Terrestrial Systems Model (CTSM).

This module defines the Cft class, which represents a single crop functional type, and provides
methods for updating PFT (Plant Functional Type) numbers and retrieving indices corresponding to
the CFT within a dataset.
"""

import numpy as np

class Cft:
    """
    Represents a single crop functional type (CFT) in the Community Terrestrial Systems Model (CTSM)

    Attributes:
        name (str): Name of the CFT.
        cft_num (int): 1-indexed CFT number in the FORTRAN style.
        pft_num (int): PFT number, updated after reading all CFTs.
        pft_ind (int): 0-indexed PFT index in the Python style.
        where (numpy.ndarray): Indices on the pft dimension corresponding to this CFT.
    """

    def __init__(self, name, cft_num):
        """
        Initialize a Cft instance.

        Parameters:
            name (str): Name of the CFT.
            cft_num (int): 1-indexed CFT number in the FORTRAN style.
        """
        self.name = name

        # 1-indexed in the FORTRAN style
        self.cft_num = cft_num
        self.pft_num = None  # Need to know max cft_num

        # 0-indexed in the Python style
        self.pft_ind = None  # Need to know pft_num
        self.where = None

    def __str__(self):
        return "\n".join(
            [
                self.name + ":",
                f"   cft_num: {self.cft_num}",
                f"   pft_num: {self.pft_num}",
                f"   pft_ind: {self.pft_ind}",
                f"   N cells: {len(self.where)}",
            ]
        )

    def update_pft(self, n_non_crop_pfts):
        """
        You don't know n_non_crop_pfts until after reading in all CFTs, so
        this function gets called once that's done in CftList.__init__().
        """
        self.pft_num = n_non_crop_pfts + self.cft_num - 1
        self.pft_ind = self.pft_num - 1

    def get_where(self, ds):
        """
        Get the indices on the pft dimension corresponding to this CFT
        """
        if self.pft_num is None:
            raise RuntimeError(
                "get_where() can't be run until after calling Crop.update_pft()"
            )
        pfts1d_itype_veg = ds["pfts1d_itype_veg"]
        if "time" in pfts1d_itype_veg.dims:
            pfts1d_itype_veg = pfts1d_itype_veg.isel(time=0)
        self.where = np.where(pfts1d_itype_veg.values == self.pft_num)[0].astype(int)
        return self.where
