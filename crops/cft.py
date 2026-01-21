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
        cft_num (int): Taken from `cft_temperate_corn` (or whatever) global attribute of CLM history
                       output. c3_crop=1, c4_crop=2, temperate_corn=3, etc.
        pft_num (int): PFT number, updated after reading all CFTs. temperate_corn=17, etc.
    """

    def __init__(self, name, cft_num):
        """
        Initialize a Cft instance.

        Parameters:
            name (str): Name of the CFT.
            cft_num (int): Taken from `cft_temperate_corn` (or whatever) global attribute of CLM
                           history output. c3_crop=1, c4_crop=2, temperate_corn=3, etc.
        """
        self.name = name

        # 1-indexed in the FORTRAN style
        self.cft_num = cft_num
        self.pft_num = None  # Need to know max cft_num

    def __eq__(self, other):
        # Check that they're both Cfts
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

    def __str__(self):
        return "\n".join(
            [
                self.name + ":",
                f"   cft_num: {self.cft_num}",
                f"   pft_num: {self.pft_num}",
            ]
        )

    def update_pft(self, n_non_crop_pfts):
        """
        You don't know n_non_crop_pfts until after reading in all CFTs, so
        this function gets called once that's done in CftList.__init__().
        """
        self.pft_num = n_non_crop_pfts + self.cft_num - 1

    def get_where(self, ds):
        """
        Get the indices on the pft dimension corresponding to this CFT
        """
        if self.pft_num is None:
            raise RuntimeError("get_where() can't be run until after calling Crop.update_pft()")
        pfts1d_itype_veg = ds["pfts1d_itype_veg"]
        if "time" in pfts1d_itype_veg.dims:
            pfts1d_itype_veg = pfts1d_itype_veg.isel(time=0)
        return np.where(pfts1d_itype_veg.values == self.pft_num)[0].astype(int)
