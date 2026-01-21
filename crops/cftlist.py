"""
Module for handling lists of Crop Functional Types (CFTs) in the Community Terrestrial Systems Model
(CTSM).

This module defines the CftList class, which represents a list of CFTs, and provides methods for
accessing and managing CFTs.
"""

from .cft import Cft
# TODO: Future-proof default: Determine from ds upon initialization.
from .crop_defaults import DEFAULT_CFTS_TO_INCLUDE


class CftList:
    """
    Represents a list of Crop Functional Types (CFTs) in the Community Terrestrial Systems Model
    (CTSM).

    Attributes:
        cft_list (list): List of Cft objects.
    """

    def __init__(self, ds, n_pfts, cfts_to_include=DEFAULT_CFTS_TO_INCLUDE):
        """
        Initialize a CftList instance.

        Parameters:
            ds (xarray.Dataset): Dataset containing crop data.
            n_pfts (int): Number of PFTs.
        """
        # Get list of all possible CFTs
        self.cft_list = []
        for i, (key, value) in enumerate(ds.attrs.items()):
            if not key.startswith("cft_"):
                continue
            cft_name = key[4:]
            self.cft_list.append(Cft(cft_name, value))

        # Ensure that all included CFTs are present
        cfts_in_file = [x.name for x in self.cft_list]
        missing_cfts = [x for x in cfts_to_include if x not in cfts_in_file]
        if missing_cfts:
            msg = "Trying to include these CFTs that aren't in the dataset: " + ", ".join(
                missing_cfts
            )
            raise KeyError(msg)

        # Figure out PFT indices
        max_cft_num = max(x.cft_num for x in self.cft_list)
        n_non_crop_pfts = n_pfts - max_cft_num + 1  # Incl. unvegetated
        for cft in self.cft_list:
            cft = cft.update_pft(n_non_crop_pfts)

        # Only include CFTs we care about
        if len(cfts_to_include) != len(set(cfts_to_include)):
            raise ValueError("Duplicate CFT(s) in cfts_to_include")
        self.cft_list = [x for x in self.cft_list if x.name in cfts_to_include]

    def __eq__(self, other):
        # Check that they're both CftLists
        if not isinstance(other, self.__class__):
            raise TypeError(f"== not supported between {self.__class__} and {type(other)}")
        result = self.cft_list == other.cft_list
        return result

    def __getitem__(self, index):
        return self.cft_list[index]

    def __str__(self):
        results = []
        for cft in self.cft_list:
            results.append(str(cft))
        return "\n".join(results)
