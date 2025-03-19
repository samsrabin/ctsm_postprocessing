"""
Module for handling lists of Crop Functional Types (CFTs) in the Community Terrestrial Systems Model
(CTSM).

This module defines the CftList class, which represents a list of CFTs, and provides methods for
accessing and managing CFTs.
"""
import os
import sys
import numpy as np

try:
    # Attempt relative import if running as part of a package
    from .cft import Cft
except ImportError:
    # Fallback to absolute import if running as a script
    sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    from crops.cft import Cft

class CftList:
    """
    Represents a list of Crop Functional Types (CFTs) in the Community Terrestrial Systems Model
    (CTSM).

    Attributes:
        cft_list (list): List of Cft objects.
    """

    def __init__(self, ds, n_pfts, cfts_to_include):
        """
        Initialize a CftList instance.

        Parameters:
            ds (xarray.Dataset): Dataset containing crop data.
            n_pfts (int): Number of PFTs.
            cfts_to_include (list): List of CFTs to include in the list.
        """
        # Get list of all possible CFTs
        self.cft_list = []
        for i, (key, value) in enumerate(ds.attrs.items()):
            if not key.startswith("cft_"):
                continue
            cft_name = key[4:]
            self.cft_list.append(Cft(cft_name, value))

        # Ensure that all CFTs in cfts_to_include are present
        cfts_in_file = [x.name for x in self.cft_list]
        missing_cfts = [x for x in cfts_to_include if x not in cfts_in_file]
        if missing_cfts:
            msg = (
                "The following are in cfts_to_include but not the dataset: "
                + ", ".join(missing_cfts)
            )
            raise KeyError(msg)

        # Figure out PFT indices
        max_cft_num = max(x.cft_num for x in self.cft_list)
        n_non_crop_pfts = n_pfts - max_cft_num + 1  # Incl. unvegetated
        for cft in self.cft_list:
            cft = cft.update_pft(n_non_crop_pfts)

        # Only include CFTs we care about
        if len(cfts_to_include) != len(np.unique(cfts_to_include)):
            raise ValueError("Duplicate CFT(s) in cfts_to_include")
        self.cft_list = [x for x in self.cft_list if x.name in cfts_to_include]

        # Figure out where the pft index is each CFT
        for cft in self.cft_list:
            cft.get_where(ds)
            if len(cft.where) == 0:
                print("Warning: No occurrences found of " + cft.name)

    def __getitem__(self, index):
        return self.cft_list[index]

    def __str__(self):
        results = []
        for cft in self.cft_list:
            results.append(str(cft))
        return "\n".join(results)
