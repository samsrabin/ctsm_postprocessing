"""
Module for identifying CTSM resolutions
"""

import numpy as np

class Resolution:
    """
    Class for defining a CTSM resolution
    """
    # pylint: disable=too-few-public-methods

    def __init__(self, name, *, lon_min, lon_max, lon_n, lat_min, lat_max, lat_n):
        """
        Initialize an instance of the Resolution class
        """
        self.name = name
        self.lon_min = lon_min
        self.lon_max = lon_max
        self.lon_n = lon_n
        self.lat_min = lat_min
        self.lat_max = lat_max
        self.lat_n = lat_n

    def is_ds_this_res(self, ds):  # pylint: disable=invalid-name
        """
        Check whether an xarray Dataset matches the conditions defined for this Resolution
        """
        ds_lon = ds["lon"]
        ds_lat = ds["lat"]
        conditions = [
            np.isclose(ds_lon.min(), self.lon_min),
            np.isclose(ds_lon.max(), self.lon_max),
            ds.sizes["lon"] == self.lon_n,
            np.isclose(ds_lat.min(), self.lat_min),
            np.isclose(ds_lat.max(), self.lat_max),
            ds.sizes["lat"] == self.lat_n,
        ]
        return all(conditions)

RESOLUTION_LIST = [
    Resolution(
        "f09",
        lon_min = 0,
        lon_max = 358.75,
        lon_n = 288,
        lat_min = -90,
        lat_max = 90,
        lat_n = 192,
    )
]

def identify_resolution(ds):  # pylint: disable=invalid-name
    """
    Identify the resolution of an xarray Dataset
    """
    for res in RESOLUTION_LIST:
        if res.is_ds_this_res(ds):
            return res
    raise KeyError("Unidentified resolution")
