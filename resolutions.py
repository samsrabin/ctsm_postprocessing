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
        lon_min=0,
        lon_max=358.75,
        lon_n=288,
        lat_min=-90,
        lat_max=90,
        lat_n=192,
    ),
    Resolution(
        "f19",
        lon_min=0,
        lon_max=357.5,
        lon_n=144,
        lat_min=-90,
        lat_max=90,
        lat_n=96,
    ),
    Resolution(
        "4x5",
        lon_min=0,
        lon_max=355,
        lon_n=72,
        lat_min=-90,
        lat_max=90,
        lat_n=46,
    ),
    Resolution(
        "10x15",
        lon_min=0,
        lon_max=345,
        lon_n=24,
        lat_min=-90,
        lat_max=90,
        lat_n=19,
    ),
    Resolution(
        "f10_for_testing",
        lon_min=263.75,
        lon_max=266.25,
        lon_n=3,
        lat_min=38.16754,
        lat_max=40.05236,
        lat_n=3,
    ),
]


def identify_resolution(ds):  # pylint: disable=invalid-name
    """
    Identify the resolution of an xarray Dataset
    """
    for res in RESOLUTION_LIST:
        if res.is_ds_this_res(ds):
            return res

    # This is what you hit if you haven't identified a resolution
    msg = "Unidentified resolution:"
    msg += f", lon_min: {ds['lon'].min().values}"
    msg += f", lon_max: {ds['lon'].max().values}"
    msg += f", lon_n: {ds.sizes['lon']}"
    msg += f", lat_min: {ds['lat'].min().values}"
    msg += f", lat_max: {ds['lat'].max().values}"
    msg += f", lat_n: {ds.sizes['lat']}"
    raise KeyError(msg)
