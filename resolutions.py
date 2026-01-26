"""
Module for identifying CTSM resolutions
"""

from __future__ import annotations

import numpy as np
import xarray as xr


class Resolution:
    """
    Class for defining a CTSM resolution.

    Attributes
    ----------
    name : str
        Name of the resolution (e.g., 'f09', 'f19', '4x5').
    lon_min : float
        Minimum longitude value.
    lon_max : float
        Maximum longitude value.
    lon_n : int
        Number of longitude grid points.
    lat_min : float
        Minimum latitude value.
    lat_max : float
        Maximum latitude value.
    lat_n : int
        Number of latitude grid points.
    """

    # pylint: disable=too-few-public-methods

    def __init__(
        self,
        name: str,
        *,
        lon_min: float,
        lon_max: float,
        lon_n: int,
        lat_min: float,
        lat_max: float,
        lat_n: int,
    ) -> None:
        """
        Initialize an instance of the Resolution class.

        Parameters
        ----------
        name : str
            Name of the resolution (e.g., 'f09', 'f19', '4x5').
        lon_min : float
            Minimum longitude value.
        lon_max : float
            Maximum longitude value.
        lon_n : int
            Number of longitude grid points.
        lat_min : float
            Minimum latitude value.
        lat_max : float
            Maximum latitude value.
        lat_n : int
            Number of latitude grid points.
        """
        self.name = name
        self.lon_min = lon_min
        self.lon_max = lon_max
        self.lon_n = lon_n
        self.lat_min = lat_min
        self.lat_max = lat_max
        self.lat_n = lat_n

    def is_ds_this_res(self, ds: xr.Dataset) -> bool:  # pylint: disable=invalid-name
        """
        Check whether an xarray Dataset matches the conditions defined for this Resolution.

        Parameters
        ----------
        ds : xarray.Dataset
            Dataset to check, must contain 'lon' and 'lat' coordinates.

        Returns
        -------
        bool
            True if the dataset matches this resolution's specifications, False otherwise.
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


def identify_resolution(ds: xr.Dataset) -> Resolution:  # pylint: disable=invalid-name
    """
    Identify the resolution of an xarray Dataset.

    Compares the dataset's longitude and latitude dimensions against known CTSM resolutions
    defined in RESOLUTION_LIST.

    Parameters
    ----------
    ds : xarray.Dataset
        Dataset to identify, must contain 'lon' and 'lat' coordinates.

    Returns
    -------
    Resolution
        The Resolution object matching the dataset's grid specifications.

    Raises
    ------
    KeyError
        If the dataset's resolution doesn't match any known resolution in RESOLUTION_LIST.
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
