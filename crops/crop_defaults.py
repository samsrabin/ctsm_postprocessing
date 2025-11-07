"""
Some useful crop default variables
"""

from types import MappingProxyType

# MappingProxyType makes this dict immutable
DEFAULT_VAR_DICT = MappingProxyType(
    {
        "hui_var": "HUI_PERHARV",
        "huifrac_var": "HUIFRAC_PERHARV",
        "gddharv_var": "GDDHARV_PERHARV",
        "gslen_var": "GSLEN_PERHARV",
    }
)

N_PFTS = 78

DEFAULT_CFTS_TO_INCLUDE = [
    "temperate_corn",
    "tropical_corn",
    "cotton",
    "rice",
    "temperate_soybean",
    "tropical_soybean",
    "sugarcane",
    "spring_wheat",
    "irrigated_temperate_corn",
    "irrigated_tropical_corn",
    "irrigated_cotton",
    "irrigated_rice",
    "irrigated_temperate_soybean",
    "irrigated_tropical_soybean",
    "irrigated_sugarcane",
    "irrigated_spring_wheat",
]

DEFAULT_CROPS_TO_INCLUDE = [
    "corn",
    "cotton",
    "rice",
    "soybean",
    "sugarcane",
    "wheat",
]
