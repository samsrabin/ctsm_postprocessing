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
