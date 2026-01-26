"""
Module for timing and profiling code execution.

This module provides the Timing class for measuring and reporting execution times
of code blocks and loops.
"""

from __future__ import annotations

from time import time


class Timing:
    """
    Class for holding, calculating, and printing information about process timing.

    This class provides methods to time individual loops and overall execution time,
    with optional verbose output.

    Attributes
    ----------
    _start_all : float
        Timestamp when the Timing instance was created.
    _start : float | None
        Timestamp when the current loop timer was started, or None if not started.
    """

    def __init__(self) -> None:
        """
        Initialize a Timing instance.

        Sets the overall start time and initializes the loop timer to None.
        """
        self._start_all: float = time()
        self._start: float | None = None

    def start(self) -> None:
        """
        Start timer for one loop.

        Records the current time as the start of a timed operation.
        """
        self._start = time()

    def end(self, what: str, verbose: bool) -> None:
        """
        End timer for one loop and optionally print elapsed time.

        Parameters
        ----------
        what : str
            Description of what was being timed.
        verbose : bool
            If True, print the elapsed time to stdout.
        """
        end = time()
        if verbose:
            print(f"{what} took {end - self._start} s")

    def end_all(self, what: str, verbose: bool = True) -> None:
        """
        End timer across all loops and optionally print total elapsed time.

        Calculates time elapsed since the Timing instance was created.

        Parameters
        ----------
        what : str
            Description of what was being timed.
        verbose : bool, optional
            If True, print the total elapsed time to stdout. Default is True.
        """
        end_all = time()
        if verbose:
            print(f"{what} took {int(end_all - self._start_all)} s.")