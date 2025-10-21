from time import time

class Timing:
    """
    For holding, calculating, and printing info about process timing
    """

    def __init__(self):
        self._start_all = time()
        self._start = None

    def start(self):
        """
        Start timer for one loop
        """
        self._start = time()

    def end(self, what, verbose):
        """
        End timer for one loop
        """
        end = time()
        if verbose:
            print(f"{what} took {end - self._start} s")

    def end_all(self, what, verbose):
        """
        End timer across all loops
        """
        end_all = time()
        if verbose:
            print(f"{what} took {int(end_all - self._start_all)} s.")