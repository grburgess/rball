# -*- coding: utf-8 -*-

"""Top-level package for rball."""

from .response_database import ResponseDatabase
from .rballlike import RBallLike
from .utils import GridGenerator


__author__ = """J. Michael Burgess"""
__email__ = "jburgess@mpe.mpg.de"

from ._version import get_versions

__version__ = get_versions()["version"]
del get_versions
