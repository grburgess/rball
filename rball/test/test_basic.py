import pytest

from rball import ResponseDatabase


def test_construction(rsp_database: ResponseDatabase):

    assert len(rsp_database.n_grid_points) == 96
