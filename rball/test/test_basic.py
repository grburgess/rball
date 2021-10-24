import pytest

from rball import ResponseDatabase


def test_construction(rsp_database: ResponseDatabase):

    assert rsp_database.n_grid_points == 98


def test_plotting(rsp_database: ResponseDatabase):

    rsp_database.plot_verticies()

    rsp_database.plot_verticies_ipv(selected_location=[10.0, 10.0])


def test_interpolation(rsp_database: ResponseDatabase):

    rsp_database.interpolate_to_position(10.0, 10.0)
