import pytest

import numpy as np

from threeML import *

from rball.utils.package_data import get_path_of_data_file

from rball import ResponseDatabase, RBallLike, response_database, GridGenerator


def test_construction(rsp_database: ResponseDatabase):

    assert rsp_database.n_grid_points == 98


def test_plotting(rsp_database: ResponseDatabase):

    rsp_database.plot_verticies()

    rsp_database.plot_verticies_ipv(selected_location=[10.0, 10.0])


def test_interpolation(rsp_database: ResponseDatabase):

    rsp_database.interpolate_to_position(10.0, 10.0)


def test_localization(rsp_database: ResponseDatabase):

    demo_plugin = RBallLike.from_ogip(
        "demo",
        observation=get_path_of_data_file("demo.pha"),
        spectrum_number=1,
        response_database=rsp_database,
    )

    source_function = Powerlaw(K=1, index=-2, piv=100)

    source_function.K.prior = Log_uniform_prior(
        lower_bound=1e-1, upper_bound=1e1
    )
    source_function.index.prior = Uniform_prior(lower_bound=-4, upper_bound=0)

    ps = PointSource("ps", 150.0, 1.0, spectral_shape=source_function)

    model = Model(ps)

    ba = BayesianAnalysis(model, DataList(demo_plugin))

    ba.set_sampler("emcee")
    ba.sampler.setup(n_walkers=50, n_iterations=1000.0, n_burnin=1000)

    ba.sample()


def test_grid_generator():

    gg = GridGenerator(refinement_levels=2)

    assert np.alltrue(gg.theta == gg.lons)

    assert np.alltrue(gg.phi == gg.lats)

    assert len(gg.xyz) == gg.n_grid_points
