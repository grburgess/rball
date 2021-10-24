import logging
import os
import shutil
from glob import glob
from pathlib import Path
import h5py

import pytest
from rball.utils.package_data import get_path_of_data_file
from rball.response_database import ResponseDatabase


@pytest.fixture(scope="session")
def rsp_database():

    file_name = get_path_of_data_file("demo_rsp_database.h5")

    with h5py.File(file_name, "r") as f:

        # this is for the temporary

        list_of_matrices = f["matrix"][()]

        theta = f["theta"][()]

        phi = f["phi"][()]

        ebounds = f["ebounds"][()]

        mc_energies = f["mc_energies"][()]

        yield ResponseDatabase(
            list_of_matrices=list_of_matrices,
            theta=theta,
            phi=phi,
            ebounds=ebounds,
            monte_carlo_energies=mc_energies,
        )
