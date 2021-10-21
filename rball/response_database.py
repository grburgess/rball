from typing import Optional, Iterable
import numpy as np
import numba as nb
import stripy
from stripy.spherical import lonlat2xyz

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import ipyvolume as ipv

from threeML.utils.OGIP.response import InstrumentResponse

import h5py


from .utils.logging import setup_logger


log = setup_logger(__name__)


class ResponseDatabase:
    def __init__(
        self,
        list_of_matrices: np.ndarray,
        theta: np.ndarray,
        phi: np.ndarray,
        ebounds: np.ndarray,
        monte_carlo_energies: np.ndarray,
    ):
        """

        :param list_of_matrices:
        :type list_of_matrices: np.ndarray
        :param theta:
        :type theta: np.ndarray
        :param phi:
        :type phi: np.ndarray
        :param ebounds:
        :type ebounds: np.ndarray
        :param monte_carlo_energies:
        :type monte_carlo_energies: np.ndarray
        :returns:

        """

        self._matrices: np.ndarray = list_of_matrices

        self._ebounds: np.ndarray = ebounds

        self._monte_carlo_energies: np.ndarray = monte_carlo_energies

        self._n_grid_points: int = self._matrices.shape[0]

        if theta.shape[0] != self._n_grid_points:

            log.error(
                f"theta points ({theta.shape[0]}) not equal to number of matrices ({self._n_grid_points})"
            )

            raise AssertionError()

        self._theta: np.ndarray = theta

        if phi.shape[0] != self._n_grid_points:

            log.error(
                f"phi points ({phi.shape[0]}) not equal to number of matrices ({self._n_grid_points})"
            )

            raise AssertionError()

        self._phi: np.ndarray = phi

        # create the triangulation

        self._generate_triangulation()

        # now intitialize the current matrix

        self._current_matrix: InstrumentResponse = InstrumentResponse(
            matrix=list_of_matrices[0],
            ebounds=self._ebounds,
            monte_carlo_energies=self._monte_carlo_energies,
        )

    @classmethod
    def from_hdf5(cls, file_name: str, use_high_gain: bool = False):

        with h5py.File(file_name, "r") as f:

            # this part is a little polar specific at the moment
            # will remove and generalize

            if use_high_gain:

                ext = "hg"

            else:

                ext = "lg"

            list_of_matrices = f[f"matrix_{ext}"][()]

            theta = f["theta"][()]

            phi = f["phi"][()]

            ebounds = f["ebounds"][()]

            mc_energies = f["mc_energies"][()]

            return cls(
                list_of_matrices=list_of_matrices,
                theta=theta,
                phi=phi,
                ebounds=ebounds,
                monte_carlo_energies=mc_energies,
            )

    @property
    def grid_points(self) -> np.ndarray:

        return self._triangulation.points

    def _generate_triangulation(self) -> None:

        self._triangulation = stripy.spherical.sTriangulation(
            lons=self._phi, lats=self._theta, permute=True, tree=True
        )

    def interpolate_to_position(self, theta: float, phi: float) -> None:

        # obtain the surrounding matricies and their normalized
        # barycenters

        bbc, tri = self._triangulation.containing_simplex_and_bcc(theta, phi)

        matrix = _linear_interpolation(bbc, self._matrices[tri])

        # update teh 3ML matrix

        self._current_matrix.replace_matrix(matrix)

    def plot_verticies(self) -> plt.Figure:

        fig, ax = plt.subplots(subplot_kw=dict(projection="3d"))

        points = self._triangulation.points
        segs = self._triangulation.identify_segments()

        ax.scatter(
            points[:, 0], points[:, 1], points[:, 2], color="k", alpha=0.1
        )

        # plot the verticies

        for s1, s2 in segs:

            ax.plot(
                [points[s1, 0], points[s2, 0]],
                [points[s1, 1], points[s2, 1]],
                [points[s1, 2], points[s2, 2]],
                color="grey",
                alpha=0.5,
            )

        xyzlim = np.array([ax.get_xlim3d(), ax.get_ylim3d(), ax.get_zlim3d()]).T
        XYZlim = [min(xyzlim[0]), max(xyzlim[1])]
        ax.set_xlim3d(XYZlim)
        ax.set_ylim3d(XYZlim)
        ax.set_zlim3d(XYZlim)
        ax.set_box_aspect(
            [
                ub - lb
                for lb, ub in (getattr(ax, f"get_{a}lim")() for a in "xyz")
            ]
        )

        return fig

    def plot_verticies_ipv(
        self, selected_location: Optional[Iterable[float]] = None
    ):
        fig = ipv.figure()

        points = self._triangulation.points
        segs = self._triangulation.identify_segments()

        scatter = ipv.pylab.scatter(
            points[:, 0],
            points[:, 1],
            points[:, 2],
            color="k",
            alpha=1,
            size=1,
            color_selected="red",
            marker="sphere",
        )

        # scatter.selected = tri[0]

        for s1, s2 in segs:
            ipv.plot(
                [points[s1, 0], points[s2, 0]],
                [points[s1, 1], points[s2, 1]],
                [points[s1, 2], points[s2, 2]],
                color="grey",
                alpha=0.5,
            )

        if selected_location is not None:

            theta, phi = selected_location

            this_point = lonlat2xyz(theta, phi)

            bbc, tri = self._triangulation.containing_simplex_and_bcc(
                theta, phi
            )

            ipv.scatter(
                points[tri, 0],
                points[tri, 1],
                points[tri, 2],
                color="red",
                marker="sphere",
            )

            ipv.pylab.scatter(*this_point, color="limegreen", marker="sphere")

        ipv.xyzlim(1.1)
        ipv.show()


#        return fig


@nb.njit(fastmath=True)
def _linear_interpolation(
    weights: np.ndarray, super_matrix: np.ndarray
) -> np.ndarray:

    return np.dot(weights, super_matrix)
