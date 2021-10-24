from os import replace
from typing import Optional, Iterable, Tuple
import numpy as np
import numba as nb
import stripy
from stripy.spherical import lonlat2xyz

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import ipyvolume as ipv
import pythreejs

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
        ::
        :returns:

        """

        self._matrices: np.ndarray = list_of_matrices

        self._ebounds: np.ndarray = ebounds

        self._monte_carlo_energies: np.ndarray = monte_carlo_energies

        self._n_grid_points: int = self._matrices.shape[0]

        self._matrix_shape = self._matrices.shape[1:]

        self._occulted_matrix = np.zeros(self._matrix_shape)

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

        log.debug(f" setting the current matrix to the first data point")

        self._current_matrix: InstrumentResponse = InstrumentResponse(
            matrix=list_of_matrices[0],
            ebounds=self._ebounds,
            monte_carlo_energies=self._monte_carlo_energies,
        )

        self.interpolate_to_position(0.1, 0.1)

    @property
    def current_response(self) -> InstrumentResponse:

        return self._current_matrix

    @property
    def grid_points(self) -> np.ndarray:

        return self._triangulation.points

    @property
    def n_grid_points(self) -> int:
        """
        the number of grid points in
        the response database
        :returns:

        """
        return self._n_grid_points

    def _generate_triangulation(self) -> None:

        log.debug("generating the triangulation")

        self._triangulation = stripy.spherical.sTriangulation(
            lons=self._phi, lats=self._theta, permute=True, tree=True
        )

    def _transform_to_instrument_coordinates(
        self, ra: float, dec: float
    ) -> Tuple[float]:
        """
        This is a stub function that should take and RA/Dec pair (in degrees)
        and convert it into the coordinate system of the intruments which may
        or may not be in motion.

        The function should return a spherical theta, phi tuple in radian

        :param ra:
        :param dec:

        :returns: (theta, phi) in radian

        """

        return np.deg2rad([dec, ra])

    def interpolate_to_position(self, ra: float, dec: float) -> None:

        theta, phi = self._transform_to_instrument_coordinates(ra, dec)

        # obtain the surrounding matricies and their normalized
        # barycenters

        bbc, tri = self._triangulation.containing_simplex_and_bcc(theta, phi)

        log.debug(f"weights: {bbc[0]}, indices: {tri[0]}")

        matrix = _linear_interpolation(bbc[0], self._matrices[tri[0]])

        # update teh 3ML matrix

        self._current_matrix.replace_matrix(matrix)

        self._current_ra: float = ra
        self._current_dec: float = dec

    @property
    def current_sky_position(self) -> Tuple[float]:
        return self._current_ra, self._current_dec

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
        """

        :param selected_location: ra, dec tuple in degree

        """

        background_color = "#2E0E49"
        grid_color = "#FDFF8D"
        grid_color2 = "#FDFEAE"
        selected_color = "#FF6563"
        point_color = "#63FFA3"

        fig = ipv.figure(width=800, height=600)
        ipv.pylab.style.box_off()
        # ipv.pylab.style.axes_off()
        ipv.pylab.style.set_style_dark()
        ipv.pylab.style.background_color(background_color)

        points = self._triangulation.points
        segs = self._triangulation.identify_segments()

        scatter = ipv.pylab.scatter(
            points[:, 0],
            points[:, 1],
            points[:, 2],
            color=grid_color,
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
                color=grid_color2,
                alpha=0.5,
            )

        if selected_location is not None:

            theta, phi = self._transform_to_instrument_coordinates(
                *selected_location
            )

            this_point = lonlat2xyz(phi, theta)

            bbc, tri = self._triangulation.containing_simplex_and_bcc(
                phi, theta
            )

            ipv.scatter(
                points[tri, 0],
                points[tri, 1],
                points[tri, 2],
                color=selected_color,
                marker="sphere",
            )

            ipv.pylab.scatter(*this_point, color=point_color, marker="sphere")

        ipv.xyzlim(1.1)
        ipv.pylab.style.box_off()

        fig.camera.up = [0, 0, 1]
        control = pythreejs.OrbitControls(controlling=fig.camera)
        fig.controls = control
        control.autoRotate = True
        fig.render_continuous = True

        ipv.show()


#        return fig


# @nb.njit(fastmath=True)
def _linear_interpolation(
    weights: np.ndarray, super_matrix: np.ndarray
) -> np.ndarray:

    shape = super_matrix[0].shape

    out = np.zeros((shape[0], shape[1]))

    for i in range(3):

        out += weights[i] * super_matrix[i]

    return out
