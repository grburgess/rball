import numpy as np
from stripy.spherical_meshes import icosahedral_mesh


class GridGenerator:
    def __init__(self, refinement_levels: int = 2):

        """
        Generate a grid of points for response generation.
        An Icosahedral with facepoint is chosen because it is closest
        to equal area while still being very fast for triangulation

        :param refinement_levels:
        :type refinement_levels: int
        :returns:

        """
        self._mesh = icosahedral_mesh(
            refinement_levels=refinement_levels, include_face_points=True
        )

    @property
    def n_grid_points(self) -> int:

        return len(self._mesh.lons)

    @property
    def lons(self) -> np.ndarray:

        return self._mesh.lons

    @property
    def lats(self) -> np.ndarray:

        return self._mesh.lats

    @property
    def phi(self) -> np.ndarray:

        return self._mesh.lons

    @property
    def theta(self) -> np.ndarray:

        return self._mesh.lats

    @property
    def xyz(self) -> np.ndarray:

        """
        return the cartesian points on the grid

        :returns:

        """
        return self._mesh.points
