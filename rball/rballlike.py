from typing import Optional

from threeML.plugins.DispersionSpectrumLike import DispersionSpectrumLike
from threeML.plugins.OGIPLike import OGIPLike
from astromodels import Model

from astromodels.functions.priors import Cosine_Prior, Uniform_prior

from .response_database import ResponseDatabase

from .utils.logging import setup_logger
from rball import response_database


log = setup_logger(__name__)


class RBallLike(DispersionSpectrumLike):
    def __init__(
        self,
        name: str,
        observation,
        response_database: ResponseDatabase,
        background=None,
        free_position: bool = True,
        **kwargs,
    ):

        self._free_position: bool = free_position

        # we replace the response of the observation

        self._response_database: ResponseDatabase = response_database

        observation._response = response_database.current_response

        super(RBallLike, self).__init__(name, observation, background, **kwargs)

    def set_model(self, likelihood_model: Model) -> None:
        """
        Set the model and free the location parameters


        :param likelihood_model:
        :return: None
        """

        # set the standard likelihood model

        super(RBallLike, self).set_model(likelihood_model)

        # now free the position
        # if it is needed

        if self._free_position:

            log.info(f"freeing the position of {self.name} and setting priors")

            for key in self._like_model.point_sources.keys():
                self._like_model.point_sources[key].position.ra.free = True
                self._like_model.point_sources[key].position.dec.free = True

                self._like_model.point_sources[
                    key
                ].position.ra.prior = Uniform_prior(
                    lower_bound=0.0, upper_bound=360
                )
                self._like_model.point_sources[
                    key
                ].position.dec.prior = Cosine_Prior(
                    lower_bound=-90.0, upper_bound=90
                )

                ra = self._like_model.point_sources[key].position.ra.value
                dec = self._like_model.point_sources[key].position.dec.value

        else:

            for key in self._like_model.point_sources.keys():

                self._like_model.point_sources[
                    key
                ].position.ra.prior = Uniform_prior(
                    lower_bound=0.0, upper_bound=360
                )
                self._like_model.point_sources[
                    key
                ].position.dec.prior = Cosine_Prior(
                    lower_bound=-90.0, upper_bound=90
                )

                ra = self._like_model.point_sources[key].position.ra.value
                dec = self._like_model.point_sources[key].position.dec.value

        self._response_database.interpolate_to_position(ra, dec)

    def get_model(self, precalc_fluxes=None):

        # Here we update the GBM drm parameters which creates and new DRM for that location
        # we should only be dealing with one source for GBM

        # update the location

        if self._free_position:

            # assumes that the is only one point source which is how it should be!
            ra, dec = self._like_model.get_point_source_position(0)

            self._response_database.interpolate_to_position(ra, dec)

        return super().get_model(precalc_fluxes)

    @classmethod
    def from_spectrumlike(
        cls,
        spectrum_like: DispersionSpectrumLike,
        response_database: ResponseDatabase,
        free_position: bool = True,
        **kwargs,
    ):
        """
        Generate a RBallLike from an existing SpectrumLike child


        :param spectrum_like: the existing spectrumlike
        :param response_database:  a response database
        :param free_position: if the position should be free
        :return:
        """

        try:

            observation = spectrum_like._observed_spectrum.to_binned_spectrum()

        except:

            observation = spectrum_like._observed_spectrum

        try:

            bkg = spectrum_like._background_spectrum.to_binned_spectrum()

        except:

            bkg = spectrum_like._background_spectrum

        return cls(
            spectrum_like.name,
            observation,
            response_database,
            bkg,
            free_position,
            **kwargs,
        )

    @classmethod
    def from_ogip(
        cls,
        name: str,
        observation: str,
        response_database: ResponseDatabase,
        background: Optional[str] = None,
        spectrum_number: Optional[int] = None,
        free_position: bool = True,
        **kwargs,
    ):
        """
        Create an RBallLike from OGIP files

        :param cls:
        :type cls:
        :param name:
        :type name: str
        :param observation:
        :type observation: str
        :param background:
        :type background: str
        :param response_database:
        :type response_database: ResponseDatabase
        :param spectrum_number:
        :type spectrum_number: Optional[int]
        :param free_position:
        :type free_position: bool
        :returns:

        """
        tmp = OGIPLike(
            name=name,
            observation=observation,
            background=background,
            response=response_database.current_response,
            spectrum_number=spectrum_number,
        )

        return cls.from_spectrumlike(
            tmp,
            response_database=response_database,
            free_position=free_position,
            **kwargs,
        )

    def get_simulated_dataset(self, new_name=None, **kwargs):
        return super().get_simulated_dataset(
            new_name=new_name,
            response_database=self._response_database,
            **kwargs,
        )
