"""
Toy Monte Carlo event generation.
Creates events following a powerlaw, applies two-flavor oscillations and
simulates a detector response.
"""

from typing import NamedTuple
import numpy as np
import pandas as pd
from scipy.stats import norm
from .histogram import Histogram

import logging

logger = logging.getLogger(__name__)
"""
Units? (GeV, km)
Type of generated events? (pd dataframe?, just a dict(var->np.ndarray as now?))
"""


class Response(NamedTuple):
    """
    simplified detector response: \mu and \sigma of a lognormal pdf
    """

    mu: float
    sigma: float


class OscPars(NamedTuple):
    """
    parameters for 2 flavor oscillation
    """

    delta_mqs: float
    sinsq_2theta: float


class Generator:
    def __init__(
        self,
        n_events: int,
        response: Response,
        pars: OscPars = None,
        rng_seed: int = None,
        name: str = None,
        verbose: bool = False,
    ) -> None:
        """
        Toy MC generator, sampling events from a power law neutrino energy
        distribution, applying two-flavor oscillation probabilities and
        a simplified detector respose

        Parameters
        ----------
        n_events : int
            number of events generated
        response : Response
            simplified detector response: \mu and \sigma of a lognormal pdf
        pars : OscPars
            Parameters for 2-flavor osciallation, by default None so that
            default values are used
        rng_seed : int
            seed for the RNG, can be
            {None, int, array_like[ints], SeedSequence, BitGenerator, Generator}
        name : str
            Name of this generator. This will be used to name the unique probability 
            columns that are added to DataFrames with events. Be sure to give one 
            unique name to each systematic set that is used.
        verbose : bool
            If True, print out information about the generated events
        """

        self.verbose = verbose
        self.__rng = np.random.default_rng(rng_seed)

        # define sample boundaries
        self.__boundaries = {
            "energy": [1.0, 1000],  # not used right now, sample from gaussian
            "cos(zen)": [-1.0, -1.0],  # fixed baseline
        }
        self.__generation(n_events)

        self.__apply_oscillation(pars)

        self.__apply_detector_response(response)
        
        self.name = name

    @property
    def detector_response(self) -> Response:
        # response that was used for generating events
        return self.__detector_response_generation

    @property
    def oscillation_pars(self) -> OscPars:
        # current oscillation parameters used in weights
        return self.__osc_pars

    @property
    def events(self) -> dict:
        return pd.DataFrame(self.__events)

    def get_histogram(self, bin_edges, variable="reco_energy") -> dict:
        """
        Get histogram for a certain binning in reconstructed energy

        Parameters
        ----------
        bin_edges : 'np.ndarray[np.float64]'
            bin edges of the histogram
        variable : str, optional
            variable to histogram, by default "reco_energy"
        Returns
        -------
        dict
            histogram (sum(weights) per bin),
            statistical uncertainties (sqrt(sum(weights**2)) per bin)
            and bin edges
        """

        hist = Histogram(bin_edges)
        hist.fill(self.__events[variable], weights=self.__events["weights"])
        return hist

    def reweight_oscillation(self, pars: OscPars):

        self.__apply_oscillation(pars)
        self.__events["weights"] = (
            self.__response_reweight * self.__events["weights_pre_detector"]
        )

    def reweight_detector_response(self, response: Response):

        self.__recalculate_response(response)
        self.__events["weights"] = (
            self.__response_reweight * self.__events["weights_pre_detector"]
        )

    def __generation(self, n_events: int) -> None:
        """
        Generates events, following a gaussian in log10(energy)
        and a uniform distribution in cos(zenith)

        Parameters
        ----------
        n_events : int
            number of events generated
        """

        self.__n_events = n_events

        mean_loge = 1.3
        width_loge = 0.5

        logenergies = self.__rng.normal(loc=mean_loge, scale=width_loge, size=n_events)
        energies = np.power(10, logenergies)

        czmin, czmax = self.__boundaries["cos(zen)"]
        cos_zens = self.__rng.uniform(low=czmin, high=czmax, size=n_events)

        if self.verbose:
            print(
                f"Generating events with log10(E / GeV) from a Gaussian with "
                f"mean {mean_loge} and wdith {width_loge} "
                f"and cos(zenith) values uniformly sampled between {czmin} and {czmax}."
            )

        # start with equal weights and without oscillation weight
        self.__events = {
            "true_energy": energies,
            "true_cos(zen)": cos_zens,
            "weights_pre_detector": np.ones_like(cos_zens),
        }

    def __apply_oscillation(self, pars: OscPars = None) -> None:
        """
        Calculate oscillation weight for every event

        Parameters
        ----------
        pars : OscPars, optional
            Oscillation parameters, by default None
        """
        # get survival properties

        if pars is None:
            # use atmospheric neutrino osciallation quantities, Fig. 3.6 in Andrii
            # Terliuk's thesis
            delta_msq_31 = 2.515e-3
            sinsq_theta_23 = 0.565
            # convert this to sin**2(2 \theta)
            theta_23 = np.arcsin(np.sqrt(sinsq_theta_23))
            sinsq_2theta_23 = np.sin(2 * theta_23) ** 2
            self.__osc_pars = OscPars(delta_msq_31, sinsq_2theta_23)
        else:
            self.__osc_pars = pars

        lengths = get_length_travelled(np.arccos(self.__events["true_cos(zen)"]))

        prob = survival_probability(
            lengths,
            self.__events["true_energy"],
            self.__osc_pars.delta_mqs,
            self.__osc_pars.sinsq_2theta,
        )
        # these are the event's weights before the detector response
        self.__events["weights_pre_detector"] = prob

        # "cache" callable for crosschecks
        def survival_prob_used(lengths, energies):
            return survival_probability(
                lengths,
                energies,
                self.__osc_pars.delta_mqs,
                self.__osc_pars.sinsq_2theta,
            )

        self.survival_prob = survival_prob_used

    def get_oscillation_reweight_factor(
        self, pars: OscPars
    ) -> "np.ndarray[np.float64]":
        """
        Calculate factor to reweight to new set of oscillation parameters for every event

        Parameters
        ----------
        pars : OscPars, optional
            Oscillation parameters, by default None

        Returns
        -------
        'np.ndarray[np.float64]'
            reweighting factor for every event
        """

        lengths = get_length_travelled(np.arccos(self.__events["true_cos(zen)"]))

        prob = survival_probability(
            lengths, self.__events["true_energy"], pars.delta_mqs, pars.sinsq_2theta
        )

        reweight_factors = prob / self.__events["weights_pre_detector"]

        return reweight_factors

    def __apply_detector_response(self, response: Response) -> None:
        """
        Smear the true neutrino energy by a log-normal distribution,
        with parameters \mu and \sigma

        Parameters
        ----------
        response : Response
            simplified detector response: \mu and \sigma of a lognormal pdf
        """

        self.__detector_response_generation = response

        smearing = self.__rng.normal(
            loc=response.mu, scale=response.sigma, size=self.__n_events
        )
        log_ereco = np.log(self.__events["true_energy"]) * smearing

        self.__events["reco_energy"] = np.exp(log_ereco)

        # actual weights are not changed
        self.__events["weights"] = np.copy(self.__events["weights_pre_detector"])
        # only when response is recalculated these factors will change
        self.__response_reweight = np.ones_like(self.__events["weights"])

        # bookkeep smearing values
        self.__smearing = smearing

    def __recalculate_response(self, response: Response) -> None:
        """
        Reweight every event to a new detector response by the ratio of the
        analytically known detector response pdf

        Parameters
        ----------
        response : Response
            New detector response for which to calculate re-weighting factors
        """
        # reweight every event to a new detector response:

        old_response = self.__detector_response_generation

        reweight = norm.pdf(
            self.__smearing, loc=response.mu, scale=response.sigma
        ) / norm.pdf(
            self.__smearing,
            loc=old_response.mu,
            scale=old_response.sigma,
        )

        self.__response_reweight = reweight


def survival_probability(
    length: "np.ndarray[np.float64]",
    energy: "np.ndarray[np.float64]",
    delta_msq: float,
    sinsq_2theta: float,
):
    """
    Survival probability in two-flavor approximation
    See Andrii Terliuk's thesis eq. 3.10

    Parameters
    ----------
    length : np.ndarray[np.float64]
        length travelled by neutrino in km
    energy : np.ndarray[np.float64]
        energy of the neutrino in GeV
    delta_msq: float
        value of the mass splitting in (eV)**2
    sinsq_2theta: float:
        value of the mixing angle

    Returns
    -------
    np.ndarray[np.float64]
        survival probability per neutrino
    """

    assert sinsq_2theta <= 1.0, "oscillation amplitudes greater than one?"
    assert sinsq_2theta >= 0.0, "negative oscillation amplitudes?"

    assert delta_msq >= 0.0, "negative square of the mass splitting?"

    phase = 1.267 * delta_msq * length / energy
    return 1 - sinsq_2theta * np.sin(phase) ** 2


def get_length_travelled(zenith: "np.ndarray[np.float64]") -> "np.ndarray[np.float64]":
    """
    Calculate length travelled by a neutrinos generated at specific zenith
    angles, using a IceCube DeepCore-like detector geometry
    Taken from Andrii Terliuk's thesis, eq. 3.33

    Parameters
    ----------
    zenith : np.ndarray[np.float64]
        zenith angles of the simulated neutrino events

    Returns
    -------
    np.ndarray[np.float64]
        length of the neutrino travelled when arriving at the detector
    """

    r_earth = 6371.0  # in km
    d_det = 2.0  # in km
    h_atm = 20.0  # in km

    def travelled_distance(zenith):
        return (r_earth - d_det) * np.cos(np.pi - zenith) + np.sqrt(
            (r_earth + h_atm) ** 2
            - (r_earth - d_det) ** 2 * (1 - np.cos(np.pi - zenith) ** 2)
        )

    return travelled_distance(zenith)  # in km

