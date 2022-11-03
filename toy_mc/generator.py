'''
Toy Monte Carlo event generation.
Creates events following a powerlaw, applies two-flavor oscillations and
simulates a detector response.
'''

from typing import NamedTuple
import numpy as np
from scipy.stats import norm

import logging

logger = logging.getLogger(__name__)
'''
Units? (GeV, km)
Type of generated events? (pd dataframe?, just a dict(var->np.ndarray as now?))
'''


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


class Generator():
    def __init__(
        self,
        n_events: int,
        index: float,
        response: Response,
        pars: OscPars = None
    ) -> None:
        """
        Toy MC generator, sampling events from a power law neutrino energy
        distribution, applying two-flavor oscillation probabilities and
        a simplified detector respose

        Parameters
        ----------
        n_events : int
            number of events generated
        index : float
            spectral index \gamma of power law, dN/dE \propto E^{-\gamma}
        response : Response
            simplified detector response: \mu and \sigma of a lognormal pdf
        pars : OscPars
            Parameters for 2-flavor osciallation, by default None so that
            default values are used
        """

        # define anchor of powerlaw
        self.__energy_anchor = 1
        # define sample boundaries
        self.__boundaries = {
            "energy": [1., 1000],
            "cos(zen)": [-1., -1.],  # fixed baseline
        }
        self.__generation(n_events, index)

        self.__apply_oscillation(pars)

        self.__apply_detector_response(response)

    def get_events(self) -> dict:
        return self.__events

    def get_histogram(self, bin_edges) -> dict:
        """
        Get histogram for a certain binning in reconstructed energy

        Parameters
        ----------
        bin_edges : 'np.ndarray[np.float64]'
            bin edges of the histogram

        Returns
        -------
        dict
            histogram (sum(weights) per bin)
            and statistical uncertainties (sqrt(sum(weights**2)) per bin)
        """

        idx = np.digitize(self.__events['reco_energy'], bin_edges)

        hist = np.bincount(idx, weights=self.__events['weights'])
        hist_unc = np.sqrt(
            np.bincount(idx, weights=np.power(self.__events['weights'], 2))
        )

        return {"hist": hist, "hist_unc": hist_unc}

    def reweight_oscillation(self, pars: OscPars):

        self.__apply_oscillation(pars)

    def reweight_detector_response(self, response: Response):

        self.__recalculate_response(response)

    def __generation(self, n_events: int, index: float) -> None:
        """
        Generates events, following a power law with index index in energy
        and a uniform distribution in cos(zenith)

        Parameters
        ----------
        n_events : int
            number of events generated
        index : float
            spectral index \gamma of power law, dN/dE \propto E^{-\gamma}
        """

        self.__n_events = n_events

        # TODO: seed the random generator properly for reproducibility?

        emin, emax = self.__boundaries['energy']
        # energies = sample_powerlaw(n_events, emin, emax, index)

        logenergies = np.random.normal(loc=1.3, scale=0.5, size=n_events)
        energies = np.power(10, logenergies)

        czmin, czmax = self.__boundaries['cos(zen)']
        cos_zens = np.random.uniform(low=czmin, high=czmax, size=n_events)

        print(f"Generating events between {emin} GeV and {emax} GeV" + \
            f" and cos(zenith) values between {czmin} and {czmax}")

        # start with equal weights and without oscillation weight
        self.__events = {
            'true_energy': energies,
            'true_cos(zen)': cos_zens,
            'weights': np.ones_like(cos_zens),
            'survival_prob': np.ones_like(cos_zens)
        }

    def __apply_oscillation(self, pars: OscPars = None) -> None:
        # get survival properties

        if pars is None:
            # use atmospheric neutrino osciallation quantities, Fig. 3.6 in Andrii
            # Terliuk's thesis
            delta_msq_31 = 2.515e-3
            sinsq_theta_23 = 0.565
            # convert this to sin**2(2 \theta)
            theta_23 = np.arcsin(np.sqrt(sinsq_theta_23))
            sinsq_2theta_23 = np.sin(2 * theta_23)**2
            self.__osc_pars = OscPars(delta_msq_31, sinsq_2theta_23)
        else:
            self.__osc_pars = pars

        lengths = get_length_travelled(
            np.arccos(self.__events['true_cos(zen)'])
        )

        prob = survival_probability(
            lengths, self.__events['true_energy'],
            self.__osc_pars.delta_mqs, self.__osc_pars.sinsq_2theta
        )

        # re-weight events:
        # if oscillation weights were applied previously divide this out
        # otherwise self.__events['survival_prob'] is just 1.
        reweight = prob / np.copy(self.__events['survival_prob'])

        # store current survival probabilities
        self.__events['survival_prob'] = prob

        # apply this factor to the event's weights
        self.__events['weights'] *= reweight

        # "cache" callable for crosschecks
        def survival_prob_used(lengths, energies):
            return survival_probability(
                lengths, energies, self.__osc_pars.delta_mqs,
                self.__osc_pars.sinsq_2theta
            )
        self.survival_prob = survival_prob_used

    def __apply_detector_response(self, response: Response) -> None:
        """
        Smear the true neutrino energy by a log-normal distribution,
        with parameters \mu and \sigma

        Parameters
        ----------
        response : Response
            simplified detector response: \mu and \sigma of a lognormal pdf
        """

        self.__detector_response = response

        smearing = np.random.normal(
            loc=response.mu, scale=response.sigma, size=self.__n_events
        )
        log10_ereco = np.log10(self.__events['true_energy']) * smearing

        self.__events['reco_energy'] = np.power(10, log10_ereco)

        # bookkeep smearing values
        self.__smearing = smearing

    def __recalculate_response(self, response: Response) -> None:
        # reweight every event to a new detector response:

        old_response = self.__detector_response

        reweight = norm.pdf(
            self.__smearing, loc=response.mu, scale=response.sigma
        ) / norm.pdf(
            self.__smearing,
            loc=old_response.mu,
            scale=old_response.sigma,
        )

        self.__events['weights'] *= reweight


def survival_probability(
    length: 'np.ndarray[np.float64]', energy: 'np.ndarray[np.float64]',
    delta_msq: float, sinsq_2theta: float
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

    assert sinsq_2theta <= 1., 'oscillation amplitudes greater than one?'
    assert sinsq_2theta >= 0., 'negative oscillation amplitudes?'

    assert delta_msq >= 0., 'negative square of the mass splitting?'

    phase = 1.267 * delta_msq * length / energy
    return 1 - sinsq_2theta * np.sin(phase)**2


def get_length_travelled(
    zenith: 'np.ndarray[np.float64]'
) -> 'np.ndarray[np.float64]':
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

    r_earth = 6371.  # in km
    d_det = 2.  # in km
    h_atm = 20.  # in km

    def travelled_distance(zenith):
        return (r_earth - d_det) * np.cos(np.pi - zenith) + \
            np.sqrt((r_earth + h_atm)**2 - (r_earth - d_det)**2 * \
                (1 - np.cos(np.pi - zenith)**2))

    return travelled_distance(zenith)  # in km


def sample_powerlaw(
    size: int, low: float, high: float, index: float
) -> 'np.ndarray[np.float64]':
    """
    Sample from a power law using the inverse transform method
    $$p(E) \propto E^{-\gamma)}$$

    Parameters
    ----------
    size : int
        size of the random sample
    low : float
        lower bound of the distribution
    high : float
        upper bound of the distribution
    index : float
        spectral index $\gamma$

    Returns
    -------
    np.ndarray[np.float64]
        sample of neutrino energies
    """

    uni = np.random.uniform(size=size)

    def inverse_cdf(y):
        # integration constant
        norm = high**(1 - index) - low**(1 - index)

        return np.power((y * norm + low**(1 - index)), 1 / (1 - index))

    return inverse_cdf(uni)
