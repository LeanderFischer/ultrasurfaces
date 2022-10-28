'''
Toy Monte Carlo event generation.
Creates events following a powerlaw, applies two-flavor oscillations and
simulates a detector response.
'''

from typing import NamedTuple
import numpy as np

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


class Generator():
    def __init__(self, n_events: int, index: float, response: Response) -> None:
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
        """

        # define anchor of powerlaw
        self.__energy_anchor = 1
        # define sample boundaries
        self.__boundaries = {
            "energy": [1., 1000],
            "cos(zen)": [-1., -1.],  # fixed baseline
        }
        self.__generation(n_events, index)

        self.__apply_oscillation()

        self.__apply_detector_response(response)

    def get_events(self) -> dict:
        return self.__events

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

        emin, emax = self.__boundaries['energy']
        energies = sample_powerlaw(n_events, emin, emax, index)

        czmin, czmax = self.__boundaries['cos(zen)']
        cos_zens = np.random.uniform(low=czmin, high=czmax, size=n_events)

        print(f"Generating events between {emin} GeV and {emax} GeV" + \
            f" and cos(zenith) values between {czmin} and {czmax}")

        weights = np.ones_like(cos_zens)

        self.__events = {
            'true_energy': energies,
            'true_cos(zen)': cos_zens,
            'weights': weights,
        }

    def __apply_oscillation(self) -> None:
        # get survival properties

        lengths = get_length_travelled(
            np.arccos(self.__events['true_cos(zen)'])
        )

        # use atmospheric neutrino osciallation quantities, Fig. 3.6 in Andrii
        # Terliuk's thesis
        delta_msq_31 = 2.515e-3
        sinsq_theta_23 = 0.565
        # convert this to sin**2(2 \theta)
        theta_23 = np.arcsin(np.sqrt(sinsq_theta_23))
        sinsq_2theta_23 = np.sin(2 * theta_23)**2
        prop = survival_probability(
            lengths, self.__events['true_energy'], delta_msq_31, sinsq_2theta_23
        )

        def survival_prob_used(lengths, energies):
            return survival_probability(
                lengths, energies, delta_msq_31, sinsq_2theta_23
            )

        # cache callable for crosschecks
        self.survival_prob = survival_prob_used

        self.__events['survival_prop'] = prop

        # apply this factor to the event's weights
        print(
            f"sum of weights generated: {np.sum(self.__events['weights']):.2f}"
        )
        self.__events['weights'] *= prop
        print(
            f"  sum of weights oscillated: {np.sum(self.__events['weights']):.2f}"
        )

    def __apply_detector_response(self, response: Response) -> None:
        """
        Smear the true neutrino energy by a log-normal distribution,
        with parameters \mu and \sigma

        Parameters
        ----------
        response : Response
            simplified detector response: \mu and \sigma of a lognormal pdf
        """

        smearing = np.random.lognormal(
            mean=response.mu, sigma=response.sigma, size=self.__n_events
        )

        self.__events['reco_energy'] = self.__events['true_energy'] * smearing


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
