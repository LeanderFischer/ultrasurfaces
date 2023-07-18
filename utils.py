import numpy as np
from toy_mc import generator
from toy_mc import histogram
from toy_mc.histogram import Histogram
from typing import List


# define set of default oscillation parameters
delta_msq_31 = 2.515e-3
sinsq_theta_23 = 0.565
# convert this to sin**2(2 \theta)
theta_23 = np.arcsin(np.sqrt(sinsq_theta_23))
sinsq_2theta_23 = np.sin(2 * theta_23)**2
default_pars = generator.OscPars(delta_msq_31, sinsq_2theta_23)


def generate_histogram_at_response(
    gen_nominal,
    df,
    response,
    bin_edges,
    use_systs=None,
    variable="reco_energy"
):
    """
    Generate a histogram at a given detector response.

    Parameters
    ----------
    gen_nominal : toy_mc.generator.Generator
        Nominal detector response MC generator corresponding
        to dataframe containing the events and gradients
    df : pandas.DataFrame
        DataFrame containing the events and gradients
    response : toy_mc.generator.Response
        Detector response to weight the events to
    bin_edges : np.ndarray
        Bin edges for the histogram
    use_systs : list of str, optional
        List of systematic names to use. If None, all systematics are used.
    variable : str, optional
        Variable to use for the histogram. Default is "reco_energy".
    """

    return generate_histogram_at_response_variable_osc(
        gen_nominal=gen_nominal,
        df=df,
        response=response,
        osc_pars=default_pars,
        bin_edges=bin_edges,
        use_systs=use_systs,
        variable=variable
    )


def chi_square(hist1, hist2):
    """Calculate the chi square between two Histogram objects.
    This takes errors into account.
    """
    # get values
    values1 = hist1.hist
    values2 = hist2.hist
    # get errors
    errors1 = hist1.hist_unc
    errors2 = hist2.hist_unc
    # calculate chi square
    chi_square = np.sum((values1 - values2)**2 / (errors1**2 + errors2**2))
    return chi_square


# TODO: use refactored
def chi_square_at_response(
    gen_nominal,
    df,
    response,
    bin_edges,
    use_systs=None,
    variable="reco_energy"
):
    """
    Calculate the chi square between the nominal
    and a given detector response."""
    return chi_square_at_response_and_osc_eventwise(
        gen_nominal=gen_nominal,
        df=df,
        response=response,
        osc_pars=default_pars,
        bin_edges=bin_edges,
        variable=variable,
    )


def get_binwise_gradients(
    nominal_dataset: generator.Generator,
    sys_datasets: List[generator.Generator],
    sys_variable: str,
    bin_variable: str,
    bin_edges: np.ndarray,
    include_intecpt: bool = True,
    degree: int = 1,
):
    """Calculate the bin-wise gradients for a given systematic variable.

    The gradients interpolate the ratio of the systematic set to the nominal set
    in each bin as a function of the systematic variable.

    Parameters
    ----------
    nominal_dataset : generator.Generator
        The nominal dataset.
    sys_datasets : List[generator.Generator]
        The systematic datasets.
    sys_variable : str
        The systematic variable to use for the interpolation. This has to be one of the attributes of the detector response.
    bin_variable : str
        The variable to bin in.
    bin_edges : np.ndarray
        The bin edges for bin_variable.
    include_intecpt : bool, optional
        Whether to include the intercept in the linear regression, by default True.
    degree : int, optional
        The degree of the polynomial to use for the regression, by default 1.

    Returns
    -------
    binwise_gradients : np.ndarray
        The bin-wise gradients and intercept values.
    """

    assert hasattr(
        nominal_dataset.detector_response, sys_variable
    ), (f"sys_variable '{sys_variable}' not in response attributes.")

    nominal_hist = nominal_dataset.get_histogram(
        bin_edges, variable=bin_variable
    )
    sys_hists = [
        sys_dataset.get_histogram(bin_edges, variable=bin_variable)
        for sys_dataset in sys_datasets
    ]
    all_hists = [nominal_hist] + sys_hists
    nominal_sys_value = getattr(nominal_dataset.detector_response, sys_variable)
    sys_values = [
        getattr(sys_dataset.detector_response, sys_variable) - nominal_sys_value
        for sys_dataset in sys_datasets
    ]
    sys_values = [0.0] + sys_values

    # iterate over bins and calculate gradients
    binwise_gradients = []
    for bin_idx in range(len(bin_edges) - 1):
        nominal_value = nominal_hist.hist[bin_idx]
        bin_values = np.asarray([hist.hist[bin_idx] for hist in all_hists])
        bin_uncertainties = np.asarray(
            [hist.hist_unc[bin_idx] for hist in all_hists]
        )
        # calculate gradients
        gradients = np.polyfit(
            sys_values,
            bin_values / nominal_value,
            degree,
            w=1 / bin_uncertainties
        )
        binwise_gradients.append(gradients)
    return np.asarray(binwise_gradients)


def get_histogram_at_response_gradient_method(
    nominal_dataset: generator.Generator,
    binwise_gradients: np.ndarray,
    sys_variable: str,
    bin_variable: str,
    bin_edges: np.ndarray,
    response: generator.Response,
) -> histogram.Histogram:
    """
    Generate a histogram at a given detector response
    using the bin-wise gradient method.

    Parameters
    ----------
    nominal_dataset : generator.Generator
        The nominal dataset.
    binwise_gradients : np.ndarray
        The bin-wise gradients.
    sys_variable : str
        The systematic variable to use for the interpolation.
        This has to be one of the attributes of the detector response.
    bin_variable : str
        The variable to bin in.
    bin_edges : np.ndarray
        The bin edges for bin_variable.
    response : generator.Response
        The response to generate the histogram at.

    Returns
    -------
    Histogram
        The histogram at the given response.
    """

    return get_histogram_at_response_gradient_method_variable_osc(
        nominal_dataset=nominal_dataset,
        binwise_gradients=binwise_gradients,
        sys_variable=sys_variable,
        bin_variable=bin_variable,
        bin_edges=bin_edges,
        response=response,
        osc_pars=default_pars,
    )


def generate_histogram_at_response_variable_osc(
    gen_nominal,
    df,
    response,
    osc_pars,
    bin_edges,
    use_systs=None,
    variable="reco_energy"
):
    """
    Generate a histogram at a given detector response and
    oscillation parameters.
    Uses eventwise gradients for estimation

    Parameters
    ----------
    gen_nominal : toy_mc.generator.Generator
        Nominal detector response MC generator corresponding
        to dataframe containing the events and gradients
    df : pandas.DataFrame
        DataFrame containing the events and gradients from the nominal MC set
    response : toy_mc.generator.Response
        Detector response to weight the events to
    osc_pars : toy_mc.generator.OscPars
        Oscillation parameters to weight the events to
    bin_edges : np.ndarray
        Bin edges for the histogram
    use_systs : list of str, optional
        List of systematic names to use. If None, all systematics are used.
    variable : str, optional
        Variable to use for the histogram. Default is "reco_energy".
    """
    if use_systs is None:
        grad_names = df.columns[df.columns.str.startswith("grad__")]
    else:
        grad_names = ["grad_{}".format(syst) for syst in use_systs]
    delta_p = np.ones(len(grad_names))
    nominal_response = gen_nominal.detector_response
    for i, grad_name in enumerate(grad_names):
        for param in grad_name.split("grad")[-1].split("__")[1:]:
            delta_p[i] *= getattr(response,
                                  param) - getattr(nominal_response, param)
    # default oscillation parameter weights
    weights = df["weights"] * np.exp(np.dot(df[grad_names], delta_p))
    # apply re-weighting to target osciallation parameters
    weights *= gen_nominal.get_oscillation_reweight_factor(osc_pars)

    hist = Histogram(bin_edges)
    hist.fill(df[variable], weights)
    return hist


def chi_square_at_response_and_osc_eventwise(
    gen_nominal, df, response, osc_pars, bin_edges, variable="reco_energy"
):
    """
    Calculate the chi square between the nominal and
    a given detector response from event-wise gradients.
    """

    # generate new independent dataset at response with high statistics using new Generator
    gen_target = generator.Generator(
        n_events=int(response.mu * 1e6),
        response=response,
        pars=osc_pars,
        rng_seed=50
    )
    # divide by 10 because we have generated 10 times more events
    hist_target = gen_target.get_histogram(bin_edges) / 10

    hist_eventwise_gradients = generate_histogram_at_response_variable_osc(
        gen_nominal,
        df,
        response,
        osc_pars,
        bin_edges,
        variable=variable,
    )
    return chi_square(hist_target, hist_eventwise_gradients)


def get_histogram_at_response_gradient_method_variable_osc(
    nominal_dataset: generator.Generator,
    binwise_gradients: np.ndarray,
    sys_variable: str,
    bin_variable: str,
    bin_edges: np.ndarray,
    response: generator.Response,
    osc_pars: generator.OscPars,
    conserve_base: bool = True,
) -> Histogram:
    """
    Generate a histogram at a given detector response and
    oscillation parameters.
    Using the bin-wise gradient method.

    Parameters
    ----------
    nominal_dataset : generator.Generator
        The nominal dataset.
    binwise_gradients : np.ndarray
        The bin-wise gradients.
    sys_variable : str
        The systematic variable to use for the interpolation. This has to be one of the attributes of the detector response.
    bin_variable : str
        The variable to bin in.
    bin_edges : np.ndarray
        The bin edges for bin_variable.
    response : generator.Response
        The response to generate the histogram at.
    osc_pars : generator.OscPars
        The osciallation parameters to calculate histogram for
    conserve_base : bool
        Whether to conserve the value at zero perturbation, by default True

    Returns
    -------
    Histogram
        The histogram at the given response.
    """

    delta_parameter = getattr(response, sys_variable) - getattr(
        nominal_dataset.detector_response, sys_variable
    )

    default_pars_temp = nominal_dataset.oscillation_pars
    nominal_dataset.reweight_oscillation(osc_pars)

    nominal_hist = nominal_dataset.get_histogram(
        bin_edges, variable=bin_variable
    )

    nominal_hist_values = nominal_hist.hist
    nominal_hist_uncertainties = nominal_hist.hist_unc

    if conserve_base:
        conserve_base_values = np.zeros_like(nominal_hist_values)
        for bin_idx in range(len(bin_edges) - 1):
            conserve_base_values[bin_idx] = np.polyval(
                binwise_gradients[bin_idx], 0.
            )
        conserve_base_values = np.asarray(conserve_base_values)
    else:
        conserve_base_values = np.ones_like(nominal_hist_values)

    bin_counts = np.zeros_like(nominal_hist_values)
    for bin_idx in range(len(bin_edges) - 1):
        bin_counts[bin_idx] = np.polyval(
            binwise_gradients[bin_idx], delta_parameter
        ) / conserve_base_values[bin_idx] * nominal_hist_values[bin_idx]

    # reset oscillation parameters
    nominal_dataset.reweight_oscillation(default_pars_temp)

    # return Histogram(bin_edges, bin_counts, np.zeros_like(bin_counts))
    return Histogram(bin_edges, bin_counts, nominal_hist_uncertainties)


def chi_square_at_response_and_osc_binwise(
    nominal_dataset: generator.Generator, binwise_gradients: np.ndarray,
    sys_variable: str, bin_variable: str, bin_edges: np.ndarray,
    response: generator.Response, osc_pars: generator.OscPars
):
    """
    Calculate the chi square between the nominal and a
    given detector response from event-wise gradients.
    """

    # generate new independent dataset at response with high statistics using new Generator
    gen_target = generator.Generator(
        n_events=int(response.mu * 1e6),
        response=response,
        pars=osc_pars,
        rng_seed=50
    )
    # divide by 10 because we have generated 10 times more events
    hist_target = gen_target.get_histogram(bin_edges) / 10

    hist_binwise_gradients = \
        get_histogram_at_response_gradient_method_variable_osc(
            nominal_dataset=nominal_dataset,
            binwise_gradients=binwise_gradients,
            sys_variable=sys_variable,
            bin_variable=bin_variable,
            bin_edges=bin_edges,
            response=response,
            osc_pars=osc_pars
        )

    return chi_square(hist_target, hist_binwise_gradients)
