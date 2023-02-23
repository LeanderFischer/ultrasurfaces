import typing
import numpy as np
from tqdm import tqdm
from scipy.optimize import minimize
from numba import njit


def make_delta_p_from_grad_names(gradient_names, sys_sets, nominal_set):
    """
    Make the delta_p matrix of polynomial features for a list of systematic sets.

    Parameters
    ----------
    gradient_names : list of str
        Names of the gradients to be produced. The names should be of the form
        `grad__{param_0}` for first order gradients and `grad__{param_0}__{param_1}`
        for second order gradients. The polynomial feature to be calculated is the
        offset from the nominal point for each parameter named in the string, multiplied.
        The output matrix is guaranteed to contain features in the named order.
    sys_sets : list of Generator
        List of toy_mc.Generator objects defining the systematic sets.
    nominal_set : Generator
        A toy_mc.Generator object defining the nominal set.
    """
    delta_p = np.ones((len(gradient_names), len(sys_sets)))
    nominal_params = nominal_set.detector_response
    for i, gradient_name in enumerate(gradient_names):
        for j, sys_set in enumerate(sys_sets):
            for param in gradient_name.split("grad")[-1].split("__")[1:]:
                # The appropriate entry for the intercept term is always just a one.
                if param == "intercept":
                    continue
                delta_p[i, j] *= getattr(sys_set.detector_response, param) - getattr(
                    nominal_params, param
                )
    return delta_p


def make_gradient_names(
    include_systematics: typing.List[str],
    poly_features: int,
    include_interactions: bool = False,
) -> typing.List[str]:
    """
    Generate gradient names for a given set of systematics and polynomial features.

    Parameters
    ----------
    include_systematics : list of str
        A list of strings containing the names of the systematics.
    poly_features : int
        The order of polynomial features. Must be a non-negative integer.
    include_interactions : bool, optional
        Whether to include interaction terms in the gradient names. Defaults to False.

    Returns
    -------
    list of str
        A list of strings containing the gradient names.

    Raises
    ------
    AssertionError
        If include_interactions is True and poly_features is less than 2.
    """
    gradient_names = list()

    if int(poly_features) == 1:
        for syst in include_systematics:
            grad_name = f"grad__{syst}"
            gradient_names.append(grad_name)
    elif int(poly_features) == 2:
        for syst in include_systematics:
            grad_name = f"grad__{syst}"
            gradient_names.append(grad_name)
        for count_0, syst_0 in enumerate(include_systematics):
            for count_1, syst_1 in enumerate(include_systematics):
                if count_1 == count_0:
                    gradient_names.append(f"grad__{syst_0}__{syst_1}")
                elif include_interactions and (count_1 >= count_0):
                    grad_name = f"grad__{syst_0}__{syst_1}"
                    gradient_names.append(grad_name)
    elif int(poly_features) > 2:
        assert (
            not include_interactions
        ), "interaction terms not supported for orders > 2"
        for syst in include_systematics:
            for i in range(poly_features):
                grad_name = f"grad" + i * f"__{syst}" + f"__{syst}"
                gradient_names.append(grad_name)

    return gradient_names


@njit
def softmax(x, w):
    """
    Calculate softmax for n_k gradients and n_i systematic sets (including nominal).

    Parameters
    ----------
    x : array of shape (n_k,)
        Gradients w.r.t systematic k.
    w : array of shape (n_k, n_i)
        Systematic variations w.r.t systematic k for each set i.

    Returns
    -------
    softmax : array of shape (n_i,)
        Softmax output.
    """
    num = np.exp(np.dot(x, w))
    denom = np.sum(num)
    return num / denom


@njit
def softmax_jac(p):
    """
    Get the Jacobian of the softmax activations given by the parameter p.
    """
    return np.outer(-p, p) + np.diag(p)


@njit
def nllh(x, w, p_obs):
    """
    Calculate negative log-likelihood of multinomial distributions, without constant offsets.

    Parameters
    ----------
    x : array of shape (n_k,)
        Gradients w.r.t systematic k.
    w : array of shape (n_k, n_i)
        Systematic variations w.r.t systematic k for each set i.
    p_obs : array of shape (n_i,)
        Observed probabilties for the event to be in set i.

    Returns
    -------
    nllh : float
        Negative log-likelihood without offset.
    """
    p_exp = softmax(x, w)
    return -np.dot(p_obs, np.log(p_exp))


@njit
def ngrad(x, w, p_obs):
    """
    Calculate gradient of negative log-likelihood of multinomial distributions.

    Parameters
    ----------
    x : array of shape (n_k,)
        Gradients w.r.t systematic k.
    w : array of shape (n_k, n_i)
        Systematic variations w.r.t systematic k for each set i.
    p_obs : array of shape (n_i,)
        Observed probabilties for the event to be in set i.

    Returns
    -------
    grad : array of shape (nk, )
        Gradient of negative log-likelihood.
    """

    p_exp = softmax(x, w)
    j = softmax_jac(p_exp)
    grad = -np.dot(p_obs / p_exp, np.dot(j, w.T))
    return grad


def fit_gradients(
    dataframe,
    prob_columns,
    delta_p,
    grad_names,
    minmizer_method="l-bfgs-b",
    disable_progress=False,
):
    """
    Fit gradients for a given dataframe.

    Parameters
    ----------
    dataframe : pandas DataFrame
        Dataframe of the nominal MC set.
    prob_columns : list of shape (n_i,)
        Keys of the probabilities to be in systematic set i.
    delta_p : array of shape (n_k, n_i)
        Systematic variations w.r.t systematic k for each set i.
    grad_names : list of shape (n_k,)
        Names of the gradients to be stored for n_k systamtics at polynomial order n_poly.
    minmizer_method : string
        Minimizer method to use in scipy.optimize.minimize.

    Returns
    -------
    df : pandas dataframe
        Dataframe with gradients written to it.
    """

    n_grads = delta_p.shape[0]
    assert n_grads == len(grad_names)
    x0 = np.zeros(n_grads)

    indices = list()
    grads_opt = list()

    total = len(dataframe)
    for row in tqdm(dataframe.itertuples(), total=total, disable=disable_progress):
        prob_obs = list()
        for key in keys:
            prob_obs.append(row._asdict()[key])

        prob_obs = np.array(p)

        res = minimize(
            nllh,
            x0,
            jac=nllh_grad,
            args=(delta_p, prob_obs),
            method=minmizer_method,
        )
        x_opt = res.x

        indices.append(row.Index)
        grads_opt.append(np.squeeze(x_opt))

    grads_opt = np.array(grads_opt)

    if n_grads == 1:
        dataframe.loc[indices, grad_names[0]] = grads_opt
    else:
        for i in range(n_grads):
            dataframe.loc[indices, grad_names[i]] = grads_opt[:, i]

    return dataframe
