import numpy as np
from copy import deepcopy

from sklearn.neighbors import KNeighborsClassifier
from sklearn import preprocessing
from sklearn.pipeline import Pipeline

from sklearn.linear_model import TheilSenRegressor
from sklearn.preprocessing import PolynomialFeatures


class ToyAnalysis_Hobo():
    """
    Histogram count prediction using simple Gradient estimation
    from discrete sets. Calculates bin-wise variations under the
    assumption of a specific set of flux parameters
    """
    def __init__(self, sets, variations, binning) -> None:
        """
        Parameters
        ----------
        sets : dict(str->generator.Generator)
            name of systematic sets and corresponding event generators
        variations : dict(str->dict())
            values of the systematic parameter variations, for every
            systematic set, following the scheme 
            dict(sys_param_name->(dict(variation_name)->sys_param_value))
        binning : np.array
            analysis histogram binning
        """

        self.__sets = sets

        self.__binning = binning

        # pick one of the baseline sets
        self.__baseline_set = [s for k, s in sets.items() if "baseline" in k][0]

        self.__baseline_reponse = self.__baseline_set.get_detector_response()
        self.__default_pars = self.__baseline_set.get_oscillation_pars()

        self.__make_gradients(variations)

    def __make_gradients(self, variations) -> None:

        gradients_histogram = {}
        # naive way of calculating finite difference: G = (hist_up - hist_low) / delta_par
        # TODO: include statistical uncertainties?
        for par, vals in variations.items():

            hist_up = self.__sets[par + '_up'].get_histogram(self.__binning
                                                            )['hist']
            hist_low = self.__sets[par + '_low'].get_histogram(self.__binning
                                                              )['hist']
            delta_par = vals['up'] - vals['low']

            gradients_histogram[par] = (hist_up - hist_low) / delta_par

        self.__gradients = gradients_histogram

    def get_histogram(self, response, osc_pars) -> dict:
        # reweight baseline set to oscillation parameters
        self.__baseline_set.reweight_oscillation(osc_pars)
        hist_base = self.__baseline_set.get_histogram(self.__binning)['hist']
        hist_base_unc = self.__baseline_set.get_histogram(self.__binning
                                                         )['hist_unc']

        # not strictly necessary: re-set generator of baseline set to
        # default oscillation pars
        self.__baseline_set.reweight_oscillation(self.__default_pars)

        # apply gradients:
        print(
            "Warning: Only gradients with respect to 'mu' are implemeted so far"
        )
        # TODO otherwise: loop over parameters...
        hist_final = hist_base + (response.mu - self.__baseline_reponse.mu
                                 ) * self.__gradients['mu']

        # return histogram in full format (see Generator), statistical
        # uncertainties from gradients not yet included...
        return {
            'hist': hist_final,
            'hist_unc': hist_base_unc,
            'bin_edges': self.__binning
        }


class Toy_Analysis_USF():

    def __init__(self, sets) -> None:

        self._sets = sets

        self.__set_variables_and_labels()

        print("Prediction llh ratio")
        self.__predict_llh_ratio()

        print("Fitting gradients")
        self.__fit_gradients()

    def __set_variables_and_labels(self):

        X_nom = np.vstack((
            self._sets['mu_baseline'].get_events()["true_energy"],
            self._sets['mu_baseline'].get_events()["reco_energy"]
        )).T

        weights_nom = self._sets['mu_baseline'].get_events()["weights"]

        X_sys = [
            np.vstack(
                (
                    sys_set.get_events()["true_energy"],
                    sys_set.get_events()["reco_energy"]
                )
            ).T for sys_set_key, sys_set in self._sets.items()
            if sys_set_key != 'mu_baseline'
        ]

        weights_sys = np.hstack(
            [
                sys_set.get_events()["weights"]
                for sys_set_key, sys_set in self._sets.items()
                if sys_set_key != 'mu_baseline'
            ]
        )

        X = np.vstack([X_nom] + X_sys)

        weights = np.hstack([weights_nom] + [weights_sys])

        y = np.hstack(
            [
                np.full(
                    len(self._sets['mu_baseline'].get_events()["true_energy"]), i
                ) for i, _ in enumerate(self._sets)
            ]
        )

        self._X = X
        self._X_nom = X_nom
        self._X_sys = X_sys
        self._y = y
        self._weights = weights
        self._weights_nom = weights_nom

    def __predict_llh_ratio(self):

        # a transformer to make all of the input data normal...
        # classifiers like this
        trafo = preprocessing.PowerTransformer(
            method='box-cox', standardize=True
        )
        X_transformed = trafo.fit_transform(self._X)
        self._X_transformed = X_transformed

        # do the classification
        knn = KNeighborsClassifier(n_neighbors=1000, weights="uniform")
        pipe = Pipeline(
            [("transform", trafo), ("classifier", knn)]
        )
        pipe.fit(self._X, self._y)
        self._pipe = pipe

        nominal_set_probabilities = pipe.predict_proba(self._X_nom)
        self._nominal_set_probabilities = nominal_set_probabilities

    def __fit_gradients(self):
        # using the theil sen regressor so that the result is robust w.r.t. outliers
        nom_events = self._sets['mu_baseline'].get_events()

        sigma_nom = self._sets['mu_baseline'].get_detector_response().sigma
        mu_nom = self._sets['mu_baseline'].get_detector_response().mu

        # the sigmas and mus for the systematic sets stay the same
        sigmas = np.array(
            [
                sys_set.get_detector_response().sigma
                for sys_set_key, sys_set in self._sets.items()
            ]
        ) - sigma_nom
        mus = np.array(
            [
                sys_set.get_detector_response().mu
                for sys_set_key, sys_set in self._sets.items()
            ]
        ) - mu_nom

        poly = PolynomialFeatures(
            2, include_bias=False, interaction_only=False
        )
        event_x = poly.fit_transform(np.vstack((mus, sigmas)).T)

        n_events = len(nom_events["weights"])
        per_event_gradients_nom = np.zeros((n_events, event_x.shape[1]))

        # looping over many events in Python is super slow...
        # but unfortunately there is no version of the TheilSenRegressor
        # that will make many different regressions at once
        for i, (true_e, reco_e) in (
            enumerate(
                zip(nom_events["true_energy"], nom_events["reco_energy"])
            )
        ):
            pred_probs = self._pipe.predict_proba(
                np.atleast_2d([true_e, reco_e])
            )[0]
            # divide out the nominal prob to get the re-scale ratio
            pred_probs = pred_probs / pred_probs[0]

            # because we fit without intercept, we must subtract the offset of 1!
            event_y = pred_probs - 1.
            reg = TheilSenRegressor(random_state=0,
                                    fit_intercept=False).fit(event_x, event_y)

            # extract the gradients from the estimator
            per_event_gradients_nom[i] = reg.coef_

        self._per_event_gradients_nom = per_event_gradients_nom

    def get_histogram(self, response, osc_pars) -> dict:

        # start prediction from baseline/nominal set
        baseline_set = self._sets['mu_baseline']
        # reweight baseline set to oscillation parameters
        baseline_set.reweight_oscillation(osc_pars)
        hist_base = baseline_set.get_histogram(self.__binning)['hist']
        hist_base_unc = baseline_set.get_histogram(self.__binning)['hist_unc']

        # not strictly necessary: re-set generator of baseline set to
        # default oscillation pars
        baseline_set.reweight_oscillation(self.__default_pars)

        # apply gradients:
        print(
            "Warning: Only gradients with respect to 'mu' are implemeted so far"
        )
        # TODO otherwise: loop over parameters...
        raise NotImplementedError("WIP")
        # baseline_response = baseline_set.get_detector_response()
        # # per-event calculation here
        # hist_final = hist_base + (response.mu - baseline_response.mu
        #                          ) * self.__gradients['mu']

        # # return histogram in full format (see Generator), statistical
        # # uncertainties from gradients not yet included...
        # return {
        #     'hist': hist_final,
        #     'hist_unc': hist_base_unc,
        #     'bin_edges': self.__binning
        # }

    def get_all_transformed_variables(self):
        return self._X_transformed

    def get_all_variables(self):
        return self._X

    def get_variables_nom(self):
        return self._X_nom

    def get_variables_sys(self):
        return self._X_sys

    def get_label(self):
        return self._y

    def get_weights(self):
        return self._weights

    def get_weights_nom(self):
        return self._weights_nom

    def get_nominal_set_probabilities(self):
        return self._nominal_set_probabilities
