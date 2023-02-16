import numpy as np
from copy import deepcopy

from sklearn.neighbors import KNeighborsClassifier
from sklearn import preprocessing
from sklearn.pipeline import Pipeline

from sklearn.linear_model import TheilSenRegressor
from sklearn.preprocessing import PolynomialFeatures

class ToyAnalysis():
    """
    Histogram count prediction using simple Gradient estimation
    from discrete sets. Calculates bin-wise variations under the
    assumption of a specific set of flux parameters
    """
    def __init__(self, sets, variations, binning, baseline_key='mu_baseline') -> None:
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

        self.__baseline_key = baseline_key

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

            hist_up = self.__sets[par+'_up'].get_histogram(self.__binning)['hist']
            hist_low = self.__sets[par+'_low'].get_histogram(self.__binning)['hist']
            delta_par = vals['up'] - vals['low']

            gradients_histogram[par] = (hist_up - hist_low) / delta_par

        self.__gradients = gradients_histogram

    def get_histogram(self, response, osc_pars) -> dict:
        # reweight baseline set to oscillation parameters
        self.__baseline_set.reweight_oscillation(osc_pars)
        hist_base = self.__baseline_set.get_histogram(self.__binning)['hist']
        hist_base_unc = self.__baseline_set.get_histogram(self.__binning)['hist_unc']

        # not strictly necessary: re-set generator of baseline set to default oscillation pars
        self.__baseline_set.reweight_oscillation(self.__default_pars)

        # apply gradients:
        print("Warning: Only gradients with respect to 'mu' are implemeted so far")
        # TODO otherwise: loop over parameters...
        hist_final = hist_base + (response.mu - self.__baseline_reponse.mu) * self.__gradients['mu']

        # return histogram in full format (see Generator), statistical uncertainties from gradients not yet included...
        return {'hist': hist_final, 'hist_unc': hist_base_unc, 'bin_edges': self.__binning}

    def knn_prepare(self):
        # prepare knn
        self.X_nom = np.vstack((
            self.__sets[self.__baseline_key].get_events()["true_energy"],
            self.__sets[self.__baseline_key].get_events()["reco_energy"]
        )).T

        self.weights_nom = self.__sets[self.__baseline_key].get_events()["weights"]

        # only create tmp storage for systematics
        X_sys_tmp = [
            np.vstack((
                sys_set.get_events()["true_energy"],
                sys_set.get_events()["reco_energy"]
                )).T
            for sys_set_key, sys_set in self.__sets.items() if sys_set_key != self.__baseline_key
        ]

        weights_sys_tmp = np.hstack([
            sys_set.get_events()["weights"]
            for sys_set_key, sys_set in self.__sets.items() if sys_set_key != self.__baseline_key
        ])

        self.X = np.vstack([self.X_nom] + X_sys_tmp)
        self.weights = np.hstack([self.weights_nom] + [weights_sys_tmp])
        self.y = np.hstack(
            [
                np.full(len(self.__sets[self.__baseline_key].get_events()["true_energy"]), i)
                for i, _ in enumerate(self.__sets)
            ]
        )

    def knn_predict(self, knn_n_neighbors=1000, knn_weights="uniform"):
        # predict probabilities for nominal events and return them
        from sklearn import preprocessing
        from sklearn.neighbors import KNeighborsClassifier
        from sklearn.pipeline import Pipeline

        # a transformer to make all of the input data normal... classifiers like this
        self.__trafo = preprocessing.PowerTransformer(method='box-cox', standardize=True)

        # make these private later
        self.X_transformed = self.__trafo.fit_transform(self.X)

        self.__knn = KNeighborsClassifier(n_neighbors=knn_n_neighbors, weights=knn_weights)
        self.__pipe = Pipeline(
            [("transform", self.__trafo), ("classifier", self.__knn)]
        )
        self.__pipe.fit(self.X, self.y)


        self.__nominal_set_probabilities = self.__pipe.predict_proba(self.X_nom)

        return self.__nominal_set_probabilities

    def knn_fit_gradients(self, ):
        # interpolate knn predictions using the theil sen regressor so that the result is robust w.r.t. outliers
        from sklearn.linear_model import TheilSenRegressor
        from sklearn.preprocessing import PolynomialFeatures
        
        nom_events = self.__sets[self.__baseline_key].get_events()

        sigma_nom = self.__sets[self.__baseline_key].get_detector_response().sigma
        mu_nom = self.__sets[self.__baseline_key].get_detector_response().mu

        # the sigmas and mus for the systematic sets stay the same
        sigmas = np.array([
            sys_set.get_detector_response().sigma for sys_set_key, sys_set in self.__sets.items()
            ]) - sigma_nom
        mus = np.array([
            sys_set.get_detector_response().mu for sys_set_key, sys_set in self.__sets.items()
            ]) - mu_nom

        poly = PolynomialFeatures(2, include_bias=False, interaction_only=False)
        event_x = poly.fit_transform(np.vstack((mus, sigmas)).T)

        n_events = len(nom_events["weights"])
        self.__knn_gradients = np.zeros((n_events, event_x.shape[1]))

        # looping over many events in Python is super slow... but unfortunately there is no
        # version of the TheilSenRegressor that will make many different regressions at once
        for i, (true_e, reco_e) in (enumerate(zip(nom_events["true_energy"], nom_events["reco_energy"]))):

            pred_probs = self.__pipe.predict_proba(np.atleast_2d([true_e, reco_e]))[0]

            # divide out the nominal prob to get the re-scale ratio
            pred_probs = pred_probs / pred_probs[0]
            
            # because we fit without intercept, we must subtract the offset of 1!
            event_y = pred_probs - 1.
            reg = TheilSenRegressor(random_state=0, fit_intercept=False).fit(event_x, event_y)
            
            # extract the gradients from the estimator
            self.__knn_gradients[i] = reg.coef_

            # break
            
        return self.__knn_gradients

