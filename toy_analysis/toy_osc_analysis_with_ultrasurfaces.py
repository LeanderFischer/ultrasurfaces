"""
Adding ultrasurfaces to the toy osc analysis

A. Trettin
"""

import os, sys

import collections
import numpy as np

from sklearn.neural_network import MLPClassifier
from sklearn import preprocessing
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import TheilSenRegressor

from scipy.stats import lognorm, truncnorm

from analysis.sandbox.atrettin.osc_analysis.simple_analysis import AnalysisParam
from analysis.sandbox.atrettin.osc_analysis.toy_osc_analysis import ToyOscAnalysis


class ToyOscAnalysisUltrasurf(ToyOscAnalysis):
    def __init__(self, *args, dropout=False, true_mean_energy=60., **kwargs):

        super().__init__(*args, **kwargs)

        # The "ultrasurface" are per-event gradients with which the events can be re-weighted
        self.has_ultrasurface = False
        # Degree of the polynomial features that the event-wise surfaces are fitted to.
        # Degree 1 means that the surfaces are linear in all terms, degree two includes
        # squared and interaction terms
        self.poly_degree = 1
        self.dropout = dropout

        self.params["scale"] = AnalysisParam(value=1.0, bounds=(0.8, 1.2), fixed=False)
        self.params["shape"] = AnalysisParam(value=0.3, bounds=(0.2, 0.4), fixed=False)
        
        self.nominal_scale = 1.0
        self.nominal_shape = 0.3
        self.true_mean_energy = true_mean_energy
        self.true_energy_range = (0.1, 1000)
        
    def _generate_mc_events(self, scale=None, shape=None, random_state=None, compute_reference_weights=True):
        """
        Generate MC events

        Overriding the base class version
        """

        events = collections.OrderedDict()

        if random_state is None:
            random_state = np.random.RandomState()

        if scale is None:
            scale = self.params["scale"].value

        if shape is None:
            shape = self.params["shape"].value
        # optionally, scale the number of events in the set by the efficiency
        # to mimick selection effects.
        num_mc_events = int(self.num_mc_events * scale if self.dropout else self.num_mc_events)
        
        # Generate uniform true E distribution
        # from https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.truncnorm.html
        myclip_a = np.log(self.true_energy_range[0])
        myclip_b = np.log(self.true_energy_range[1])
        my_std = (myclip_b - myclip_a) / 10.  # make it clipped to 2 sigma on each side
        my_mean = np.log(self.true_mean_energy)
        
        a, b = (myclip_a - my_mean) / my_std, (myclip_b - my_mean) / my_std
        true_log_energy = truncnorm.rvs(a, b, loc=my_mean, scale=my_std, size=num_mc_events)
        true_energy = np.exp(true_log_energy)

        # Reco energy is distributed log-normally, influenced by shape and scale parameters
        reco_energy = lognorm.rvs(
            shape, loc=0, scale=true_energy * scale, size=num_mc_events, random_state=random_state
        )

        weights = np.ones_like(true_energy)

        # Stash
        events["true_energy"] = true_energy
        events["reco_energy"] = reco_energy
        events["initial_weights"] = weights
        events["weights"] = weights.copy()
        
        # The gradients w.r.t. the systematic parameters (or polynomial features thereof) will be filled
        # when ultrasurfaces are fit.
        events["gradients"] = np.full_like(weights, np.nan)
        
        # reference weights are "standard" oscillated weights
        if compute_reference_weights:
            events["reference_weights"] = self.compute_weights(events) 
        return events

    def _pipeline(self, events):
        """
        Reweight events to get template
        """
        
        if self.has_ultrasurface:
            delta_mu = self.params["scale"].value - self.nominal_scale
            delta_sig = self.params["shape"].value - self.nominal_shape
            
            poly_features = preprocessing.PolynomialFeatures(
                self.poly_degree, include_bias=False
            ).fit_transform(
                np.atleast_2d([delta_mu, delta_sig])
            )[0]
            
            events["weights"] = events["initial_weights"] * (
                1. + np.dot(events["gradients"], poly_features)
            )
        # Run standard osc analysis pipeline
        hist = super()._pipeline(events=events)

        return hist

    def fit_ultrasurface(self, poly_degree=1, random_state=None):
        
        if random_state is None:
            random_state = np.random.RandomState()
        
        model = self
        model.poly_degree = poly_degree
        # Generate MC
        print("generating MC...")
        nominal_set = {"scale": self.params["scale"].value, "shape": self.params["shape"].value}
        sys_sets = [
            # on-axis variations
            {"scale":1.1, "shape":0.3 },
            {"scale":0.9, "shape":0.3 },
            {"scale":1.0, "shape":0.25},
            {"scale":1.0, "shape":0.35},
            # off-axis variations
            {"scale":1.1, "shape":0.25},
            {"scale":1.1, "shape":0.35},
            {"scale":0.9, "shape":0.25},
            {"scale":0.9, "shape":0.35},
        ]

        for dataset in sys_sets + [nominal_set]:
            dataset["events"] = model.generate_mc_events(
                random_state=random_state, **dataset
            )
            dataset["template"] = model.get_template()
        
        print("assembling dataset...")
        # Build the ML-friendly dataset
        X_nom = np.vstack((nominal_set["events"]["true_energy"], nominal_set["events"]["reco_energy"])).T

        X_sys = [
            np.vstack((sys_set["events"]["true_energy"], sys_set["events"]["reco_energy"])).T
            for sys_set in sys_sets
        ]

        X = np.vstack([X_nom] + X_sys)
        # The target label is the set number of every event
        y = np.hstack(
            [np.full(len(dataset["events"]["true_energy"]), i)
             for i, dataset in enumerate([nominal_set] + sys_sets)]
        )

        # shuffle data (always a good idea!)
        random_idx = random_state.permutation(len(X))
        X = X[random_idx]
        y = y[random_idx]

        # Build the ML model
        mlp = MLPClassifier(
            solver='adam',
            alpha=1e-3,
            hidden_layer_sizes=(20, 10, 5),
            random_state=random_state
        )

        # a transformer to make all of the input data normal... classifiers like this
        pt = preprocessing.PowerTransformer(method='box-cox', standardize=True)

        # actually fit! This takes a while...
        print("fitting classifier...")
        pipe = make_pipeline(pt, mlp)
        pipe.fit(X, y)
        
        # last step is to calculate the event-wise gradients that predict the ML prediction
        nom_events = model.events

        sigma_nom = nominal_set["shape"]
        mu_nom = nominal_set["scale"]
        
        # the sigmas and mus for the systematic sets stay the same
        sigmas = np.array([dataset["shape"] for dataset in [nominal_set] + sys_sets]) - sigma_nom
        mus = np.array([dataset["scale"] for dataset in [nominal_set] + sys_sets]) - mu_nom
        
        poly = preprocessing.PolynomialFeatures(self.poly_degree, include_bias=False)
        event_x = poly.fit_transform(np.vstack((mus, sigmas)).T)

        n_events = len(nom_events["weights"])
        nom_events["gradients"] = np.zeros((n_events, event_x.shape[1]))
        
        print("fitting gradients to classifier predictions...")
        # looping over many events in Python is super slow... but unfortunately there is no
        # version of the TheilSenRegressor that will make many different regressions at once
        for i, (true_e, reco_e) in enumerate(zip(nom_events["true_energy"], nom_events["reco_energy"])):
            pred_probs = pipe.predict_proba(np.atleast_2d([true_e, reco_e]))[0]
            # divide out the nominal prob to get the re-scale ratio
            pred_probs = pred_probs / pred_probs[0]

            # because we fit without intercept, we must subtract the offset of 1!
            event_y = pred_probs - 1
            reg = TheilSenRegressor(random_state=0, fit_intercept=False).fit(event_x, event_y)

            # extract the gradients from the estimator
            nom_events["gradients"][i] = reg.coef_

        print("Ultrasurface fit finished!")
        self.has_ultrasurface = True

