"""
Toy physics model and fitter implementation

Tom Stuttard
"""

import os, sys, datetime

from utils.plotting.standard_modules import *

from utils.plotting.animation import AnimatedFigure
from utils.maths.distribution import Distribution
from utils.maths.fitting import fit, Scaling
from scipy import optimize
from scipy.stats import poisson, norm
from utils.maths.stats import get_chi2_critical_values_for_sigma
from utils.cache_tools import Cachable
from utils.filesys_tools import get_file_stem


#
# Test statistics
#


def get_poission_negative_log_likelihood(observed_hist, expected_hist):

    O = observed_hist.hist
    E = expected_hist.hist

    # PISA version
    neg_llh = np.sum(E - (O * np.log(E)))
    neg_llh -= np.sum(O - (O * np.log(O)))

    # # Hand-dervied version
    # from scipy.special import gamma
    # def _poison_llh(O,E) :
    #     return  np.sum( O*np.log(E) - E - np.log(gamma(O+1.)) )
    # llh_v2 = _poison_llh(O,E)
    # llh_v2 -= _poison_llh(O,O)
    # neg_llh_v2  = -1. * llh_v2
    # # print(neg_llh, neg_llh_v2)
    # assert np.isclose(neg_llh, neg_llh_v2)

    return neg_llh
    # return neg_llh_v2


#
# Main analysis class
#


class AnalysisBase(Cachable):
    """
    Base class for a physics model, where the model has free parameters that can be fit to (pseudo)data

    This is basically a super simple version of PISA
    """

    def __init__(self, cache_dir=None, physics_params=None):

        # Caching
        if cache_dir is None:
            cache_dir = os.path.join(
                os.path.dirname(__file__), ".cache", self.__class__.__name__
            )
        Cachable.__init__(self, cache_dir=cache_dir)

        # Store args
        self.physics_params = physics_params

        # Core state
        self.params = collections.OrderedDict()
        self.reset()

        # Init some defaults
        self.set_metric("poisson_llh", get_poission_negative_log_likelihood)

        # # Check physics params
        # if self.physics_params is not None :
        #     for param_name in self.physics_params :
        #     assert param_name in self.params

    def set_metric(self, name, func):
        self.metric_name = name
        self._metric_func = func

    def generate_mc_events(self, *args, **kwargs):

        # TODO For some reason the hash of the realoded events object is not the same as the original. Need to investigate why, but for the time being have disabled events caching...
        # Load cached events if available
        # events, func_call_hash = self.load_cached_results("generate_mc_events", locals())

        # # Run function if no cached results available
        # if events is None :
        if True:

            # Generate evets
            events = self._generate_mc_events(*args, **kwargs)

            # Check them
            assert isinstance(
                events, collections.abc.Mapping
            ), "`_generate_mc_events` must return a dict"
            assert len(events) > 0, "`_generate_mc_events` has returned no data"
            num_events = None
            for k, v in events.items():
                assert isinstance(
                    k, str
                ), "`_generate_mc_events` events dict key must be strings"
                assert isinstance(
                    v, np.ndarray
                ), "`_generate_mc_events` events dict svalues must be numpy arrays"
                if num_events is None:
                    num_events = v.size
                else:
                    assert (
                        v.size == num_events
                    ), "All arryas must have same number of events"

            # Save to cache
            # self.save_results_to_cache("generate_mc_events", func_call_hash, events)

        # Store as member
        self.events = events

        return events

    def _generate_mc_events(self):
        raise Exception(
            "Derived class must overload the `_generate_mc_events` function"
        )

    def pipeline(self, *args, **kwargs):
        hist = self._pipeline(events=self.events, *args, **kwargs)
        assert isinstance(hist, Histogram), "`pipeline` must return a `Histogram`"
        return hist

    def _pipeline(self, events):
        raise Exception("Derived class must overload the `_pipeline` function")

    def get_template(self):
        return self.pipeline()

    def get_asimov_data(self):
        return self.pipeline()

    def get_trial_data(self, trial_index):
        """
        Get trial data, e.g. with statistical fluctuations
        """

        # TODO option to regenerate MC (with the target num events) instead of rvs
        # TODO trial index

        # First get Asimov hist
        hist = self.pipeline()

        # Fluctuate bin counts
        random_state = np.random.RandomState(trial_index + 1)
        hist._hist = poisson.rvs(hist._hist, random_state=random_state)

        return hist

    def reset(self):

        # Reset minimization state variables
        self._num_iterations = 0
        self._animate = False

        # Reset params
        for p in self.free_params.values():
            p.reset()
            p.value = p.nominal_value

    def _minimizer_callback(self, x, data):
        """
        x = params
        args = other useful stuff
        """

        # Set the param values
        free_params = list(self.free_params.values())
        assert x.size == len(free_params)
        for i in range(x.size):
            free_params[i].scaled_value = x[i]

        # Get the template
        template = self.get_template()

        # Clip
        template = template.clip(a_min=1.0e-5, a_max=np.inf)
        data = data.clip(a_min=1.0e-5, a_max=np.inf)

        # Metric
        metric = self._metric_func(expected_hist=template, observed_hist=data)

        # Take any priors into account
        metric_penalty = 0.0
        for p in self.params.values():
            metric_penalty += p.prior_penalty()
        metric += metric_penalty

        # Animation
        if self._animate:
            self._animated_fig.get_ax().clear()
            self.plot_data_template_comparison(
                ax=self._animated_fig.get_ax(), data=data, template=template
            )
            self._animated_fig.get_ax().set_title("-LLH=%0.3g" % metric)
            self._animated_fig.snapshot()

        # Counter
        self._num_iterations += 1

        return metric

    def plot_data_template_comparison(self, ax, data, template):
        plot_hist(ax=ax, hist=template, color="red", errors="band", label="Template")
        plot_hist(
            ax=ax,
            hist=data,
            color="black",
            errors="band",
            linestyle="None",
            label="Data",
        )  # TODO errorbar

    def _fit(self, data, options=None, animate=False, minimizer_algorithm="SLSQP"):
        """
        Fitting function
        """

        # TODO take a copy of the params and pass to pipeline? rather than resetting

        start_time = datetime.datetime.now()

        # Get some default options (need to be well tuned to the specififc problem normally though
        # or else get things like 'Inequality constraints incompatible' )
        # if options is None :
        #     options = { "ftol":1.e-9, "eps":1.e-9 }

        # Reset params
        self.reset()

        # Handle animation
        self._animate = animate  # Needs to be a member to pass to the callback
        if self._animate:
            self._animated_fig = AnimatedFigure(nx=1, ny=1)
            self._animated_fig.start_capture(outfile="animation.mp4")

        # Get the (scaled) param bounds and initial values
        param_initial_guesses = [
            p.scaled_nominal_value for p in list(self.free_params.values())
        ]
        param_bounds = [p.scaled_bounds for p in list(self.free_params.values())]

        # Perform the minimization
        minimizer_results = optimize.minimize(
            self._minimizer_callback,
            x0=param_initial_guesses,
            bounds=param_bounds,
            args=data,
            method=minimizer_algorithm,
            options=options,
        )

        # Finish animation
        if self._animate:
            self._animated_fig.stop_capture()
            self._animated_fig = None

        # Extract results
        success = minimizer_results["success"]
        if not success:
            print("WARNING : Fit failed with error '%s'" % minimizer_results["message"])
        metric = minimizer_results["fun"]  # if success else np.NaN

        # Get best fit template and param values
        best_fit_template = self.get_template()
        best_fit_params = copy.deepcopy(self.params)

        # Time logging
        time_taken = datetime.datetime.now() - start_time

        # Collect results
        results = {
            "success": success,
            "metric": metric,
            "best_fit_template": best_fit_template,
            "best_fit_params": best_fit_params,
            "time_taken": time_taken,
            "num_iterations": self._num_iterations,
        }

        # Reset
        self.reset()

        return results

    def fit(self, *args, fit_null=False, **kwargs):

        # Load cached results if available
        results, func_call_hash = self.load_cached_results("fit", locals())

        # Run function if no cached results available
        if results is None:

            #
            # Perform fit
            #

            fit_results = self._fit(*args, **kwargs)

            results = collections.OrderedDict()
            results["fit"] = fit_results

            #
            # Perform null fit
            #

            if fit_null:

                assert self.physics_params is not None

                # Fix physics params
                for param_name in self.physics_params:
                    assert param_name in self.params
                    self.params[param_name].fixed = True

                # Fit
                null_fit_results = self._fit(*args, **kwargs)
                results["null_fit"] = null_fit_results

                # Unfix physics params
                for param_name in self.physics_params:
                    assert param_name in self.params
                    self.params[param_name].fixed = False

                # Compute mismodelling
                assert self.metric_name == "poisson_llh"
                ndof = len(self.physics_params)
                mismod_test_stat = 2.0 * (
                    null_fit_results["metric"] - fit_results["metric"]
                )  # Wilks theorem

                results["mismodeling"] = {"ndof": ndof, "test_stat": mismod_test_stat}

            # Save to cache
            self.save_results_to_cache("fit", func_call_hash, results)

        # Done
        return results

    def profile(self, data, scan, **fit_kw):
        """
        Run a profile metric scan (such as a profile likelihood)
        """

        # Load cached results if available
        results, func_call_hash = self.load_cached_results("profile", locals())

        # Run function if no cached results available
        if results is None:

            # Check scan
            self._check_scan(scan)
            assert len(scan) == 1, ">1 scan params not yet supported"

            # First do a fit with all params free
            free_fit_results = self._fit(data=data, **fit_kw)
            assert free_fit_results["success"]

            # Fix the scan params
            for param_name in scan.keys():
                assert not self.params[param_name].fixed
                self.params[param_name].fixed = True

            # Scan
            scan_results = []
            for param_name, param_values in scan.items():
                for param_val in param_values:

                    print("Profile scan point : %s" % param_val)

                    # Set param
                    self.params[param_name].value = param_val

                    # Fit
                    scan_point_results = self._fit(data=data, **fit_kw)
                    scan_results.append(scan_point_results)

            # Unfix the scan params and reset original vals
            for param_name in scan.keys():
                self.params[param_name].fixed = False
            self.reset()

            # Store the results
            results = {
                "free_fit_results": free_fit_results,
                "scan_points": scan,
                "scan_results": scan_results,
            }

            # Save to cache
            self.save_results_to_cache("profile", func_call_hash, results)

        # Done
        return results

    def _check_scan(self, scan):
        """
        Check the user-defined scan points
        """
        assert isinstance(scan, collections.abc.Mapping)
        for n, v in scan.items():
            assert n in self.free_params
            assert isinstance(v, np.ndarray)

    @property
    def free_params(self):
        return collections.OrderedDict(
            [(n, p) for n, p in self.params.items() if not p.fixed]
        )

    @property
    def fixed_params(self):
        return collections.OrderedDict(
            [(n, p) for n, p in self.params.items() if p.fixed]
        )

    def plot_fit(self):
        """
        Plot fit results
        """

        raise Exception("Needs updating")

        add_heading_page("Fit")

        all_figs = []

        #
        # Plot data vs template (single trial only)
        #

        trial_index = 0
        trial = self.trial_data[trial_index]

        fig = Figure(title="Trial %i" % trial_index)
        all_figs.append(fig)

        plot_hist(
            ax=fig.get_ax(),
            hist=trial["fit"]["best_fit_template"],
            color="red",
            errors="band",
            label="Fitted template",
        )
        plot_hist(
            ax=fig.get_ax(),
            hist=trial["data"],
            color="black",
            errors="bar",
            linestyle="None",
            label="Data",
        )

        fig.quick_format()

        #
        # Plot fit values for selected params
        #

        nx, ny = get_grid_dims(n=len(self.free_params))
        fig = Figure(
            nx=nx, ny=ny, title="Fitted values (%i trials)" % len(self.trial_data)
        )
        all_figs.append(fig)

        for i, param_name in enumerate(self.free_params.keys()):

            ax = fig.get_ax(i=i)

            param_fit_vals = [
                t["fit"]["best_fit_params"][param_name].value for t in self.trial_data
            ]

            hist = Histogram(
                ndim=1, bins=generate_bins(param_fit_vals, num=20), x=param_fit_vals
            )
            plot_hist(ax=ax, hist=hist, errors="band", label="Trial fits")

            ax.axvline(
                x=self.trial_data[0]["truth_values"][param_name],
                color="purple",
                label="Truth",
            )

            percentiles = np.percentile(
                param_fit_vals, [50.0 - (68.0 / 2.0), 50.0, 50.0 + (68.0 / 2.0)]
            )
            ax.axvline(x=percentiles[1], color="grey", linestyle="--", label=r"Median")
            ax.axvline(
                x=percentiles[0], color="grey", linestyle=":", label=r"$1 \sigma$"
            )
            ax.axvline(x=percentiles[2], color="grey", linestyle=":")

            ax.set_xlabel(param_name)

        fig.hide_unused_axes()
        fig.quick_format(ylabel="Num trials", ylim=(0.0, None))

        # Done
        return all_figs

    def plot_profile(self):
        """
        Plot profile results
        """

        raise Exception("Needs updating")

        add_heading_page("Profile")

        all_figs = []

        #
        # Plot profile
        #

        fig = Figure(nx=2, title="Profile scan (%i trials)" % len(self.trial_data))
        all_figs.append(fig)

        if self.num_trials > 0:

            #
            # Plot prifle for trials
            #

            # Get data
            ndim = len(self.trial_data[0]["profile"]["scan_points"])
            assert ndim == 1, "Only 1D plotting supported currently"
            scan_param_name = list(self.trial_data[0]["profile"]["scan_points"].keys())[
                0
            ]
            scan_points = list(self.trial_data[0]["profile"]["scan_points"].values())[0]
            scan_param_truth = self.trial_data[0]["truth_values"][scan_param_name]

            # Loop over trials
            trial_scan_metrics, trial_scan_test_stats = [], []
            for i_trial, trial in enumerate(self.trial_data):

                # Plot the metric
                scan_metrics = np.array(
                    [r["metric"] for r in trial["profile"]["scan_results"]]
                )
                fig.get_ax(x=0).plot(
                    scan_points,
                    scan_metrics,
                    color="orange",
                    alpha=0.1,
                    label=("Trials" if i_trial == 0 else None),
                )

                # Plot the test stat
                scan_test_stats = 2.0 * (
                    scan_metrics - trial["profile"]["free_fit_results"]["metric"]
                )
                fig.get_ax(x=1).plot(
                    scan_points,
                    scan_test_stats,
                    color="orange",
                    alpha=0.1,
                    label=("Trials" if i_trial == 0 else None),
                )

                # Also store for median calculation later
                trial_scan_metrics.append(scan_metrics)
                trial_scan_test_stats.append(scan_test_stats)

            # Median
            scan_metrics_median = np.median(trial_scan_metrics, axis=0)
            fig.get_ax(x=0).plot(
                scan_points, scan_metrics_median, color="red", label="Median"
            )
            scan_test_stats_median = np.median(trial_scan_test_stats, axis=0)
            fig.get_ax(x=1).plot(
                scan_points, scan_test_stats_median, color="red", label="Median"
            )

        if self.asimov_data is not None:

            #
            # Plot profle for trials
            #

            ndim = len(self.asimov_data["profile"]["scan_points"])
            assert ndim == 1, "Only 1D plotting supported currently"

            scan_param_name = list(self.asimov_data["profile"]["scan_points"].keys())[0]
            scan_points = list(self.asimov_data["profile"]["scan_points"].values())[0]

            scan_param_truth = self.asimov_data["truth_values"][
                scan_param_name
            ]  # TODO check matches trials

            scan_metrics = np.array(
                [r["metric"] for r in self.asimov_data["profile"]["scan_results"]]
            )
            fig.get_ax(x=0).plot(
                scan_points, scan_metrics, color="black", linestyle="--", label="Asimov"
            )

            scan_test_stats = 2.0 * (
                scan_metrics - self.asimov_data["profile"]["free_fit_results"]["metric"]
            )
            fig.get_ax(x=1).plot(
                scan_points,
                scan_test_stats,
                color="black",
                linestyle="--",
                label="Asimov",
            )

        # Overlay sigma lines
        critical_vals = get_chi2_critical_values_for_sigma(ndim, [1, 2, 3])
        color_scale = ColorScale("Greys_r", len(critical_vals) + 1)
        for k, v in critical_vals.items():
            fig.get_ax(x=1).axhline(
                v, color=color_scale.get_next(), label=r"$%i \sigma$" % k
            )

        # Formatting
        for ax in fig.get_all_ax():
            ax.axvline(scan_param_truth, color="purple", label="Truth")

        fig.get_ax(x=0).set_ylabel(r"$\rm{LLH}$")
        fig.get_ax(x=1).set_ylabel(r"$-2 \Delta \rm{LLH}$")

        fig.hide_unused_axes()
        fig.quick_format(xlabel=scan_param_name, ylim=(0.0, None))

        #
        # Plot free fits
        #

        fig = Figure(title="Free fits")
        all_figs.append(fig)

        scan_param_free_fit_results = [
            t["profile"]["free_fit_results"]["best_fit_params"][scan_param_name].value
            for t in self.trial_data
        ]

        percentiles = np.percentile(
            scan_param_free_fit_results,
            [50.0 - (68.0 / 2.0), 50.0, 50.0 + (68.0 / 2.0)],
        )

        hist = Histogram(
            ndim=1,
            bins=generate_bins(scan_param_free_fit_results, num=20),
            x=scan_param_free_fit_results,
        )
        plot_hist(ax=fig.get_ax(), hist=hist, errors="band", label="Trial fits")

        fig.get_ax().axvline(
            x=self.trial_data[0]["truth_values"][scan_param_name],
            color="purple",
            label="Truth",
        )

        fig.get_ax().axvline(
            x=percentiles[1], color="grey", linestyle="--", label=r"Median"
        )
        fig.get_ax().axvline(
            x=percentiles[0], color="grey", linestyle=":", label=r"$1 \sigma$"
        )
        fig.get_ax().axvline(x=percentiles[2], color="grey", linestyle=":")

        fig.quick_format(xlabel=scan_param_name)

        # Done
        return all_figs


class AnalysisParam(object):
    """
    Class representating a parameter in the model
    """

    # TODO Replace with utils.maths.fitting.Param?
    # TODO add a prior

    def __init__(self, value, bounds=None, fixed=False, prior_sigma=None):
        self.nominal_value = copy.deepcopy(value)
        self.value = value
        self.bounds = bounds
        self.fixed = fixed
        self.prior_sigma = prior_sigma  # This specifies a Gaussian prior

        if self.bounds is None:
            assert self.fixed  # TODO enforce as part of a setter
        else:
            assert len(self.bounds) == 2
            assert np.all(np.isfinite(self.bounds))
            self.scaling = Scaling(min_val=self.bounds[0], max_val=self.bounds[1])

    def reset(self):
        self.value = self.nominal_value

    @property
    def scaled_value(self):
        assert not self.fixed
        return self.scaling.scale(self.value)

    @scaled_value.setter
    def scaled_value(self, sv):
        assert not self.fixed
        self.value = self.scaling.unscale(sv)

    @property
    def scaled_nominal_value(self):
        assert not self.fixed
        return self.scaling.scale(self.nominal_value)

    @property
    def scaled_bounds(self):
        assert not self.fixed
        return tuple(
            [self.scaling.scale(self.bounds[0]), self.scaling.scale(self.bounds[1])]
        )

    def prior_penalty(self):
        """
        Penalty term from a Gaussian prior
        This assumes the metric is LLH
        """
        if self.prior_sigma is not None:
            x = self.value
            m = self.nominal_value
            s = self.prior_sigma
            return (x - m) ** 2 / (
                2 * s ** 2
            )  # Removed - sign, sign using -llh as metric
        else:
            return 0.0


class Hypersurface(Cachable):
    """
    A simple hypersurface implementation

    Actually is only a line right now...
    """

    def __init__(self, bins, gradient_bounds, intercept_bounds):

        self.bins = bins

        assert self.bins.ndim == 1
        shape = self.bins.size - 1

        self.gradient = np.full(shape, np.NaN)
        self.intercept = np.full(shape, np.NaN)

        self.gradient_bounds = gradient_bounds
        self.intercept_bounds = intercept_bounds

    @property
    def shape(self):
        return self.gradient.shape

    def _func(self, value, gradient, intercept):
        return (gradient * value) + intercept

    def fit(self, nominal_mc, sys_mc):
        """
        Each MC set should be defined as:
            {
                "hist" : the histogram resulting from the MC
                "value" : the parameter value
            }
        """

        # TODO caching

        # Norm to the nominal
        self.hists = [nominal_mc["hist"]] + [s["hist"] for s in sys_mc]
        self.hists = [h / nominal_mc["hist"] for h in self.hists]

        # Get the param values
        x = [nominal_mc["value"]] + [s["value"] for s in sys_mc]

        # Steer fit
        p0 = [0.0, 0.0]  # TODO steerable
        bounds = [
            (self.gradient_bounds[0], self.intercept_bounds[0]),
            (self.gradient_bounds[1], self.intercept_bounds[1]),
        ]

        # Loop over bins
        for bin_idx in np.ndindex(self.shape):

            # Skip any with no data in the nominal hist
            if nominal_mc["hist"].hist[bin_idx] == 0.0:
                continue

            # Get the normalised bin counts
            y = [h.hist[bin_idx] for h in self.hists]
            y_sigma = [h.sigma[bin_idx] for h in self.hists]

            # Define callback
            def callback(x, *p):
                return self._func(value=x, gradient=p[0], intercept=p[1])

            # Fit
            popt, pcov = optimize.curve_fit(
                callback,
                x,
                y,
                sigma=y_sigma,
                p0=p0,
                bounds=bounds,
                maxfev=1000000,
                # method="dogbox", # lm, trf, dogbox
            )

            # Write fit results to the member variables
            self.gradient[bin_idx] = popt[0]
            self.intercept[bin_idx] = popt[1]

    def evaluate(self, value):
        return self._func(value=value, gradient=self.gradient, intercept=self.intercept)

    def __call__(self, value):
        return self.evaluate(value=value)

    def plot(self, ax, x, bin_idx, **kw):
        y = [self.evaluate(xx)[bin_idx] for xx in x]
        ax.plot(x, y, **kw)


#
# Simple example
#


class ExampleAnalysis(AnalysisBase):
    def __init__(self):

        super(ExampleAnalysis, self).__init__()

        # Define the free params of a normal distribution
        self.params["norm"] = AnalysisParam(value=1.0, bounds=(0.0, 10.0), fixed=False)
        self.params["mean"] = AnalysisParam(
            value=100.0, bounds=(80.0, 120.0), fixed=False
        )
        self.params["sigma"] = AnalysisParam(value=5.0, bounds=(0.1, 10.0), fixed=False)

    def _generate_mc_events(self, random_state=None):

        events = collections.OrderedDict()

        if random_state is None:
            random_state = np.random.RandomState()

        num_events = 1000000
        num_events_weighted = 100000

        events["true_x"] = random_state.uniform(
            self.params["mean"].bounds[0], self.params["mean"].bounds[1], num_events
        )
        # events["reco_x"] = andom_state.normal(events["true_x"], 5.)
        events["reco_x"] = events["true_x"]

        events["weights"] = np.full_like(
            events["true_x"], float(num_events_weighted) / float(num_events)
        )

        return events

    def _pipeline(self, events):

        reco_x = events["reco_x"]
        weights = events["weights"]

        # Re-weight to desired Gaussian
        weight_mod = np.exp(
            -0.5
            * np.square(
                (reco_x - self.params["mean"].value) / self.params["sigma"].value
            )
        )
        weight_mod *= self.params["norm"].value / np.nanmax(weight_mod)
        new_weights = weight_mod * weights

        # Make hist
        hist = Histogram(
            ndim=1,
            uof=False,
            bins=get_bins(
                self.params["mean"].bounds[0], self.params["mean"].bounds[1], num=20
            ),
            x=reco_x,
            weights=new_weights,
        )

        return hist


if __name__ == "__main__":

    from utils.script_tools import ScriptWrapper
    from utils.filesys_tools import replace_file_ext

    with ScriptWrapper(replace_file_ext(__file__, ".log")) as script:

        # Create model
        model = ExampleAnalysis()
        model.generate_mc_events()

        # Define a profile scan

        # Get some Asimov data
        data = model.get_asimov_data()  # TODO store true params

        # Fit the data
        results = model.fit(data)

        # Profile the data
        scan = {"sigma": np.linspace(3.0, 7.0, num=5)}
        model.profile(data=data, scan=scan)

        # Done
        print("")
        dump_figures_to_pdf(replace_file_ext(__file__, ".pdf"))
