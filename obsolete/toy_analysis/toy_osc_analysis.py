"""
Toy oscillation analysis

This is DeepCore-like, but only in 1D (reco energy)
"""

import os, sys
import numpy as np

# from utils.plotting.standard_modules import *

from simple_analysis import (
    AnalysisBase,
    AnalysisParam,
    get_bins,
    Histogram,
)


#
# Analysis class
#


def spectral_index_scale(true_energy, egy_pivot, delta_index):
    return np.power((true_energy / egy_pivot), delta_index)


def calc_osc_survival_prob(true_baseline, true_energy, theta, deltam2):
    # return np.full_like(true_energy, 1.)
    return 1.0 - (
        np.square(np.sin(2.0 * theta))
        * np.square(np.sin((1.27 * deltam2 * true_baseline) / true_energy))
    )


class ToyOscAnalysis(AnalysisBase):

    # TODO inetrgate with toy PISA stage (maybe that call common functions)

    def __init__(self, num_mc_events, num_data_events):

        super(ToyOscAnalysis, self).__init__(physics_params=["theta", "deltam2"])

        self.num_mc_events = num_mc_events
        self.num_data_events = num_data_events

        self.reco_energy_bins = get_bins(10.0, 50.0, width=2.5)
        self.true_energy_range = (0.1, 100.0)

        self.baseline = 12700.0  # Earth diameter

        self.params["norm"] = AnalysisParam(value=1.0, bounds=(0.1, 3.0), fixed=False)
        self.params["delta_gamma"] = AnalysisParam(
            value=-1.0, bounds=(-2.0, 0.0), fixed=False
        )  # TODO use a (weaker) atmo power law
        self.params["theta"] = AnalysisParam(
            value=45.0, bounds=(0.0, 90.0), fixed=False
        )
        self.params["deltam2"] = AnalysisParam(
            value=2.5e-3, bounds=(2.0e-3, 3.0e-3), fixed=False
        )

    def _generate_mc_events(self, random_state=None, compute_reference_weights=True):
        """
        Generate MC events
        """

        events = collections.OrderedDict()

        if random_state is None:
            random_state = np.random.RandomState()

        # Generate uniform true E distribution
        true_energy = random_state.uniform(
            self.true_energy_range[0],
            self.true_energy_range[1],
            size=self.num_mc_events,
        )

        # Get reco energy, which has some smearing (assuming 10% resolution)
        reco_energy = random_state.normal(true_energy, 0.1 * true_energy)

        # Stash
        events["true_energy"] = true_energy
        events["reco_energy"] = reco_energy
        # reference weights are "standard" oscillated weights
        if compute_reference_weights:
            events["reference_weights"] = self.compute_weights(events) 
        return events
    
    def compute_weights(self, events):
        if "weights" in events:
            weights = events["weights"].copy()
        else:
            weights = np.ones_like(events["true_energy"])

        # Apply oscillations
        Posc = calc_osc_survival_prob(
            true_baseline=self.baseline,
            true_energy=events["true_energy"],
            theta=np.deg2rad(self.params["theta"].value),
            deltam2=self.params["deltam2"].value,
        )
        weights *= Posc

        # Apply spectral index shift
        weights *= spectral_index_scale(
            true_energy=events["true_energy"],
            egy_pivot=5.0,
            delta_index=self.params["delta_gamma"].value,
        )

        # Correct weighting to target num events
        generated_num_events = events["true_energy"].size
        weights *= float(self.num_data_events) / float(self.num_mc_events)

        # Correct for norm
        weights *= self.params["norm"].value
        return weights
    
    def _pipeline(self, events):
        """
        Reweight events to get template
        """

        bins = self.reco_energy_bins

        weights = self.compute_weights(events)

        # Make hist
        hist = Histogram(
            ndim=1, uof=False, bins=bins, x=events["reco_energy"], weights=weights
        )

        return hist


#
# Main
#

if __name__ == "__main__":

    # from utils.script_tools import ScriptWrapper
    # from utils.filesys_tools import replace_file_ext

    with ScriptWrapper(replace_file_ext(__file__, ".log")) as script:

        #
        # Init
        #

        # Random number seeding
        random_state = np.random.RandomState(123345)

        # Instantiate analysis
        model = ToyOscAnalysis(num_mc_events=100000, num_data_events=10000)

        # Generate the MC
        model.generate_mc_events(random_state=random_state)

        #
        # Sanity plots
        #

        add_heading_page("Sanity checks", figsize=(4, 2))

        # Osc prob
        fig = Figure(figsize=(6, 4))
        E = np.linspace(5.0, 100.0, num=1000)
        P = calc_osc_survival_prob(
            true_baseline=model.baseline,
            true_energy=E,
            theta=np.deg2rad(model.params["theta"].value),
            deltam2=model.params["deltam2"].value,
        )
        fig.get_ax().plot(E, P, color="orange")
        fig.quick_format(
            xlabel=r"$E_{\rm{true}}$ [GeV]", ylabel=r"$P_{\mu\mu}$", ylim=(0.0, None)
        )

        #
        # Asimov fit
        #

        add_heading_page("Asimov fit", figsize=(4, 2))

        # Generate pseuodata
        data = model.get_asimov_data()

        # Fit
        fit_results = model.fit(data, animate=False)

        # Report results
        print("Asimov fit results:")
        for n, p in fit_results["fit"]["best_fit_params"].items():
            print(f"  {n} = {p.value}")
        print("  -LLH = %0.3g" % fit_results["fit"]["metric"])
        print("  Num interations = %i" % fit_results["fit"]["num_iterations"])
        print("  Time taken = %s" % fit_results["fit"]["time_taken"])

        # Plot data and best fit map
        fig = Figure(figsize=(6, 4))
        model.plot_data_template_comparison(
            ax=fig.get_ax(), data=data, template=fit_results["fit"]["best_fit_template"]
        )
        fig.quick_format(
            xlabel=r"$E_{\rm{reco}}$ [GeV]",
            ylabel="Num events",
            ylim=(0.0, None),
            legend_kw={"fontsize": 8},
        )
        # TODO data-fit pulls

        #
        # LLH profile
        #

        if True:

            add_heading_page("Asimov LLH profile", figsize=(4, 2))

            # Define scan
            scan = {"theta": np.linspace(40.0, 50.0, num=11)}

            # Run the profile
            profile_results = model.profile(data, scan=scan)

            # Plot the profile
            # TODO

        #
        # Trial fits
        #

        if True:

            add_heading_page("Trial fits", figsize=(4, 2))

            # Steering
            num_trials = 10

            # Init hists of trial results
            num_bins = 20
            fit_results_hists = collections.OrderedDict()
            fit_results_hists["theta"] = Histogram(
                ndim=1, uof=False, bins=get_bins(40.0, 50.0, num=num_bins)
            )
            fit_results_hists["deltam2"] = Histogram(
                ndim=1, uof=False, bins=get_bins(2.4e-3, 2.6e-3, num=num_bins)
            )
            fit_results_hists["norm"] = Histogram(
                ndim=1, uof=False, bins=get_bins(0.8, 1.2, num=num_bins)
            )
            fit_results_hists["delta_gamma"] = Histogram(
                ndim=1, uof=False, bins=get_bins(-0.2, +0.2, num=num_bins)
            )
            metric_hist = Histogram(
                ndim=1, uof=False, bins=get_bins(0.0, 20.0, num=num_bins)
            )

            # Plot steering
            num_trials_to_plot = min(10, num_trials)
            trial_colors = ColorScale("hsv", num_trials_to_plot)
            fig_hists = Figure(ny=2, figsize=(6, 8), row_headings=["Data", "Best fit"])

            # Loop over trials
            print("\nStarting trials...")
            for i_trial in range(num_trials):

                print(f"Trial {i_trial}")

                # Generate pseuodata
                data = model.get_trial_data(trial_index=i_trial)

                # Fit
                fit_results = model.fit(data, animate=False)

                # Histogram the param fit values
                metric_hist.fill([fit_results["fit"]["metric"]])
                for n, p in fit_results["fit"]["best_fit_params"].items():
                    print(f"  {n} = {p.value}")
                    fit_results_hists[n].fill(x=[p.value])

                # Plot data and fit map
                if i_trial < num_trials_to_plot:
                    plot_hist(
                        ax=fig_hists.get_ax(y=0),
                        hist=data,
                        color=trial_colors.get(i_trial),
                        label=f"Trial {i_trial}",
                    )
                    plot_hist(
                        ax=fig_hists.get_ax(y=1),
                        hist=fit_results["fit"]["best_fit_template"],
                        color=trial_colors.get(i_trial),
                        label=f"Trial {i_trial}",
                    )
                    # TODO data-fit pulls

            # Plot formatting
            fig_hists.quick_format(
                xlabel=r"$E_{\rm{reco}}$ [GeV]",
                ylabel="Num events",
                ylim=(0.0, None),
                legend_kw={"fontsize": 8},
            )

            # Plot metric
            fig = Figure(figsize=(6, 4))
            plot_hist(ax=fig.get_ax(), hist=metric_hist, errors=False)
            fig.quick_format(xlabel="-LLH", ylabel="Num trials")

            # Plot param fits
            nx, ny = get_grid_dims(n=len(fit_results_hists))
            fig = Figure(nx=nx, ny=ny)
            for i, (k, h) in enumerate(fit_results_hists.items()):
                ax = fig.get_ax(i=i)
                plot_hist(ax=ax, hist=h, errors=False)
                ax.set_xlabel("Fit %s" % k)
            fig.hide_unused_axes()
            fig.quick_format(ylabel="Num trials")

        #
        # Done
        #

        print("")
        dump_figures_to_pdf(replace_file_ext(__file__, ".pdf"))
