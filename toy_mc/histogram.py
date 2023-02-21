import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import pint


class Histogram:
    def __init__(self, bin_edges, hist=None, hist_unc=None):
        if hist is not None:
            if len(hist) != len(hist_unc):
                raise ValueError("hist and hist_unc must have the same length")
            if len(bin_edges) != len(hist) + 1:
                raise ValueError("bin_edges must have length hist + 1")

        self._bin_edges = bin_edges
        self._hist = hist
        self._hist_unc = hist_unc

    @property
    def bin_edges(self):
        return self._bin_edges

    @property
    def hist(self):
        return self._hist

    @hist.setter
    def hist(self, value):
        self._hist = value

    @property
    def hist_unc(self):
        return self._hist_unc

    @hist_unc.setter
    def hist_unc(self, value):
        self._hist_unc = value

    @property
    def hist_with_unc(self):
        """Returns the histogram data with uncertainties"""
        hist_with_unc = pint.Quantity(self.hist, "dimensionless")
        hist_with_unc.uncertainty = self.hist_unc
        return hist_with_unc

    def fill(self, samples, weights=None):
        if weights is None:
            hist, bin_edges, _ = stats.binned_statistic(samples, samples, bins=self.bin_edges)
            self.hist = hist
            self.hist_unc = np.sqrt(hist)
        else:
            hist, bin_edges, bin_idx = stats.binned_statistic(
                samples, weights, bins=self.bin_edges
            )
            hist_sq, _, _ = stats.binned_statistic(
                samples, weights**2, bins=bin_edges
            )
            self.hist = hist
            self.hist_unc = np.sqrt(hist_sq)

    def __add__(self, other):
        if not np.array_equal(self.bin_edges, other.bin_edges):
            raise ValueError("bin_edges must be the same for both histograms")

        new_hist_with_unc = self.hist_with_unc + other.hist_with_unc
        new_hist = new_hist_with_unc.nominal_value
        new_hist_unc = new_hist_with_unc.std_dev

        return Histogram(new_hist, new_hist_unc, self.bin_edges)

    def __sub__(self, other):
        if not np.array_equal(self.bin_edges, other.bin_edges):
            raise ValueError("bin_edges must be the same for both histograms")

        new_hist_with_unc = self.hist_with_unc - other.hist_with_unc
        new_hist = new_hist_with_unc.nominal_value
        new_hist_unc = new_hist_with_unc.std_dev

        return Histogram(new_hist, new_hist_unc, self.bin_edges)

    def __mul__(self, other):
        if not np.array_equal(self.bin_edges, other.bin_edges):
            raise ValueError("bin_edges must be the same for both histograms")

        new_hist_with_unc = self.hist_with_unc * other.hist_with_unc
        new_hist = new_hist_with_unc.nominal_value
        new_hist_unc = new_hist_with_unc.std_dev

        return Histogram(new_hist, new_hist_unc, self.bin_edges)

    def __truediv__(self, other):
        if not np.array_equal(self.bin_edges, other.bin_edges):
            raise ValueError("bin_edges must be the same for both histograms")

        new_hist_with_unc = self.hist_with_unc / other.hist_with_unc
        new_hist = new_hist_with_unc.nominal_value
        new_hist_unc = new_hist_with_unc.std_dev

        return Histogram(new_hist, new_hist_unc, self.bin_edges)


def plot_histogram(histogram):
    bin_centers = 0.5 * (histogram.bin_edges[1:] + histogram.bin_edges[:-1])
    bin_widths = np.diff(histogram.bin_edges)
    bin_edges = histogram.bin_edges

    fig, ax = plt.subplots()

    step = ax.step(
        bin_edges,
        np.append(histogram.hist, histogram.hist[-1]),
        where="post",
        color="black",
    )

    color = step[0].get_color()
    ax.fill_between(
        bin_edges,
        np.append(
            histogram.hist - histogram.hist_unc,
            histogram.hist[-1] - histogram.hist_unc[-1],
        ),
        np.append(
            histogram.hist + histogram.hist_unc,
            histogram.hist[-1] + histogram.hist_unc[-1],
        ),
        step="post",
        alpha=0.2,
        color=color,
    )

    ax.set_xlabel("Bin edges")
    ax.set_ylabel("Counts")
    plt.show()
