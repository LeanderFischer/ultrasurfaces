import numpy as np
import matplotlib.pyplot as plt
import scipy.stats
import uncertainties.unumpy as unp


class Histogram:
    def __init__(self, bin_edges, hist=None, hist_unc=None):
        self.bin_edges = bin_edges
        if hist is None:
            self._hist = unp.uarray(
                np.zeros(len(bin_edges) - 1), np.zeros(len(bin_edges) - 1)
            )
        else:
            self._hist = unp.uarray(hist, hist_unc)

    @property
    def hist(self):
        return unp.nominal_values(self._hist)

    @hist.setter
    def hist(self, value):
        self._hist = unp.uarray(value, unp.std_devs(self._hist))

    @property
    def hist_unc(self):
        return unp.std_devs(self._hist)

    @hist_unc.setter
    def hist_unc(self, value):
        self._hist = unp.uarray(unp.nominal_values(self._hist), value)

    def fill(self, samples, weights=None):
        hist, _ = np.histogram(samples, bins=self.bin_edges, weights=weights)
        hist_unc = (
            np.sqrt(hist)
            if weights is None
            else np.sqrt(
                np.histogram(samples, bins=self.bin_edges, weights=weights**2)[0]
            )
        )
        self.hist = hist
        self.hist_unc = hist_unc

    def __add__(self, other):
        if not isinstance(other, Histogram):
            raise ValueError("Can only add Histogram to Histogram.")
        if not np.array_equal(self.bin_edges, other.bin_edges):
            raise ValueError("Bin edges must be equal to add Histogram instances")
        new_hist = self._hist + other._hist
        return Histogram(
            self.bin_edges,
            hist=unp.nominal_values(new_hist),
            hist_unc=unp.std_devs(new_hist),
        )

    def __sub__(self, other):
        if not isinstance(other, Histogram):
            raise ValueError("Can only subtract Histogram from Histogram.")
        if not np.array_equal(self.bin_edges, other.bin_edges):
            raise ValueError("Bin edges must be equal to add Histogram instances")
        new_hist = self._hist - other._hist
        return Histogram(
            self.bin_edges,
            hist=unp.nominal_values(new_hist),
            hist_unc=unp.std_devs(new_hist),
        )

    def __mul__(self, other):
        if isinstance(other, (int, float)):
            new_hist = self._hist * other
        elif isinstance(other, Histogram):
            if not np.array_equal(self.bin_edges, other.bin_edges):
                raise ValueError("Bin edges must be equal to add Histogram instances")
            new_hist = self._hist * other._hist

        return Histogram(
            self.bin_edges,
            hist=unp.nominal_values(new_hist),
            hist_unc=unp.std_devs(new_hist),
        )

    def __truediv__(self, other):
        if isinstance(other, (int, float)):
            new_hist = self._hist / other
        elif isinstance(other, Histogram):
            if not np.array_equal(self.bin_edges, other.bin_edges):
                raise ValueError("Bin edges must be equal to add Histogram instances")
            new_hist = self._hist / other._hist
        return Histogram(
            self.bin_edges,
            hist=unp.nominal_values(new_hist),
            hist_unc=unp.std_devs(new_hist),
        )


def plot_histogram(histogram, ax=None, show_errorband=True, **plot_kwargs):
    bin_edges = histogram.bin_edges

    if ax is None:
        fig, ax = plt.subplots()

    step = ax.step(
        bin_edges,
        np.append(histogram.hist, histogram.hist[-1]),
        where="post",
        **plot_kwargs
    )

    if not show_errorband:
        return

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


if __name__ == "__main__":
    import numpy as np

    # Unit tests for Histogram class
    # Test initialization without hist and hist_unc
    edges = np.linspace(0, 1, 6)
    h = Histogram(edges)
    assert np.array_equal(h.hist, np.zeros(5))
    assert np.array_equal(h.hist_unc, np.zeros(5))

    # Test initialization with hist and hist_unc
    hist = np.array([1, 2, 3, 2, 1])
    hist_unc = np.sqrt(hist)
    h = Histogram(edges, hist=hist, hist_unc=hist_unc)
    assert np.array_equal(h.hist, hist)
    assert np.array_equal(h.hist_unc, hist_unc)

    # Test setting hist
    hist2 = np.array([2, 3, 4, 3, 2])
    h.hist = hist2
    assert np.array_equal(h.hist, hist2)

    # Test setting hist_unc
    hist_unc3 = np.array([0.1, 0.2, 0.3, 0.2, 0.1])
    h.hist_unc = hist_unc3
    assert np.array_equal(h.hist_unc, hist_unc3)

    # Test fill method
    samples = np.random.uniform(0, 1, size=100)
    h.fill(samples)
    assert np.isclose(np.sum(h.hist), 100)
    assert np.isclose(np.sum(h.hist_unc**2), 100)
    assert np.allclose(h.hist, np.histogram(samples, bins=edges)[0])
    assert np.allclose(h.hist_unc, np.sqrt(np.histogram(samples, bins=edges)[0]))

    # Test fill method with weights
    weights = np.random.normal(1, 0.1, size=100)
    h.fill(samples, weights=weights)
    assert np.isclose(np.sum(h.hist), np.sum(weights))
    assert np.isclose(np.sum(h.hist_unc**2), np.sum(weights**2))

    # Test addition
    h2 = Histogram(edges, h.hist, h.hist_unc)
    assert np.allclose(h2.hist, h.hist)
    assert np.allclose(h2.hist_unc, h.hist_unc)

    h3 = h + h2
    assert np.allclose(h3.hist, 2 * h.hist)

    h3 = h - h2
    assert np.allclose(h3.hist, 0 * h.hist)
    h3 = h * h2
    assert np.allclose(h3.hist, h.hist**2)
    h3 = h / h2
    assert np.allclose(h3.hist, np.ones_like(h.hist))
