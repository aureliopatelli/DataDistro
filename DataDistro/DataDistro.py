import numpy as np
import pandas as pd
from scipy.stats import norm, lognorm


class DataDistro:
    """
    A class for computing data distributions with options for outlier removal and standardization.

    Parameters
    ----------
    data : array-like
        Input data for distribution. Stored as the original and a working copy.
    """

    def __init__(self, data):
        """
        Initialize the DataDistro with input data.

        Converts input to a NumPy array of floats and stores both original and working copies.
        """
        self.original = np.asarray(data)
        self.data = self.original.copy().astype(float)
        self._df = None

    def remove_outliers_zscore(self, thresh=3.0):
        """
        Remove outliers based on the Z-score method.

        Parameters
        ----------
        thresh : float, optional
            Z-score threshold. Values with |z| > thresh are considered outliers and removed.

        Returns
        -------
        self : DataDistro
            Returns self to allow method chaining.
        """
        # Compute mean and standard deviation
        mean, std = np.mean(self.data), np.std(self.data)
        # Compute Z-scores
        z_scores = (self.data - mean) / std
        # Filter data within threshold
        self.data = self.data[np.abs(z_scores) <= thresh]
        return self

    def remove_outliers_iqr(self, k=1.5):
        """
        Remove outliers based on the interquartile range (IQR) method.

        Parameters
        ----------
        k : float, optional
            Multiplier for the IQR. Points outside [Q1 - k*IQR, Q3 + k*IQR] are removed.

        Returns
        -------
        self : DataDistro
            Returns self to allow method chaining.
        """
        # Calculate first and third quartiles
        q1, q3 = np.percentile(self.data, [25, 75])
        iqr = q3 - q1
        # Determine bounds
        lower, upper = q1 - k * iqr, q3 + k * iqr
        # Filter data within bounds
        self.data = self.data[(self.data >= lower) & (self.data <= upper)]
        return self

    def standardize_zscore(self):
        """
        Standardize the data to zero mean and unit variance (Z-score scaling).

        Returns
        -------
        self : DataDistro
            Returns self to allow method chaining.
        """
        self.data = (self.data - np.mean(self.data)) / np.std(self.data)
        return self

    def trim_extremes(self, minx=None, maxx=None):
        """
        Remove a fixed number or percentage of the smallest and largest values.

        Parameters
        ----------
        minx : float, optional
        minx : float, optional

        Returns
        -------
        self : DataDistro
            Returns self to allow method chaining.

        """

        maxx = maxx if maxx is not None else self.data.max()
        minx = minx if minx is not None else self.data.min()

        # Trim left and right
        self.data = self.data[(self.data <= maxx) & (self.data >= minx)]

        return self


    def remove_extremes(self, left=None, right=None):
        """
        Remove a the left and right extremes

        Parameters
        ----------
        left : int, optional
            number of elements to remove on the left
        right : int, optional
            Exact number of elements to remove from each end.

        Returns
        -------
        self : DataDistro
            Returns self to allow method chaining.

        """
        data_sorted = np.sort(self.data)
        n = len(data_sorted)

        if left is None:
            left = 0
        if right is None:
            right = 1
        # Trim left and right
        self.data = data_sorted[left:-right]
        return self

    def remove_top_bottom(self, top=None, bottom=None):
        """
        Remove a the left and right extremes

        Parameters
        ----------
        left : int, optional
            number of elements to remove on the left
        right : int, optional
            Exact number of elements to remove from each end.

        Returns
        -------
        self : DataDistro
            Returns self to allow method chaining.

        """
        xmin = self.data.min()
        xmax = self.data.max()
        if self._df is None:
            df = self.distribution(N=50, minx=xmin, norm=True)
        df = self._df
        x = df.index
        y = df['density'].values

        top_x = xmin
        if top is not None:
            top_x = np.interp(top, y[::-1], x[::-1])
        bottom_x = xmax
        if top is not None:
            bottom_x = np.interp(bottom, y[::-1], x[::-1])

        # Trim left and right
        self.data = self.data[(self.data <= bottom_x) & (self.data >= top_x)]

        return self

    def standardize_minmax(self, feature_range=(0, 1)):
        """
        Scale the data to a specified range [min, max] (min-max scaling).

        Parameters
        ----------
        feature_range : tuple of float, optional
            Desired range of transformed data. Default is (0, 1).

        Returns
        -------
        self : DataDistro
            Returns self to allow method chaining.
        """
        min_val, max_val = feature_range
        dmin, dmax = np.min(self.data), np.max(self.data)
        # Compute scaling factor
        scale = (max_val - min_val) / (dmax - dmin)
        # Apply transformation
        self.data = min_val + (self.data - dmin) * scale
        return self

    def distribution(self, dx=0.001, maxx=None, minx=None, log=False, N=None, norm=False):
        """
        Compute the histogram distribution of the current data.

        Dispatches to linear or logarithmic bin calculation helper methods.

        Parameters
        ----------
        dx : float, optional
            Bin width for linear bins or log-step for logarithmic bins.
        maxx : float, optional
            Upper bound for binning; defaults to max(data).
        minx : float, optional
            Lower bound for binning; defaults to min(data).
        log : bool, optional
            If True, use logarithmic binning.
        N : int, optional
            Explicit number of bins; overrides dx-based calculation if provided.
        norm : bool, optional
            If True, return a density (normalized) distribution instead of raw counts.

        Returns
        -------
        pd.DataFrame
            A DataFrame indexed by bin centers, with columns:
            - 'count' for raw counts (if norm=False)
            - 'density' for normalized density (if norm=True)
        """
        # Determine bin bounds
        data = self.data
        maxx = maxx if maxx is not None else data.max()
        minx = minx if minx is not None else data.min()

        # Compute bins and counts via helper methods
        if log:
            centers, counts, width = self._compute_log_bins(data, minx, maxx, dx, N)
        else:
            centers, counts, width = self._compute_linear_bins(data, minx, maxx, dx, N)

        # Build output DataFrame
        df = pd.DataFrame(counts, index=centers, columns=['count'])

        # Store for sampling
        total = counts.sum()
        if total == 0:
            raise ValueError("Empty distribution: no data in bins.")
        self._last_edges = centers
        self._last_probabilities = counts / total

        # Normalize to density if requested
        if norm:
            df['density'] = df['count'] / (df['count'].sum() * width)
            df = df.drop(columns=['count'])

        self._df = df

        return df

    def _compute_linear_bins(self, data, minx, maxx, dx, N):
        """
        Helper to compute linear histogram bins.

        Parameters
        ----------
        data : array-like
            Data to bin.
        minx, maxx : float
            Binning bounds.
        dx : float
            Bin width.
        N : int or None
            Number of bins; if None, computed from dx.

        Returns
        -------
        centers : ndarray
            Bin center positions.
        counts : ndarray
            Raw bin counts.
        dx : float
            Final bin width used.
        """
        # Determine edges from N or dx
        if N is not None:
            edges = np.linspace(minx, maxx, N + 1)
            dx = edges[1] - edges[0]
        else:
            N = int((maxx - minx) / dx)
            edges = np.arange(minx, maxx + dx, dx)
        # Compute histogram
        counts, edges = np.histogram(data, bins=edges)
        # Shift centers: integer data gets full dx, floats get half-dx offset
        shift = dx if data.dtype == 'int64' else dx / 2
        centers = edges[:-1] + shift
        return centers, counts, dx

    def _compute_log_bins(self, data, minx, maxx, dx, N):
        """
        Helper to compute logarithmic histogram bins.

        Parameters
        ----------
        data : array-like
            Data to bin.
        minx, maxx : float
            Binning bounds.
        dx : float
            Approximate log-step size (in log-space).
        N : int or None
            Number of bins; if None, computed from dx.

        Returns
        -------
        centers : ndarray
            Geometric mean positions of each bin.
        densities : ndarray
            Counts divided by bin widths (approximate densities).
        mean_width : float
            Average bin width (for normalization in distribution()).
        """
        # Determine number of bins from log-step if needed
        if N is None:
            N = int((np.log(maxx) - np.log(minx)) / dx)
        # Create geometric sequence of edges
        edges = np.geomspace(minx, maxx, num=N, endpoint=True)
        counts, _ = np.histogram(data, bins=edges)
        # Bin centers as midpoint in linear space
        centers = (edges[:-1] + edges[1:]) / 2
        # Compute widths and densities
        widths = np.diff(edges)
        densities = counts / widths
        # Return density vector for uniform interface
        return centers, densities, widths

    def reset(self):
        """
        Reset the working data to the original input.

        Returns
        -------
        self : DataDistro
            Returns self to allow method chaining.
        """
        self.data = self.original.copy().astype(float)
        return self

    def sample(self, size=1, random_state=None):
        """
        Draw random samples from the last computed distribution.

        Parameters
        ----------
        size : int
            Number of samples to draw.
        random_state : int or None
            Random seed for reproducibility.

        Returns
        -------
        samples : ndarray
            Random values drawn according to the histogram probabilities.

        Raises
        ------
        ValueError
            If no distribution has been computed yet.
        """
        if self._last_edges is None or self._last_probabilities is None:
            raise ValueError("No distribution stored: call distribution() first.")
        rng = np.random.RandomState(random_state) if random_state is not None else np.random
        # Choose bins
        bins = rng.choice(len(self._last_probabilities), size=size, p=self._last_probabilities)
        # Sample uniformly within each chosen bin
        samples = []
        for b in bins:
            left, right = self._last_edges[b], self._last_edges[b + 1]
            samples.append(left + rng.rand() * (right - left))
        return np.array(samples)

    def fit_gaussian(self):
        """
        Fit a Gaussian (normal) distribution to the current data.

        Uses maximum likelihood to estimate mean and standard deviation.

        Returns
        -------
        params : tuple
            (mu, sigma) estimated from the data.
        """
        # Fit parameters
        mu, sigma = norm.fit(self.data)
        self._gaussian_params = (mu, sigma)

        # Compute empirical density
        if self._df is None:
            df = self.distribution(N=50, norm=True)
        x = self._df.index.values
        y_true = self._df['density'].values
        # Gaussian pdf
        y_pred = norm.pdf(x, mu, sigma)
        # R²
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - y_true.mean()) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
        self._gaussian_quality = r2
        return {'mu': mu, 'sigma': sigma, 'r2': r2}

    def fit_powerlaw(self, xmin=None):
        """
        Fit a continuous power-law to the tail of the distribution.

        Parameters
        ----------
        xmin : float, optional
            Lower cutoff for power-law behavior. Defaults to data.min().

        Returns
        -------
        params : tuple
            (alpha, xmin) where alpha is the exponent.

        Raises
        ------
        ValueError
            If data has non-positive values or insufficient tail points.
        """

        data = self.data
        if len(data) > 5:
            xmin = xmin if xmin is not None else data.min()
            tail = data[data >= xmin]
            if len(tail) < 2 or np.any(tail <= 0):
                raise ValueError("Insufficient tail data.")
            # MLE alpha
            n = len(tail)
            alpha = 1 + n / np.sum(np.log(tail / xmin))
            self._powerlaw_params = (alpha, xmin)

            # Empirical density on tail bins
            if self._df is None:
                df = self.distribution(N=50, minx=xmin, norm=True)
            x = self._df.index.values
            y_true = self._df['density'].values
            # Power-law pdf: (alpha-1)/xmin * (x/xmin)^(-alpha)
            y_pred = (alpha - 1) / xmin * (x / xmin) ** (-alpha)
            ss_res = np.sum((y_true - y_pred) ** 2)
            ss_tot = np.sum((y_true - y_true.mean()) ** 2)
            r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
            self._powerlaw_quality = r2
            return {'alpha': alpha, 'xmin': xmin, 'r2': r2}
        else:
            return {'alpha': np.nan, 'xmin': np.nan, 'r2': np.nan}

    def fit_lognormal(self):
        """Fit log-normal distribution and compute R² on normalized histogram density."""
        # Fit parameters: shape (s), loc, scale
        data = self.data
        if np.any(data <= 0):
            raise ValueError("Data must be positive to fit log-normal.")
        s, loc, scale = lognorm.fit(data, floc=0)
        self._lognormal_params = (s, loc, scale)
        # Ensure histogram computed
        if self._df is None:
            self.distribution(N=50, norm=True)
        x = self._df.index.values
        y_true = self._df['density'].values
        # Lognormal PDF
        y_pred = lognorm.pdf(x, s, loc=loc, scale=scale)
        ss_res = np.sum((y_true - y_pred)**2)
        ss_tot = np.sum((y_true - y_true.mean())**2)
        r2 = 1 - ss_res/ss_tot if ss_tot>0 else np.nan
        self._lognormal_quality = r2
        return {'shape':s, 'loc':loc, 'scale':scale, 'r2':r2}