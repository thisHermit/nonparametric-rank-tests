import numpy as np
from collections import namedtuple
from scipy import special
from scipy import stats
from scipy.stats import norm, chi2
import numpy as np
from scipy.stats import chi2_contingency
from scipy.stats import norm, chi2
from scipy.stats import ttest_ind

def _order_ranks(ranks, j):
    # Reorder ascending order `ranks` according to `j`
    ordered_ranks = np.empty(j.shape, dtype=ranks.dtype)
    np.put_along_axis(ordered_ranks, j, ranks, axis=-1)
    return ordered_ranks

def _rankdata(x, method, return_ties=False):
    shape = x.shape

    kind = 'mergesort' if method == 'ordinal' else 'quicksort'
    j = np.argsort(x, axis=-1, kind=kind)
    ordinal_ranks = np.broadcast_to(np.arange(1, shape[-1]+1, dtype=int), shape)

    if method == 'ordinal':
        return _order_ranks(ordinal_ranks, j)  # never return ties

    y = np.take_along_axis(x, j, axis=-1)
    i = np.concatenate([np.ones(shape[:-1] + (1,), dtype=np.bool_),
                       y[..., :-1] != y[..., 1:]], axis=-1)

    indices = np.arange(y.size)[i.ravel()]
    counts = np.diff(indices, append=y.size)

    if method == 'min':
        ranks = ordinal_ranks[i]
    elif method == 'max':
        ranks = ordinal_ranks[i] + counts - 1
    elif method == 'average':
        ranks = ordinal_ranks[i] + (counts - 1)/2
    elif method == 'dense':
        ranks = np.cumsum(i, axis=-1)[i]

    ranks = np.repeat(ranks, counts).reshape(shape)
    ranks = _order_ranks(ranks, j)

    if return_ties:
        t = np.zeros(shape, dtype=float)
        t[i] = counts
        return ranks, t
    return ranks

def _broadcast_concatenate(x, y, axis):
    '''Broadcast then concatenate arrays, leaving concatenation axis last'''
    x = np.moveaxis(x, axis, -1)
    y = np.moveaxis(y, axis, -1)
    z = np.broadcast(x[..., 0], y[..., 0])
    x = np.broadcast_to(x, z.shape + (x.shape[-1],))
    y = np.broadcast_to(y, z.shape + (y.shape[-1],))
    z = np.concatenate((x, y), axis=-1)
    return x, y, z


class _MWU:
    '''Distribution of MWU statistic under the null hypothesis'''

    def __init__(self, n1, n2):
        self._reset(n1, n2)

    def set_shapes(self, n1, n2):
        n1, n2 = min(n1, n2), max(n1, n2)
        if (n1, n2) == (self.n1, self.n2):
            return

        self.n1 = n1
        self.n2 = n2
        self.s_array = np.zeros(0, dtype=int)
        self.configurations = np.zeros(0, dtype=np.uint64)

    def reset(self):
        self._reset(self.n1, self.n2)

    def _reset(self, n1, n2):
        self.n1 = None
        self.n2 = None
        self.set_shapes(n1, n2)

    def pmf(self, k):
        pmfs = self.build_u_freqs_array(np.max(k))
        return pmfs[k]

    def cdf(self, k):
        '''Cumulative distribution function'''
        pmfs = self.build_u_freqs_array(np.max(k))
        cdfs = np.cumsum(pmfs)
        return cdfs[k]

    def sf(self, k):
        '''Survival function'''
        kc = np.asarray(self.n1*self.n2 - k)  # complement of k
        i = k < kc
        if np.any(i):
            kc[i] = k[i]
            cdfs = np.asarray(self.cdf(kc))
            cdfs[i] = 1. - cdfs[i] + self.pmf(kc[i])
        else:
            cdfs = np.asarray(self.cdf(kc))
        return cdfs[()]

    # build_sigma_array and build_u_freqs_array adapted from code
    # by @toobaz with permission. Thanks to @andreasloe for the suggestion.
    # See https://github.com/scipy/scipy/pull/4933#issuecomment-1898082691
    def build_sigma_array(self, a):
        n1, n2 = self.n1, self.n2
        if a + 1 <= self.s_array.size:
            return self.s_array[1:a+1]

        s_array = np.zeros(a + 1, dtype=int)

        for d in np.arange(1, n1 + 1):
            # All multiples of d, except 0:
            indices = np.arange(d, a + 1, d)
            # \epsilon_d = 1:
            s_array[indices] += d

        for d in np.arange(n2 + 1, n2 + n1 + 1):
            # All multiples of d, except 0:
            indices = np.arange(d, a + 1, d)
            # \epsilon_d = -1:
            s_array[indices] -= d

        # We don't need 0:
        self.s_array = s_array
        return s_array[1:]

    def build_u_freqs_array(self, maxu):
        """
        Build all the array of frequencies for u from 0 to maxu.
        Assumptions:
          n1 <= n2
          maxu <= n1 * n2 / 2
        """
        n1, n2 = self.n1, self.n2
        total = special.binom(n1 + n2, n1)

        if maxu + 1 <= self.configurations.size:
            return self.configurations[:maxu + 1] / total

        s_array = self.build_sigma_array(maxu)

        # Start working with ints, for maximum precision and efficiency:
        configurations = np.zeros(maxu + 1, dtype=np.uint64)
        configurations_is_uint = True
        uint_max = np.iinfo(np.uint64).max
        # How many ways to have U=0? 1
        configurations[0] = 1

        for u in np.arange(1, maxu + 1):
            coeffs = s_array[u - 1::-1]
            new_val = np.dot(configurations[:u], coeffs) / u
            if new_val > uint_max and configurations_is_uint:
                # OK, we got into numbers too big for uint64.
                # So now we start working with floats.
                # By doing this since the beginning, we would have lost precision.
                # (And working on python long ints would be unbearably slow)
                configurations = configurations.astype(float)
                configurations_is_uint = False
            configurations[u] = new_val

        self.configurations = configurations
        return configurations / total


_mwu_state = _MWU(0, 0)


def _get_mwu_z(U, n1, n2, t, axis=0, continuity=True):
    '''Standardized MWU statistic'''
    # Follows mannwhitneyu [2]
    mu = n1 * n2 / 2
    n = n1 + n2

    # Tie correction according to [2], "Normal approximation and tie correction"
    # "A more computationally-efficient form..."
    tie_term = (t**3 - t).sum(axis=-1)
    s = np.sqrt(n1*n2/12 * ((n + 1) - tie_term/(n*(n-1))))

    numerator = U - mu

    # Continuity correction.
    # Because SF is always used to calculate the p-value, we can always
    # _subtract_ 0.5 for the continuity correction. This always increases the
    # p-value to account for the rest of the probability mass _at_ q = U.
    if continuity:
        numerator -= 0.5

    # no problem evaluating the norm SF at an infinity
    with np.errstate(divide='ignore', invalid='ignore'):
        z = numerator / s
    return z


def _mwu_input_validation(x, y, use_continuity, alternative, axis, method):
    ''' Input validation and standardization for mannwhitneyu '''
    # Would use np.asarray_chkfinite, but infs are OK
    x, y = np.atleast_1d(x), np.atleast_1d(y)
    if np.isnan(x).any() or np.isnan(y).any():
        raise ValueError('`x` and `y` must not contain NaNs.')
    if np.size(x) == 0 or np.size(y) == 0:
        raise ValueError('`x` and `y` must be of nonzero size.')

    bools = {True, False}
    if use_continuity not in bools:
        raise ValueError(f'`use_continuity` must be one of {bools}.')

    alternatives = {"two-sided", "less", "greater"}
    alternative = alternative.lower()
    if alternative not in alternatives:
        raise ValueError(f'`alternative` must be one of {alternatives}.')

    axis_int = int(axis)
    if axis != axis_int:
        raise ValueError('`axis` must be an integer.')

    if not isinstance(method, stats.PermutationMethod):
        methods = {"asymptotic", "exact", "auto"}
        method = method.lower()
        if method not in methods:
            raise ValueError(f'`method` must be one of {methods}.')

    return x, y, use_continuity, alternative, axis_int, method


def _mwu_choose_method(n1, n2, ties):
    """Choose method 'asymptotic' or 'exact' depending on input size, ties"""

    # if both inputs are large, asymptotic is OK
    if n1 > 8 and n2 > 8:
        return "asymptotic"

    # if there are any ties, asymptotic is preferred
    if ties:
        return "asymptotic"

    return "exact"


MannwhitneyuResult = namedtuple('MannwhitneyuResult', ('statistic', 'pvalue'))


def mannwhitneyu(x, y, use_continuity=True, alternative="two-sided",
                 axis=0, method="auto"):

    x, y, use_continuity, alternative, axis_int, method = (
        _mwu_input_validation(x, y, use_continuity, alternative, axis, method))

    x, y, xy = _broadcast_concatenate(x, y, axis)

    n1, n2 = x.shape[-1], y.shape[-1]

    # Follows [2]
    ranks, t = _rankdata(xy, 'average', return_ties=True)  # method 2, step 1
    R1 = ranks[..., :n1].sum(axis=-1)                      # method 2, step 2
    U1 = R1 - n1*(n1+1)/2                                  # method 2, step 3
    U2 = n1 * n2 - U1                                      # as U1 + U2 = n1 * n2

    if alternative == "greater":
        U, f = U1, 1  # U is the statistic to use for p-value, f is a factor
    elif alternative == "less":
        U, f = U2, 1  # Due to symmetry, use SF of U2 rather than CDF of U1
    else:
        U, f = np.maximum(U1, U2), 2  # multiply SF by two for two-sided test

    if method == "auto":
        method = _mwu_choose_method(n1, n2, np.any(t > 1))

    if method == "exact":
        _mwu_state.set_shapes(n1, n2)
        p = _mwu_state.sf(U.astype(int))
    elif method == "asymptotic":
        z = _get_mwu_z(U, n1, n2, t, continuity=use_continuity)
        p = stats.norm.sf(z)
    else:  # `PermutationMethod` instance (already validated)
        def statistic(x, y, axis):
            return mannwhitneyu(x, y, use_continuity=use_continuity,
                                alternative=alternative, axis=axis,
                                method="asymptotic").statistic

        res = stats.permutation_test((x, y), statistic, axis=axis,
                                     **method._asdict(), alternative=alternative)
        p = res.pvalue
        f = 1

    p *= f

    # Ensure that test statistic is not greater than 1
    # This could happen for exact test when U = m*n/2
    p = np.clip(p, 0, 1)

    return MannwhitneyuResult(U1, p)

def van_der_waerden_test(*samples):
    """
    Perform the Van der Waerden test for k independent samples.

    Parameters:
    *samples : array-like
        Variable number of arrays, each representing an independent sample.

    Returns:
    statistic : float
        The test statistic.
    p_value : float
        The p-value of the test.
    """
    # Combine all samples into a single array
    data = np.concatenate(samples)
    n_total = len(data)
    k = len(samples)
    sample_sizes = [len(sample) for sample in samples]

    # Rank the combined data
    ranks = np.argsort(np.argsort(data)) + 1

    # Compute normal scores
    normal_scores = norm.ppf((ranks - 0.5) / n_total)

    # Calculate the sum of normal scores for each sample
    sum_scores = []
    start = 0
    for size in sample_sizes:
        sum_scores.append(np.sum(normal_scores[start:start + size]))
        start += size

    # Compute the test statistic
    sum_scores = np.array(sum_scores)
    n_i = np.array(sample_sizes)
    A_bar = sum_scores / n_i
    A_bar_total = np.sum(sum_scores) / n_total
    S_square = np.sum((normal_scores - A_bar_total) ** 2) / (n_total - 1)
    T = np.sum(n_i * (A_bar - A_bar_total) ** 2) / S_square

    # Calculate the p-value
    p_value = 1 - chi2.cdf(T, k - 1)

    return T, p_value

def median_test(*samples):
    """
    Perform the Median Test for k independent samples.

    Parameters:
    *samples : array-like
        Variable number of arrays, each representing an independent sample.

    Returns:
    statistic : float
        The test statistic.
    p_value : float
        The p-value of the test.
    """
    # Combine all samples into a single array
    data = np.concatenate(samples)
    grand_median = np.median(data)

    # Create a contingency table
    contingency_table = []
    for sample in samples:
        above_median = np.sum(sample > grand_median)
        below_or_equal_median = np.sum(sample <= grand_median)
        contingency_table.append([above_median, below_or_equal_median])

    # Perform the chi-squared test
    contingency_table = np.array(contingency_table)
    chi2, p_value, _, _ = chi2_contingency(contingency_table, correction=False)

    return chi2, p_value

def fisher_yates_terry_hoeffding_test(*samples):
    """
    Perform the Fisher-Yates-Terry-Hoeffding (FYTH) test for k independent samples.

    Parameters:
    *samples : array-like
        Variable number of arrays, each representing an independent sample.

    Returns:
    statistic : float
        The test statistic.
    p_value : float
        The p-value of the test.
    """
    # Combine all samples into a single array
    data = np.concatenate(samples)
    n_total = len(data)
    k = len(samples)
    sample_sizes = [len(sample) for sample in samples]

    # Rank the combined data
    ranks = np.argsort(np.argsort(data)) + 1

    # Compute expected normal scores
    expected_normal_scores = norm.ppf((ranks - 0.5) / n_total)

    # Calculate the sum of expected normal scores for each sample
    sum_scores = []
    start = 0
    for size in sample_sizes:
        sum_scores.append(np.sum(expected_normal_scores[start:start + size]))
        start += size

    # Compute the test statistic
    sum_scores = np.array(sum_scores)
    n_i = np.array(sample_sizes)
    A_bar = sum_scores / n_i
    A_bar_total = np.sum(sum_scores) / n_total
    S_square = np.sum((expected_normal_scores - A_bar_total) ** 2) / (n_total - 1)
    T = np.sum(n_i * (A_bar - A_bar_total) ** 2) / S_square

    # Calculate the p-value
    p_value = 1 - chi2.cdf(T, k - 1)

    return T, p_value

def permutation_test(sample1, sample2, num_permutations=10000, alternative='two-sided'):
    """
    Perform a permutation test to compare two independent samples.

    Parameters:
    sample1 : array-like
        First sample data.
    sample2 : array-like
        Second sample data.
    num_permutations : int, optional
        Number of permutations to perform (default is 10,000).
    alternative : {'two-sided', 'greater', 'less'}, optional
        Defines the alternative hypothesis (default is 'two-sided').

    Returns:
    p_value : float
        The p-value of the test.
    observed_diff : float
        The observed difference in means between the two samples.
    """
    # Combine the data
    combined = np.concatenate([sample1, sample2])
    n1 = len(sample1)
    observed_diff = np.mean(sample1) - np.mean(sample2)

    # Generate permutation samples
    perm_diffs = np.zeros(num_permutations)
    for i in range(num_permutations):
        np.random.shuffle(combined)
        perm_sample1 = combined[:n1]
        perm_sample2 = combined[n1:]
        perm_diffs[i] = np.mean(perm_sample1) - np.mean(perm_sample2)

    # Calculate p-value
    if alternative == 'two-sided':
        p_value = np.mean(np.abs(perm_diffs) >= np.abs(observed_diff))
    elif alternative == 'greater':
        p_value = np.mean(perm_diffs >= observed_diff)
    elif alternative == 'less':
        p_value = np.mean(perm_diffs <= observed_diff)
    else:
        raise ValueError("alternative must be 'two-sided', 'greater', or 'less'")

    return observed_diff, p_value

def two_sample_t_test(sample1, sample2, equal_var=True, alternative='two-sided'):
    """
    Perform a two-sample t-test to compare the means of two independent samples.

    Parameters:
    sample1 : array-like
        First sample data.
    sample2 : array-like
        Second sample data.
    equal_var : bool, optional
        If True (default), perform a standard independent 2 sample t-test that assumes equal population variances.
        If False, perform Welch’s t-test, which does not assume equal population variances.
    alternative : {'two-sided', 'less', 'greater'}, optional
        Defines the alternative hypothesis (default is 'two-sided').

    Returns:
    statistic : float
        The calculated t-statistic.
    p_value : float
        The p-value of the test.
    """
    # Perform the t-test
    t_stat, p_value = ttest_ind(sample1, sample2, equal_var=equal_var, alternative=alternative)
    return t_stat, p_value
