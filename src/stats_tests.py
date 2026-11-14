"""Small hand-written statistical tests (numpy/scipy only, no statsmodels).

- adf_test:           Augmented Dickey-Fuller unit-root test (constant only).
- diebold_mariano:    Diebold-Mariano test of equal predictive accuracy, with the
                      Harvey-Leybourne-Newbold small-sample correction.
"""
import numpy as np
from scipy import stats

# MacKinnon asymptotic critical values, constant-only case ("c").
ADF_CRITICAL_VALUES = {'1%': -3.43, '5%': -2.86, '10%': -2.57}


def _adf_regression(dy, y_lag, p, start):
    """OLS of dy_t on [1, y_{t-1}, dy_{t-1..t-p}] for t >= start (indices into dy).

    Returns (ssr, nobs, t_stat_of_gamma).
    """
    n = len(dy)
    t = np.arange(start, n)
    cols = [np.ones(len(t)), y_lag[t]]
    for i in range(1, p + 1):
        cols.append(dy[t - i])
    Z = np.column_stack(cols)
    target = dy[t]
    beta, _, rank, _ = np.linalg.lstsq(Z, target, rcond=None)
    resid = target - Z @ beta
    ssr = float(resid @ resid)
    nobs, k = Z.shape
    dof = nobs - k
    if dof <= 0 or rank < k or ssr <= 0:
        return ssr, nobs, np.nan
    sigma2 = ssr / dof
    cov = sigma2 * np.linalg.inv(Z.T @ Z)
    return ssr, nobs, float(beta[1] / np.sqrt(cov[1, 1]))


def adf_test(x, max_lag=None):
    """Augmented Dickey-Fuller test with a constant (no trend).

    Regression: dy_t = c + gamma*y_{t-1} + sum_{i=1..p} phi_i*dy_{t-i} + e_t.
    H0: unit root (gamma = 0, non-stationary); H1: stationary (gamma < 0).
    The lag order p is chosen by AIC over 0..max_lag, all candidates being fitted on
    the same sample; the chosen model is then re-estimated on its own full sample.
    max_lag defaults to int(12 * (n / 100) ** 0.25).

    The statistic is compared with the MacKinnon asymptotic critical values for the
    constant-only case (1%: -3.43, 5%: -2.86, 10%: -2.57). p-values are NOT computed.
    Reject H0 (evidence of stationarity) when the statistic is below the critical value.

    Returns a dict: stat, lags, nobs, crit_1, crit_5, crit_10, reject_5pct.
    """
    y = np.asarray(x, dtype=float)
    y = y[~np.isnan(y)]
    n = len(y)
    if max_lag is None:
        max_lag = int(12 * (n / 100) ** 0.25)
    # keep enough observations for the regression
    max_lag = max(0, min(max_lag, (n - 3) // 2 - 1))
    if n < 10:
        raise ValueError("series too short for an ADF test")

    dy = np.diff(y)
    y_lag = y[:-1]

    # AIC on a common sample (first max_lag observations dropped)
    best_p, best_aic = 0, np.inf
    for p in range(max_lag + 1):
        ssr, nobs, _ = _adf_regression(dy, y_lag, p, max_lag)
        if ssr <= 0 or nobs <= 0:
            continue
        aic = nobs * np.log(ssr / nobs) + 2 * (p + 2)
        if aic < best_aic:
            best_aic, best_p = aic, p

    _, nobs, tstat = _adf_regression(dy, y_lag, best_p, best_p)
    return {
        'stat': tstat,
        'lags': best_p,
        'nobs': nobs,
        'crit_1': ADF_CRITICAL_VALUES['1%'],
        'crit_5': ADF_CRITICAL_VALUES['5%'],
        'crit_10': ADF_CRITICAL_VALUES['10%'],
        'reject_5pct': bool(tstat < ADF_CRITICAL_VALUES['5%']) if np.isfinite(tstat) else False,
    }


def diebold_mariano(y_true, pred_1, pred_2, h=1):
    """Diebold-Mariano test of equal predictive accuracy, squared-error loss.

    Loss differential d_t = (y - pred_1)^2 - (y - pred_2)^2.
    SIGN CONVENTION: a NEGATIVE statistic means forecast 1 has the smaller squared
    error (pred_1 is better); a positive one means pred_2 is better.
    H0: E[d_t] = 0, two-sided.

    The variance of the mean of d uses Newey-West (Bartlett weights) with h-1 lags.
    The statistic is multiplied by the Harvey-Leybourne-Newbold factor
    sqrt((n + 1 - 2h + h(h-1)/n) / n) and compared with a t distribution with n-1
    degrees of freedom.

    Returns a dict: stat, p_value, n, h, mean_loss_diff. If the loss differential is
    identically (numerically) zero, or the variance estimate is not positive, stat and
    p_value are NaN.
    """
    y = np.asarray(y_true, dtype=float)
    e1 = y - np.asarray(pred_1, dtype=float)
    e2 = y - np.asarray(pred_2, dtype=float)
    d = e1 ** 2 - e2 ** 2
    d = d[np.isfinite(d)]
    n = len(d)
    out = {'stat': np.nan, 'p_value': np.nan, 'n': n, 'h': h,
           'mean_loss_diff': float(d.mean()) if n else np.nan}
    if n < 3 or h < 1 or h >= n:
        return out
    dbar = d.mean()
    dc = d - dbar
    var = float(dc @ dc) / n
    for k in range(1, h):
        gamma_k = float(dc[k:] @ dc[:-k]) / n
        var += 2 * (1 - k / h) * gamma_k
    scale = max(float(np.abs(d).max()), 1e-300)
    if not np.isfinite(var) or var <= (1e-12 * scale) ** 2:
        return out
    dm = dbar / np.sqrt(var / n)
    stat = dm * np.sqrt((n + 1 - 2 * h + h * (h - 1) / n) / n)  # HLN correction
    out['stat'] = float(stat)
    out['p_value'] = float(2 * stats.t.sf(abs(stat), df=n - 1))
    return out
