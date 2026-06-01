import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import warnings
warnings.filterwarnings('ignore')
from pandas.tseries.offsets import MonthEnd
from itertools import combinations
from scipy.optimize import minimize

plt.rcParams['figure.figsize'] = (12, 5)
plt.rcParams['axes.grid'] = True
plt.rcParams['grid.alpha'] = 0.3

# ══════════════════════════════════════════════════════════════════════════════
# LOAD DATA
# ══════════════════════════════════════════════════════════════════════════════

FILE = "Final_Project2_Data.xlsx"

# ── Returns ───────────────────────────────────────────────────────────────────
returns = (
    pd.read_excel(FILE)
    .rename(columns={"Total returns in local currency": "Date"})
    .set_index("Date")
    .sort_index()
)

assert returns.isnull().sum().sum() == 0, "Returns contain missing values"

COUNTRIES = list(returns.columns)

# ── PE Ratios ─────────────────────────────────────────────────────────────────
pe_clean = (
    pd.read_excel(FILE, sheet_name="Country equity PE ratios")
    .pipe(lambda df: df.rename(columns={df.columns[0]: "Date"}))
    .set_index("Date")
    .sort_index()
    .reindex(returns.index)
    .ffill()
    .dropna(how="any")
)

# remove nonpositive PE values if any, then forward-fill again
pe_clean = pe_clean.where(pe_clean > 0).ffill().dropna(how="any")

assert pe_clean.isnull().sum().sum() == 0, "PE data contain missing values"
assert list(pe_clean.columns) == COUNTRIES , "pe_data.columns != returns.columns" 



# ══════════════════════════════════════════════════════════════════════════════
# DEFINE TRAIN / BACKTEST PERIODS CLEANLY
# ══════════════════════════════════════════════════════════════════════════════

# Backtest window: fixed out-of-sample period
BACKTEST_START = pd.Timestamp("2016-01-31")
BACKTEST_END   = pd.Timestamp("2025-12-31")

# Lookback / lag requirements
RETURN_LOOKBACK = 36
PE_LOOKBACK     = 36

# If a PE signal uses PE_{t-1-L}, then it needs 36 months plus 1 lag.
# Use 37-month warmup if you want all PE-based signals to be safely computable.
PE_TOTAL_WARMUP = PE_LOOKBACK + 1

# ══════════════════════════════════════════════════════════════════════════════
# 1. FULL HISTORIES
# ══════════════════════════════════════════════════════════════════════════════

full_history_returns = returns.copy()
full_history_pe      = pe_clean.copy()

# Make sure PE is clean and aligned to returns index
full_history_pe = (
    full_history_pe
    .reindex(full_history_returns.index)
    .ffill()
    .where(lambda df: df > 0)
    .ffill()
    .dropna(how="any")
)

# ══════════════════════════════════════════════════════════════════════════════
# 2. BACKTEST DATASETS
# ══════════════════════════════════════════════════════════════════════════════

backtest_returns = full_history_returns.loc[BACKTEST_START:BACKTEST_END].copy()
backtest_pe      = full_history_pe.loc[BACKTEST_START:BACKTEST_END].copy()

# Force common backtest index across returns and PE
common_backtest_idx = (
    backtest_returns.index
    .intersection(backtest_pe.index)
    .sort_values()
)

backtest_returns = backtest_returns.loc[common_backtest_idx]
backtest_pe      = backtest_pe.loc[common_backtest_idx]

assert backtest_returns.index.equals(backtest_pe.index), (
    "Backtest returns and PE indexes do not match."
)

# This is the common target date object for all out-of-sample testing
backtest_dates = backtest_returns.copy()

# ══════════════════════════════════════════════════════════════════════════════
# 3. FULL TRAINING HISTORIES
#    These are all data available before the backtest starts.
#    They are used as historical pools for computing training signals.
# ══════════════════════════════════════════════════════════════════════════════

full_train_history_returns = full_history_returns.loc[:BACKTEST_START - MonthEnd(1)].copy()
full_train_history_pe      = full_history_pe.loc[:BACKTEST_START - MonthEnd(1)].copy()

# ══════════════════════════════════════════════════════════════════════════════
# 4. TRAINING EVALUATION DATASETS AFTER WARMUP
#    These are the dates on which we actually evaluate/optimize factors.
# ══════════════════════════════════════════════════════════════════════════════

# Return-based training begins after 36 months of return history
return_train_start = full_train_history_returns.index.min() + MonthEnd(RETURN_LOOKBACK)
return_train_end   = BACKTEST_START - MonthEnd(1)

train_returns = full_train_history_returns.loc[return_train_start:return_train_end].copy()

# PE-based training begins after enough PE history for all PE-based signals
pe_train_start = full_train_history_pe.index.min() + MonthEnd(PE_TOTAL_WARMUP)
pe_train_end   = BACKTEST_START - MonthEnd(1)

train_pe = full_train_history_pe.loc[pe_train_start:pe_train_end].copy()

# Common carry/value training dates: must exist in returns and PE
carry_train_idx = (
    train_returns.index
    .intersection(train_pe.index)
    .sort_values()
)

carry_train_returns = train_returns.loc[carry_train_idx]
carry_train_pe      = train_pe.loc[carry_train_idx]

assert carry_train_returns.index.equals(carry_train_pe.index), (
    "Carry training returns and PE indexes do not match."
)

# These are the target-date objects to pass into functions
train_dates       = train_returns.copy()          # for momentum / vol
carry_train_dates = carry_train_returns.copy()    # for value / carry
backtest_dates    = backtest_returns.copy()       # for all backtests

# ══════════════════════════════════════════════════════════════════════════════
# 5. SANITY CHECKS
# ══════════════════════════════════════════════════════════════════════════════

assert train_dates.index[-1] < backtest_dates.index[0], (
    "Overlap between return train and backtest."
)

assert carry_train_dates.index[-1] < backtest_dates.index[0], (
    "Overlap between carry train and backtest."
)

# Check enough return history before first return training date
first_return_train_date = train_dates.index[0]
return_history_available = full_history_returns.loc[:first_return_train_date - MonthEnd(1)]

assert len(return_history_available) >= RETURN_LOOKBACK, (
    "Not enough return history before first return training date."
)

# Check enough PE history before first carry training date
first_carry_train_date = carry_train_dates.index[0]
pe_history_available = full_history_pe.loc[:first_carry_train_date - MonthEnd(1)]

assert len(pe_history_available) >= PE_TOTAL_WARMUP, (
    "Not enough PE history before first carry training date."
)

# Check enough PE history before first carry backtest date
first_backtest_date = backtest_dates.index[0]
pe_history_before_backtest = full_history_pe.loc[:first_backtest_date - MonthEnd(1)]

assert len(pe_history_before_backtest) >= PE_TOTAL_WARMUP, (
    "Not enough PE history before first backtest date."
)

# ══════════════════════════════════════════════════════════════════════════════
# 6. PRINT SUMMARY
# ══════════════════════════════════════════════════════════════════════════════

print("PERIOD SUMMARY")
print("=" * 80)

print("\nFULL HISTORIES")
print(f"  Returns full history:       {full_history_returns.index[0].date()} → {full_history_returns.index[-1].date()} | {len(full_history_returns)} months")
print(f"  PE full history:            {full_history_pe.index[0].date()} → {full_history_pe.index[-1].date()} | {len(full_history_pe)} months")

print("\nTRAINING HISTORIES")
print(f"  Returns train history:      {full_train_history_returns.index[0].date()} → {full_train_history_returns.index[-1].date()} | {len(full_train_history_returns)} months")
print(f"  PE train history:           {full_train_history_pe.index[0].date()} → {full_train_history_pe.index[-1].date()} | {len(full_train_history_pe)} months")

print("\nTRAINING EVALUATION WINDOWS")
print(f"  Momentum/Vol train dates:   {train_dates.index[0].date()} → {train_dates.index[-1].date()} | {len(train_dates)} months")
print(f"  Carry train dates:          {carry_train_dates.index[0].date()} → {carry_train_dates.index[-1].date()} | {len(carry_train_dates)} months")

print("\nBACKTEST WINDOW")
print(f"  Backtest returns:           {backtest_returns.index[0].date()} → {backtest_returns.index[-1].date()} | {len(backtest_returns)} months")
print(f"  Backtest PE:                {backtest_pe.index[0].date()} → {backtest_pe.index[-1].date()} | {len(backtest_pe)} months")
print(f"  Common backtest dates:      {backtest_dates.index[0].date()} → {backtest_dates.index[-1].date()} | {len(backtest_dates)} months")

# ══════════════════════════════════════════════════════════════════════════════
# CHECK FOR MISSING MONTHS
# ══════════════════════════════════════════════════════════════════════════════

def assert_no_missing_months(df, name):
    """
    Assert that a DataFrame has a complete month-end index with no missing months.
    """
    idx = df.index.sort_values()

    expected_idx = pd.date_range(
        start=idx.min(),
        end=idx.max(),
        freq="M"
    )

    missing_months = expected_idx.difference(idx)

    assert len(missing_months) == 0, (
        f"{name} has missing months: "
        f"{[d.date() for d in missing_months]}"
    )

    print(
        f"✓ {name}: no missing months "
        f"({idx.min().date()} → {idx.max().date()} | {len(idx)} months)"
    )


# Check full histories
assert_no_missing_months(full_history_returns, "Full returns history")
assert_no_missing_months(full_history_pe, "Full PE history")

# Check training histories
assert_no_missing_months(full_train_history_returns, "Returns training history")
assert_no_missing_months(full_train_history_pe, "PE training history")

# Check evaluation windows
assert_no_missing_months(train_dates, "Momentum/Vol training dates")
assert_no_missing_months(carry_train_dates, "Carry training dates")
assert_no_missing_months(backtest_dates, "Backtest dates")

# Check aligned backtest panels
assert_no_missing_months(backtest_returns, "Backtest returns")
assert_no_missing_months(backtest_pe, "Backtest PE")

def build_asset_cov_matrices(full_returns, target_dates, look_back=36):
    """
    Compute rolling annualized covariance matrices for each date in target_dates.

    For each date t, estimates the covariance matrix from the look_back months
    of returns ending strictly at t-1:

        Sigma_t = 12 * Cov(r_{t-look_back}, ..., r_{t-1})

    The multiplication by 12 converts the monthly sample covariance to an
    annualized covariance matrix under the assumption of i.i.d. monthly returns.
    No data from month t or later is used, ensuring strict look-ahead prevention.

    Parameters
    ----------
    full_returns : DataFrame
        Complete return history used as the lookback pool. Must contain all
        dates in the range [target_dates.min() - look_back months, target_dates.max()].
    target_dates : DataFrame
        Dates for which covariance matrices are required. Index must be
        month-end timestamps.
    look_back : int
        Number of prior months used per estimate. Default is 36.

    Returns
    -------
    cov_dict      : dict[Timestamp, DataFrame]   annualized covariance matrices
    vol_dict      : dict[Timestamp, Series]      annualized volatilities (sqrt of diagonal)
    corr_dict     : dict[Timestamp, DataFrame]   correlation matrices
    window_map_df : DataFrame                    audit trail mapping each date to
                                                 its estimation window [start, end]
    """
    cov_dict, vol_dict, corr_dict, window_map = {}, {}, {}, []

    for t in target_dates.index:
        start = t - MonthEnd(look_back)   # e.g. 36 months back
        end   = t - MonthEnd(1)           # last month before t (no look-ahead)

        assert start in full_returns.index, f"{start.date()} not in returns index"
        assert end   in full_returns.index, f"{end.date()} not in returns index"

        window = full_returns.loc[start:end]
        
        assert len(window) == look_back, (
            f"Expected {look_back} rows at {t.date()}, got {len(window)}"
        )
        cov_t  = window.cov() * 12
        vol_t  = pd.Series(np.sqrt(np.diag(cov_t)), index=cov_t.index, name=t)
        corr_t = window.corr()

        cov_dict[t]  = cov_t
        vol_dict[t]  = vol_t
        corr_dict[t] = corr_t
    
        window_map.append({'Date': t, 'cov_start': start, 'cov_end': end})

    window_map_df = pd.DataFrame(window_map).set_index('Date')
    return cov_dict, vol_dict, corr_dict, window_map_df

def scale_factor_to_target_vol(unscaled_weights_df, cov_dict, target_vol=0.01):
    """
    Scale FMP weights so that ex-ante annualized portfolio volatility equals target_vol.

    For each date t, computes the raw portfolio volatility using the pre-estimated
    annualized covariance matrix, then rescales the weight vector uniformly:

        w_t^scaled = w_t^raw * (target_vol / sqrt(w_t^raw' * Sigma_t * w_t^raw))

    This guarantees that sqrt(w_t^scaled' * Sigma_t * w_t^scaled) = target_vol
    exactly, making all FMPs comparable on a common risk basis before combination.

    Parameters
    ----------
    unscaled_weights_df : DataFrame
        Raw FMP weights. Index must be a subset of cov_dict keys.
    cov_dict : dict[Timestamp, ndarray]
        Annualized covariance matrices keyed by date, as returned by
        build_asset_cov_matrices.
    target_vol : float
        Target annualized volatility. Default is 0.01 (1%).

    Returns
    -------
    scaled_weights_df : DataFrame
        Rescaled weights achieving target_vol ex-ante at every date.
    portfolio_risk_s  : Series
        Raw ex-ante volatility before scaling at each date, retained for audit.
    """

    cols = unscaled_weights_df.columns
    scaled_rows, risks = [], []

    for t in unscaled_weights_df.index:
        w     = unscaled_weights_df.loc[t].values
        cov_t = cov_dict[t].loc[cols, cols].values

        raw_vol = np.sqrt(w @ cov_t @ w)
        assert raw_vol > 0, f"Zero raw vol at {t.date()} — check weights"

        scaled_rows.append(w * (target_vol / raw_vol))
        risks.append(raw_vol)

    scaled_weights_df = pd.DataFrame(
        scaled_rows, index=unscaled_weights_df.index, columns=cols
    )
    portfolio_risk_s = pd.Series(
        risks, index=unscaled_weights_df.index, name='raw_vol_before_scaling'
    )
    return scaled_weights_df, portfolio_risk_s

def compute_factor_returns(scaled_weights_df, target_dates, name='factor_return'):
    """
    Compute realized monthly portfolio returns as the inner product of
    scaled weights and realized asset returns:

        r_t^{portfolio} = w_t' * r_t

    Weights formed at date t use only information through t-1 (enforced
    upstream in signal construction and covariance estimation). The realized
    return r_t is the contemporaneous monthly return, so no look-ahead is
    introduced here.

    A strict index alignment check is enforced — any mismatch between the
    weight index and the returns index raises an assertion error rather than
    silently filling with NaN.

    Parameters
    ----------
    scaled_weights_df : DataFrame
        Volatility-scaled FMP weights, indexed by rebalance date.
    target_dates : DataFrame
        Realized monthly returns for the same dates and assets.
    name : str
        Label for the returned Series.

    Returns
    -------
    Series
        Monthly portfolio returns indexed by date.
    """

    assert scaled_weights_df.index.equals(target_dates.index), (
        "Index mismatch between weights and returns — check your backtest_dates slice"
    )
    cols = scaled_weights_df.columns
    factor_returns = (target_dates[cols] * scaled_weights_df).sum(axis=1)
    factor_returns.name = name
    return factor_returns

def combine_fmps(weight_dfs, cov_dict, fmp_weights=None, target_vol=0.01):
    """
    Combine N individually-scaled FMPs into a single portfolio and rescale
    to target_vol.

        w_combined = sum_i (alpha_i * w_i)
        w_final    = scale(w_combined, target_vol)

    Each FMP is assumed to already be scaled to 1% vol. The mixing weights
    alpha_i determine factor allocations and must sum to 1. The combined
    portfolio is then rescaled to target_vol to account for diversification
    reducing ex-ante volatility below 1%.

    Parameters
    ----------
    weight_dfs  : list of DataFrame   individually vol-scaled FMP weights
    cov_dict    : dict                annualized covariance matrices
    fmp_weights : list of float       mixing weights alpha_i (default: equal)
    target_vol  : float               target annualized vol (default: 0.01)

    Returns
    -------
    combined_raw_df : DataFrame  weighted sum of FMP weights before rescaling
    final_df        : DataFrame  final portfolio weights at target_vol
    final_risk_s    : Series     ex-ante vol of combined portfolio before rescaling
    """
    n = len(weight_dfs)
    if fmp_weights is None:
        fmp_weights = [1.0 / n] * n
    assert abs(sum(fmp_weights) - 1.0) < 1e-9, "FMP mixing weights must sum to 1"

    combined_raw_df = sum(w * df for w, df in zip(fmp_weights, weight_dfs))

    final_df, final_risk_s = scale_factor_to_target_vol(
        combined_raw_df, cov_dict, target_vol=target_vol
    )
    return combined_raw_df, final_df, final_risk_s

def evaluate_combination_scipy(names, scaled_dict, cov_dict, target_dates):
    """
    Optimize nonnegative factor-combination weights to maximize annualized IR.

    The optimizer chooses nonnegative factor sleeve weights that sum to 1.
    For each candidate factor-weight vector, the selected FMPs are combined,
    rescaled to the target volatility inside combine_fmps, and evaluated by
    realized training-period IR.
    """

    n = len(names)

    if n == 0:
        raise ValueError("names must contain at least one factor.")

    missing = [name for name in names if name not in scaled_dict]
    if missing:
        raise KeyError(f"Missing factors from scaled_dict: {missing}")

    # Dates where covariance matrices and realized returns both exist
    valid_dates = pd.Index(cov_dict.keys()).intersection(target_dates.index)

    if len(valid_dates) == 0:
        raise ValueError("No overlapping dates between cov_dict and target_dates.")

    # Align each selected FMP to valid dates
    wdfs = [
        scaled_dict[name].loc[scaled_dict[name].index.intersection(valid_dates)]
        for name in names
    ]

    # Common index across all selected FMPs
    common_idx = wdfs[0].index
    for df in wdfs[1:]:
        common_idx = common_idx.intersection(df.index)

    common_idx = common_idx.sort_values()

    if len(common_idx) == 0:
        raise ValueError("No common dates across selected FMPs, covariance matrices, and returns.")

    # Final aligned objects
    wdfs = [df.loc[common_idx] for df in wdfs]
    ret_sub = target_dates.loc[common_idx]
    cov_sub = {t: cov_dict[t] for t in common_idx}

    def safe_normalize(weights):
        """
        Clip small numerical violations and normalize to sum to 1.
        If the optimizer returns a degenerate vector, fall back to equal weight.
        """
        weights = np.asarray(weights, dtype=float)
        weights = np.clip(weights, 0.0, 1.0)

        total = weights.sum()

        if total <= 1e-12 or np.isnan(total):
            return np.ones(len(weights)) / len(weights)

        return weights / total

    def compute_ir_for_weights(weights):
        """
        Combine FMPs using candidate factor weights and compute realized annualized IR.
        """
        weights = safe_normalize(weights)

        _, fw, _ = combine_fmps(
            weight_dfs=wdfs,
            cov_dict=cov_sub,
            fmp_weights=list(weights)
        )

        ret = compute_factor_returns(
            scaled_weights_df=fw,
            target_dates=ret_sub.loc[fw.index],
            name="c"
        )

        ann_ret = ret.mean() * 12
        ann_vol = ret.std() * np.sqrt(12)

        if ann_vol <= 1e-12 or np.isnan(ann_vol):
            return np.nan

        return ann_ret / ann_vol

    # Single-factor case
    if n == 1:
        ir = compute_ir_for_weights([1.0])
        return ir, (1.0,)

    def neg_ir(weights):
        ir = compute_ir_for_weights(weights)

        if np.isnan(ir):
            return 1e6

        return -ir

    constraints = {
        "type": "eq",
        "fun": lambda w: np.sum(w) - 1.0
    }

    bounds = [(0.0, 1.0)] * n
    x0 = np.ones(n) / n

    result = minimize(
        neg_ir,
        x0,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"ftol": 1e-9, "maxiter": 1000}
    )

    if not result.success:
        print(f"Optimization warning: {result.message}")

    best_weights = safe_normalize(result.x)
    best_ir = compute_ir_for_weights(best_weights)

    return best_ir, tuple(best_weights)

def perf_summary(r, weights_df=None):
    """
    Compute annualized performance statistics for a return series.

    Parameters
    ----------
    r : pd.Series
        Monthly simple returns.

    weights_df : pd.DataFrame, optional
        Asset weights indexed by date. If provided, average turnover is computed as:
            Turnover_t = sum_i |w_{i,t} - w_{i,t-1}|

    Returns
    -------
    pd.Series with keys:
        Ann. Return, Ann. Vol, IR, Max DD, Avg DD, Win Rate, Turnover
    """
    r = r.dropna()

    ann_ret = r.mean() * 12
    ann_vol = r.std() * np.sqrt(12)
    ir = ann_ret / ann_vol if ann_vol != 0 and not np.isnan(ann_vol) else np.nan

    cum = (1 + r).cumprod()
    dd = cum / cum.cummax() - 1

    out = {
        "Ann. Return": ann_ret,
        "Ann. Vol": ann_vol,
        "IR": ir,
        "Max DD": dd.min(),
        "Avg DD": dd.mean(),
        "Win Rate": (r > 0).mean(),
    }

    if weights_df is not None:
        w = weights_df.reindex(r.index).dropna(how="any")
        turnover = w.diff().abs().sum(axis=1).dropna().mean()
        out["Turnover"] = turnover

    return pd.Series(out)

def build_momentum_factor(full_returns, target_dates, start_lag, end_lag, use_ranks=False):
    """
    Cross-sectional momentum: cumulative return over [t-start_lag, t-end_lag].

    Signal:
        Mom_{i,t} = prod(1 + r_{i,t-start_lag}, ..., 1 + r_{i,t-end_lag}) - 1

    The end_lag skips the most recent month (typically end_lag=2) to avoid
    short-term reversal contamination.

    Weighting (use_ranks):
        False — demean raw cumulative returns cross-sectionally. Signal magnitude
                is preserved: a country up 40% gets proportionally more weight
                than one up 5%. Sensitive to outliers.
        True  — rank countries by cumulative return, then demean ranks. Weights
                reflect rank position only, not magnitude. More robust in small
                universes where one outlier can dominate raw weights.
    """
    rows = []
    for t in target_dates.index:
        start  = t - MonthEnd(start_lag)
        end    = t - MonthEnd(end_lag)
        assert start in full_returns.index
        assert end   in full_returns.index
        window = full_returns.loc[start:end]
        signal = (1 + window).prod(axis=0) - 1
        signal.name = t
        rows.append({'t': t, 'start': start, 'end': end, 'signal': signal})

    signals_df = pd.DataFrame({r['t']: r['signal'] for r in rows}).T
    signals_df.index.name = 'Date'
    signals_df.insert(0, 'start', [r['start'] for r in rows])
    signals_df.insert(1, 'end',   [r['end']   for r in rows])

    sig             = signals_df.iloc[:, 2:]
    sig_for_weights = sig.rank(axis=1, ascending=True) if use_ranks else sig
    demeaned        = sig_for_weights.sub(sig_for_weights.mean(axis=1), axis=0)
    abs_sum         = demeaned.abs().sum(axis=1).replace(0, np.nan)
    return signals_df, demeaned.div(abs_sum, axis=0).fillna(0.0)


def build_tsmom_factor(full_returns, target_dates, look_back=12):
    """
    Time-series momentum-style FMP.

    Signal:
        S_{i,t} = prod_{s=t-L}^{t-1}(1 + r_{i,s}) - 1

    The signal is each country's own cumulative return over the prior L months.
    Positive values indicate positive own-history momentum; negative values
    indicate negative own-history momentum.

    Unlike cross-sectional momentum, the raw signal is not based on each
    country's rank relative to other countries. However, the final FMP is still
    dollar-neutral because the signals are demeaned cross-sectionally and
    normalized to unit gross exposure.
    """
    rows = []
    for t in target_dates.index:
        start  = t - MonthEnd(look_back)
        end    = t - MonthEnd(1)
        assert start in full_returns.index
        assert end   in full_returns.index
        window = full_returns.loc[start:end]
        signal = (1 + window).prod(axis=0) - 1
        signal = np.sign(signal) * signal.abs()
        signal.name = t
        rows.append({'t': t, 'start': start, 'end': end, 'signal': signal})

    signals_df = pd.DataFrame({r['t']: r['signal'] for r in rows}).T
    signals_df.index.name = 'Date'
    signals_df.insert(0, 'start', [r['start'] for r in rows])
    signals_df.insert(1, 'end',   [r['end']   for r in rows])

    sig      = signals_df.iloc[:, 2:]
    demeaned = sig.sub(sig.mean(axis=1), axis=0)
    abs_sum  = demeaned.abs().sum(axis=1).replace(0, np.nan)
    return signals_df, demeaned.div(abs_sum, axis=0).fillna(0.0)


def build_riskadjmom_factor(full_returns, target_dates,
                             start_lag=12, end_lag=2, use_ranks=True):
    """
    Risk-adjusted momentum: cumulative return scaled by realized volatility.

    Signal:
        signal_{i,t} = Mom_{i,t} / (sqrt(12) * SD(r_{i,t-start_lag}, ..., r_{i,t-end_lag}))

    Penalizes countries whose momentum is driven by high volatility rather
    than persistent directional moves. Similar in spirit to a realized Sharpe
    ratio over the momentum window.

    Weighting (use_ranks):
        False — raw ratio signal. Sensitive to near-zero volatility inflating
                the signal for low-vol countries.
        True  — rank the ratio before demeaning. Preferred as it removes
                sensitivity to the scale of the volatility estimate.
    """
    rows = []
    for t in target_dates.index:
        start  = t - MonthEnd(start_lag)
        end    = t - MonthEnd(end_lag)
        assert start in full_returns.index
        assert end   in full_returns.index
        window  = full_returns.loc[start:end]
        cum_ret = (1 + window).prod(axis=0) - 1
        vol     = window.std(axis=0) * np.sqrt(12)
        signal  = cum_ret / vol.replace(0, np.nan)
        signal.name = t
        rows.append({'t': t, 'start': start, 'end': end, 'signal': signal})

    signals_df = pd.DataFrame({r['t']: r['signal'] for r in rows}).T
    signals_df.index.name = 'Date'
    signals_df.insert(0, 'start', [r['start'] for r in rows])
    signals_df.insert(1, 'end',   [r['end']   for r in rows])

    sig             = signals_df.iloc[:, 2:]
    sig_for_weights = sig.rank(axis=1, ascending=True) if use_ranks else sig
    demeaned        = sig_for_weights.sub(sig_for_weights.mean(axis=1), axis=0)
    abs_sum         = demeaned.abs().sum(axis=1).replace(0, np.nan)
    return signals_df, demeaned.div(abs_sum, axis=0).fillna(0.0)


def build_autocorr_factor(full_returns, target_dates, look_back=36, use_ranks=True):
    """
    Return autocorrelation: buy countries with most persistent return trends.

    Signal:
        signal_{i,t} = corr(r_{i,t-look_back}, ..., r_{i,t-2}, r_{i,t-look_back+1}, ..., r_{i,t-1})

    Positive autocorrelation indicates that recent gains tend to follow prior
    gains — a country whose returns are self-reinforcing. Complements standard
    momentum by capturing persistence rather than just direction.

    Weighting (use_ranks):
        False — raw autocorrelation values, bounded in [-1, 1] so outliers are
                less of a concern than for other signals.
        True  — rank autocorrelations before demeaning. Preferred for consistency
                with other factor constructions in this framework.
    """
    rows = []
    for t in target_dates.index:
        start  = t - MonthEnd(look_back)
        end    = t - MonthEnd(1)
        assert start in full_returns.index
        assert end   in full_returns.index
        window = full_returns.loc[start:end]
        signal = window.apply(lambda x: x.autocorr(lag=1))
        signal.name = t
        rows.append({'t': t, 'start': start, 'end': end, 'signal': signal})

    signals_df = pd.DataFrame({r['t']: r['signal'] for r in rows}).T
    signals_df.index.name = 'Date'
    signals_df.insert(0, 'start', [r['start'] for r in rows])
    signals_df.insert(1, 'end',   [r['end']   for r in rows])

    sig             = signals_df.iloc[:, 2:]
    sig_for_weights = sig.rank(axis=1, ascending=True) if use_ranks else sig
    demeaned        = sig_for_weights.sub(sig_for_weights.mean(axis=1), axis=0)
    abs_sum         = demeaned.abs().sum(axis=1).replace(0, np.nan)
    return signals_df, demeaned.div(abs_sum, axis=0).fillna(0.0)

def build_lowvol_factor(full_returns, target_dates, look_back, dollar_neutral=True):
    """
    Low volatility FMP.

    Signal:
        signal_{i,t} = -sqrt(12) * SD(r_{i,t-L}, ..., r_{i,t-1})

        Lower total volatility => higher signal => larger long weight.

    Weighting (dollar_neutral):
        True  — demean the mapped ranks to force zero net exposure,
                isolating the pure cross-sectional volatility premium.
                Used in the main analysis.
        False — percentile ranks mapped to [-1, +1] without demeaning.
                Produces a net long tilt toward low-vol countries.
                Used in Appendix A.

    Parameters
    ----------
    full_returns   : DataFrame  complete return history
    target_dates   : DataFrame  portfolio formation dates
    look_back      : int        volatility estimation window in months
    dollar_neutral : bool       whether to demean weights (default True)

    Returns
    -------
    signals_df          : DataFrame  raw volatility signals
    unscaled_weights_df : DataFrame  gross-normalized FMP weights
    """
    rows = []

    for t in target_dates.index:
        start = t - MonthEnd(look_back)
        end   = t - MonthEnd(1)
        assert start in full_returns.index, f"{start.date()} not in returns index"
        assert end   in full_returns.index, f"{end.date()} not in returns index"
        window = full_returns.loc[start:end]
        assert len(window) == look_back, (
            f"Expected {look_back} rows at {t.date()}, got {len(window)}"
        )

        signal = -(window.std(axis=0) * np.sqrt(12))
        signal.name = t
        rows.append({"t": t, "start": start, "end": end, "signal": signal})

    signals_df = pd.DataFrame({r["t"]: r["signal"] for r in rows}).T
    signals_df.index.name = "Date"
    signals_df.insert(0, "start", [r["start"] for r in rows])
    signals_df.insert(1, "end",   [r["end"]   for r in rows])

    sig     = signals_df.iloc[:, 2:]
    weights = 2 * sig.rank(axis=1, pct=True) - 1

    if dollar_neutral:
        weights = weights.sub(weights.mean(axis=1), axis=0)

    abs_sum = weights.abs().sum(axis=1).replace(0, np.nan)
    unscaled_weights_df = weights.div(abs_sum, axis=0).fillna(0.0)

    return signals_df, unscaled_weights_df

def build_ivol_factor(full_returns, target_dates, look_back, dollar_neutral=True):
    """
    Idiosyncratic volatility FMP.

    Signal:
        For each country i at month t, regress returns on an equal-weighted
        market proxy over the prior look_back months:

            r_{i,s} = alpha_i + beta_i * r_{mkt,s} + epsilon_{i,s},
                s in [t - look_back, t - 1]

        where r_{mkt,s} = mean_j(r_{j,s}). The annualized residual volatility
        is negated so lower IVOL maps to a higher signal:

            signal_{i,t} = -sqrt(12) * sqrt( sum(epsilon^2) / (T - 2) )

        Lower idiosyncratic volatility => higher signal => larger long weight.

    Weighting (dollar_neutral):
        True  — demean mapped ranks to force zero net exposure,
                isolating the pure idiosyncratic volatility premium.
                Used in the main analysis.
        False — percentile ranks mapped to [-1, +1] without demeaning.
                Produces a net long tilt toward low-IVOL countries.
                Used in Appendix A.

    Parameters
    ----------
    full_returns   : DataFrame  complete return history
    target_dates   : DataFrame  portfolio formation dates
    look_back      : int        regression window in months (must be > 2)
    dollar_neutral : bool       whether to demean weights (default True)

    Returns
    -------
    signals_df          : DataFrame  raw IVOL signals
    unscaled_weights_df : DataFrame  gross-normalized FMP weights
    """
    assert look_back > 2, "look_back must be greater than 2 for IVOL regression."

    rows = []
    # Equal-weighted market proxy
    market = full_returns.mean(axis=1)

    for t in target_dates.index:
        start = t - MonthEnd(look_back)
        end   = t - MonthEnd(1)
        assert start in full_returns.index, f"{start.date()} not in returns index"
        assert end   in full_returns.index, f"{end.date()} not in returns index"
        window = full_returns.loc[start:end]
        mkt_w  = market.loc[start:end]
        assert len(window) == look_back, (
            f"Expected {look_back} rows at {t.date()}, got {len(window)}"
        )

        ivol_signal = {}
        for country in full_returns.columns:
            y = window[country].values
            x = mkt_w.values
            X = np.column_stack([np.ones(len(x)), x])
            beta = np.linalg.lstsq(X, y, rcond=None)[0]
            resid = y - X @ beta
            sigma_eps = np.sqrt(np.sum(resid**2) / (len(resid) - 2))
            # negative residual vol: lower IVOL = higher signal
            ivol_signal[country] = -sigma_eps * np.sqrt(12)

        signal = pd.Series(ivol_signal, name=t)
        rows.append({
            "t": t,
            "start": start,
            "end": end,
            "signal": signal
        })

    signals_df = pd.DataFrame({r["t"]: r["signal"] for r in rows}).T
    signals_df.index.name = "Date"
    signals_df.insert(0, "start", [r["start"] for r in rows])
    signals_df.insert(1, "end",   [r["end"] for r in rows])

    sig = signals_df.iloc[:, 2:]
    weights = 2 * sig.rank(axis=1, pct=True) - 1
    if dollar_neutral:
        weights = weights.sub(weights.mean(axis=1), axis=0)
    abs_sum = weights.abs().sum(axis=1).replace(0, np.nan)
    unscaled_weights_df = weights.div(abs_sum, axis=0).fillna(0.0)

    return signals_df, unscaled_weights_df

def build_volvol_factor(full_returns, target_dates, look_back=36, dollar_neutral=True):
    """
    Vol-of-vol FMP: buy countries with the most stable volatility.

    Signal:
        signal_{i,t} = -SD(vol_{i,t-6}, ..., vol_{i,t-1})

    where vol is estimated over 6-month rolling subwindows. Lower vol-of-vol
    indicates more predictable risk and a more reliable low-vol position.
    """
    rows = []
    for t in target_dates.index:
        start  = t - MonthEnd(look_back)
        end    = t - MonthEnd(1)
        assert start in full_returns.index
        assert end   in full_returns.index
        window       = full_returns.loc[start:end]
        rolling_vols = window.rolling(6).std().dropna()
        signal       = -(rolling_vols.std(axis=0) * np.sqrt(12))
        signal.name  = t
        rows.append({'t': t, 'start': start, 'end': end, 'signal': signal})

    signals_df = pd.DataFrame({r['t']: r['signal'] for r in rows}).T
    signals_df.index.name = 'Date'
    signals_df.insert(0, 'start', [r['start'] for r in rows])
    signals_df.insert(1, 'end',   [r['end']   for r in rows])

    sig      = signals_df.iloc[:, 2:]
    weights  = 2 * sig.rank(axis=1, pct=True) - 1
    if dollar_neutral:
        weights = weights.sub(weights.mean(axis=1), axis=0)
    abs_sum  = weights.abs().sum(axis=1).replace(0, np.nan)
    return signals_df, weights.div(abs_sum, axis=0).fillna(0.0)


def build_downvol_factor(full_returns, target_dates, look_back=36, dollar_neutral=True):
    """
    Downside volatility FMP: buy countries with the lowest downside variation.

    Signal:
        D_{i,s} = r_{i,s} * 1{r_{i,s} < 0}

        signal_{i,t} = -sqrt(12) * SD(D_{i,t-L}, ..., D_{i,t-1})

    Positive return months are set to zero and negative return months are retained,
    so the estimate captures variation in downside returns.
    """
    rows = []
    for t in target_dates.index:
        start    = t - MonthEnd(look_back)
        end      = t - MonthEnd(1)
        assert start in full_returns.index
        assert end   in full_returns.index
        window   = full_returns.loc[start:end]
        downside = window[window < 0].fillna(0)
        signal   = -(downside.std(axis=0) * np.sqrt(12))
        signal.name = t
        rows.append({'t': t, 'start': start, 'end': end, 'signal': signal})

    signals_df = pd.DataFrame({r['t']: r['signal'] for r in rows}).T
    signals_df.index.name = 'Date'
    signals_df.insert(0, 'start', [r['start'] for r in rows])
    signals_df.insert(1, 'end',   [r['end']   for r in rows])

    sig     = signals_df.iloc[:, 2:]
    weights = 2 * sig.rank(axis=1, pct=True) - 1
    if dollar_neutral:
        weights = weights.sub(weights.mean(axis=1), axis=0)
    abs_sum = weights.abs().sum(axis=1).replace(0, np.nan)
    return signals_df, weights.div(abs_sum, axis=0).fillna(0.0)


def build_lowbeta_factor(full_returns, target_dates, look_back=36, dollar_neutral=True):
    """
    Low-beta FMP: buy countries with the lowest market beta.

    Signal:
        signal_{i,t} = -beta_i

    where beta_i is estimated from OLS regression of country returns on the
    equal-weighted market proxy over the prior look_back months. Targets
    countries with the least systematic market exposure.
    """
    rows   = []
    market = full_returns.mean(axis=1)
    for t in target_dates.index:
        start  = t - MonthEnd(look_back)
        end    = t - MonthEnd(1)
        assert start in full_returns.index
        assert end   in full_returns.index
        window = full_returns.loc[start:end]
        mkt_w  = market.loc[start:end]
        betas  = {}
        for country in full_returns.columns:
            y    = window[country].values
            X    = np.column_stack([np.ones(len(mkt_w)), mkt_w.values])
            beta = np.linalg.lstsq(X, y, rcond=None)[0]
            betas[country] = -beta[1]
        signal = pd.Series(betas, name=t)
        rows.append({'t': t, 'start': start, 'end': end, 'signal': signal})

    signals_df = pd.DataFrame({r['t']: r['signal'] for r in rows}).T
    signals_df.index.name = 'Date'
    signals_df.insert(0, 'start', [r['start'] for r in rows])
    signals_df.insert(1, 'end',   [r['end']   for r in rows])

    sig     = signals_df.iloc[:, 2:]
    weights = 2 * sig.rank(axis=1, pct=True) - 1
    if dollar_neutral:
        weights = weights.sub(weights.mean(axis=1), axis=0)
    abs_sum = weights.abs().sum(axis=1).replace(0, np.nan)
    return signals_df, weights.div(abs_sum, axis=0).fillna(0.0)


def build_voltrend_factor(full_returns, target_dates, look_back=24, dollar_neutral=True):
    """
    Vol trend FMP: buy countries where volatility is falling.

    Signal:
        signal_{i,t} = -(vol_recent - vol_past)

    where vol_recent and vol_past are estimated over the first and second
    halves of the look_back window respectively. Falling volatility signals
    an improving risk environment.
    """
    rows = []
    for t in target_dates.index:
        start  = t - MonthEnd(look_back)
        end    = t - MonthEnd(1)
        assert start in full_returns.index
        assert end   in full_returns.index
        window   = full_returns.loc[start:end]
        half     = look_back // 2
        vol_past = window.iloc[:half].std()  * np.sqrt(12)
        vol_now  = window.iloc[-half:].std() * np.sqrt(12)
        signal   = -(vol_now - vol_past)
        signal.name = t
        rows.append({'t': t, 'start': start, 'end': end, 'signal': signal})

    signals_df = pd.DataFrame({r['t']: r['signal'] for r in rows}).T
    signals_df.index.name = 'Date'
    signals_df.insert(0, 'start', [r['start'] for r in rows])
    signals_df.insert(1, 'end',   [r['end']   for r in rows])

    sig     = signals_df.iloc[:, 2:]
    weights = 2 * sig.rank(axis=1, pct=True) - 1
    if dollar_neutral:
        weights = weights.sub(weights.mean(axis=1), axis=0)
    abs_sum = weights.abs().sum(axis=1).replace(0, np.nan)
    return signals_df, weights.div(abs_sum, axis=0).fillna(0.0)



# ── Covariance matrices ───────────────────────────────────────────────────────
cov_dict_train, _, _, _ = build_asset_cov_matrices(
    full_returns=full_history_returns,
    target_dates=train_dates,
    look_back=36
)

# ── Momentum FMPs ─────────────────────────────────────────────────────────────
momentum_specs = [
    ('Mom 3-1 Raw',   3,  2, False),
    ('Mom 3-1 Rank',  3,  2, True),
    ('Mom 6-1 Raw',   6,  2, False),
    ('Mom 6-1 Rank',  6,  2, True),
    ('Mom 9-1 Raw',   9,  2, False),
    ('Mom 9-1 Rank',  9,  2, True),
    ('Mom 12-1 Raw',  12, 2, False),
    ('Mom 12-1 Rank', 12, 2, True),
    ('Mom 18-1 Raw',  18, 2, False),
    ('Mom 18-1 Rank', 18, 2, True),
]

mom_scaled  = {}
mom_unscaled = {}
mom_raw_vols = {}
mom_returns = {}

for name, start_lag, end_lag, use_ranks in momentum_specs:
    _, unscaled_w = build_momentum_factor(
        full_history_returns,
        train_dates,
        start_lag=start_lag,
        end_lag=end_lag,
        use_ranks=use_ranks
    )

    valid = (
        unscaled_w.index
        .intersection(pd.Index(cov_dict_train.keys()))
        .intersection(train_dates.index)
        .sort_values()
    )

    unscaled_w = unscaled_w.loc[valid]
    cov_sub = {t: cov_dict_train[t] for t in valid}

    scaled_w, raw_vol = scale_factor_to_target_vol(
        unscaled_weights_df=unscaled_w,
        cov_dict=cov_sub,
        target_vol=0.01
    )

    mom_unscaled[name] = unscaled_w
    mom_scaled[name] = scaled_w
    mom_raw_vols[name] = raw_vol

    mom_returns[name] = compute_factor_returns(
        scaled_weights_df=scaled_w,
        target_dates=train_dates.loc[scaled_w.index],
        name=name
    )

for name, func, kwargs in [
    ('TSMom 12m',   build_tsmom_factor,      {'look_back': 12}),
    ('RiskAdj Mom', build_riskadjmom_factor, {'start_lag': 12, 'end_lag': 2}),
    ('AutoCorr',    build_autocorr_factor,   {'look_back': 36}),
]:
    _, unscaled_w = func(
        full_history_returns,
        train_dates,
        **kwargs
    )

    valid = (
        unscaled_w.index
        .intersection(pd.Index(cov_dict_train.keys()))
        .intersection(train_dates.index)
        .sort_values()
    )

    unscaled_w = unscaled_w.loc[valid]
    cov_sub = {t: cov_dict_train[t] for t in valid}

    scaled_w, raw_vol = scale_factor_to_target_vol(
        unscaled_weights_df=unscaled_w,
        cov_dict=cov_sub,
        target_vol=0.01
    )

    mom_unscaled[name] = unscaled_w
    mom_scaled[name] = scaled_w
    mom_raw_vols[name] = raw_vol

    mom_returns[name] = compute_factor_returns(
        scaled_weights_df=scaled_w,
        target_dates=train_dates.loc[scaled_w.index],
        name=name
    )

mom_returns_df = pd.DataFrame(mom_returns).dropna(how='any')

# ── Performance summary ───────────────────────────────────────────────────────
mom_summary = mom_returns_df.apply(perf_summary).T.sort_values('IR', ascending=False)

fmt = mom_summary.copy()
for c in ['Ann. Return', 'Ann. Vol', 'Max DD', 'Avg DD', 'Win Rate']:
    fmt[c] = fmt[c].map(lambda x: f"{x:.2%}")
fmt['IR'] = fmt['IR'].map(lambda x: f"{x:.3f}")

print(
    f"Momentum Factor Performance — Training Period "
    f"({train_dates.index[0].date()} to {train_dates.index[-1].date()})"
)
display(fmt)

# ── Cumulative returns ────────────────────────────────────────────────────────
(1 + mom_returns_df).cumprod().plot(
    figsize=(13, 6),
    title=f"Momentum Factors — Cumulative Returns "
          f"({train_dates.index[0].date()} to {train_dates.index[-1].date()})"
)
plt.axhline(1, color='black', linewidth=0.8, linestyle=':')
plt.ylabel('Growth of $1')
plt.legend(ncol=3, fontsize=8)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# ── Subperiod IR analysis ─────────────────────────────────────────────────────
def safe_ir(r):
    r = r.dropna()
    ann_ret = r.mean() * 12
    ann_vol = r.std() * np.sqrt(12)
    return ann_ret / ann_vol if ann_vol != 0 and not np.isnan(ann_vol) else np.nan

subperiods = [
    ('1973–1983', '1973', '1983'),
    ('1983–1993', '1983', '1993'),
    ('1993–2003', '1993', '2003'),
    ('2003–2015', '2003', '2015'),
]

subperiod_ir = {}

for name, ret in mom_returns.items():
    subperiod_ir[name] = {
        label: safe_ir(ret.loc[start:end])
        for label, start, end in subperiods
    }

print("Subperiod IR Analysis")
display(pd.DataFrame(subperiod_ir).T.round(3))

# ── Positive IR factors ───────────────────────────────────────────────────────
positive_ir = mom_summary[mom_summary['IR'] > 0].index.tolist()
mom_pos_df = mom_returns_df[positive_ir]
mom_pos_corr = mom_pos_df.corr()

print("Positive IR Momentum Factors")
display(mom_summary.loc[positive_ir, ['IR']].style.format({'IR': '{:.3f}'}))

# ── Correlation heatmap ───────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 8))
im = ax.imshow(mom_pos_corr.values, cmap='RdYlGn', vmin=-1, vmax=1, aspect='auto')
plt.colorbar(im, ax=ax, label='Correlation')

ax.set_xticks(range(len(positive_ir)))
ax.set_yticks(range(len(positive_ir)))
ax.set_xticklabels(positive_ir, rotation=45, ha='right', fontsize=9)
ax.set_yticklabels(positive_ir, fontsize=9)

for i in range(len(positive_ir)):
    for j in range(len(positive_ir)):
        ax.text(
            j,
            i,
            f"{mom_pos_corr.values[i, j]:.2f}",
            ha='center',
            va='center',
            fontsize=8,
            color='black' if abs(mom_pos_corr.values[i, j]) < 0.8 else 'white'
        )

ax.set_title(
    'Momentum Factors — Pairwise Correlations (Positive IR Only)',
    fontsize=12,
    fontweight='bold'
)
plt.tight_layout()
plt.show()

# ── Highly correlated pairs ───────────────────────────────────────────────────
threshold = 0.85

pairs = [
    {
        'Factor A': n1,
        'Factor B': n2,
        'Correlation': round(mom_pos_corr.loc[n1, n2], 3)
    }
    for i, n1 in enumerate(positive_ir)
    for j, n2 in enumerate(positive_ir)
    if j > i and mom_pos_corr.loc[n1, n2] > threshold
]

print(f"Highly Correlated Pairs (r > {threshold})")
display(pd.DataFrame(pairs))

# ── Exhaustive search — all combinations up to size 4 ─────────────────────────
all_results = []

for size in [1, 2, 3, 4]:
    combos = list(combinations(positive_ir, size))

    for combo in combos:
        ir, weights = evaluate_combination_scipy(
            names=list(combo),
            scaled_dict=mom_scaled,
            cov_dict=cov_dict_train,
            target_dates=train_dates
        )

        all_results.append({
            'size': size,
            'factors': combo,
            'weights': weights,
            'IR': ir,
        })


# ── Summary table: best combination by size ───────────────────────────────────
prev_ir = None
rows = []
best_by_size = {}

for size in [1, 2, 3, 4]:
    size_results = [r for r in all_results if r['size'] == size]

    if len(size_results) == 0:
        continue

    best = max(size_results, key=lambda x: x['IR'])
    best_by_size[size] = best

    imp = f"+{best['IR'] - prev_ir:.4f}" if prev_ir is not None else "—"
    w_str = " / ".join([f"{w:.2f}" for w in best['weights']])
    factors = ", ".join(best['factors'])

    rows.append({
        'Size': size,
        'IR': round(best['IR'], 4),
        'Improvement': imp,
        'Factors': factors,
        'Weights': w_str,
    })

    prev_ir = best['IR']

search_summary = pd.DataFrame(rows).set_index('Size')

print("Exhaustive Search Summary — Best Combination per Size")
display(search_summary)

# ── Select final momentum combination ─────────────────────────────────────────
# Chosen based on training IR and marginal improvement from adding factors.
selected_size = 3

best_result = best_by_size[selected_size]

best_mom_factors = list(best_result['factors'])
best_mom_weights = list(best_result['weights'])

# ── Build final selected Momentum FMP ─────────────────────────────────────────
mom_common_idx = mom_scaled[best_mom_factors[0]].index

for name in best_mom_factors[1:]:
    mom_common_idx = mom_common_idx.intersection(mom_scaled[name].index)

mom_common_idx = (
    mom_common_idx
    .intersection(pd.Index(cov_dict_train.keys()))
    .intersection(train_dates.index)
    .sort_values()
)

mom_wdfs = [mom_scaled[n].loc[mom_common_idx] for n in best_mom_factors]
cov_sub_mom = {t: cov_dict_train[t] for t in mom_common_idx}
ret_sub_mom = train_dates.loc[mom_common_idx]

_, mom_final_w, _ = combine_fmps(
    weight_dfs=mom_wdfs,
    cov_dict=cov_sub_mom,
    fmp_weights=best_mom_weights
)

mom_final_ret = compute_factor_returns(
    scaled_weights_df=mom_final_w,
    target_dates=ret_sub_mom.loc[mom_final_w.index],
    name='Momentum FMP'
)

ir = mom_final_ret.mean() * 12 / (mom_final_ret.std() * np.sqrt(12))
cum = (1 + mom_final_ret).cumprod()
dd = (cum / cum.cummax() - 1).min()
wr = (mom_final_ret > 0).mean()

print("Final Momentum FMP")
display(pd.DataFrame({
    'Factors': [", ".join(best_mom_factors)],
    'Weights': [" / ".join(f"{w:.2f}" for w in best_mom_weights)],
    'IR': [f"{ir:.4f}"],
    'Max DD': [f"{dd:.2%}"],
    'Win Rate': [f"{wr:.2%}"],
}).set_index('Factors'))

# ── Build all volatility FMPs ─────────────────────────────────────────────────
vol_specs = [
    ('LowVol 6m',    build_lowvol_factor,   {'look_back': 6}),
    ('LowVol 12m',   build_lowvol_factor,   {'look_back': 12}),
    ('LowVol 24m',   build_lowvol_factor,   {'look_back': 24}),
    ('LowVol 36m',   build_lowvol_factor,   {'look_back': 36}),
    ('IVOL 12m',     build_ivol_factor,     {'look_back': 12}),
    ('IVOL 24m',     build_ivol_factor,     {'look_back': 24}),
    ('IVOL 36m',     build_ivol_factor,     {'look_back': 36}),
    ('VolVol 36m',   build_volvol_factor,   {'look_back': 36}),
    ('DownVol 36m',  build_downvol_factor,  {'look_back': 36}),
    ('LowBeta 36m',  build_lowbeta_factor,  {'look_back': 36}),
    ('VolTrend 24m', build_voltrend_factor, {'look_back': 24}),
]

vol_unscaled = {}
vol_scaled   = {}
vol_raw_vols = {}
vol_returns  = {}

for name, func, kwargs in vol_specs:
    _, unscaled_w = func(returns, train_dates, **kwargs)

    # Align to dates with covariance matrices and realized returns
    valid = (
        unscaled_w.index
        .intersection(pd.Index(cov_dict_train.keys()))
        .intersection(train_dates.index)
        .sort_values()
    )

    unscaled_w = unscaled_w.loc[valid]
    cov_sub    = {t: cov_dict_train[t] for t in valid}

    scaled_w, raw_vol = scale_factor_to_target_vol(
        unscaled_weights_df=unscaled_w,
        cov_dict=cov_sub,
        target_vol=0.01
    )

    vol_unscaled[name] = unscaled_w
    vol_scaled[name]   = scaled_w
    vol_raw_vols[name] = raw_vol

    vol_returns[name] = compute_factor_returns(
        scaled_weights_df=scaled_w,
        target_dates=train_dates.loc[scaled_w.index],
        name=name
    )

vol_returns_df = pd.DataFrame(vol_returns).dropna(how="any")

# ── Performance summary ───────────────────────────────────────────────────────
vol_summary = vol_returns_df.apply(perf_summary).T.sort_values('IR', ascending=False)

fmt = vol_summary.copy()
for c in ['Ann. Return', 'Ann. Vol', 'Max DD', 'Avg DD', 'Win Rate']:
    fmt[c] = fmt[c].map(lambda x: f"{x:.2%}")
fmt['IR'] = fmt['IR'].map(lambda x: f"{x:.3f}")

print("Volatility Factor Performance — Training Period (1973–2015)")
display(fmt)

# ── Cumulative returns ────────────────────────────────────────────────────────
(1 + vol_returns_df).cumprod().plot(
    figsize=(13, 6),
    title='Volatility Factors — Cumulative Returns (Training Period 1973–2015)'
)
plt.axhline(1, color='black', linewidth=0.8, linestyle=':')
plt.ylabel('Growth of $1')
plt.legend(ncol=3, fontsize=8)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# ── Positive IR factors ───────────────────────────────────────────────────────
positive_ir_vol = vol_summary[vol_summary['IR'] > 0].index.tolist()

print(f"Positive IR Volatility Factors ({len(positive_ir_vol)} of {len(vol_specs)})")
display(vol_summary.loc[positive_ir_vol, ['IR']].style.format({'IR': '{:.3f}'}))

# ── Correlation heatmap ───────────────────────────────────────────────────────
if len(positive_ir_vol) > 1:
    vol_pos_df   = vol_returns_df[positive_ir_vol]
    vol_pos_corr = vol_pos_df.corr()

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(vol_pos_corr.values, cmap='RdYlGn', vmin=-1, vmax=1, aspect='auto')
    plt.colorbar(im, ax=ax, label='Correlation')
    ax.set_xticks(range(len(positive_ir_vol)))
    ax.set_yticks(range(len(positive_ir_vol)))
    ax.set_xticklabels(positive_ir_vol, rotation=45, ha='right', fontsize=9)
    ax.set_yticklabels(positive_ir_vol, fontsize=9)

    for i in range(len(positive_ir_vol)):
        for j in range(len(positive_ir_vol)):
            ax.text(
                j, i, f"{vol_pos_corr.values[i, j]:.2f}",
                ha='center', va='center', fontsize=8,
                color='black' if abs(vol_pos_corr.values[i, j]) < 0.8 else 'white'
            )

    ax.set_title(
        'Volatility Factors — Pairwise Correlations (Positive IR Only)',
        fontsize=12,
        fontweight='bold'
    )
    plt.tight_layout()
    plt.show()

else:
    print("Correlation heatmap skipped: fewer than two positive-IR volatility factors.")

# ── Subperiod IR analysis ─────────────────────────────────────────────────────
def safe_ir(r):
    r = r.dropna()
    ann_ret = r.mean() * 12
    ann_vol = r.std() * np.sqrt(12)
    return ann_ret / ann_vol if ann_vol != 0 and not np.isnan(ann_vol) else np.nan

subperiod_ir_vol = {}

for name in positive_ir_vol:
    subperiod_ir_vol[name] = {
        label: safe_ir(vol_returns[name].loc[start:end])
        for label, start, end in subperiods
    }

print("Subperiod IR Analysis — Volatility Factors")
display(pd.DataFrame(subperiod_ir_vol).T.round(3))

# ── Select final volatility sleeve ────────────────────────────────────────────
positive_ir_vol = vol_summary[vol_summary["IR"] > 0].index.tolist()

print(f"Positive IR Volatility Factors ({len(positive_ir_vol)} of {len(vol_scaled)})")
display(vol_summary.loc[positive_ir_vol, ["IR"]].style.format({"IR": "{:.3f}"}))

if len(positive_ir_vol) == 0:
    print("No positive-IR volatility factors. No volatility sleeve selected.")
    best_vol_factors = []
    best_vol_weights = []
    vol_final_w = None
    vol_final_ret = None

elif len(positive_ir_vol) == 1:
    best_vol_factors = positive_ir_vol
    best_vol_weights = [1.0]

    vol_name = best_vol_factors[0]

    # Align dates
    vol_common_idx = (
        vol_scaled[vol_name].index
        .intersection(pd.Index(cov_dict_train.keys()))
        .intersection(train_dates.index)
        .sort_values()
    )

    vol_final_w = vol_scaled[vol_name].loc[vol_common_idx]

    vol_final_ret = compute_factor_returns(
        scaled_weights_df=vol_final_w,
        target_dates=train_dates.loc[vol_final_w.index],
        name="Volatility FMP"
    )

    vol_ir = vol_final_ret.mean() * 12 / (vol_final_ret.std() * np.sqrt(12))
    vol_cum = (1 + vol_final_ret).cumprod()
    vol_dd = (vol_cum / vol_cum.cummax() - 1).min()
    vol_wr = (vol_final_ret > 0).mean()

    print("\nFinal Volatility FMP")
    display(pd.DataFrame({
        "Factors": [vol_name],
        "Weights": ["1.00"],
        "IR": [f"{vol_ir:.4f}"],
        "Max DD": [f"{vol_dd:.2%}"],
        "Win Rate": [f"{vol_wr:.2%}"],
    }).set_index("Factors"))

else:
    print("More than one positive-IR volatility factor. Run the volatility optimizer.")

# ══════════════════════════════════════════════════════════════════════════════
# TEST WHETHER VOLATILITY IMPROVES THE SELECTED MOMENTUM SLEEVE
# ══════════════════════════════════════════════════════════════════════════════

print("TESTING MOMENTUM + VOLATILITY COMBINATION")
print("=" * 70)

# ── Step 1: Optimize sleeve weights between selected Momentum and Vol FMPs ────
momvol_scaled = {
    "Momentum FMP": mom_final_w,
    "Volatility FMP": vol_final_w
}

ir_momvol, w_momvol = evaluate_combination_scipy(
    names=["Momentum FMP", "Volatility FMP"],
    scaled_dict=momvol_scaled,
    cov_dict=cov_dict_train,
    target_dates=train_dates
)

print("Optimized Momentum + Volatility Sleeve Weights")
print("-" * 70)
print(f"Momentum FMP weight   : {w_momvol[0]:.4f}")
print(f"Volatility FMP weight : {w_momvol[1]:.4f}")
print(f"Combined training IR  : {ir_momvol:.4f}")


# ── Step 2: Align Momentum, Volatility, covariance, and realized returns ──────
overlap_idx = (
    mom_final_w.index
    .intersection(vol_final_w.index)
    .intersection(pd.Index(cov_dict_train.keys()))
    .intersection(train_dates.index)
    .sort_values()
)

mom_w_align = mom_final_w.loc[overlap_idx]
vol_w_align = vol_final_w.loc[overlap_idx]
cov_sub     = {t: cov_dict_train[t] for t in overlap_idx}
ret_sub     = train_dates.loc[overlap_idx]

print(
    f"Overlap window        : {overlap_idx[0].date()} → "
    f"{overlap_idx[-1].date()} | {len(overlap_idx)} months"
)


# ── Step 3: Build final optimized Momentum + Vol portfolio ───────────────────
_, momvol_final_w, momvol_raw_vol = combine_fmps(
    weight_dfs=[mom_w_align, vol_w_align],
    cov_dict=cov_sub,
    fmp_weights=list(w_momvol)
)

momvol_final_ret = compute_factor_returns(
    scaled_weights_df=momvol_final_w,
    target_dates=ret_sub.loc[momvol_final_w.index],
    name="Momentum + Vol"
)


# ── Step 4: Effective final factor weights before final asset-level scaling ───
print("\nEffective Final Portfolio Factor Weights")
print("-" * 70)

if "best_mom_factors" in globals() and "best_mom_weights" in globals():
    for factor, weight in zip(best_mom_factors, best_mom_weights):
        print(f"{factor:<25}: {w_momvol[0] * weight:.4f}")

if "best_vol_factors" in globals() and len(best_vol_factors) > 0:
    for factor, weight in zip(best_vol_factors, best_vol_weights):
        print(f"{factor:<25}: {w_momvol[1] * weight:.4f}")
else:
    print(f"{'Volatility FMP':<25}: {w_momvol[1]:.4f}")


# ── Step 5: Compare Momentum-only vs Vol-only vs Momentum+Vol ────────────────
comparison_rets = pd.DataFrame({
    "Momentum FMP":   mom_final_ret.reindex(momvol_final_ret.index),
    "Volatility FMP": vol_final_ret.reindex(momvol_final_ret.index),
    "Momentum + Vol": momvol_final_ret
}).dropna(how="any")

comparison_summary = comparison_rets.apply(perf_summary).T

fmt = comparison_summary.copy()
for c in ["Ann. Return", "Ann. Vol", "Max DD", "Avg DD", "Win Rate"]:
    fmt[c] = fmt[c].map(lambda x: f"{x:.2%}")
fmt["IR"] = fmt["IR"].map(lambda x: f"{x:.3f}")

print("\nMOMENTUM VS MOMENTUM + VOLATILITY — TRAINING PERIOD")
print("=" * 70)
display(fmt)


# ── Step 6: Incremental effect of adding Volatility ──────────────────────────
mom_ir    = comparison_summary.loc["Momentum FMP", "IR"]
momvol_ir = comparison_summary.loc["Momentum + Vol", "IR"]

mom_dd    = comparison_summary.loc["Momentum FMP", "Max DD"]
momvol_dd = comparison_summary.loc["Momentum + Vol", "Max DD"]

mom_avg_dd    = comparison_summary.loc["Momentum FMP", "Avg DD"]
momvol_avg_dd = comparison_summary.loc["Momentum + Vol", "Avg DD"]

corr_mom_vol = comparison_rets["Momentum FMP"].corr(
    comparison_rets["Volatility FMP"]
)

print("\nIncremental Effect of Adding Volatility")
print("-" * 70)
print(f"IR improvement          : {momvol_ir - mom_ir:.4f}")
print(f"Max DD improvement      : {momvol_dd - mom_dd:.2%}")
print(f"Avg DD improvement      : {momvol_avg_dd - mom_avg_dd:.2%}")
print(f"Corr(Mom, Vol)          : {corr_mom_vol:.3f}")


# ── Step 7: Save final portfolio objects under clear names ───────────────────
final_train_w   = momvol_final_w.copy()
final_train_ret = momvol_final_ret.copy()
final_weights   = {
    "Momentum FMP": w_momvol[0],
    "Volatility FMP": w_momvol[1],
}

print("\nSaved Objects")
print("-" * 70)
print("final_train_w   = final optimized Momentum + Vol asset weights")
print("final_train_ret = final optimized Momentum + Vol training returns")
print("final_weights   = optimized Momentum/Vol sleeve weights")

# ══════════════════════════════════════════════════════════════════════════════
# TRAINING PERIOD OUTPUTS — GROSS RETURNS, STATISTICS, WEIGHTS, COVARIANCE
# ══════════════════════════════════════════════════════════════════════════════

print("TRAINING PERIOD OUTPUTS")
print("=" * 80)

# ── Align Momentum, Volatility, and Final Portfolio Returns ──────────────────
training_idx = final_train_ret.index

training_rets = pd.DataFrame({
    "Momentum FMP":   mom_final_ret.reindex(training_idx),
    "Volatility FMP": vol_final_ret.reindex(training_idx),
    "Final Portfolio": final_train_ret.reindex(training_idx),
}).dropna(how="any")

print(
    f"Training evaluation window: {training_rets.index[0].date()} → "
    f"{training_rets.index[-1].date()} | {len(training_rets)} months"
)

# ══════════════════════════════════════════════════════════════════════════════
# 1. GROSS RETURNS — GROWTH OF $1
# ══════════════════════════════════════════════════════════════════════════════

gross_returns_train = (1 + training_rets).cumprod()
gross_returns_train.index.name = "Date"

print("\nGROSS RETURNS — Growth of $1")
display(gross_returns_train.tail())

fig, ax = plt.subplots(figsize=(13, 5))
gross_returns_train.plot(ax=ax, linewidth=1.5)
ax.axhline(1, color="black", linewidth=0.8, linestyle=":")
ax.set_title("Gross Returns — Training Period", fontsize=13)
ax.set_ylabel("Growth of $1")
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()


# ══════════════════════════════════════════════════════════════════════════════
# 2. PORTFOLIO STATISTICS
# ══════════════════════════════════════════════════════════════════════════════

def turnover(weights_df):
    """
    Average monthly turnover:
        Turnover_t = sum_i |w_{i,t} - w_{i,t-1}|
    """
    return weights_df.diff().abs().sum(axis=1).dropna().mean()


portfolio_statistics_train = training_rets.apply(perf_summary).T

# Add turnover
portfolio_statistics_train.loc["Momentum FMP", "Turnover"] = turnover(
    mom_final_w.reindex(training_rets.index).dropna(how="any")
)

portfolio_statistics_train.loc["Volatility FMP", "Turnover"] = turnover(
    vol_final_w.reindex(training_rets.index).dropna(how="any")
)

portfolio_statistics_train.loc["Final Portfolio", "Turnover"] = turnover(
    final_train_w.reindex(training_rets.index).dropna(how="any")
)

print("\nPORTFOLIO STATISTICS — Training Period")
display(portfolio_statistics_train)

fmt_stats = portfolio_statistics_train.copy()
for c in ["Ann. Return", "Ann. Vol", "Max DD", "Avg DD", "Win Rate", "Turnover"]:
    fmt_stats[c] = fmt_stats[c].map(lambda x: f"{x:.2%}")
fmt_stats["IR"] = fmt_stats["IR"].map(lambda x: f"{x:.3f}")

print("\nPORTFOLIO STATISTICS — Formatted")
display(fmt_stats)


# ══════════════════════════════════════════════════════════════════════════════
# 3. RAW FACTOR WEIGHTS — BEFORE 1% VOLATILITY SCALING
# ══════════════════════════════════════════════════════════════════════════════

# Momentum raw sleeve weights before 1% vol scaling
# This combines the selected raw momentum candidate FMPs using best_mom_weights.
mom_raw_common_idx = mom_unscaled[best_mom_factors[0]].index

for factor in best_mom_factors[1:]:
    mom_raw_common_idx = mom_raw_common_idx.intersection(mom_unscaled[factor].index)

mom_raw_common_idx = mom_raw_common_idx.intersection(training_rets.index).sort_values()

momentum_raw_weights_train = sum(
    w * mom_unscaled[f].loc[mom_raw_common_idx]
    for f, w in zip(best_mom_factors, best_mom_weights)
)

momentum_raw_weights_train.index.name = "Date"


# Volatility raw sleeve weights before 1% vol scaling
# If only one vol factor was selected, this is just that raw volatility FMP.
vol_raw_common_idx = vol_unscaled[best_vol_factors[0]].index

for factor in best_vol_factors[1:]:
    vol_raw_common_idx = vol_raw_common_idx.intersection(vol_unscaled[factor].index)

vol_raw_common_idx = vol_raw_common_idx.intersection(training_rets.index).sort_values()

volatility_raw_weights_train = sum(
    w * vol_unscaled[f].loc[vol_raw_common_idx]
    for f, w in zip(best_vol_factors, best_vol_weights)
)

volatility_raw_weights_train.index.name = "Date"


print("\nRAW FACTOR WEIGHTS — Momentum FMP, before 1% volatility scaling")
display(momentum_raw_weights_train.tail())

print("\nRAW FACTOR WEIGHTS — Volatility FMP, before 1% volatility scaling")
display(volatility_raw_weights_train.tail())


# ══════════════════════════════════════════════════════════════════════════════
# 4. FACTOR 1% VOLATILITY WEIGHTS
# ══════════════════════════════════════════════════════════════════════════════

factor1_1pct_vol_weights_train = mom_final_w.reindex(training_rets.index).dropna(how="any")
factor2_1pct_vol_weights_train = vol_final_w.reindex(training_rets.index).dropna(how="any")

factor1_1pct_vol_weights_train.index.name = "Date"
factor2_1pct_vol_weights_train.index.name = "Date"

print("\nFACTOR 1 — Momentum FMP 1% Volatility Weights")
display(factor1_1pct_vol_weights_train.tail())

print("\nFACTOR 2 — Volatility FMP 1% Volatility Weights")
display(factor2_1pct_vol_weights_train.tail())


# Chart selected asset weights over time
fig, ax = plt.subplots(figsize=(13, 5))
factor1_1pct_vol_weights_train.plot(ax=ax, linewidth=1)
ax.axhline(0, color="black", linewidth=0.8)
ax.set_title("Factor 1 — Momentum FMP 1% Volatility Weights", fontsize=13)
ax.set_ylabel("Asset Weight")
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

fig, ax = plt.subplots(figsize=(13, 5))
factor2_1pct_vol_weights_train.plot(ax=ax, linewidth=1)
ax.axhline(0, color="black", linewidth=0.8)
ax.set_title("Factor 2 — Volatility FMP 1% Volatility Weights", fontsize=13)
ax.set_ylabel("Asset Weight")
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()


# ══════════════════════════════════════════════════════════════════════════════
# 5. FINAL 1% VOLATILITY PORTFOLIO WEIGHTS
# ══════════════════════════════════════════════════════════════════════════════

final_1pct_vol_portfolio_weights_train = final_train_w.reindex(training_rets.index).dropna(how="any")
final_1pct_vol_portfolio_weights_train.index.name = "Date"

print("\nFINAL 1% VOLATILITY PORTFOLIO WEIGHTS")
display(final_1pct_vol_portfolio_weights_train.tail())

fig, ax = plt.subplots(figsize=(13, 5))
final_1pct_vol_portfolio_weights_train.plot(ax=ax, linewidth=1)
ax.axhline(0, color="black", linewidth=0.8)
ax.set_title("Final 1% Volatility Portfolio Weights — Training Period", fontsize=13)
ax.set_ylabel("Asset Weight")
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()


# ══════════════════════════════════════════════════════════════════════════════
# 6. FINAL TIME PERIOD ONLY — COVARIANCE, VOLATILITY, CORRELATION
# ══════════════════════════════════════════════════════════════════════════════

final_train_date = final_1pct_vol_portfolio_weights_train.index[-1]

final_covariance_train = cov_dict_train[final_train_date].copy()
final_covariance_train.index.name = "Country"

final_volatility_train = pd.Series(
    np.sqrt(np.diag(final_covariance_train)),
    index=final_covariance_train.index,
    name="Annualized Volatility"
)

final_correlation_train = final_covariance_train.div(final_volatility_train, axis=0).div(final_volatility_train, axis=1)

print(f"\nFINAL TRAINING DATE: {final_train_date.date()}")

print("\nCOVARIANCES — Annualized covariance matrix used at final training date")
display(final_covariance_train)

print("\nVOLATILITIES — Annualized asset volatilities extracted from covariance matrix")
display(final_volatility_train.to_frame())

print("\nCORRELATIONS — Correlation matrix extracted from covariance matrix")
display(final_correlation_train)


# ══════════════════════════════════════════════════════════════════════════════
# 7. OPTIONAL: STORE EVERYTHING IN A DICTIONARY FOR EASY EXPORT LATER
# ══════════════════════════════════════════════════════════════════════════════

training_outputs = {
    "Gross Returns": gross_returns_train,
    "Portfolio Statistics": portfolio_statistics_train,
    "Momentum Raw Weights": momentum_raw_weights_train,
    "Volatility Raw Weights": volatility_raw_weights_train,
    "Factor1 1% Vol Weights": factor1_1pct_vol_weights_train,
    "Factor2 1% Vol Weights": factor2_1pct_vol_weights_train,
    "Final 1% Vol Portfolio Weights": final_1pct_vol_portfolio_weights_train,
    "Final Covariances": final_covariance_train,
    "Final Volatilities": final_volatility_train.to_frame(),
    "Final Correlations": final_correlation_train,
}

print("\nSaved all training outputs in dictionary: training_outputs")

# ══════════════════════════════════════════════════════════════════════════════
# BACKTEST — BUILD SELECTED MOMENTUM, VOLATILITY, AND FINAL PORTFOLIO
# ══════════════════════════════════════════════════════════════════════════════

print("BACKTEST: BUILDING FINAL PORTFOLIO")
print("=" * 80)

# ── Build covariance matrices — backtest period ──────────────────────────────
cov_dict_backtest, _, _, _ = build_asset_cov_matrices(
    full_returns=full_history_returns,
    target_dates=backtest_dates,
    look_back=36
)

print(f"Backtest window: {backtest_dates.index[0].date()} → {backtest_dates.index[-1].date()} | {len(backtest_dates)} months")
print(f"Covariance matrices built: {len(cov_dict_backtest)}")


# ══════════════════════════════════════════════════════════════════════════════
# 1. REBUILD SELECTED MOMENTUM FACTORS ON BACKTEST DATES
# ══════════════════════════════════════════════════════════════════════════════

momentum_spec_lookup = {
    "Mom 3-1 Raw":   (build_momentum_factor, {"start_lag": 3,  "end_lag": 2, "use_ranks": False}),
    "Mom 3-1 Rank":  (build_momentum_factor, {"start_lag": 3,  "end_lag": 2, "use_ranks": True}),
    "Mom 6-1 Raw":   (build_momentum_factor, {"start_lag": 6,  "end_lag": 2, "use_ranks": False}),
    "Mom 6-1 Rank":  (build_momentum_factor, {"start_lag": 6,  "end_lag": 2, "use_ranks": True}),
    "Mom 9-1 Raw":   (build_momentum_factor, {"start_lag": 9,  "end_lag": 2, "use_ranks": False}),
    "Mom 9-1 Rank":  (build_momentum_factor, {"start_lag": 9,  "end_lag": 2, "use_ranks": True}),
    "Mom 12-1 Raw":  (build_momentum_factor, {"start_lag": 12, "end_lag": 2, "use_ranks": False}),
    "Mom 12-1 Rank": (build_momentum_factor, {"start_lag": 12, "end_lag": 2, "use_ranks": True}),
    "Mom 18-1 Raw":  (build_momentum_factor, {"start_lag": 18, "end_lag": 2, "use_ranks": False}),
    "Mom 18-1 Rank": (build_momentum_factor, {"start_lag": 18, "end_lag": 2, "use_ranks": True}),
    "TSMom 12m":     (build_tsmom_factor, {"look_back": 12}),
    "RiskAdj Mom":   (build_riskadjmom_factor, {"start_lag": 12, "end_lag": 2}),
    "AutoCorr":      (build_autocorr_factor, {"look_back": 36}),
}

mom_bt_unscaled = {}
mom_bt_scaled   = {}
mom_bt_returns  = {}

for name in best_mom_factors:
    func, kwargs = momentum_spec_lookup[name]

    _, unscaled_w = func(
        full_history_returns,
        backtest_dates,
        **kwargs
    )

    valid = (
        unscaled_w.index
        .intersection(pd.Index(cov_dict_backtest.keys()))
        .intersection(backtest_dates.index)
        .sort_values()
    )

    unscaled_w = unscaled_w.loc[valid]
    cov_sub = {t: cov_dict_backtest[t] for t in valid}

    scaled_w, _ = scale_factor_to_target_vol(
        unscaled_weights_df=unscaled_w,
        cov_dict=cov_sub,
        target_vol=0.01
    )

    mom_bt_unscaled[name] = unscaled_w
    mom_bt_scaled[name]   = scaled_w

    mom_bt_returns[name] = compute_factor_returns(
        scaled_weights_df=scaled_w,
        target_dates=backtest_dates.loc[scaled_w.index],
        name=name
    )

print("\nSelected Momentum Factors — Backtest")
for f, w in zip(best_mom_factors, best_mom_weights):
    print(f"  {f:<25}: {w:.4f}")


# ── Combine selected momentum factors into Momentum FMP ──────────────────────
mom_bt_common_idx = mom_bt_scaled[best_mom_factors[0]].index

for f in best_mom_factors[1:]:
    mom_bt_common_idx = mom_bt_common_idx.intersection(mom_bt_scaled[f].index)

mom_bt_common_idx = (
    mom_bt_common_idx
    .intersection(pd.Index(cov_dict_backtest.keys()))
    .intersection(backtest_dates.index)
    .sort_values()
)

mom_bt_wdfs = [mom_bt_scaled[f].loc[mom_bt_common_idx] for f in best_mom_factors]
cov_sub_mom_bt = {t: cov_dict_backtest[t] for t in mom_bt_common_idx}
ret_sub_mom_bt = backtest_dates.loc[mom_bt_common_idx]

_, mom_bt_final_w, _ = combine_fmps(
    weight_dfs=mom_bt_wdfs,
    cov_dict=cov_sub_mom_bt,
    fmp_weights=list(best_mom_weights)
)

mom_bt_final_ret = compute_factor_returns(
    scaled_weights_df=mom_bt_final_w,
    target_dates=ret_sub_mom_bt.loc[mom_bt_final_w.index],
    name="Momentum FMP"
)

print("✓ Momentum FMP built")


# ══════════════════════════════════════════════════════════════════════════════
# 2. REBUILD SELECTED VOLATILITY FACTOR(S) ON BACKTEST DATES
# ══════════════════════════════════════════════════════════════════════════════

vol_spec_lookup = {
    "LowVol 6m":    (build_lowvol_factor,   {"look_back": 6}),
    "LowVol 12m":   (build_lowvol_factor,   {"look_back": 12}),
    "LowVol 24m":   (build_lowvol_factor,   {"look_back": 24}),
    "LowVol 36m":   (build_lowvol_factor,   {"look_back": 36}),
    "IVOL 12m":     (build_ivol_factor,     {"look_back": 12}),
    "IVOL 24m":     (build_ivol_factor,     {"look_back": 24}),
    "IVOL 36m":     (build_ivol_factor,     {"look_back": 36}),
    "VolVol 36m":   (build_volvol_factor,   {"look_back": 36}),
    "DownVol 36m":  (build_downvol_factor,  {"look_back": 36}),
    "LowBeta 36m":  (build_lowbeta_factor,  {"look_back": 36}),
    "VolTrend 24m": (build_voltrend_factor, {"look_back": 24}),
}

vol_bt_unscaled = {}
vol_bt_scaled   = {}
vol_bt_returns  = {}

for name in best_vol_factors:
    func, kwargs = vol_spec_lookup[name]

    _, unscaled_w = func(
        full_history_returns,
        backtest_dates,
        **kwargs
    )

    valid = (
        unscaled_w.index
        .intersection(pd.Index(cov_dict_backtest.keys()))
        .intersection(backtest_dates.index)
        .sort_values()
    )

    unscaled_w = unscaled_w.loc[valid]
    cov_sub = {t: cov_dict_backtest[t] for t in valid}

    scaled_w, _ = scale_factor_to_target_vol(
        unscaled_weights_df=unscaled_w,
        cov_dict=cov_sub,
        target_vol=0.01
    )

    vol_bt_unscaled[name] = unscaled_w
    vol_bt_scaled[name]   = scaled_w

    vol_bt_returns[name] = compute_factor_returns(
        scaled_weights_df=scaled_w,
        target_dates=backtest_dates.loc[scaled_w.index],
        name=name
    )

print("\nSelected Volatility Factor(s) — Backtest")
for f, w in zip(best_vol_factors, best_vol_weights):
    print(f"  {f:<25}: {w:.4f}")


# ── Combine selected volatility factors into Volatility FMP ──────────────────
vol_bt_common_idx = vol_bt_scaled[best_vol_factors[0]].index

for f in best_vol_factors[1:]:
    vol_bt_common_idx = vol_bt_common_idx.intersection(vol_bt_scaled[f].index)

vol_bt_common_idx = (
    vol_bt_common_idx
    .intersection(pd.Index(cov_dict_backtest.keys()))
    .intersection(backtest_dates.index)
    .sort_values()
)

vol_bt_wdfs = [vol_bt_scaled[f].loc[vol_bt_common_idx] for f in best_vol_factors]
cov_sub_vol_bt = {t: cov_dict_backtest[t] for t in vol_bt_common_idx}
ret_sub_vol_bt = backtest_dates.loc[vol_bt_common_idx]

_, vol_bt_final_w, _ = combine_fmps(
    weight_dfs=vol_bt_wdfs,
    cov_dict=cov_sub_vol_bt,
    fmp_weights=list(best_vol_weights)
)

vol_bt_final_ret = compute_factor_returns(
    scaled_weights_df=vol_bt_final_w,
    target_dates=ret_sub_vol_bt.loc[vol_bt_final_w.index],
    name="Volatility FMP"
)

print("✓ Volatility FMP built")


# ══════════════════════════════════════════════════════════════════════════════
# 3. BUILD FINAL BACKTEST PORTFOLIO USING LOCKED TRAINING WEIGHTS
# ══════════════════════════════════════════════════════════════════════════════

bt_common_idx = (
    mom_bt_final_w.index
    .intersection(vol_bt_final_w.index)
    .intersection(pd.Index(cov_dict_backtest.keys()))
    .intersection(backtest_dates.index)
    .sort_values()
)

mom_bt_align = mom_bt_final_w.loc[bt_common_idx]
vol_bt_align = vol_bt_final_w.loc[bt_common_idx]
cov_sub_bt   = {t: cov_dict_backtest[t] for t in bt_common_idx}
ret_sub_bt   = backtest_dates.loc[bt_common_idx]

_, final_backtest_w, final_backtest_raw_vol = combine_fmps(
    weight_dfs=[mom_bt_align, vol_bt_align],
    cov_dict=cov_sub_bt,
    fmp_weights=list(w_momvol)   # locked training Momentum/Vol weights
)

final_backtest_ret = compute_factor_returns(
    scaled_weights_df=final_backtest_w,
    target_dates=ret_sub_bt.loc[final_backtest_w.index],
    name="Final Portfolio"
)

print("\nFinal Momentum + Vol Backtest Portfolio")
print("=" * 80)
print(f"Momentum sleeve weight   : {w_momvol[0]:.4f}")
print(f"Volatility sleeve weight : {w_momvol[1]:.4f}")
print(f"Backtest window          : {final_backtest_ret.index[0].date()} → {final_backtest_ret.index[-1].date()} | {len(final_backtest_ret)} months")
print("✓ Final backtest portfolio built")

# ══════════════════════════════════════════════════════════════════════════════
# BACKTEST OUTPUTS — GROSS RETURNS, STATISTICS, WEIGHTS, COVARIANCE
# ══════════════════════════════════════════════════════════════════════════════

print("BACKTEST OUTPUTS")
print("=" * 80)

# ── Align Momentum, Volatility, and Final Portfolio Returns ──────────────────
backtest_idx = final_backtest_ret.index

backtest_rets = pd.DataFrame({
    "Momentum FMP": mom_bt_final_ret.reindex(backtest_idx),
    "Volatility FMP": vol_bt_final_ret.reindex(backtest_idx),
    "Final Portfolio": final_backtest_ret.reindex(backtest_idx),
}).dropna(how="any")

print(
    f"Backtest evaluation window: {backtest_rets.index[0].date()} → "
    f"{backtest_rets.index[-1].date()} | {len(backtest_rets)} months"
)


# ══════════════════════════════════════════════════════════════════════════════
# 1. GROSS RETURNS — GROWTH OF $1
# ══════════════════════════════════════════════════════════════════════════════

gross_returns_backtest = (1 + backtest_rets).cumprod()
gross_returns_backtest.index.name = "Date"

print("\nGROSS RETURNS — Growth of $1")
display(gross_returns_backtest.tail())

fig, ax = plt.subplots(figsize=(13, 5))
gross_returns_backtest.plot(ax=ax, linewidth=1.5)
ax.axhline(1, color="black", linewidth=0.8, linestyle=":")
ax.set_title("Gross Returns — Backtest Period", fontsize=13)
ax.set_ylabel("Growth of $1")
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()


# ══════════════════════════════════════════════════════════════════════════════
# 2. PORTFOLIO STATISTICS
# ══════════════════════════════════════════════════════════════════════════════

def turnover(weights_df):
    """
    Average monthly turnover:
        Turnover_t = sum_i |w_{i,t} - w_{i,t-1}|
    """
    return weights_df.diff().abs().sum(axis=1).dropna().mean()


portfolio_statistics_backtest = backtest_rets.apply(perf_summary).T

portfolio_statistics_backtest.loc["Momentum FMP", "Turnover"] = turnover(
    mom_bt_final_w.reindex(backtest_rets.index).dropna(how="any")
)

portfolio_statistics_backtest.loc["Volatility FMP", "Turnover"] = turnover(
    vol_bt_final_w.reindex(backtest_rets.index).dropna(how="any")
)

portfolio_statistics_backtest.loc["Final Portfolio", "Turnover"] = turnover(
    final_backtest_w.reindex(backtest_rets.index).dropna(how="any")
)

print("\nPORTFOLIO STATISTICS — Backtest Period")
display(portfolio_statistics_backtest)

fmt_stats_bt = portfolio_statistics_backtest.copy()
for c in ["Ann. Return", "Ann. Vol", "Max DD", "Avg DD", "Win Rate", "Turnover"]:
    fmt_stats_bt[c] = fmt_stats_bt[c].map(lambda x: f"{x:.2%}")
fmt_stats_bt["IR"] = fmt_stats_bt["IR"].map(lambda x: f"{x:.3f}")

print("\nPORTFOLIO STATISTICS — Formatted")
display(fmt_stats_bt)


# ══════════════════════════════════════════════════════════════════════════════
# 3. RAW FACTOR WEIGHTS — BEFORE 1% VOLATILITY SCALING
# ══════════════════════════════════════════════════════════════════════════════

# Momentum raw sleeve weights before 1% volatility scaling
mom_raw_bt_idx = mom_bt_unscaled[best_mom_factors[0]].index

for factor in best_mom_factors[1:]:
    mom_raw_bt_idx = mom_raw_bt_idx.intersection(mom_bt_unscaled[factor].index)

mom_raw_bt_idx = mom_raw_bt_idx.intersection(backtest_rets.index).sort_values()

momentum_raw_weights_backtest = sum(
    w * mom_bt_unscaled[f].loc[mom_raw_bt_idx]
    for f, w in zip(best_mom_factors, best_mom_weights)
)

momentum_raw_weights_backtest.index.name = "Date"


# Volatility raw sleeve weights before 1% volatility scaling
vol_raw_bt_idx = vol_bt_unscaled[best_vol_factors[0]].index

for factor in best_vol_factors[1:]:
    vol_raw_bt_idx = vol_raw_bt_idx.intersection(vol_bt_unscaled[factor].index)

vol_raw_bt_idx = vol_raw_bt_idx.intersection(backtest_rets.index).sort_values()

volatility_raw_weights_backtest = sum(
    w * vol_bt_unscaled[f].loc[vol_raw_bt_idx]
    for f, w in zip(best_vol_factors, best_vol_weights)
)

volatility_raw_weights_backtest.index.name = "Date"


print("\nRAW FACTOR WEIGHTS — Momentum FMP, before 1% volatility scaling")
display(momentum_raw_weights_backtest.tail())

print("\nRAW FACTOR WEIGHTS — Volatility FMP, before 1% volatility scaling")
display(volatility_raw_weights_backtest.tail())


# ══════════════════════════════════════════════════════════════════════════════
# 4. FACTOR 1% VOLATILITY WEIGHTS
# ══════════════════════════════════════════════════════════════════════════════

factor1_1pct_vol_weights_backtest = mom_bt_final_w.reindex(backtest_rets.index).dropna(how="any")
factor2_1pct_vol_weights_backtest = vol_bt_final_w.reindex(backtest_rets.index).dropna(how="any")

factor1_1pct_vol_weights_backtest.index.name = "Date"
factor2_1pct_vol_weights_backtest.index.name = "Date"

print("\nFACTOR 1 — Momentum FMP 1% Volatility Weights")
display(factor1_1pct_vol_weights_backtest.tail())

print("\nFACTOR 2 — Volatility FMP 1% Volatility Weights")
display(factor2_1pct_vol_weights_backtest.tail())


fig, ax = plt.subplots(figsize=(13, 5))
factor1_1pct_vol_weights_backtest.plot(ax=ax, linewidth=1)
ax.axhline(0, color="black", linewidth=0.8)
ax.set_title("Factor 1 — Momentum FMP 1% Volatility Weights — Backtest", fontsize=13)
ax.set_ylabel("Asset Weight")
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

fig, ax = plt.subplots(figsize=(13, 5))
factor2_1pct_vol_weights_backtest.plot(ax=ax, linewidth=1)
ax.axhline(0, color="black", linewidth=0.8)
ax.set_title("Factor 2 — Volatility FMP 1% Volatility Weights — Backtest", fontsize=13)
ax.set_ylabel("Asset Weight")
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()


# ══════════════════════════════════════════════════════════════════════════════
# 5. FINAL 1% VOLATILITY PORTFOLIO WEIGHTS
# ══════════════════════════════════════════════════════════════════════════════

final_1pct_vol_portfolio_weights_backtest = final_backtest_w.reindex(backtest_rets.index).dropna(how="any")
final_1pct_vol_portfolio_weights_backtest.index.name = "Date"

print("\nFINAL 1% VOLATILITY PORTFOLIO WEIGHTS")
display(final_1pct_vol_portfolio_weights_backtest.tail())

fig, ax = plt.subplots(figsize=(13, 5))
final_1pct_vol_portfolio_weights_backtest.plot(ax=ax, linewidth=1)
ax.axhline(0, color="black", linewidth=0.8)
ax.set_title("Final 1% Volatility Portfolio Weights — Backtest Period", fontsize=13)
ax.set_ylabel("Asset Weight")
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()


# ══════════════════════════════════════════════════════════════════════════════
# 6. FINAL TIME PERIOD ONLY — COVARIANCE, VOLATILITY, CORRELATION
# ══════════════════════════════════════════════════════════════════════════════

final_backtest_date = final_1pct_vol_portfolio_weights_backtest.index[-1]

final_covariance_backtest = cov_dict_backtest[final_backtest_date].copy()
final_covariance_backtest.index.name = "Country"

final_volatility_backtest = pd.Series(
    np.sqrt(np.diag(final_covariance_backtest)),
    index=final_covariance_backtest.index,
    name="Annualized Volatility"
)

final_correlation_backtest = final_covariance_backtest.div(
    final_volatility_backtest, axis=0
).div(
    final_volatility_backtest, axis=1
)

print(f"\nFINAL BACKTEST DATE: {final_backtest_date.date()}")

print("\nCOVARIANCES — Annualized covariance matrix used at final backtest date")
display(final_covariance_backtest)

print("\nVOLATILITIES — Annualized asset volatilities extracted from covariance matrix")
display(final_volatility_backtest.to_frame())

print("\nCORRELATIONS — Correlation matrix extracted from covariance matrix")
display(final_correlation_backtest)


# ══════════════════════════════════════════════════════════════════════════════
# 7. STORE EVERYTHING IN A DICTIONARY FOR EASY EXPORT LATER
# ══════════════════════════════════════════════════════════════════════════════

backtest_outputs = {
    "Gross Returns": gross_returns_backtest,
    "Portfolio Statistics": portfolio_statistics_backtest,
    "Momentum Raw Weights": momentum_raw_weights_backtest,
    "Volatility Raw Weights": volatility_raw_weights_backtest,
    "Factor1 1% Vol Weights": factor1_1pct_vol_weights_backtest,
    "Factor2 1% Vol Weights": factor2_1pct_vol_weights_backtest,
    "Final 1% Vol Portfolio Weights": final_1pct_vol_portfolio_weights_backtest,
    "Final Covariances": final_covariance_backtest,
    "Final Volatilities": final_volatility_backtest.to_frame(),
    "Final Correlations": final_correlation_backtest,
}

print("\nSaved all backtest outputs in dictionary: backtest_outputs")

# ══════════════════════════════════════════════════════════════════════════════
# PORTFOLIO STATISTICS — BACKTEST PERIOD
# ══════════════════════════════════════════════════════════════════════════════

backtest_idx = final_backtest_ret.index

backtest_rets = pd.DataFrame({
    "Momentum FMP": mom_bt_final_ret.reindex(backtest_idx),
    "Volatility FMP": vol_bt_final_ret.reindex(backtest_idx),
    "Final Portfolio": final_backtest_ret.reindex(backtest_idx),
}).dropna(how="any")

portfolio_statistics_backtest = pd.DataFrame({
    "Momentum FMP": perf_summary(
        backtest_rets["Momentum FMP"],
        weights_df=mom_bt_final_w
    ),
    "Volatility FMP": perf_summary(
        backtest_rets["Volatility FMP"],
        weights_df=vol_bt_final_w
    ),
    "Final Portfolio": perf_summary(
        backtest_rets["Final Portfolio"],
        weights_df=final_backtest_w
    ),
}).T

print("Portfolio Statistics — Backtest Period")
display(portfolio_statistics_backtest)

# ── Formatted output ─────────────────────────────────────────────────────────
portfolio_statistics_backtest_fmt = portfolio_statistics_backtest.copy()

for c in ["Ann. Return", "Ann. Vol", "Max DD", "Avg DD", "Win Rate", "Turnover"]:
    portfolio_statistics_backtest_fmt[c] = portfolio_statistics_backtest_fmt[c].map(lambda x: f"{x:.2%}")

portfolio_statistics_backtest_fmt["IR"] = portfolio_statistics_backtest_fmt["IR"].map(lambda x: f"{x:.3f}")

print("Portfolio Statistics — Backtest Period, Formatted")
display(portfolio_statistics_backtest_fmt)

# ══════════════════════════════════════════════════════════════════════════════
# BACKTEST CHARTS
# ══════════════════════════════════════════════════════════════════════════════

colors = ['#2196F3', '#FF9800', '#4CAF50']

# ── Figure 1: Gross Returns ───────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(13, 5))
gross_returns_backtest.plot(ax=ax, linewidth=1.5, color=colors)
ax.axhline(1, color='black', linewidth=0.8, linestyle=':')
ax.set_title('Gross Returns — Backtest Period (2016–2025)', fontsize=13)
ax.set_ylabel('Growth of $1')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('backtest_gross_returns.png', dpi=150, bbox_inches='tight')
plt.show()

# ── Figure 2: Raw Factor Weights — side by side ───────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(16, 5), sharex=True, sharey=True)

for ax, (name, w_df) in zip(
    axes,
    [('Factor 1 — Momentum FMP\nRaw Weights', momentum_raw_weights_backtest),
     ('Factor 2 — Volatility FMP\nRaw Weights', volatility_raw_weights_backtest)]
):
    for col in w_df.columns:
        ax.plot(w_df.index, w_df[col], linewidth=0.8, label=col)
    ax.axhline(0, color='black', linewidth=0.5)
    ax.set_title(name, fontsize=10)
    ax.legend(ncol=2, fontsize=7)
    ax.grid(True, alpha=0.3)
    ax.set_ylabel('Asset Weight')

plt.suptitle('Raw Factor Weights — Before 1% Volatility Scaling (2016–2025)',
             fontsize=13)
plt.tight_layout()
plt.savefig('backtest_raw_weights.png', dpi=150, bbox_inches='tight')
plt.show()

# ── Figure 3: Factor 1% Volatility Weights + Final Portfolio — side by side ───
fig, axes = plt.subplots(1, 3, figsize=(20, 5), sharex=True, sharey=True)

for ax, (name, w_df) in zip(
    axes,
    [('Factor 1 — Momentum FMP\n1% Volatility Weights',
      factor1_1pct_vol_weights_backtest),
     ('Factor 2 — Volatility FMP\n1% Volatility Weights',
      factor2_1pct_vol_weights_backtest),
     ('Final Portfolio\n1% Volatility Weights',
      final_1pct_vol_portfolio_weights_backtest)]
):
    for col in w_df.columns:
        ax.plot(w_df.index, w_df[col], linewidth=0.8, label=col)
    ax.axhline(0, color='black', linewidth=0.5)
    ax.set_title(name, fontsize=10)
    ax.legend(ncol=2, fontsize=7)
    ax.grid(True, alpha=0.3)
    ax.set_ylabel('Asset Weight')

plt.suptitle('Backtest Asset Weights — Factor and Final 1% Volatility Portfolios',
             fontsize=13)
plt.tight_layout()
plt.savefig('backtest_1pct_vol_weights.png', dpi=150, bbox_inches='tight')
plt.show()

# ══════════════════════════════════════════════════════════════════════════════
# ONE-OFF: OPTIMIZE DIRECTLY OVER INDIVIDUAL SIGNALS (NO FAMILY AGGREGATION)
# ══════════════════════════════════════════════════════════════════════════════

from itertools import combinations

# All positive-IR individual signals — using mom_scaled from training cell
individual_signals = {
    'Mom 12-1 Rank': mom_scaled['Mom 12-1 Rank'],
    'RiskAdj Mom':   mom_scaled['RiskAdj Mom'],
    'Mom 9-1 Rank':  mom_scaled['Mom 9-1 Rank'],
    'Mom 12-1 Raw':  mom_scaled['Mom 12-1 Raw'],
    'TSMom 12m':     mom_scaled['TSMom 12m'],
    'Mom 18-1 Rank': mom_scaled['Mom 18-1 Rank'],
    'Mom 9-1 Raw':   mom_scaled['Mom 9-1 Raw'],
    'Mom 18-1 Raw':  mom_scaled['Mom 18-1 Raw'],
    'Mom 6-1 Raw':   mom_scaled['Mom 6-1 Raw'],
    'Mom 6-1 Rank':  mom_scaled['Mom 6-1 Rank'],
    'VolVol 36m':    vol_scaled['VolVol 36m'],
}

all_names   = list(individual_signals.keys())
best_result = {'ir': -np.inf, 'combo': None, 'weights': None}
all_results = []

for size in range(1, 5):
    for combo in combinations(all_names, size):
        ir, weights = evaluate_combination_scipy(
            list(combo),
            individual_signals,
            cov_dict_train,
            train_dates
        )
        all_results.append({'size': size, 'factors': combo, 'IR': ir, 'weights': weights})
        if ir > best_result['ir']:
            best_result = {'ir': ir, 'combo': combo, 'weights': weights}

print("DIRECT SIGNAL OPTIMIZATION — Best combination per size")
print("=" * 70)
for size in range(1, 5):
    size_results = [r for r in all_results if r['size'] == size]
    best = max(size_results, key=lambda x: x['IR'])
    w_str = ' / '.join([f"{w:.2f}" for w in best['weights']])
    print(f"Size {size}: {' + '.join(best['factors'])}")
    print(f"         Weights: {w_str} | IR: {best['IR']:.4f}")
    print()

print(f"Overall best: {' + '.join(best_result['combo'])}")
print(f"Weights: {[round(w,4) for w in best_result['weights']]}")
print(f"IR: {best_result['ir']:.4f}")
