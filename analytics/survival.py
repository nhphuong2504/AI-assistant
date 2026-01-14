import pandas as pd
import numpy as np
from dataclasses import dataclass
from lifelines import KaplanMeierFitter, CoxPHFitter
from typing import Dict, List, Tuple, Optional, Any

# --------------------
# GLOBAL MODEL ASSUMPTIONS
# --------------------
CUTOFF_DATE = "2011-12-09"
INACTIVITY_DAYS = 90


@dataclass
class CovariateTable:
    df: pd.DataFrame
    cutoff_date: pd.Timestamp


def build_covariate_table(
    transactions: pd.DataFrame,
    cutoff_date: str = CUTOFF_DATE,
    inactivity_days: int = INACTIVITY_DAYS,
) -> CovariateTable:
    """
    Builds a customer-level table at a fixed cutoff date (inclusive),
    using invoice-level orders.

    Output columns (core):
      customer_id, duration, event,
      frequency, recency, tenure_days, gap_days,
      monetary_value, aov, orders_per_month,
      product_diversity, is_uk
    """
    df = transactions.copy()
    df["invoice_date"] = pd.to_datetime(df["invoice_date"])
    cutoff = pd.to_datetime(cutoff_date)

    # Observation window (inclusive)
    df = df[df["invoice_date"] <= cutoff]
    df = df[df["customer_id"].notna()]
    df = df[df["revenue"] > 0]

    # Invoice-level orders
    orders = (
        df.groupby(["customer_id", "invoice_no"], as_index=False)
          .agg(
              order_date=("invoice_date", "min"),
              order_value=("revenue", "sum"),
          )
    )

    g = orders.groupby("customer_id")
    first = g["order_date"].min()
    last = g["order_date"].max()
    n_orders = g["invoice_no"].nunique()

    # Time quantities (days)
    tenure_days = (cutoff - first).dt.days
    recency = (last - first).dt.days
    gap_days = (cutoff - last).dt.days  # "recent visits" proxy: lower = more recent

    # Survival event: churn if inactive for >= inactivity_days at cutoff
    event = (gap_days >= inactivity_days).astype(int)

    # Duration: if churned, event time = last + inactivity_days; else censored at cutoff
    duration = pd.Series(index=first.index, dtype=float)
    duration[event == 1] = (
        last[event == 1]
        + pd.to_timedelta(inactivity_days, unit="D")
        - first[event == 1]
    ).dt.days
    duration[event == 0] = tenure_days[event == 0]

    # Behavioral covariates
    frequency = (n_orders - 1).astype(float)

    monetary_value = g["order_value"].mean()
    monetary_value = monetary_value.where(n_orders >= 2, np.nan)

    total_revenue = g["order_value"].sum()
    aov = total_revenue / n_orders

    orders_per_month = n_orders / (tenure_days / 30.0)
    orders_per_month = orders_per_month.replace([np.inf, -np.inf], np.nan)

    product_diversity = (
        df.groupby("customer_id")["stock_code"]
          .nunique()
          .reindex(first.index)
          .fillna(0)
          .astype(float)
    )

    is_uk = (
        df.groupby("customer_id")["country"]
          .apply(lambda x: int((x == "United Kingdom").any()))
          .reindex(first.index)
          .fillna(0)
          .astype(int)
    )

    cov = pd.DataFrame({
        "customer_id": first.index,
        "duration": duration,
        "event": event.astype(int),
        "frequency": frequency,
        "recency": recency.astype(float),
        "tenure_days": tenure_days.astype(float),
        "gap_days": gap_days.astype(float),
        "monetary_value": monetary_value.astype(float),
        "aov": aov.astype(float),
        "orders_per_month": orders_per_month.astype(float),
        "product_diversity": product_diversity,
        "is_uk": is_uk,
    }).reset_index(drop=True)

    # Safety filters
    cov = cov[
        (cov["duration"] > 0) &
        (cov["tenure_days"] > 0) &
        (cov["gap_days"] >= 0)
    ].copy()

    return CovariateTable(df=cov, cutoff_date=cutoff)


def fit_km_all(covariates: pd.DataFrame) -> KaplanMeierFitter:
    kmf = KaplanMeierFitter()
    kmf.fit(
        durations=covariates["duration"],
        event_observed=covariates["event"],
        label="All customers",
    )
    return kmf

def _km_curve(df_group: pd.DataFrame, label: str) -> List[Dict[str, float]]:
    kmf = KaplanMeierFitter()
    kmf.fit(
        durations=df_group["duration"],
        event_observed=df_group["event"],
        label=label,
    )
    return [
        {"time": float(t), "survival": float(s)}
        for t, s in zip(kmf.timeline, kmf.survival_function_.iloc[:, 0])
    ]


def add_tertile_group(df: pd.DataFrame, col: str, new_col: str) -> Tuple[pd.DataFrame, List[str]]:
    """
    Adds a low/medium/high group column based on tertiles of `col`.
    Robust to ties via rank.
    """
    x = df[col].copy()

    # Handle missing or infinite values
    x = x.replace([np.inf, -np.inf], np.nan)

    # If too many NaNs, you'll see fewer rows in stratified plots
    ok = x.notna()
    if ok.sum() < 10:
        df[new_col] = np.nan
        return df, []

    # Rank to break ties for qcut stability
    xr = x[ok].rank(method="first")

    # qcut into 3 bins
    try:
        bins = pd.qcut(xr, q=3, labels=["low", "medium", "high"])
    except ValueError:
        # If qcut fails due to duplicates, drop duplicate edges
        bins = pd.qcut(xr, q=3, labels=["low", "medium", "high"], duplicates="drop")

    df[new_col] = np.nan
    df.loc[ok, new_col] = bins.astype(str)

    return df, ["low", "medium", "high"]


def km_stratified(
    cov: pd.DataFrame,
    stratify: str,
) -> Tuple[Dict[str, List[Dict[str, float]]], Dict[str, float], Dict[str, int]]:
    """
    Returns:
      curves: {group_name: [{"time":..., "survival":...}, ...]}
      churn_rates: {group_name: churn_rate}
      sizes: {group_name: n_customers}
    """
    df = cov.copy()

    if stratify == "is_uk":
        df["group"] = df["is_uk"].map({1: "UK", 0: "Non-UK"}).astype(str)

        group_order = ["UK", "Non-UK"]

    elif stratify == "orders_per_month":
        df, group_order = add_tertile_group(df, "orders_per_month", "group")

    elif stratify == "aov":
        df, group_order = add_tertile_group(df, "aov", "group")

    elif stratify == "monetary_value":
        # monetary_value is NaN for one-time buyers by construction.
        # We'll stratify repeat-buyers only (otherwise the bins are meaningless).
        df = df[df["monetary_value"].notna()].copy()
        df, group_order = add_tertile_group(df, "monetary_value", "group")

    else:
        raise ValueError(f"Unknown stratify='{stratify}'")

    # Drop rows where grouping failed
    df = df[df["group"].notna()].copy()

    curves: Dict[str, List[Dict[str, float]]] = {}
    churn_rates: Dict[str, float] = {}
    sizes: Dict[str, int] = {}

    # If group_order empty (e.g., too many NaNs), fallback to whatever exists
    if not group_order:
        group_order = sorted(df["group"].unique().tolist())

    for gname in group_order:
        gdf = df[df["group"] == gname]
        if len(gdf) < 20:
            # skip tiny groups
            continue
        curves[gname] = _km_curve(gdf, gname)
        churn_rates[gname] = float(gdf["event"].mean())
        sizes[gname] = int(len(gdf))

    return curves, churn_rates, sizes

# --------------------
# Cox model helpers
# --------------------
def _zscore(s: pd.Series) -> pd.Series:
    mu = s.mean()
    sd = s.std(ddof=0)
    if sd == 0 or np.isnan(sd):
        return s * 0.0
    return (s - mu) / sd


def build_cox_design(
    cov: pd.DataFrame,
) -> pd.DataFrame:
    """
    Builds a Cox-ready dataframe with transforms and standardization.

    Covariates included:
      z_gap_days
      z_orders_per_month (log1p, z)
      z_tenure_days
      z_product_diversity (log1p, z)
    """
    df = cov.copy()

    # Log transforms for heavy-tailed covariates
    df["log_orders_per_month"] = np.log1p(df["orders_per_month"].clip(lower=0))
    df["log_product_diversity"] = np.log1p(df["product_diversity"].clip(lower=0))

    # Standardize continuous covariates
    df["z_orders_per_month"] = _zscore(df["log_orders_per_month"])
    df["z_tenure_days"] = _zscore(df["tenure_days"])
    df["z_gap_days"] = _zscore(df["gap_days"])
    df["z_product_diversity"] = _zscore(df["log_product_diversity"])

    cols = [
        "duration",
        "event",
        "z_gap_days",
        "z_orders_per_month",
        "z_tenure_days",
        "z_product_diversity",
    ]

    out = df[cols].dropna().copy()
    return out


def fit_cox(df_cox: pd.DataFrame, penalizer: float = 0.1) -> CoxPHFitter:
    cph = CoxPHFitter(penalizer=penalizer)
    cph.fit(df_cox, duration_col="duration", event_col="event")
    return cph



def cox_summary_json(cph: CoxPHFitter) -> List[Dict[str, float]]:
    """
    Returns hazard ratios and stats in a JSON-friendly format.
    """
    s = cph.summary.reset_index().rename(columns={"index": "covariate"})
    s["hazard_ratio"] = np.exp(s["coef"])
    keep = ["covariate", "coef", "hazard_ratio", "se(coef)", "p", "coef lower 95%", "coef upper 95%"]
    s = s[keep]
    return s.to_dict(orient="records")


def predict_churn_probabilities(
    cov: pd.DataFrame,
    cph: CoxPHFitter,
    horizon_days: int,
    include_churned: bool = False,
    prob_threshold_red: float = 0.7,
    prob_threshold_amber_low: float = 0.4,
) -> pd.DataFrame:
    """
    Predicts per-customer conditional churn probability at a specified horizon using a fitted Cox model.
    
    Computes forward-looking, conditional churn probability from the cutoff date:
    P(churn within H | alive at cutoff) = 1 - S(t₀ + H) / S(t₀)
    
    where t₀ = tenure_days (time from first order to cutoff).
    
    Args:
        cov: Covariate table with customer_id, tenure_days, event, and all required covariates
        cph: Fitted CoxPHFitter model
        horizon_days: Prediction horizon in days from cutoff (e.g., 30, 60, 90)
        include_churned: If False, exclude customers already churned at cutoff (event == 1)
        prob_threshold_red: Probability threshold for Red segment (default: 0.7)
        prob_threshold_amber_low: Lower probability threshold for Amber segment (default: 0.4)
    
    Returns:
        DataFrame with columns: customer_id, hazard_score, churn_prob_cond_{horizon_days}d, segment
        - hazard_score: Partial hazard (linear predictor) for continuous risk ranking (higher = higher risk)
        - churn_prob_cond_{horizon_days}d: Conditional churn probability at specified horizon
        - segment: Risk segment (Red/Amber/Green) based on hazard_score percentiles and churn probability
        Probabilities are conditional on being alive at cutoff.
    """
    # Filter out already-churned customers unless explicitly requested
    if not include_churned:
        cov = cov[cov["event"] == 0].copy()
    
    # Build Cox design matrix for remaining customers
    df_cox = build_cox_design(cov)
    
    # Get covariates (exclude duration and event)
    covariate_cols = [c for c in df_cox.columns if c not in ["duration", "event"]]
    
    # Merge customer_id and tenure_days back into df_cox for proper mapping
    # The index of df_cox should match cov index (after dropna in build_cox_design)
    df_cox_with_meta = df_cox.copy()
    df_cox_with_meta["customer_id"] = cov.loc[df_cox.index, "customer_id"].values
    df_cox_with_meta["tenure_days"] = cov.loc[df_cox.index, "tenure_days"].values
    
    # Prepare customer covariates DataFrame (without duration/event for prediction)
    customer_covariates = df_cox[covariate_cols].copy()
    
    # Predict partial hazard (linear predictor) for continuous ranking
    # Higher partial hazard = higher risk
    partial_hazards = cph.predict_partial_hazard(customer_covariates)
    
    # Predict survival functions for all customers at once
    # This returns a DataFrame where each column is a customer's survival function over time
    # The survival function S(t) gives probability of surviving from time 0 (first order) to time t
    survival_functions = cph.predict_survival_function(customer_covariates)
    
    # Get time index (same for all customers)
    times = survival_functions.index.values
    time_min = times.min()
    time_max = times.max()
    
    # Get all customer metadata as arrays for vectorized operations
    customer_ids = df_cox_with_meta.loc[customer_covariates.index, "customer_id"].values
    t0_array = df_cox_with_meta.loc[customer_covariates.index, "tenure_days"].values.astype(float)
    hazard_scores = partial_hazards.values.flatten()
    
    # Convert survival functions to numpy array for vectorized interpolation
    # Shape: (n_times, n_customers)
    survival_array = survival_functions.values
    
    # Vectorized interpolation for S(t₀) for all customers at once
    # Clip t0 values to valid range for interpolation
    t0_clipped = np.clip(t0_array, time_min, time_max)
    # Use np.interp for vectorized 1D interpolation
    # For each customer, interpolate their survival function at their t0
    # This is vectorized across customers using list comprehension (necessary because each customer has different t0)
    s_t0_array = np.array([
        np.interp(t0_clipped[i], times, survival_array[:, i])
        for i in range(len(customer_ids))
    ])
    
    # Handle edge case: if t0 exceeds max time, use last survival value
    s_t0_array = np.where(t0_array > time_max, survival_array[-1, :], s_t0_array)
    
    # Handle edge case: if S(t₀) is very small or zero, mark for special handling
    very_small_s_t0 = s_t0_array < 1e-10
    
    # Compute t₀ + H for all customers (single horizon)
    t_target_array = t0_array + horizon_days
    t_target_clipped = np.clip(t_target_array, time_min, time_max)
    
    # Vectorized interpolation for S(t₀ + H) for all customers
    # This is vectorized across customers using list comprehension (necessary because each customer has different t_target)
    s_t_target_array = np.array([
        np.interp(t_target_clipped[i], times, survival_array[:, i])
        for i in range(len(customer_ids))
    ])
    
    # Handle edge case: if t_target exceeds max time, use last survival value
    s_t_target_array = np.where(t_target_array > time_max, survival_array[-1, :], s_t_target_array)
    
    # Compute conditional survival: S(t₀ + H) / S(t₀)
    # Avoid division by zero
    conditional_survival = np.divide(
        s_t_target_array,
        s_t0_array,
        out=np.zeros_like(s_t_target_array),
        where=(s_t0_array > 1e-10)
    )
    
    # Clamp to [0, 1] for numerical stability
    conditional_survival = np.clip(conditional_survival, 0.0, 1.0)
    
    # Conditional churn probability: P(churn within H | alive at cutoff) = 1 - S(t₀ + H) / S(t₀)
    churn_prob_array = 1.0 - conditional_survival
    
    # For customers with very small S(t₀), set churn probability to 1.0
    churn_prob_array[very_small_s_t0] = 1.0
    
    # Compute segmentation based on hazard_score percentiles and churn probability
    # Percentiles for hazard_score ranking
    p70_hazard = np.percentile(hazard_scores, 70)
    p90_hazard = np.percentile(hazard_scores, 90)
    
    # Initialize segment array
    segment_array = np.full(len(customer_ids), "Green", dtype=object)
    
    # Red segment: top 10% hazard_score (>= p90) AND churn_prob >= threshold
    is_red = (hazard_scores >= p90_hazard) & (churn_prob_array >= prob_threshold_red)
    segment_array[is_red] = "Red"
    
    # Amber segment: (p70-p90 hazard_score OR churn_prob 0.4-0.7) AND not Red
    is_amber = (
        (
            (hazard_scores >= p70_hazard) & (hazard_scores < p90_hazard)  # p70-p90 hazard_score
            | (churn_prob_array >= prob_threshold_amber_low) & (churn_prob_array < prob_threshold_red)  # OR prob 0.4-0.7
        )
        & ~is_red  # AND not already Red
    )
    segment_array[is_amber] = "Amber"
    
    # Green segment: everything else (already initialized)
    
    # Build results DataFrame
    results_dict = {
        "customer_id": customer_ids,
        "hazard_score": hazard_scores,
        f"churn_prob_cond_{horizon_days}d": churn_prob_array,
        "segment": segment_array,
    }
    
    return pd.DataFrame(results_dict)


def get_customer_survival_curve(
    cov: pd.DataFrame,
    cph: CoxPHFitter,
    customer_id: str,
    include_churned: bool = False,
) -> Dict[str, Any]:
    """
    Gets the conditional survival curve and expected remaining lifetime for a specific customer.
    
    Computes conditional survival from cutoff: S(t0+u)/S(t0) where u is time from cutoff.
    Uses build_cox_design to retrieve customer covariates, ensuring exact same scaling as
    predict_churn_probabilities.
    
    Args:
        cov: Covariate table with customer_id, tenure_days, event, and all required covariates
        cph: Fitted CoxPHFitter model
        customer_id: Customer ID to look up
        include_churned: If False, exclude already-churned customers
    
    Returns:
        Dictionary with:
        - customer_id: Customer ID
        - found: Whether customer was found
        - tenure_days: Time from first order to cutoff
        - survival_curve: List of {time, conditional_survival} points (conditional on being alive at cutoff)
        - expected_remaining_lifetime: Expected remaining lifetime in days (area under conditional survival curve)
        - error: Error message if customer not found or already churned
    """
    # Convert customer_id to string for comparison
    customer_id_str = str(customer_id)
    cov_customer_id = cov["customer_id"].astype(str)
    
    # Filter out already-churned customers unless explicitly requested (same as predict_churn_probabilities)
    if not include_churned:
        cov = cov[cov["event"] == 0].copy()
    
    # Find customer in covariate table
    customer_row = cov[cov_customer_id == customer_id_str]
    
    if len(customer_row) == 0:
        return {
            "customer_id": customer_id_str,
            "found": False,
            "error": "Customer not found or already churned at cutoff",
        }
    
    customer_row = customer_row.iloc[0]
    t0 = float(customer_row["tenure_days"])
    
    # Build Cox design matrix exactly like predict_churn_probabilities
    # This ensures we use the same scaling and same customers
    df_cox = build_cox_design(cov)
    
    if len(df_cox) == 0:
        return {
            "customer_id": customer_id_str,
            "found": False,
            "error": "Unable to build Cox design matrix",
        }
    
    # Find customer in Cox design matrix (after dropna in build_cox_design)
    customer_idx = customer_row.name
    if customer_idx not in df_cox.index:
        return {
            "customer_id": customer_id_str,
            "found": False,
            "error": "Customer covariates missing or invalid (dropped in build_cox_design)",
        }
    
    # Get covariates (exclude duration and event) - exactly like predict_churn_probabilities
    covariate_cols = [c for c in df_cox.columns if c not in ["duration", "event"]]
    
    # Extract this customer's covariates from the Cox design matrix
    # This ensures exact same scaling as used in predict_churn_probabilities
    customer_covariates = df_cox.loc[[customer_idx], covariate_cols].copy()
    
    # Predict survival function for this customer
    survival_function = cph.predict_survival_function(customer_covariates)
    
    # Get the survival function (first and only column)
    survival_fn = survival_function.iloc[:, 0]
    times = survival_fn.index.values
    
    # Get S(t0) - survival probability at cutoff
    # Use np.interp for consistency with predict_churn_probabilities
    time_min = times.min()
    time_max = times.max()
    t0_clipped = np.clip(t0, time_min, time_max)
    s_t0 = float(np.interp(t0_clipped, times, survival_fn.values))
    if t0 > time_max:
        s_t0 = float(survival_fn.iloc[-1])
    
    # Handle edge case: if S(t0) is very small or zero
    if s_t0 < 1e-10:
        return {
            "customer_id": customer_id_str,
            "found": True,
            "tenure_days": t0,
            "survival_curve": [{"time": 0.0, "survival": 0.0}],
            "expected_remaining_lifetime": 0.0,
        }
    
    # Extract survival curve from t0 onwards and convert to conditional
    # Find times >= t0
    mask = times >= t0
    times_from_t0 = times[mask]
    survival_from_t0 = survival_fn.values[mask]
    
    # If no times >= t0, extend with last value
    if len(times_from_t0) == 0:
        times_from_t0 = np.array([t0])
        survival_from_t0 = np.array([s_t0])
    
    # Convert to conditional survival: S(t0+u)/S(t0) where u is time from cutoff
    conditional_survival = survival_from_t0 / s_t0
    conditional_survival = np.clip(conditional_survival, 0.0, 1.0)
    
    # Build conditional survival curve points (time relative to cutoff)
    survival_curve = [
        {"time": float(t - t0), "survival": float(s)}
        for t, s in zip(times_from_t0, conditional_survival)
    ]
    
    # Calculate expected remaining lifetime from conditional survival curve
    # This is the area under the conditional survival curve
    if len(times_from_t0) > 1:
        # Time differences (u values, time from cutoff)
        u_values = times_from_t0 - t0
        du = np.diff(u_values)
        # Average conditional survival probabilities
        avg_conditional = (conditional_survival[:-1] + conditional_survival[1:]) / 2
        # Area under curve (trapezoidal rule)
        expected_remaining_lifetime = float(np.sum(du * avg_conditional))
    else:
        # If only one point, use a conservative estimate
        if conditional_survival[0] > 0:
            expected_remaining_lifetime = float(conditional_survival[0] * 365)  # Conservative estimate
        else:
            expected_remaining_lifetime = 0.0
    
    return {
        "customer_id": customer_id_str,
        "found": True,
        "tenure_days": t0,
        "survival_curve": survival_curve,
        "expected_remaining_lifetime": expected_remaining_lifetime,
    }