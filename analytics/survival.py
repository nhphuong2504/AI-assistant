import pandas as pd
import numpy as np
from dataclasses import dataclass

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
    frequency = (n_orders).astype(float)

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
