import pandas as pd
from dataclasses import dataclass

@dataclass
class SurvivalData:
    df: pd.DataFrame
    cutoff_date: pd.Timestamp
    inactivity_days: int


def build_survival_table(
    transactions: pd.DataFrame,
    cutoff_date: str,
    inactivity_days: int = 90,
) -> SurvivalData:
    df = transactions.copy()
    df["invoice_date"] = pd.to_datetime(df["invoice_date"])
    cutoff = pd.to_datetime(cutoff_date)

    # invoice-level orders
    orders = (
        df[df["invoice_date"] <= cutoff]
        .groupby(["customer_id", "invoice_no"], as_index=False)
        .agg(order_date=("invoice_date", "min"))
    )

    g = orders.groupby("customer_id")

    first = g["order_date"].min()
    last = g["order_date"].max()

    # time since last purchase at cutoff
    gap = (cutoff - last).dt.days

    # churn indicator
    churned = gap >= inactivity_days

    # survival duration:
    # if churned → last_purchase + inactivity_days
    # else → censored at cutoff
    duration = pd.Series(index=first.index, dtype=float)
    duration[churned] = (last[churned] + pd.to_timedelta(inactivity_days, unit="D") - first[churned]).dt.days
    duration[~churned] = (cutoff - first[~churned]).dt.days

    survival_df = pd.DataFrame({
        "customer_id": first.index,
        "duration": duration,
        "event": churned.astype(int),
        "first_purchase": first,
        "last_purchase": last,
        "gap_days": gap,
    }).reset_index(drop=True)

    # safety
    survival_df = survival_df[survival_df["duration"] > 0]

    return SurvivalData(
        df=survival_df,
        cutoff_date=cutoff,
        inactivity_days=inactivity_days,
    )
