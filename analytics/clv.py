import pandas as pd
import numpy as np
from dataclasses import dataclass
from lifetimes.utils import summary_data_from_transaction_data
from lifetimes import BetaGeoFitter, GammaGammaFitter


@dataclass
class CLVResult:
    rfm: pd.DataFrame
    bgnbd: BetaGeoFitter
    gg: GammaGammaFitter


def build_rfm(transactions: pd.DataFrame, cutoff_date: str) -> pd.DataFrame:
    df = transactions.copy()
    df["invoice_date"] = pd.to_datetime(df["invoice_date"])
    cutoff = pd.to_datetime(cutoff_date)

    # explicit pre-cutoff
    df = df[df["invoice_date"] <= cutoff]
    df = df[df["customer_id"].notna()]
    df = df[df["revenue"] > 0]

    # invoice-level orders
    orders = (
        df.groupby(["customer_id", "invoice_no"], as_index=False)
          .agg(
              order_date=("invoice_date", "min"),
              order_value=("revenue", "sum"),
          )
    )

    # customer-level aggregates
    g = orders.groupby("customer_id", as_index=True)

    first = g["order_date"].min()
    last = g["order_date"].max()
    n_orders = g["invoice_no"].nunique()

    # lifetimes-style fields (in days)
    T = (cutoff - first).dt.days.astype(float)
    recency = (last - first).dt.days.astype(float)

    # BG/NBD frequency is repeat purchases = orders - 1
    frequency = (n_orders - 1).astype(float)

    # Gamma-Gamma uses avg order value, only for repeat buyers
    monetary_value = g["order_value"].mean()
    monetary_value = monetary_value.where(n_orders >= 2, np.nan)

    rfm = pd.DataFrame({
        "frequency": frequency,
        "recency": recency,
        "T": T,
        "monetary_value": monetary_value,
    })

    # safety: keep valid
    rfm = rfm[(rfm["T"] > 0) & (rfm["recency"] >= 0) & (rfm["frequency"] >= 0)]

    return rfm


def fit_models(rfm: pd.DataFrame, penalizer: float = 0.001) -> CLVResult:
    bgnbd = BetaGeoFitter(penalizer_coef=penalizer)
    bgnbd.fit(rfm["frequency"], rfm["recency"], rfm["T"])

    rfm_gg = rfm[(rfm["frequency"] > 0) & (rfm["monetary_value"] > 0)].copy()

    gg = GammaGammaFitter(penalizer_coef=penalizer)
    gg.fit(rfm_gg["frequency"], rfm_gg["monetary_value"])

    return CLVResult(rfm=rfm, bgnbd=bgnbd, gg=gg)



def predict_clv(
    clv: CLVResult,
    horizon_days: int = 180,
    discount_rate: float = 0.0
) -> pd.DataFrame:
    rfm = clv.rfm.copy()

    # expected transactions
    rfm["pred_purchases"] = clv.bgnbd.conditional_expected_number_of_purchases_up_to_time(
        horizon_days, rfm["frequency"], rfm["recency"], rfm["T"]
    )

    # expected avg profit (monetary)
    mask = (rfm["frequency"] > 0) & (rfm["monetary_value"] > 0)
    rfm["pred_aov"] = np.nan
    rfm.loc[mask, "pred_aov"] = clv.gg.conditional_expected_average_profit(
        rfm.loc[mask, "frequency"], rfm.loc[mask, "monetary_value"]
    )

    # CLV: expected purchases * expected AOV
    rfm["clv"] = rfm["pred_purchases"] * rfm["pred_aov"]

    # optional discounting (very light but keeping it for now)
    if discount_rate and discount_rate > 0:
        rfm["clv"] = rfm["clv"] / (1 + discount_rate)

    out = rfm.reset_index().rename(columns={"index": "customer_id"})
    return out
