## get_clv

Purpose: Retrieves Customer Lifetime Value (CLV) predictions and RFM (Recency, Frequency, Monetary) metrics for customers at the fixed cutoff date of 2011-12-09. Predictions are computed over a specified horizon period.

Use when: the user asks about CLV, future value, RFM, or 'top value customers'.

Signature: `get_clv(horizon_days: int, customer_ids: list[str] | None = None) -> dict`

Arguments:

| Name | Type | Required | Description |
|------|------|----------|-------------|
| horizon_days | int | Yes | Prediction horizon in days. Must be between 30 and 365 days. |
| customer_ids | list[str] \| None | No | Optional list of customer IDs to filter results. If None, returns all customers. |

Behavior notes:

- Cutoff date is fixed at 2011-12-09 (inclusive) and cannot be changed
- CLV predictions use BG/NBD model for purchase frequency and Gamma-Gamma model for average order value
- Results are sorted by CLV in descending order
- Only customers with valid transaction history up to the cutoff date are included
- If horizon_days is outside 30-365, the backend will raise a validation error

Return shape:

Returns a JSON object with metadata and a customers list of per-customer records.

```json
{
  "cutoff_date": "2011-12-09",
  "horizon_days": 180,
  "customers": [
    {
      "customer_id": "12345",
      "frequency": 5.0,
      "recency": 120.0,
      "T": 200.0,
      "monetary_value": 450.50,
      "pred_purchases": 2.3,
      "pred_aov": 425.75,
      "clv": 979.23
    }
  ],
  "summary": {
    "customers_total": 1000,
    "customers_with_repeat": 750,
    "clv_mean": 690.93,
    "clv_max": 15000.00
  }
}
```

## get_risk_score

Purpose: Computes churn risk scores for customers at the fixed cutoff date of 2011-12-09. Risk scores are derived from a Cox proportional hazards model and represent relative churn risk for prioritization purposes.

Use when: the user asks about risk, high-risk customers, or who is likely to churn.

Signature: `get_risk_score(customer_ids: list[str] | None = None) -> dict`

Arguments:

| Name | Type | Required | Description |
|------|------|----------|-------------|
| customer_ids | list[str] \| None | No | Optional list of customer IDs to filter results. If None, returns all customers. |

Behavior notes:

- Cutoff date is fixed at 2011-12-09 (inclusive) and cannot be changed
- Risk scores are computed using only historical data up to the cutoff date (leakage-free)
- Risk buckets are assigned based on percentiles: High (90-100%), Medium (70-90%), Low (0-70%)
- Higher risk_score values indicate higher churn risk
- Risk scores are relative measures for prioritization, not absolute probabilities

Return shape:

Returns a JSON object with metadata and a customers list of per-customer records.

```json
{
  "cutoff_date": "2011-12-09",
  "inactivity_days": 90,
  "customers": [
    {
      "customer_id": "12345",
      "n_orders": 3.0,
      "log_monetary_value": 5.2,
      "product_diversity": 12.0,
      "risk_score": 2.45,
      "risk_rank": 150,
      "risk_percentile": 85.5,
      "risk_bucket": "Medium"
    }
  ],
  "summary": {
    "n_customers": 1000,
    "risk_score_mean": 1.85,
    "risk_score_max": 5.20,
    "risk_bucket_counts": {
      "High": 100,
      "Medium": 200,
      "Low": 700
    }
  }
}
```

## get_churn_probability

Purpose: Predicts conditional churn probability for active customers over a specified prediction horizon. Computes the probability that a customer will churn within the next X days, given they have survived to the cutoff date of 2011-12-09.

Use when: the user asks 'probability to churn/stay' over a time window.

Signature: `get_churn_probability(prediction_horizon_days: int, customer_ids: list[str] | None = None) -> dict`

Arguments:

| Name | Type | Required | Description |
|------|------|----------|-------------|
| prediction_horizon_days | int | Yes | Prediction horizon in days. Must be between 7 and 365 days. |
| customer_ids | list[str] \| None | No | Optional list of customer IDs to filter results. If None, returns all active customers. |

Behavior notes:

- Cutoff date is fixed at 2011-12-09 (inclusive) and cannot be changed
- Only active customers are included in results. Active customers are those without a churn event before the cutoff and with less than 90 days of inactivity at 2011-12-09
- Churn probability is computed as: P(churn in next X days | survived to t0) = 1 - S(t0 + X) / S(t0)
- Uses individual survival functions from the Cox proportional hazards model
- Results are sorted by churn probability in descending order
- If prediction_horizon_days is outside 7-365, the backend will raise a validation error

Return shape:

Returns a JSON object with metadata and a customers list of per-customer records.

```json
{
  "cutoff_date": "2011-12-09",
  "inactivity_days": 90,
  "X_days": 90,
  "customers": [
    {
      "customer_id": "12345",
      "t0": 200.0,
      "X_days": 90,
      "churn_probability": 0.25,
      "survival_at_t0": 0.85,
      "survival_at_t0_plus_X": 0.64
    }
  ],
  "summary": {
    "n_customers": 750,
    "churn_probability_mean": 0.15,
    "churn_probability_median": 0.12,
    "churn_probability_max": 0.95,
    "churn_probability_min": 0.01,
    "survival_at_t0_mean": 0.78,
    "survival_at_t0_plus_X_mean": 0.65
  }
}
```

## get_expected_lifetime

Purpose: Computes the expected remaining lifetime in days for active customers over a fixed 365-day window. This represents the restricted expected remaining lifetime, accounting for the customer's current tenure and behavioral covariates.

Use when: the user asks 'how long will this customer remain active'.

Signature: `get_expected_lifetime(customer_ids: list[str] | None = None) -> dict`

Arguments:

| Name | Type | Required | Description |
|------|------|----------|-------------|
| customer_ids | list[str] \| None | No | Optional list of customer IDs to filter results. If None, returns all active customers. |

Behavior notes:

- Cutoff date is fixed at 2011-12-09 (inclusive) and cannot be changed
- Horizon window is fixed at 365 days
- Only active customers are included in results. Active customers are defined as in get_churn_probability (no churn event before cutoff and < 90 days inactivity at 2011-12-09)
- Expected lifetime is computed using numerical integration of the survival function
- Results are sorted by expected remaining lifetime in descending order
- Values are bounded between 0 and 365 days

Return shape:

Returns a JSON object with metadata and a customers list of per-customer records.

```json
{
  "cutoff_date": "2011-12-09",
  "inactivity_days": 90,
  "H_days": 365,
  "customers": [
    {
      "customer_id": "12345",
      "t0": 200.0,
      "H_days": 365,
      "expected_remaining_life_days": 280.5
    }
  ],
  "summary": {
    "n_customers": 750,
    "expected_lifetime_mean": 220.3,
    "expected_lifetime_median": 195.0,
    "expected_lifetime_max": 365.0,
    "expected_lifetime_min": 45.2,
    "t0_mean": 180.5
  }
}
```

## get_segmentation

Purpose: Provides customer segmentation combining risk labels and expected remaining lifetime. Each customer is assigned to a segment with corresponding action tags and recommended actions for targeted marketing strategies.

Use when: the user asks about segments, action tags, or recommended actions.

Signature: `get_segmentation(customer_ids: list[str] | None = None) -> dict`

Arguments:

| Name | Type | Required | Description |
|------|------|----------|-------------|
| customer_ids | list[str] \| None | No | Optional list of customer IDs to filter results. If None, returns all active customers. |

Behavior notes:

- Cutoff date is fixed at 2011-12-09 (inclusive) and cannot be changed
- Only active customers are included in results. Active customers are defined as in get_churn_probability (no churn event before cutoff and < 90 days inactivity at 2011-12-09)
- Segments are combinations of risk_label (High/Medium/Low) and life_bucket (Short/Medium/Long)
- Life buckets are determined by quantiles (33rd and 67th percentiles) of expected remaining lifetime
- Action tags and recommended actions are automatically assigned based on segment
- Results are sorted by risk label (High first) then by expected remaining lifetime (descending)

Return shape:

Returns a JSON object with metadata and a customers list of per-customer records.

```json
{
  "cutoff_date": "2011-12-09",
  "inactivity_days": 90,
  "H_days": 365,
  "customers": [
    {
      "customer_id": "12345",
      "risk_label": "High",
      "t0": 200.0,
      "erl_365_days": 280.5,
      "life_bucket": "Long",
      "segment": "High-Long",
      "action_tag": "Priority Save",
      "recommended_action": "High-touch retention; targeted offers; outreach."
    }
  ],
  "cutoffs": {
    "q33": 150.0,
    "q67": 250.0,
    "H_days": 365
  },
  "summary": {
    "n_customers": 750,
    "segment_counts": {
      "High-Long": 25,
      "High-Medium": 50,
      "High-Short": 25,
      "Medium-Long": 75,
      "Medium-Medium": 100,
      "Medium-Short": 75,
      "Low-Long": 150,
      "Low-Medium": 150,
      "Low-Short": 100
    },
    "risk_label_counts": {
      "High": 100,
      "Medium": 250,
      "Low": 400
    },
    "life_bucket_counts": {
      "Short": 200,
      "Medium": 300,
      "Long": 250
    },
    "action_tag_counts": {
      "Priority Save": 25,
      "Save": 50,
      "Let Churn": 25,
      "Growth Retain": 75,
      "Nurture": 100,
      "Monitor": 75,
      "VIP": 150,
      "Maintain": 150,
      "Sunset": 100
    },
    "erl_mean": 220.3,
    "erl_median": 195.0
  }
}
```

