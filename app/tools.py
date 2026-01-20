# app/tools.py - Tool functions for OpenAI function calling

from typing import List, Optional, Dict, Any
import pandas as pd
from app.db import run_query_internal
from analytics.clv import build_rfm, fit_models, predict_clv, PURCHASE_SCALE, REVENUE_SCALE, CUTOFF_DATE as CLV_CUTOFF_DATE
from analytics.survival import (
    build_covariate_table,
    fit_cox_baseline,
    score_customers,
    predict_churn_probability,
    expected_remaining_lifetime,
    build_segmentation_table,
    CUTOFF_DATE,
    INACTIVITY_DAYS,
)


def get_clv(horizon_days: int, customer_ids: Optional[List[str]] = None) -> Dict[str, Any]:
    """
    CLV predictions and RFM metrics at fixed cutoff 2011-12-09.
    
    Args:
        horizon_days: Prediction horizon in days (30-365)
        customer_ids: Optional list of customer IDs to filter results
    
    Returns:
        Dictionary with cutoff_date, horizon_days, customers list, and summary
    """
    # Validate horizon_days
    if not (30 <= horizon_days <= 365):
        raise ValueError("horizon_days must be between 30 and 365")
    
    # Load transaction data
    sql = """
    SELECT customer_id, invoice_no, invoice_date, revenue
    FROM transactions
    WHERE customer_id IS NOT NULL
    """
    rows, _ = run_query_internal(sql, max_rows=2_000_000)
    df = pd.DataFrame(rows)
    
    # Build RFM and fit models on all data (for proper model calibration)
    rfm = build_rfm(df)
    models = fit_models(rfm)
    
    # Get unscaled predictions to calculate target totals
    pred_unscaled = predict_clv(models, horizon_days=horizon_days, aov_fallback="global_mean")
    
    # Calculate target totals using hard-coded scales
    pred_total_purchases = pred_unscaled["pred_purchases"].sum()
    pred_total_revenue = pred_unscaled["clv"].sum(skipna=True)
    
    target_purchases = pred_total_purchases * PURCHASE_SCALE if PURCHASE_SCALE != 1.0 else None
    target_revenue = pred_total_revenue * REVENUE_SCALE if REVENUE_SCALE != 1.0 else None
    
    # Get scaled predictions
    pred = predict_clv(
        models,
        horizon_days=horizon_days,
        scale_to_target_purchases=target_purchases,
        scale_to_target_revenue=target_revenue,
        aov_fallback="global_mean"
    )
    
    pred = pred.sort_values("clv", ascending=False)
    
    # Filter by customer_ids if provided (after predictions for proper model calibration)
    if customer_ids:
        # Convert customer_ids to match the type in the dataframe
        pred = pred[pred["customer_id"].astype(str).isin([str(cid) for cid in customer_ids])]
    
    # Select output columns
    output_cols = ["customer_id", "frequency", "recency", "T", "monetary_value", "pred_purchases", "pred_aov", "clv"]
    customers = pred[output_cols].to_dict(orient="records")
    
    # Create summary
    summary = {
        "customers_total": int(len(pred)),
        "customers_with_repeat": int((pred["frequency"] > 0).sum()),
        "clv_mean": float(pred["clv"].mean(skipna=True)) if len(pred) > 0 else 0.0,
        "clv_max": float(pred["clv"].max(skipna=True)) if len(pred) > 0 else 0.0,
    }
    
    return {
        "cutoff_date": CLV_CUTOFF_DATE,
        "horizon_days": horizon_days,
        "customers": customers,
        "summary": summary,
    }


def get_risk_score(customer_ids: Optional[List[str]] = None) -> Dict[str, Any]:
    """
    Churn risk scores at cutoff 2011-12-09.
    
    Args:
        customer_ids: Optional list of customer IDs to filter results
    
    Returns:
        Dictionary with cutoff_date, inactivity_days, customers list, and summary
    """
    # Load transaction data
    sql = """
    SELECT customer_id, invoice_no, invoice_date, revenue, stock_code, country
    FROM transactions
    WHERE customer_id IS NOT NULL
    """
    rows, _ = run_query_internal(sql, max_rows=2_000_000)
    df = pd.DataFrame(rows)
    
    # Build covariate table for model fitting
    cov = build_covariate_table(
        transactions=df,
        inactivity_days=INACTIVITY_DAYS,
    ).df
    
    # Fit Cox model with standard covariates
    cox_result = fit_cox_baseline(
        covariates=cov,
        covariate_cols=['n_orders', 'log_monetary_value', 'product_diversity'],
        train_frac=0.8,
        random_state=42,
        penalizer=0.1,
    )
    cox_model = cox_result['model']
    
    # Score customers
    scored = score_customers(
        model=cox_model,
        transactions=df,
    )
    
    # Filter by customer_ids if provided
    if customer_ids:
        scored = scored[scored["customer_id"].astype(str).isin([str(cid) for cid in customer_ids])]
    
    # Create summary
    summary = {
        "n_customers": int(len(scored)),
        "risk_score_mean": float(scored["risk_score"].mean()) if len(scored) > 0 else 0.0,
        "risk_score_max": float(scored["risk_score"].max()) if len(scored) > 0 else 0.0,
        "risk_bucket_counts": scored["risk_bucket"].value_counts().to_dict(),
    }
    
    return {
        "cutoff_date": CUTOFF_DATE,
        "inactivity_days": INACTIVITY_DAYS,
        "customers": scored.to_dict(orient="records"),
        "summary": summary,
    }


def get_churn_probability(prediction_horizon_days: int, customer_ids: Optional[List[str]] = None) -> Dict[str, Any]:
    """
    Churn probability predictions over a specified horizon.
    
    Args:
        prediction_horizon_days: Prediction horizon in days (7-365)
        customer_ids: Optional list of customer IDs to filter results
    
    Returns:
        Dictionary with cutoff_date, inactivity_days, X_days, customers list, and summary
    """
    # Validate prediction_horizon_days
    if not (7 <= prediction_horizon_days <= 365):
        raise ValueError("prediction_horizon_days must be between 7 and 365")
    
    # Load transaction data
    sql = """
    SELECT customer_id, invoice_no, invoice_date, revenue, stock_code, country
    FROM transactions
    WHERE customer_id IS NOT NULL
    """
    rows, _ = run_query_internal(sql, max_rows=2_000_000)
    df = pd.DataFrame(rows)
    
    # Build covariate table for model fitting
    cov = build_covariate_table(
        transactions=df,
        inactivity_days=INACTIVITY_DAYS,
    ).df
    
    # Fit Cox model with standard covariates
    cox_result = fit_cox_baseline(
        covariates=cov,
        covariate_cols=['n_orders', 'log_monetary_value', 'product_diversity'],
        train_frac=0.8,
        random_state=42,
        penalizer=0.1,
    )
    cox_model = cox_result['model']
    
    # Predict churn probabilities
    predictions = predict_churn_probability(
        model=cox_model,
        transactions=df,
        X_days=prediction_horizon_days,
        inactivity_days=INACTIVITY_DAYS,
    )
    
    # Filter by customer_ids if provided
    if customer_ids:
        predictions = predictions[predictions["customer_id"].astype(str).isin([str(cid) for cid in customer_ids])]
    
    # Create summary
    summary = {
        "n_customers": int(len(predictions)),
        "churn_probability_mean": float(predictions["churn_probability"].mean()) if len(predictions) > 0 else 0.0,
        "churn_probability_median": float(predictions["churn_probability"].median()) if len(predictions) > 0 else 0.0,
        "churn_probability_max": float(predictions["churn_probability"].max()) if len(predictions) > 0 else 0.0,
        "churn_probability_min": float(predictions["churn_probability"].min()) if len(predictions) > 0 else 0.0,
        "survival_at_t0_mean": float(predictions["survival_at_t0"].mean()) if len(predictions) > 0 else 0.0,
        "survival_at_t0_plus_X_mean": float(predictions["survival_at_t0_plus_X"].mean()) if len(predictions) > 0 else 0.0,
    }
    
    return {
        "cutoff_date": CUTOFF_DATE,
        "inactivity_days": INACTIVITY_DAYS,
        "X_days": prediction_horizon_days,
        "customers": predictions.to_dict(orient="records"),
        "summary": summary,
    }


def get_expected_lifetime(customer_ids: Optional[List[str]] = None) -> Dict[str, Any]:
    """
    Expected remaining lifetime in days over fixed 365-day window.
    
    Args:
        customer_ids: Optional list of customer IDs to filter results
    
    Returns:
        Dictionary with cutoff_date, inactivity_days, H_days, customers list, and summary
    """
    # Load transaction data
    sql = """
    SELECT customer_id, invoice_no, invoice_date, revenue, stock_code, country
    FROM transactions
    WHERE customer_id IS NOT NULL
    """
    rows, _ = run_query_internal(sql, max_rows=2_000_000)
    df = pd.DataFrame(rows)
    
    # Build covariate table for model fitting
    cov = build_covariate_table(
        transactions=df,
        inactivity_days=INACTIVITY_DAYS,
    ).df
    
    # Fit Cox model with standard covariates
    cox_result = fit_cox_baseline(
        covariates=cov,
        covariate_cols=['n_orders', 'log_monetary_value', 'product_diversity'],
        train_frac=0.8,
        random_state=42,
        penalizer=0.1,
    )
    cox_model = cox_result['model']
    
    # Compute expected remaining lifetime
    H_days = 365
    expected_lifetimes = expected_remaining_lifetime(
        model=cox_model,
        covariates_df=cov,
        H_days=H_days,
        inactivity_days=INACTIVITY_DAYS,
    )
    
    # Filter by customer_ids if provided
    if customer_ids:
        expected_lifetimes = expected_lifetimes[expected_lifetimes["customer_id"].astype(str).isin([str(cid) for cid in customer_ids])]
    
    # Create summary
    summary = {
        "n_customers": int(len(expected_lifetimes)),
        "expected_lifetime_mean": float(expected_lifetimes["expected_remaining_life_days"].mean()) if len(expected_lifetimes) > 0 else 0.0,
        "expected_lifetime_median": float(expected_lifetimes["expected_remaining_life_days"].median()) if len(expected_lifetimes) > 0 else 0.0,
        "expected_lifetime_max": float(expected_lifetimes["expected_remaining_life_days"].max()) if len(expected_lifetimes) > 0 else 0.0,
        "expected_lifetime_min": float(expected_lifetimes["expected_remaining_life_days"].min()) if len(expected_lifetimes) > 0 else 0.0,
        "t0_mean": float(expected_lifetimes["t0"].mean()) if len(expected_lifetimes) > 0 else 0.0,
    }
    
    return {
        "cutoff_date": CUTOFF_DATE,
        "inactivity_days": INACTIVITY_DAYS,
        "H_days": H_days,
        "customers": expected_lifetimes.to_dict(orient="records"),
        "summary": summary,
    }


def get_segmentation(customer_ids: Optional[List[str]] = None) -> Dict[str, Any]:
    """
    Customer segmentation with risk labels and expected lifetime.
    
    Args:
        customer_ids: Optional list of customer IDs to filter results
    
    Returns:
        Dictionary with cutoff_date, inactivity_days, H_days, customers list, cutoffs, and summary
    """
    # Load transaction data
    sql = """
    SELECT customer_id, invoice_no, invoice_date, revenue, stock_code, country
    FROM transactions
    WHERE customer_id IS NOT NULL
    """
    rows, _ = run_query_internal(sql, max_rows=2_000_000)
    df = pd.DataFrame(rows)
    
    # Build covariate table
    cov = build_covariate_table(
        transactions=df,
        inactivity_days=INACTIVITY_DAYS,
    ).df
    
    # Fit Cox model with standard covariates
    cox_result = fit_cox_baseline(
        covariates=cov,
        covariate_cols=['n_orders', 'log_monetary_value', 'product_diversity'],
        train_frac=0.8,
        random_state=42,
        penalizer=0.1,
    )
    cox_model = cox_result['model']
    
    # Build segmentation table
    H_days = 365
    segmentation_df, cutoffs = build_segmentation_table(
        model=cox_model,
        transactions=df,
        covariates_df=cov,
        H_days=H_days,
    )
    
    # Filter by customer_ids if provided
    if customer_ids:
        segmentation_df = segmentation_df[segmentation_df["customer_id"].astype(str).isin([str(cid) for cid in customer_ids])]
    
    # Create summary
    summary = {
        "n_customers": int(len(segmentation_df)),
        "segment_counts": segmentation_df['segment'].value_counts().to_dict(),
        "risk_label_counts": segmentation_df['risk_label'].value_counts().to_dict(),
        "life_bucket_counts": segmentation_df['life_bucket'].value_counts().to_dict(),
        "action_tag_counts": segmentation_df['action_tag'].value_counts().to_dict(),
        "erl_mean": float(segmentation_df['erl_365_days'].mean()) if len(segmentation_df) > 0 else 0.0,
        "erl_median": float(segmentation_df['erl_365_days'].median()) if len(segmentation_df) > 0 else 0.0,
    }
    
    return {
        "cutoff_date": CUTOFF_DATE,
        "inactivity_days": INACTIVITY_DAYS,
        "H_days": H_days,
        "customers": segmentation_df.to_dict(orient="records"),
        "cutoffs": cutoffs,
        "summary": summary,
    }


# Registry for easy lookup
TOOL_FUNCTIONS = {
    "get_clv": get_clv,
    "get_risk_score": get_risk_score,
    "get_churn_probability": get_churn_probability,
    "get_expected_lifetime": get_expected_lifetime,
    "get_segmentation": get_segmentation,
}

# OpenAI function calling schemas
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_clv",
            "description": "Retrieves Customer Lifetime Value (CLV) predictions and RFM metrics for customers at the fixed cutoff date of 2011-12-09. Use when the user asks about CLV, future value, RFM, or 'top value customers'.",
            "parameters": {
                "type": "object",
                "properties": {
                    "horizon_days": {
                        "type": "integer",
                        "description": "Prediction horizon in days. Must be between 30 and 365 days.",
                        "minimum": 30,
                        "maximum": 365
                    },
                    "customer_ids": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Optional list of customer IDs to filter results. If None, returns all customers."
                    }
                },
                "required": ["horizon_days"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_risk_score",
            "description": "Computes churn risk scores for customers at the fixed cutoff date of 2011-12-09. Use when the user asks about risk, high-risk customers, or who is likely to churn.",
            "parameters": {
                "type": "object",
                "properties": {
                    "customer_ids": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Optional list of customer IDs to filter results. If None, returns all customers."
                    }
                },
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_churn_probability",
            "description": "Predicts conditional churn probability for active customers over a specified prediction horizon. Computes the probability that a customer will churn within the next X days, given they have survived to the cutoff date of 2011-12-09. Use when the user asks 'probability to churn/stay' over a time window.",
            "parameters": {
                "type": "object",
                "properties": {
                    "prediction_horizon_days": {
                        "type": "integer",
                        "description": "Prediction horizon in days. Must be between 7 and 365 days.",
                        "minimum": 7,
                        "maximum": 365
                    },
                    "customer_ids": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Optional list of customer IDs to filter results. If None, returns all active customers."
                    }
                },
                "required": ["prediction_horizon_days"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_expected_lifetime",
            "description": "Computes the expected remaining lifetime in days for active customers over a fixed 365-day window. This represents the restricted expected remaining lifetime, accounting for the customer's current tenure and behavioral covariates. Use when the user asks 'how long will this customer remain active'.",
            "parameters": {
                "type": "object",
                "properties": {
                    "customer_ids": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Optional list of customer IDs to filter results. If None, returns all active customers."
                    }
                },
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_segmentation",
            "description": "Provides customer segmentation combining risk labels and expected remaining lifetime. Each customer is assigned to a segment with corresponding action tags and recommended actions for targeted marketing strategies. Use when the user asks about segments, action tags, or recommended actions.",
            "parameters": {
                "type": "object",
                "properties": {
                    "customer_ids": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Optional list of customer IDs to filter results. If None, returns all active customers."
                    }
                },
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "text_to_sql",
            "description": "Generates and executes SQL queries for raw transaction data. Use ONLY for raw data queries like 'revenue by country', 'transactions in December', 'how many transactions', 'total revenue', 'list all transactions'. DO NOT use for CLV, risk, churn, or prediction questions - use analytics tools instead.",
            "parameters": {
                "type": "object",
                "properties": {
                    "question": {
                        "type": "string",
                        "description": "The user's question that requires a SQL query to answer."
                    }
                },
                "required": ["question"]
            }
        }
    },
]
