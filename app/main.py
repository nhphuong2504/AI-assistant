from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field
from typing import Any, Dict, List, Optional
from app.db import get_schema, run_query, run_query_internal
from app.llm import generate_sql, generate_analytics_answer
import pandas as pd
import json
from analytics.clv import build_rfm, fit_models, predict_clv, PURCHASE_SCALE, REVENUE_SCALE, CUTOFF_DATE as CLV_CUTOFF_DATE
from analytics.survival import (
    build_covariate_table,
    fit_km_all,
    fit_cox_baseline,
    score_customers,
    predict_churn_probability,
    expected_remaining_lifetime,
    build_segmentation_table,
    CUTOFF_DATE,
    INACTIVITY_DAYS,
)

app = FastAPI(title="Retail Data Assistant API", version="0.1")

#uvicorn app.main:app --reload --host 127.0.0.1 --port 8000


class QueryRequest(BaseModel):
    sql: str = Field(..., description="Read-only SQL query (SELECT / WITH ... SELECT)")
    limit: int = Field(500, ge=1, le=5000, description="Default LIMIT applied if SQL has none")


class QueryResponse(BaseModel):
    columns: List[str]
    rows: List[Dict[str, Any]]
    row_count: int

class AskRequest(BaseModel):
    question: str = Field(..., min_length=1, description="Natural language question")


class AskResponse(BaseModel):
    question: str
    sql: str
    answer: str
    columns: List[str]
    rows: List[Dict[str, Any]]
    row_count: int
    chart: Optional[Dict[str, Any]] = None
    used_tools: Optional[List[str]] = None
    debug_info: Optional[Dict[str, Any]] = None

class CLVRequest(BaseModel):
    horizon_days: int = Field(180, ge=1, le=3650)
    limit_customers: int = Field(5000, ge=10, le=200000)


class CLVResponse(BaseModel):
    cutoff_date: str
    horizon_days: int
    top_customers: List[Dict[str, Any]]
    summary: Dict[str, Any]


class KMResponse(BaseModel):
    cutoff_date: str
    inactivity_days: int
    n_customers: int
    churn_rate: float
    survival_curve: List[Dict[str, float]]


class ScoreCustomersResponse(BaseModel):
    cutoff_date: str
    inactivity_days: int
    n_customers: int
    scored_customers: List[Dict[str, Any]]
    summary: Dict[str, Any]


class ChurnProbabilityResponse(BaseModel):
    cutoff_date: str
    inactivity_days: int
    X_days: int
    n_customers: int
    predictions: List[Dict[str, Any]]
    summary: Dict[str, Any]


class ExpectedLifetimeResponse(BaseModel):
    cutoff_date: str
    inactivity_days: int
    H_days: int
    n_customers: int
    expected_lifetimes: List[Dict[str, Any]]
    summary: Dict[str, Any]


class SegmentationResponse(BaseModel):
    cutoff_date: str
    inactivity_days: int
    H_days: int
    n_customers: int
    segments: List[Dict[str, Any]]
    cutoffs: Dict[str, float]
    summary: Dict[str, Any]




@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}


@app.get("/schema")
def schema() -> Dict[str, Any]:
    try:
        return get_schema()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/query", response_model=QueryResponse)
def query(req: QueryRequest) -> QueryResponse:
    try:
        rows, cols = run_query(req.sql, limit=req.limit)
        return QueryResponse(columns=cols, rows=rows, row_count=len(rows))
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest) -> AskResponse:
    schema = get_schema()

    # First attempt: try analytics with tool calling
    try:
        analytics_result = generate_analytics_answer(schema, req.question)
        
        # If analytics used tools, check if text_to_sql was used
        if analytics_result.get("tool_calls_made", 0) > 0:
            used_tools = analytics_result.get("used_tools", [])
            
            # If text_to_sql was used, extract SQL and execute it
            if "text_to_sql" in used_tools:
                # Extract SQL from debug messages (tool result)
                debug_messages = analytics_result.get("debug_messages", [])
                sql_result = None
                
                # Find the text_to_sql tool result in debug messages
                for msg in debug_messages:
                    if msg.get("role") == "tool" and msg.get("name") == "text_to_sql":
                        try:
                            content = json.loads(msg.get("content", "{}"))
                            if "sql" in content:
                                sql_result = content
                                break
                        except (json.JSONDecodeError, TypeError):
                            pass
                
                # If we found SQL, execute it
                if sql_result and "sql" in sql_result:
                    sql = sql_result["sql"]
                    answer = sql_result.get("answer", analytics_result["answer"])
                    chart = sql_result.get("chart", None)
                    
                    # Execute the SQL query
                    rows, cols = run_query(sql, limit=500)
                    return AskResponse(
                        question=req.question,
                        sql=sql,
                        answer=answer,
                        columns=cols,
                        rows=rows,
                        row_count=len(rows),
                        chart=chart,
                        used_tools=used_tools,
                        debug_info={
                            "tool_calls_made": analytics_result.get("tool_calls_made", 0),
                        },
                    )
            
            # Analytics answer (non-SQL tools) - return with empty SQL result structure
            return AskResponse(
                question=req.question,
                sql="",  # No SQL for analytics answers
                answer=analytics_result["answer"],
                columns=[],
                rows=[],
                row_count=0,
                chart=None,
                used_tools=used_tools,
                debug_info={
                    "tool_calls_made": analytics_result.get("tool_calls_made", 0),
                    "debug_messages": analytics_result.get("debug_messages", [])
                },
            )
    except Exception as e_analytics:
        # Analytics failed, fall through to SQL fallback
        # Log the exception for debugging (in production, use proper logging)
        import traceback
        print(f"Analytics attempt failed: {e_analytics}")
        print(traceback.format_exc())
        pass

    # Fallback: generate SQL (original behavior)
    try:
        gen = generate_sql(schema, req.question)
        sql = gen["sql"]
        answer = gen["answer"]
        chart = gen.get("chart", None)

        rows, cols = run_query(sql, limit=500)
        return AskResponse(
            question=req.question,
            sql=sql,
            answer=answer,
            columns=cols,
            rows=rows,
            row_count=len(rows),
            chart=chart,
        )

    except Exception as e1:
        # 1 repair attempt: include the error message and regenerate
        try:
            repair_question = (
                f"{req.question}\n\n"
                f"NOTE: The previous query failed with error: {str(e1)}\n"
                f"Please generate a corrected SELECT query."
            )
            gen = generate_sql(schema, repair_question)
            sql = gen["sql"]
            answer = gen["answer"]
            chart = gen.get("chart", None)

            rows, cols = run_query(sql, limit=500)
            return AskResponse(
                question=req.question,
                sql=sql,
                answer=answer,
                columns=cols,
                rows=rows,
                row_count=len(rows),
                chart=chart,
            )
        except Exception as e2:
            raise HTTPException(status_code=400, detail=f"Ask failed: {str(e2)}")

@app.post("/clv", response_model=CLVResponse)
def clv(req: CLVRequest) -> CLVResponse:
    # Pull only what we need
    sql = """
    SELECT customer_id, invoice_no, invoice_date, revenue
    FROM transactions
    WHERE customer_id IS NOT NULL
    """
    rows, _ = run_query_internal(sql, max_rows=2_000_000)  # internal call, not user SQL
    df = pd.DataFrame(rows)

    rfm = build_rfm(df)
    models = fit_models(rfm)
    
    # First get unscaled predictions to calculate target totals
    pred_unscaled = predict_clv(models, horizon_days=req.horizon_days, aov_fallback="global_mean")
    
    # Calculate target totals using hard-coded scales
    pred_total_purchases = pred_unscaled["pred_purchases"].sum()
    pred_total_revenue = pred_unscaled["clv"].sum(skipna=True)
    
    target_purchases = pred_total_purchases * PURCHASE_SCALE if PURCHASE_SCALE != 1.0 else None
    target_revenue = pred_total_revenue * REVENUE_SCALE if REVENUE_SCALE != 1.0 else None
    
    # Get scaled predictions
    pred = predict_clv(
        models, 
        horizon_days=req.horizon_days,
        scale_to_target_purchases=target_purchases,
        scale_to_target_revenue=target_revenue,
        aov_fallback="global_mean"
    )

    pred = pred.sort_values("clv", ascending=False)

    # Return all customers (or up to limit_customers if less than total)
    customer_limit = min(req.limit_customers, len(pred))
    all_customers = pred.head(customer_limit)[["customer_id", "frequency", "recency", "T", "monetary_value", "pred_purchases", "pred_aov", "clv"]]
    summary = {
        "customers_total": int(len(pred)),
        "customers_with_repeat": int((pred["frequency"] > 0).sum()),
        "clv_mean": float(pred["clv"].mean(skipna=True)) if len(pred) > 0 else 0.0,  # Mean of all customers
        "clv_max": float(all_customers["clv"].max()) if len(all_customers) > 0 else 0.0,
    }

    return CLVResponse(
        cutoff_date=CLV_CUTOFF_DATE,
        horizon_days=req.horizon_days,
        top_customers=all_customers.to_dict(orient="records"),
        summary=summary,
    )


@app.post("/survival/km", response_model=KMResponse)
def km_all(inactivity_days: int = Query(INACTIVITY_DAYS, description="Inactivity days threshold for churn definition")) -> KMResponse:
    """
    Fit Kaplan-Meier survival model on all customers.
    """
    sql = """
    SELECT customer_id, invoice_no, invoice_date, revenue, stock_code, country
    FROM transactions
    WHERE customer_id IS NOT NULL
    """
    rows, _ = run_query_internal(sql, max_rows=2_000_000)
    df = pd.DataFrame(rows)

    cov = build_covariate_table(
        transactions=df,
        inactivity_days=inactivity_days,
    ).df

    kmf = fit_km_all(cov)

    curve = [
        {"time": float(t), "survival": float(s)}
        for t, s in zip(kmf.timeline, kmf.survival_function_.iloc[:, 0])
    ]

    return KMResponse(
        cutoff_date=CUTOFF_DATE,
        inactivity_days=inactivity_days,
        n_customers=len(cov),
        churn_rate=float(cov["event"].mean()),
        survival_curve=curve,
    )


@app.post("/survival/score", response_model=ScoreCustomersResponse)
def score_customers_endpoint(
    inactivity_days: int = Query(INACTIVITY_DAYS, description="Inactivity days threshold for churn definition"),
) -> ScoreCustomersResponse:
    """
    Score customers using a fitted Cox model to predict churn risk.
    """
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
        inactivity_days=inactivity_days,
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

    # Create summary
    summary = {
        "n_customers": int(len(scored)),
        "risk_score_mean": float(scored["risk_score"].mean()),
        "risk_score_max": float(scored["risk_score"].max()),
        "risk_bucket_counts": scored["risk_bucket"].value_counts().to_dict(),
    }

    return ScoreCustomersResponse(
        cutoff_date=CUTOFF_DATE,
        inactivity_days=inactivity_days,
        n_customers=len(scored),
        scored_customers=scored.to_dict(orient="records"),
        summary=summary,
    )


@app.post("/survival/churn-probability", response_model=ChurnProbabilityResponse)
def churn_probability_endpoint(
    inactivity_days: int = Query(INACTIVITY_DAYS, description="Inactivity days threshold for churn definition"),
    X_days: int = Query(90, ge=1, le=3650, description="Prediction horizon in days"),
) -> ChurnProbabilityResponse:
    """
    Predict conditional churn probability for active customers.
    Computes P(churn in next X days | survived to cutoff).
    """
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
        inactivity_days=inactivity_days,
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
        X_days=X_days,
        inactivity_days=inactivity_days,
    )

    # Create summary
    summary = {
        "n_customers": int(len(predictions)),
        "churn_probability_mean": float(predictions["churn_probability"].mean()),
        "churn_probability_median": float(predictions["churn_probability"].median()),
        "churn_probability_max": float(predictions["churn_probability"].max()),
        "churn_probability_min": float(predictions["churn_probability"].min()),
        "survival_at_t0_mean": float(predictions["survival_at_t0"].mean()),
        "survival_at_t0_plus_X_mean": float(predictions["survival_at_t0_plus_X"].mean()),
    }

    return ChurnProbabilityResponse(
        cutoff_date=CUTOFF_DATE,
        inactivity_days=inactivity_days,
        X_days=X_days,
        n_customers=len(predictions),
        predictions=predictions.to_dict(orient="records"),
        summary=summary,
    )


@app.post("/survival/expected-lifetime", response_model=ExpectedLifetimeResponse)
def expected_lifetime_endpoint(
    inactivity_days: int = Query(INACTIVITY_DAYS, description="Inactivity days threshold for churn definition"),
    H_days: int = Query(365, ge=1, le=3650, description="Horizon in days for restricted expectation"),
) -> ExpectedLifetimeResponse:
    """
    Compute restricted expected remaining lifetime for active customers.
    """
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
        inactivity_days=inactivity_days,
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
    expected_lifetimes = expected_remaining_lifetime(
        model=cox_model,
        covariates_df=cov,
        H_days=H_days,
        inactivity_days=inactivity_days,
    )

    # Create summary
    summary = {
        "n_customers": int(len(expected_lifetimes)),
        "expected_lifetime_mean": float(expected_lifetimes["expected_remaining_life_days"].mean()),
        "expected_lifetime_median": float(expected_lifetimes["expected_remaining_life_days"].median()),
        "expected_lifetime_max": float(expected_lifetimes["expected_remaining_life_days"].max()),
        "expected_lifetime_min": float(expected_lifetimes["expected_remaining_life_days"].min()),
        "t0_mean": float(expected_lifetimes["t0"].mean()),
    }

    return ExpectedLifetimeResponse(
        cutoff_date=CUTOFF_DATE,
        inactivity_days=inactivity_days,
        H_days=H_days,
        n_customers=len(expected_lifetimes),
        expected_lifetimes=expected_lifetimes.to_dict(orient="records"),
        summary=summary,
    )


@app.post("/survival/segmentation", response_model=SegmentationResponse)
def segmentation_endpoint(
    inactivity_days: int = Query(INACTIVITY_DAYS, description="Inactivity days threshold for churn definition"),
    H_days: int = Query(365, ge=1, le=3650, description="Horizon in days for expected remaining lifetime"),
) -> SegmentationResponse:
    """
    Build segmentation table combining risk labels and expected remaining lifetime.
    """
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
        inactivity_days=inactivity_days,
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
    segmentation_df, cutoffs = build_segmentation_table(
        model=cox_model,
        transactions=df,
        covariates_df=cov,
        H_days=H_days,
    )

    # Create summary
    summary = {
        "n_customers": int(len(segmentation_df)),
        "segment_counts": segmentation_df['segment'].value_counts().to_dict(),
        "risk_label_counts": segmentation_df['risk_label'].value_counts().to_dict(),
        "life_bucket_counts": segmentation_df['life_bucket'].value_counts().to_dict(),
        "action_tag_counts": segmentation_df['action_tag'].value_counts().to_dict(),
        "erl_mean": float(segmentation_df['erl_365_days'].mean()),
        "erl_median": float(segmentation_df['erl_365_days'].median()),
    }

    return SegmentationResponse(
        cutoff_date=CUTOFF_DATE,
        inactivity_days=inactivity_days,
        H_days=H_days,
        n_customers=len(segmentation_df),
        segments=segmentation_df.to_dict(orient="records"),
        cutoffs=cutoffs,
        summary=summary,
    )

