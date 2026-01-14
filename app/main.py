from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field
from typing import Any, Dict, List, Optional
from app.db import get_schema, run_query, run_query_internal
from app.llm import generate_sql
import pandas as pd
from analytics.clv import build_rfm, fit_models, predict_clv
from analytics.survival import (
    build_covariate_table,
    fit_km_all,
    CUTOFF_DATE,
    INACTIVITY_DAYS,
    build_cox_design,
    fit_cox,
    cox_summary_json,
    predict_churn_probabilities,
    get_customer_survival_curve,
)

from analytics.survival import km_stratified



from app.db import get_schema, run_query

app = FastAPI(title="Retail Data Assistant API", version="0.1")


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

class CLVRequest(BaseModel):
    cutoff_date: str = Field("2011-09-30", description="Calibration cutoff date (YYYY-MM-DD)")
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

class KMStratResponse(BaseModel):
    cutoff_date: str
    inactivity_days: int
    stratify: str
    groups: Dict[str, Dict[str, float]]  # {group: {"n":..., "churn_rate":...}}
    curves: Dict[str, List[Dict[str, float]]]  # {group: curve_points}

class CoxResponse(BaseModel):
    cutoff_date: str
    inactivity_days: int
    population: str
    n_customers: int
    covariates: List[str]
    summary: List[Dict[str, Any]]

class ChurnProbResponse(BaseModel):
    cutoff_date: str
    inactivity_days: int
    n_customers: int
    customers: List[Dict[str, Any]]  # List of customer churn probabilities

class CustomerSurvivalResponse(BaseModel):
    cutoff_date: str
    inactivity_days: int
    customer_id: str
    found: bool
    tenure_days: Optional[float] = None
    survival_curve: Optional[List[Dict[str, float]]] = None
    expected_remaining_lifetime: Optional[float] = None
    error: Optional[str] = None

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

    # First attempt: generate SQL
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

    rfm = build_rfm(df, cutoff_date=req.cutoff_date)
    models = fit_models(rfm)
    pred = predict_clv(models, horizon_days=req.horizon_days)

    pred = pred.sort_values("clv", ascending=False)

    top = pred.head(50)[["customer_id", "frequency", "recency", "T", "monetary_value", "pred_purchases", "pred_aov", "clv"]]
    summary = {
        "customers_total": int(len(pred)),
        "customers_with_repeat": int((pred["frequency"] > 0).sum()),
        "clv_mean": float(top["clv"].mean()) if len(top) else 0.0,
        "clv_max": float(top["clv"].max()) if len(top) else 0.0,
    }

    return CLVResponse(
        cutoff_date=req.cutoff_date,
        horizon_days=req.horizon_days,
        top_customers=top.to_dict(orient="records"),
        summary=summary,
    )


@app.post("/survival/km", response_model=KMResponse)
def km_all(inactivity_days: int = Query(INACTIVITY_DAYS, description="Inactivity days threshold for churn definition")) -> KMResponse:
    sql = """
    SELECT customer_id, invoice_no, invoice_date, revenue, stock_code, country
    FROM transactions
    WHERE customer_id IS NOT NULL
    """
    rows, _ = run_query_internal(sql, max_rows=2_000_000)
    df = pd.DataFrame(rows)

    cov = build_covariate_table(
        transactions=df,
        cutoff_date=CUTOFF_DATE,
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

@app.post("/survival/km_strat", response_model=KMStratResponse)
def km_strat(
    stratify: str = Query(..., description="Stratification variable"),
    inactivity_days: int = Query(INACTIVITY_DAYS, description="Inactivity days threshold for churn definition")
) -> KMStratResponse:
    # allowed stratifications
    allowed = {"is_uk", "orders_per_month", "aov", "monetary_value"}
    if stratify not in allowed:
        raise ValueError(f"stratify must be one of {sorted(allowed)}")

    sql = """
    SELECT customer_id, invoice_no, invoice_date, revenue, stock_code, country
    FROM transactions
    WHERE customer_id IS NOT NULL
    """
    rows, _ = run_query_internal(sql, max_rows=2_000_000)
    df = pd.DataFrame(rows)

    cov = build_covariate_table(
        transactions=df,
        cutoff_date=CUTOFF_DATE,
        inactivity_days=inactivity_days,
    ).df

    curves, churn_rates, sizes = km_stratified(cov, stratify=stratify)

    groups = {
        g: {"n": float(sizes[g]), "churn_rate": float(churn_rates[g])}
        for g in curves.keys()
    }

    return KMStratResponse(
        cutoff_date=CUTOFF_DATE,
        inactivity_days=inactivity_days,
        stratify=stratify,
        groups=groups,
        curves=curves,
    )

@app.post("/survival/cox", response_model=CoxResponse)
def cox_model(
    inactivity_days: int = Query(INACTIVITY_DAYS, description="Inactivity days threshold for churn definition"),
) -> CoxResponse:
    sql = """
    SELECT customer_id, invoice_no, invoice_date, revenue, stock_code, country
    FROM transactions
    WHERE customer_id IS NOT NULL
    """
    rows, _ = run_query_internal(sql, max_rows=2_000_000)
    df = pd.DataFrame(rows)

    cov = build_covariate_table(
        transactions=df,
        cutoff_date=CUTOFF_DATE,
        inactivity_days=inactivity_days,
    ).df

    df_cox = build_cox_design(cov=cov)

    # Fit Cox
    cph = fit_cox(df_cox, penalizer=0.1)
    summary = cox_summary_json(cph)

    population = "all_customers"
    covariates = [x for x in df_cox.columns if x not in {"duration", "event"}]

    return CoxResponse(
        cutoff_date=CUTOFF_DATE,
        inactivity_days=inactivity_days,
        population=population,
        n_customers=int(len(df_cox)),
        covariates=covariates,
        summary=summary,
    )

@app.post("/survival/churn_prob", response_model=ChurnProbResponse)
def churn_probabilities(
    inactivity_days: int = Query(INACTIVITY_DAYS, description="Inactivity days threshold for churn definition"),
    horizon_days: int = Query(30, ge=1, le=3650, description="Prediction horizon in days from cutoff"),
    prob_threshold_red: float = Query(0.7, ge=0.0, le=1.0, description="Probability threshold for Red segment (default: 0.7)"),
    prob_threshold_amber_low: float = Query(0.4, ge=0.0, le=1.0, description="Lower probability threshold for Amber segment (default: 0.4)"),
) -> ChurnProbResponse:
    """
    Compute per-customer conditional churn probability at a specified horizon using Cox model.
    
    Returns forward-looking, conditional churn probability:
    P(churn within H | alive at cutoff) = 1 - S(t₀ + H) / S(t₀)
    
    Customers already churned at cutoff are excluded by default.
    Includes segmentation: Red (high risk), Amber (medium risk), Green (low risk).
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
        cutoff_date=CUTOFF_DATE,
        inactivity_days=inactivity_days,
    ).df

    df_cox = build_cox_design(cov=cov)

    # Fit Cox model
    cph = fit_cox(df_cox, penalizer=0.1)
    
    # Predict conditional churn probability for alive customers only
    churn_pred = predict_churn_probabilities(
        cov=cov,
        cph=cph,
        horizon_days=horizon_days,
        include_churned=False,  # Exclude already-churned customers
        prob_threshold_red=prob_threshold_red,
        prob_threshold_amber_low=prob_threshold_amber_low,
    )
    
    # Merge with some customer attributes for context
    # Only merge with customers that are in churn_pred (alive customers)
    customer_info = cov[["customer_id", "tenure_days", "orders_per_month", "aov", "gap_days"]].copy()
    result_df = churn_pred.merge(customer_info, on="customer_id", how="left")
    
    # Sort by conditional churn probability (highest risk first)
    churn_col = f"churn_prob_cond_{horizon_days}d"
    result_df = result_df.sort_values(churn_col, ascending=False)

    return ChurnProbResponse(
        cutoff_date=CUTOFF_DATE,
        inactivity_days=inactivity_days,
        n_customers=int(len(result_df)),
        customers=result_df.to_dict(orient="records"),
    )

@app.get("/survival/customer/{customer_id}", response_model=CustomerSurvivalResponse)
def customer_survival(
    customer_id: str,
    inactivity_days: int = Query(INACTIVITY_DAYS, description="Inactivity days threshold for churn definition"),
) -> CustomerSurvivalResponse:
    """
    Get survival curve and expected remaining lifetime for a specific customer.
    
    Returns the survival curve from cutoff onwards and the expected remaining lifetime.
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
        cutoff_date=CUTOFF_DATE,
        inactivity_days=inactivity_days,
    ).df

    df_cox = build_cox_design(cov=cov)

    # Fit Cox model
    cph = fit_cox(df_cox, penalizer=0.1)
    
    # Get survival curve for the customer
    result = get_customer_survival_curve(
        cov=cov,
        cph=cph,
        customer_id=customer_id,
        include_churned=False,
    )
    
    return CustomerSurvivalResponse(
        cutoff_date=CUTOFF_DATE,
        inactivity_days=inactivity_days,
        **result,
    )