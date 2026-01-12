from fastapi import FastAPI, HTTPException
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
def km_all(inactivity_days: int = INACTIVITY_DAYS) -> KMResponse:
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
    stratify: str,
    inactivity_days: int = INACTIVITY_DAYS
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
    inactivity_days: int = INACTIVITY_DAYS,
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