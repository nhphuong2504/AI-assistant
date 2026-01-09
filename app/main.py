from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Any, Dict, List, Optional
from app.db import get_schema, run_query
from app.llm import generate_sql


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
