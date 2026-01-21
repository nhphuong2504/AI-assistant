import os
import json
from typing import Any, Dict
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

SYSTEM = """You are a data assistant for an Online Retail SQLite database.
You must produce a SINGLE valid JSON object with keys:
- sql: string (a single SELECT query, may start with WITH, no semicolons)
- answer: string (short explanation of what the query does)
- chart: object or null. If chart is not null, include:
    - type: one of ["bar","line","scatter"]
    - x: column name
    - y: column name
    - title: string

Rules:
- Only query the provided tables/columns.
- Use table: transactions (cleaned data).
- Always use LIMIT 500 unless aggregation already produces small output.
- Never use destructive SQL. Only SELECT/WITH SELECT.
"""

def build_prompt(schema: Dict[str, Any], question: str) -> str:
    return f"""SCHEMA (SQLite):
{json.dumps(schema, indent=2)}

USER QUESTION:
{question}

Return JSON only. No markdown. No extra keys.
"""


def generate_sql(schema: Dict[str, Any], question: str) -> Dict[str, Any]:
    prompt = build_prompt(schema, question)

    resp = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": prompt},
        ],
        temperature=0,
    )

    text = resp.choices[0].message.content.strip()

    # Parse JSON strictly
    try:
        obj = json.loads(text)
    except Exception as e:
        raise ValueError(f"LLM did not return valid JSON. Raw:\n{text}") from e

    # Basic key checks
    for k in ["sql", "answer", "chart"]:
        if k not in obj:
            raise ValueError(f"Missing key '{k}' in LLM output. Raw:\n{text}")

    return obj
