import os
import json
from typing import Any, Dict, List, Optional
from dotenv import load_dotenv
from openai import OpenAI
from app.tools import TOOLS, TOOL_FUNCTIONS

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

ANALYTICS_SYSTEM = """You are a retail analytics assistant with access to specialized tools for CLV, churn risk, churn probability, expected lifetime, customer segmentation, and SQL queries.

TOOL SELECTION RULES:

ANALYTICS TOOLS - Use these for predictions, risk analysis, CLV, and customer insights:

1. get_clv - REQUIRED for:
   - Questions mentioning "CLV", "Customer Lifetime Value", "lifetime value"
   - Questions about "top customers by CLV", "top value customers", "best customers by value"
   - Questions asking to rank/order customers by CLV or predicted value
   - Questions about RFM (Recency, Frequency, Monetary) metrics
   - Example: "What are the top 10 customers by CLV?" → call get_clv(horizon_days=365)

2. get_risk_score - REQUIRED for:
   - Questions mentioning "risk", "high-risk customers", "at-risk customers", "churn risk"
   - Questions asking "who is likely to churn" or "customers likely to churn"
   - Example: "Who are the high-risk customers?" → call get_risk_score()

3. get_churn_probability - Use when asking about probability to churn/stay over a time window.

4. get_expected_lifetime - Use when asking "how long will this customer remain active".

5. get_segmentation - Use when asking about segments, action tags, or recommended actions.

SQL TOOL - Use text_to_sql ONLY for:
- Raw transaction data queries: "revenue by country", "total revenue", "sum of revenue"
- Transaction counts: "how many transactions", "transactions in December", "number of transactions"
- Historical data queries: "list all transactions", "show transactions by date"
- Simple aggregations over raw data (not predictions or analytics)

CRITICAL:
- DO NOT use analytics tools (get_clv, get_risk_score, etc.) for SQL-style questions
- DO NOT use text_to_sql for CLV, risk, churn, or prediction questions
- For "revenue by country" or "transactions in December" → use text_to_sql
- For "top customers by CLV" or "high-risk customers" → use analytics tools
- Extract customer_ids and horizons from the question
- Default horizons: CLV=365 days, churn=90 days

Cutoff date is always 2011-12-09 across all analytics tools.
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


def text_to_sql_tool(schema: Dict[str, Any], question: str) -> Dict[str, Any]:
    """Text-to-SQL wrapper tool that returns dict for consistency with other tools."""
    try:
        result = generate_sql(schema, question)
        return result
    except Exception as e:
        return {"error": str(e)}


# Add SQL tool to registry
TOOL_FUNCTIONS["text_to_sql"] = text_to_sql_tool


def _detect_required_tool(question: str) -> Optional[str]:
    """Detect which tool should be used based on question keywords.
    
    Only forces tool usage for clear analytics questions.
    SQL queries (revenue by country, transaction counts, etc.) should not trigger tools.
    """
    question_lower = question.lower()
    
    # Exclude SQL-style questions - don't force tools for these
    sql_indicators = [
        "revenue by", "total revenue", "sum of revenue",
        "how many transactions", "count of transactions", "number of transactions",
        "transactions in", "transactions by", "transactions for",
        "list all", "show all", "all customers", "all transactions"
    ]
    if any(indicator in question_lower for indicator in sql_indicators):
        return None  # Let LLM decide, don't force analytics tools
    
    # CLV detection - must be specific about CLV/predictions
    clv_keywords = [
        "clv", "customer lifetime value", "lifetime value",
        "top customers by clv", "top value customers", "best customers by value",
        "most valuable customers", "future value", "predicted value", 
        "rfm", "recency frequency monetary"
    ]
    # Only trigger if question is clearly about CLV/predictions, not just "top customers"
    if "clv" in question_lower or "customer lifetime value" in question_lower:
        return "get_clv"
    if "top customers" in question_lower and ("clv" in question_lower or "value" in question_lower or "by clv" in question_lower):
        return "get_clv"
    if "rfm" in question_lower:
        return "get_clv"
    
    # Risk score detection - must be specific about risk/churn
    risk_keywords = [
        "high-risk customers", "at-risk customers", "risk score",
        "customers likely to churn", "who might churn", "who is likely to churn",
        "churn risk"
    ]
    if any(kw in question_lower for kw in risk_keywords):
        return "get_risk_score"
    
    # Churn probability detection
    if "churn probability" in question_lower or "probability to churn" in question_lower:
        return "get_churn_probability"
    
    # Expected lifetime detection
    if "expected lifetime" in question_lower or ("how long will" in question_lower and "customer" in question_lower):
        return "get_expected_lifetime"
    
    # Segmentation detection
    if "segment" in question_lower or "action tag" in question_lower or "recommended action" in question_lower:
        return "get_segmentation"
    
    return None


def generate_analytics_answer(
    schema: Dict[str, Any],
    question: str,
    chat_history: List[Dict] = []
) -> Dict[str, Any]:
    """Generate answer using tool calling with analytics tools.
    
    Note: schema is kept for backward compatibility but not included in messages
    to reduce token usage. Schema is only needed if text_to_sql tool is called.
    """
    # Detect if a specific tool should be forced
    required_tool = _detect_required_tool(question)
    
    # Don't include schema in messages to reduce token usage
    # Schema will be available if text_to_sql tool is called
    messages = [
        {"role": "system", "content": ANALYTICS_SYSTEM},
    ] + chat_history + [
        {"role": "user", "content": question}
    ]
    
    # Determine tool_choice: force tool if we detected one, otherwise auto
    tool_choice = "auto"
    if required_tool:
        # Force the specific tool to be called
        tool_choice = {"type": "function", "function": {"name": required_tool}}
    
    # First model call with tools
    response = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        tools=TOOLS,
        tool_choice=tool_choice,
        temperature=0.1
    )
    
    response_message = response.choices[0].message
    messages.append(response_message.model_dump(exclude_none=True))
    
    # Debug: Check if LLM chose not to use tools
    if not response_message.tool_calls:
        # If no tool calls and LLM provided content, it's trying to answer without tools
        # This shouldn't happen for analytics questions - log for debugging
        if response_message.content:
            print(f"DEBUG: LLM chose not to use tools. Required tool was: {required_tool}. Content: {response_message.content[:200]}...")
    
    # Tool loop (max 3 to prevent infinite loops)
    max_loops = 3
    used_tools = []
    for loop in range(max_loops):
        tool_calls = response_message.tool_calls
        if not tool_calls:
            break
        
        for tool_call in tool_calls:
            function_name = tool_call.function.name
            try:
                function_args = json.loads(tool_call.function.arguments)
            except json.JSONDecodeError as e:
                tool_result = {"error": f"Invalid JSON in tool arguments: {e}"}
                messages.append({
                    "tool_call_id": tool_call.id,
                    "role": "tool",
                    "name": function_name,
                    "content": json.dumps(tool_result, indent=2)
                })
                continue
            
            used_tools.append(function_name)
            
            # Route to correct tool with error handling
            try:
                if function_name == "text_to_sql":
                    # text_to_sql needs schema from outer scope
                    tool_result = text_to_sql_tool(schema, function_args.get("question", question))
                elif function_name in TOOL_FUNCTIONS:
                    tool_result = TOOL_FUNCTIONS[function_name](**function_args)
                else:
                    tool_result = {"error": f"Unknown tool: {function_name}"}
            except Exception as e:
                tool_result = {"error": f"Tool execution failed: {str(e)}"}
            
            # Truncate large tool results to avoid token limit issues
            # Keep summary and limit customers list to first 50
            if isinstance(tool_result, dict) and "customers" in tool_result:
                truncated_result = tool_result.copy()
                if len(truncated_result["customers"]) > 50:
                    truncated_result["customers"] = truncated_result["customers"][:50]
                    truncated_result["_truncated"] = True
                    truncated_result["_total_customers"] = len(tool_result["customers"])
                tool_result = truncated_result
            
            # All tools return dict, so this is safe
            tool_result_str = json.dumps(tool_result, indent=2)
            
            # Limit tool result size to ~50k characters to avoid token limits
            max_result_size = 50000
            if len(tool_result_str) > max_result_size:
                tool_result_str = tool_result_str[:max_result_size] + f"\n... (truncated, original size: {len(json.dumps(tool_result, indent=2))} chars)"
            
            messages.append({
                "tool_call_id": tool_call.id,
                "role": "tool",
                "name": function_name,
                "content": tool_result_str
            })
        
        # Next model call
        response = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            tools=TOOLS,
            tool_choice="auto",
            temperature=0.1
        )
        response_message = response.choices[0].message
        messages.append(response_message.model_dump(exclude_none=True))
    
    return {
        "answer": response_message.content or "No answer generated",
        "used_tools": used_tools,
        "tool_calls_made": len(used_tools),
        "debug_messages": messages[-4:] if len(messages) >= 4 else messages
    }
