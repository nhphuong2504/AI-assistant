import os
import json
import re
import time
from typing import Any, Dict, List, Optional
from dotenv import load_dotenv
from openai import OpenAI, APIError, RateLimitError, APIConnectionError, APITimeoutError
from app.db import ensure_select_only

load_dotenv()

MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Analytics function definitions with fixed parameters
ANALYTICS_FUNCTIONS = [
    {
        "type": "function",
        "function": {
            "name": "predict_customer_lifetime_value",
            "description": "Predict Customer Lifetime Value (CLV) using BG/NBD and Gamma-Gamma models. Use this for questions about: customer lifetime value, CLV, future customer value, predicted revenue per customer, customer worth, or which customers are most valuable. Calibration cutoff date is fixed at 2011-12-09.",
            "parameters": {
                "type": "object",
                "properties": {
                    "horizon_days": {
                        "type": "integer",
                        "description": "Prediction horizon in days (default: 90)",
                        "minimum": 1,
                        "maximum": 365,
                        "default": 90
                    },
                    "limit_customers": {
                        "type": "integer",
                        "description": "Maximum number of customers to return (default: 10)",
                        "minimum": 1,
                        "maximum": 5000,
                        "default": 10
                    }
                },
                "required": ["horizon_days"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "score_churn_risk",
            "description": "Score customers by churn risk using survival analysis (Cox model). Use this for questions about: churn risk, customer retention risk, which customers are likely to churn, risk scoring, or high-risk customers. Cutoff date is fixed at 2011-12-09, inactivity days is fixed at 90.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "predict_churn_probability",
            "description": "Predict the probability that active customers will churn in the next X days. Use this for questions about: churn probability, likelihood of leaving, retention probability, or probability of churn. Cutoff date is fixed at 2011-12-09, inactivity days is fixed at 90.",
            "parameters": {
                "type": "object",
                "properties": {
                    "X_days": {
                        "type": "integer",
                        "description": "Prediction horizon in days (default: 90)",
                        "minimum": 1,
                        "maximum": 365,
                        "default": 90
                    }
                },
                "required": ["X_days"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "customer_segmentation",
            "description": "Build customer segmentation combining risk labels and expected remaining lifetime. Use this for questions about: customer segments, segmentation, customer groups, risk-based segments, or action recommendations for customers. Cutoff date is fixed at 2011-12-09, inactivity days is fixed at 90.",
            "parameters": {
                "type": "object",
                "properties": {
                    "H_days": {
                        "type": "integer",
                        "description": "Horizon in days for expected remaining lifetime (default: 365)",
                        "minimum": 1,
                        "maximum": 3650,
                        "default": 365
                    }
                },
                "required": []
            }
        }
    }
]

# SQL function definition
SQL_FUNCTION = {
    "type": "function",
    "function": {
        "name": "execute_sql_query",
        "description": "Execute a SQL SELECT query to answer questions about historical data, aggregations, filtering, reporting, or data exploration. Use this for descriptive questions that don't require predictive modeling, such as: revenue by country, top customers, sales trends, product analysis, or any data aggregation/filtering questions.",
        "parameters": {
            "type": "object",
            "properties": {
                "sql": {
                    "type": "string",
                    "description": "A valid SQL SELECT query (may start with WITH for CTEs, no semicolons). Must only query the transactions table. Always use LIMIT 500 unless aggregation already produces small output."
                },
                "explanation": {
                    "type": "string",
                    "description": "Brief explanation of what the query does and what it answers"
                }
            },
            "required": ["sql", "explanation"]
        }
    }
}

# Combine all functions
ALL_FUNCTIONS = ANALYTICS_FUNCTIONS + [SQL_FUNCTION]

# Updated system prompt for routing
SYSTEM_ROUTER = """You are a data assistant for an Online Retail SQLite database.
You can answer questions using either SQL queries or specialized analytics functions.

Available tools:
1. execute_sql_query - for descriptive questions, aggregations, filtering, reporting, data exploration
2. Analytics functions - for predictive modeling, CLV prediction, churn analysis, survival analysis

When to use execute_sql_query:
- Questions about historical data, aggregations, counts, sums, averages
- Filtering, grouping, sorting data
- Reporting on what happened
- Simple data exploration
- Revenue by country/month/product
- Top customers/products
- Sales trends

When to use analytics functions:
- Questions about predictions, future values, probabilities
- Customer lifetime value, CLV, future revenue
- Churn risk, retention risk, churn probability
- Survival analysis, retention curves
- Customer segmentation based on risk/lifetime

Always choose the most appropriate tool. If the question can be answered with SQL, use execute_sql_query. If it requires predictive modeling or advanced analytics, use the appropriate analytics function.
"""

SYSTEM = """You are a data assistant for an Online Retail SQLite database.
You must produce a SINGLE valid JSON object with keys:
- sql: string (a single SELECT query, may start with WITH, no semicolons)
- answer: string (short explanation of what the query does)

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


def generate_sql(schema: Dict[str, Any], question: str, max_retries: int = 3, retry_delay: float = 1.0) -> Dict[str, Any]:
    """
    Generate SQL query from natural language question with resilience.
    
    Args:
        schema: Database schema
        question: Natural language question
        max_retries: Maximum number of retry attempts
        retry_delay: Delay between retries in seconds
        
    Returns:
        Dictionary with 'sql' and 'answer' keys
        
    Raises:
        ValueError: If LLM output is invalid after all retries
        APIError: If API calls fail after all retries
    """
    prompt = build_prompt(schema, question)
    last_error = None
    
    for attempt in range(max_retries):
        try:
            resp = client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": SYSTEM},
                    {"role": "user", "content": prompt},
                ],
                temperature=0,
                timeout=30.0,  # 30 second timeout
                response_format={"type": "json_object"},  # Strict JSON mode
            )

            text = resp.choices[0].message.content.strip()
            
            # With strict JSON mode, we shouldn't need to extract from markdown blocks
            # But keep the extraction as a fallback for robustness
            if "```json" in text:
                text = text.split("```json")[1].split("```")[0].strip()
            elif "```" in text:
                text = text.split("```")[1].split("```")[0].strip()

            # Parse JSON strictly
            try:
                obj = json.loads(text)
            except json.JSONDecodeError as e:
                if attempt < max_retries - 1:
                    # Retry on JSON parse error
                    time.sleep(retry_delay * (attempt + 1))
                    continue
                raise ValueError(f"LLM did not return valid JSON. Raw:\n{text}") from e

            # Basic key checks
            missing_keys = [k for k in ["sql", "answer"] if k not in obj]
            if missing_keys:
                if attempt < max_retries - 1:
                    # Retry on missing keys
                    time.sleep(retry_delay * (attempt + 1))
                    continue
                raise ValueError(f"Missing key(s) '{', '.join(missing_keys)}' in LLM output. Raw:\n{text}")

            # Validate SQL before returning
            try:
                validate_sql(obj["sql"])
            except ValueError as ve:
                if attempt < max_retries - 1:
                    # Retry on validation error - the LLM might generate invalid SQL
                    time.sleep(retry_delay * (attempt + 1))
                    continue
                raise ValueError(f"Generated SQL failed validation: {str(ve)}") from ve

            return obj
            
        except (RateLimitError, APIConnectionError, APITimeoutError) as e:
            last_error = e
            if attempt < max_retries - 1:
                # Exponential backoff for API errors
                wait_time = retry_delay * (2 ** attempt)
                time.sleep(wait_time)
                continue
            raise
        except APIError as e:
            last_error = e
            if attempt < max_retries - 1:
                time.sleep(retry_delay * (attempt + 1))
                continue
            raise
        except Exception as e:
            last_error = e
            if attempt < max_retries - 1:
                time.sleep(retry_delay * (attempt + 1))
                continue
            raise ValueError(f"Unexpected error in generate_sql: {str(e)}") from e
    
    # Should not reach here, but handle just in case
    raise ValueError(f"Failed to generate SQL after {max_retries} attempts. Last error: {str(last_error)}")


def route_question(
    schema: Dict[str, Any], 
    question: str, 
    max_retries: int = 3, 
    retry_delay: float = 1.0
) -> Dict[str, Any]:
    """
    Route question using function calling - LLM decides between SQL or analytics.
    
    Args:
        schema: Database schema
        question: Natural language question
        max_retries: Maximum number of retry attempts
        retry_delay: Delay between retries in seconds
        
    Returns:
        Dictionary with:
        - type: "sql" or "analytics"
        - If type="sql": sql, answer
        - If type="analytics": function_name, parameters, answer
    """
    prompt = build_prompt(schema, question)
    last_error = None
    
    for attempt in range(max_retries):
        try:
            resp = client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": SYSTEM_ROUTER},
                    {"role": "user", "content": prompt},
                ],
                tools=ALL_FUNCTIONS,
                tool_choice="auto",  # Let LLM decide which tool to use
                temperature=0,
                timeout=30.0,
            )
            
            message = resp.choices[0].message
            
            # Check if LLM wants to call a function
            if message.tool_calls and len(message.tool_calls) > 0:
                tool_call = message.tool_calls[0]  # Take first tool call
                function_name = tool_call.function.name
                
                try:
                    function_args = json.loads(tool_call.function.arguments)
                except json.JSONDecodeError as e:
                    if attempt < max_retries - 1:
                        time.sleep(retry_delay * (attempt + 1))
                        continue
                    raise ValueError(f"Invalid JSON in function arguments: {str(e)}")
                
                if function_name == "execute_sql_query":
                    # SQL function was chosen
                    sql = function_args.get("sql", "").strip()
                    explanation = function_args.get("explanation", "")
                    
                    # Validate SQL before returning
                    try:
                        validate_sql(sql)
                    except ValueError as ve:
                        if attempt < max_retries - 1:
                            time.sleep(retry_delay * (attempt + 1))
                            continue
                        raise ValueError(f"Generated SQL failed validation: {str(ve)}")
                    
                    return {
                        "type": "sql",
                        "sql": sql,
                        "answer": explanation or "SQL query generated to answer your question."
                    }
                else:
                    # Analytics function was chosen
                    return {
                        "type": "analytics",
                        "function_name": function_name,
                        "parameters": function_args,
                        "answer": f"Using {function_name} to answer your question."
                    }
            else:
                # No function call - this shouldn't happen often, but handle it
                # Fallback: try to generate SQL the old way
                if attempt < max_retries - 1:
                    time.sleep(retry_delay * (attempt + 1))
                    continue
                raise ValueError("LLM did not call any function. Please rephrase your question.")
                
        except (RateLimitError, APIConnectionError, APITimeoutError) as e:
            last_error = e
            if attempt < max_retries - 1:
                wait_time = retry_delay * (2 ** attempt)
                time.sleep(wait_time)
                continue
            raise
        except APIError as e:
            last_error = e
            if attempt < max_retries - 1:
                time.sleep(retry_delay * (attempt + 1))
                continue
            raise
        except Exception as e:
            last_error = e
            if attempt < max_retries - 1:
                time.sleep(retry_delay * (attempt + 1))
                continue
            raise ValueError(f"Failed to route question: {str(e)}")
    
    raise ValueError(f"Failed to route question after {max_retries} attempts. Last error: {str(last_error)}")


SYSTEM_RESULT_TO_TEXT = """You are a data assistant that converts SQL query results into natural language answers.
Given a user's question and the query results, provide a clear, comprehensive natural language answer that directly addresses the question.
Be specific with numbers, dates, and key findings. If the results are empty, explain that clearly.
Do not just summarize the query - explain what the actual data shows."""


def result_to_text(
    question: str, 
    columns: List[str], 
    rows: List[Dict[str, Any]], 
    row_count: int,
    max_retries: int = 3,
    retry_delay: float = 1.0,
    large_result_threshold: int = 100
) -> str:
    """
    Convert SQL query results into natural language text with resilience.
    Handles large result sets by providing statistical summaries.
    
    Args:
        question: The original user question
        columns: List of column names
        rows: List of dictionaries representing the rows
        row_count: Number of rows returned
        max_retries: Maximum number of retry attempts
        retry_delay: Delay between retries in seconds
        large_result_threshold: Threshold for considering results "large"
        
    Returns:
        Natural language explanation of the results
    """
    # Format the results for the LLM
    if row_count == 0:
        results_text = "The query returned no results (empty result set)."
    elif row_count > large_result_threshold:
        # For large results, provide statistical summary instead of raw data
        results_text = f"The query returned {row_count} row(s).\n\n"
        results_text += f"Columns: {', '.join(columns)}\n\n"
        
        # Calculate statistics for numeric columns
        numeric_stats = {}
        sample_rows = rows[:20]  # Sample for analysis
        
        for col in columns:
            try:
                values = [row.get(col) for row in sample_rows if row.get(col) is not None]
                if values:
                    # Try to determine if numeric
                    numeric_vals = []
                    for v in values:
                        try:
                            if isinstance(v, (int, float)):
                                numeric_vals.append(float(v))
                            elif isinstance(v, str):
                                # Try to parse as number
                                numeric_vals.append(float(v.replace(',', '')))
                        except (ValueError, AttributeError):
                            pass
                    
                    if numeric_vals:
                        numeric_stats[col] = {
                            "min": min(numeric_vals),
                            "max": max(numeric_vals),
                            "mean": sum(numeric_vals) / len(numeric_vals),
                            "sample_count": len(numeric_vals)
                        }
            except Exception:
                pass
        
        if numeric_stats:
            results_text += "Statistical summary of numeric columns:\n"
            for col, stats in numeric_stats.items():
                results_text += f"  {col}: min={stats['min']:.2f}, max={stats['max']:.2f}, mean={stats['mean']:.2f}\n"
            results_text += "\n"
        
        # Show sample rows
        results_text += f"Sample of first 20 rows (out of {row_count} total):\n"
        results_text += json.dumps(sample_rows[:20], indent=2)
        
        # Add summary statistics
        results_text += f"\n\nTotal rows: {row_count}"
        if numeric_stats:
            results_text += f"\nNumeric columns analyzed: {len(numeric_stats)}"
    else:
        # For smaller results, show all data
        results_text = f"The query returned {row_count} row(s).\n\n"
        results_text += f"Columns: {', '.join(columns)}\n\n"
        results_text += "Results:\n"
        results_text += json.dumps(rows, indent=2)
    
    prompt = f"""USER QUESTION:
{question}

QUERY RESULTS:
{results_text}

Provide a natural language answer that directly addresses the user's question based on these results.
Be specific with numbers, dates, and findings. If there are no results, explain that clearly.
For large result sets, focus on key patterns, trends, and summary statistics rather than listing all individual rows.
"""
    
    last_error = None
    for attempt in range(max_retries):
        try:
            resp = client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": SYSTEM_RESULT_TO_TEXT},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.3,  # Slightly higher temperature for more natural language
                timeout=60.0,  # 60 second timeout for larger results
            )
            
            return resp.choices[0].message.content.strip()
            
        except (RateLimitError, APIConnectionError, APITimeoutError) as e:
            last_error = e
            if attempt < max_retries - 1:
                # Exponential backoff for API errors
                wait_time = retry_delay * (2 ** attempt)
                time.sleep(wait_time)
                continue
            raise
        except APIError as e:
            last_error = e
            if attempt < max_retries - 1:
                time.sleep(retry_delay * (attempt + 1))
                continue
            raise
        except Exception as e:
            last_error = e
            if attempt < max_retries - 1:
                time.sleep(retry_delay * (attempt + 1))
                continue
            raise ValueError(f"Unexpected error in result_to_text: {str(e)}") from e
    
    # Fallback response if all retries fail
    if row_count == 0:
        return "The query returned no results."
    elif row_count > large_result_threshold:
        return f"The query returned {row_count} rows. Due to the large result set, please review the data table for details."
    else:
        return f"The query returned {row_count} row(s) with columns: {', '.join(columns)}."


def improve_repair_question(original_question: str, error_message: str, sql: Optional[str] = None) -> str:
    """
    Improve repair question by analyzing common SQLite errors and providing specific guidance.
    
    Args:
        original_question: The original user question
        error_message: The error message from SQLite
        sql: The SQL query that failed (optional, for context)
        
    Returns:
        Improved repair question with specific error guidance
    """
    error_lower = error_message.lower()
    
    # Build base repair question
    repair_parts = [
        original_question,
        "",
        f"ERROR: {error_message}",
        ""
    ]
    
    # Add specific guidance based on error type
    if "no such column" in error_lower:
        repair_parts.append("The query references a column that doesn't exist.")
        repair_parts.append("Please check the schema and use the correct column name.")
        if sql:
            # Try to extract the problematic column name
            match = re.search(r"no such column:\s*([^\s]+)", error_lower)
            if match:
                bad_column = match.group(1)
                repair_parts.append(f"The column '{bad_column}' does not exist. Check available columns in the schema.")
    
    elif "no such table" in error_lower:
        repair_parts.append("The query references a table that doesn't exist.")
        repair_parts.append("Available table: 'transactions' (use this table name).")
        if sql:
            match = re.search(r"no such table:\s*([^\s]+)", error_lower)
            if match:
                bad_table = match.group(1)
                repair_parts.append(f"The table '{bad_table}' does not exist. Use 'transactions' instead.")
    
    elif "ambiguous column name" in error_lower:
        repair_parts.append("The column name appears in multiple tables in the query.")
        repair_parts.append("Please qualify the column name with the table name (e.g., 'transactions.column_name').")
    
    elif "misuse of aggregate" in error_lower or "aggregate" in error_lower:
        repair_parts.append("There's an issue with aggregate function usage (COUNT, SUM, AVG, etc.).")
        repair_parts.append("Remember: columns in SELECT must either be aggregated or appear in GROUP BY.")
        repair_parts.append("If using GROUP BY, all non-aggregated columns must be in the GROUP BY clause.")
    
    elif "syntax error" in error_lower:
        repair_parts.append("There's a SQL syntax error in the query.")
        repair_parts.append("Common issues: missing commas, incorrect quotes, unmatched parentheses, or invalid keywords.")
        if sql:
            repair_parts.append("Please review the SQL syntax carefully.")
    
    elif "datatype mismatch" in error_lower or "type" in error_lower:
        repair_parts.append("There's a data type mismatch in the query.")
        repair_parts.append("Check that you're comparing/computing compatible data types.")
        repair_parts.append("Date comparisons should use proper date format (YYYY-MM-DD).")
    
    elif "subquery" in error_lower:
        repair_parts.append("There's an issue with a subquery.")
        repair_parts.append("Ensure subqueries are properly enclosed in parentheses and return the expected number of columns.")
    
    elif "limit" in error_lower:
        repair_parts.append("There's an issue with the LIMIT clause.")
        repair_parts.append("LIMIT should be a positive integer. Place it at the end of the query.")
    
    elif "order by" in error_lower:
        repair_parts.append("There's an issue with the ORDER BY clause.")
        repair_parts.append("Columns in ORDER BY must exist in the SELECT list or be valid column names.")
    
    elif "group by" in error_lower:
        repair_parts.append("There's an issue with the GROUP BY clause.")
        repair_parts.append("All non-aggregated columns in SELECT must appear in GROUP BY.")
    
    elif "join" in error_lower:
        repair_parts.append("There's an issue with a JOIN operation.")
        repair_parts.append("Ensure JOIN conditions are correct and tables exist.")
        repair_parts.append("Note: The main table is 'transactions'.")
    
    else:
        # Generic guidance for unknown errors
        repair_parts.append("Please review the SQL query and ensure it follows SQLite syntax.")
        repair_parts.append("Remember: Only SELECT queries are allowed. Use table 'transactions'.")
    
    # Add general reminders
    repair_parts.append("")
    repair_parts.append("IMPORTANT:")
    repair_parts.append("- Use only the 'transactions' table")
    repair_parts.append("- Check column names match the schema exactly")
    repair_parts.append("- Ensure proper SQL syntax")
    repair_parts.append("- Return a valid SELECT query (no semicolons)")
    
    if sql:
        repair_parts.append("")
        repair_parts.append(f"Previous SQL (for reference): {sql[:200]}...")
    
    return "\n".join(repair_parts)


def validate_sql(sql: str) -> None:
    """
    Validate SQL query to block malicious/invalid SQL before execution.
    Raises ValueError with descriptive message if validation fails.
    """
    if not sql or not sql.strip():
        raise ValueError("SQL query cannot be empty.")
    
    sql_upper = sql.upper().strip()
    
    # Must start with SELECT or WITH (CTE)
    if not (sql_upper.startswith("SELECT") or sql_upper.startswith("WITH")):
        raise ValueError("Only SELECT queries (including WITH/CTE) are allowed.")
    
    # Use the existing ensure_select_only validation from db module
    # This checks for forbidden keywords (INSERT, UPDATE, DELETE, etc.) and multiple statements
    try:
        ensure_select_only(sql)
    except ValueError as e:
        raise ValueError(f"SQL validation failed: {str(e)}")
    
    # Additional checks for suspicious/malicious patterns
    suspicious_patterns = [
        # SQL injection attempts
        (r"'\s*;\s*--", "SQL injection pattern detected (comment injection)"),
        (r"'\s*OR\s*'\s*=\s*'", "SQL injection pattern detected (OR 1=1 style)"),
        (r"'\s*UNION\s+ALL\s+SELECT", "SQL injection pattern detected (UNION injection)"),
        # Dangerous SQLite functions
        (r'\bLOAD_EXTENSION\s*\(', "LOAD_EXTENSION is not allowed"),
        (r'\.read\s*\(', "File read operations are not allowed"),
        (r'\.import\s+', "Import operations are not allowed"),
        # System table access attempts
        (r'sqlite_master\s*WHERE\s*type\s*=\s*["\']table["\']', "Direct sqlite_master access is restricted"),
        # Function calls that could be dangerous
        (r'\bEXEC\s*\(', "EXEC() calls are not allowed"),
        (r'\bEXECUTE\s*\(', "EXECUTE() calls are not allowed"),
    ]
    
    for pattern, message in suspicious_patterns:
        if re.search(pattern, sql, re.IGNORECASE | re.DOTALL):
            raise ValueError(f"Malicious SQL pattern detected: {message}")
    
    # Check for excessive nesting or complexity (basic check to prevent DoS)
    if sql.count('(') > 100 or sql.count(')') > 100:
        raise ValueError("SQL query is too complex (excessive nesting).")
    
    # Check for suspicious string concatenation that might be used for injection
    # This pattern looks for string concatenation that could be used to bypass validation
    if re.search(r"'\s*\+\s*[^']", sql, re.IGNORECASE) or re.search(r"[^']\s*\+\s*'", sql, re.IGNORECASE):
        # Allow simple concatenation but flag suspicious patterns
        if re.search(r"'\s*\+\s*SELECT", sql, re.IGNORECASE):
            raise ValueError("Suspicious string concatenation with SELECT detected.")
