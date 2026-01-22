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


def extract_customer_ids_from_question(question: str) -> List[str]:
    """
    Extract customer IDs from a question using pattern matching.
    
    Args:
        question: The user question
        
    Returns:
        List of customer IDs found in the question (as strings)
    """
    customer_ids = []
    
    # Patterns to match customer IDs in various formats
    # Order matters - more specific patterns first
    patterns = [
        r'\bcustomer\s+(?:id\s+)?(?:is\s+)?(?:#)?(\d+)',  # "customer 12345", "customer id 12345", "customer 14646"
        r'\bcustomer\s+(?:#)?(\d+)',  # "customer #12345"
        r'\bcust\s+(?:#)?(\d+)',  # "cust 12345"
        r'\bid\s+(?:is\s+)?(?:#)?(\d+)',  # "id 12345", "id is 12345"
        r'customer\s+(?:with\s+)?id\s+(?:of\s+)?(?:#)?(\d+)',  # "customer with id 12345"
        r'for\s+customer\s+(?:#)?(\d+)',  # "for customer 12345"
        r'customer\s+(?:#)?(\d+)\'?s',  # "customer 12345's"
        r'customer\s+(?:#)?(\d+)\s+(?:has|have|is|was|in)',  # "customer 12345 has", "customer 14646 in"
    ]
    
    # Try each pattern
    for pattern in patterns:
        matches = re.findall(pattern, question, re.IGNORECASE)
        customer_ids.extend(matches)
    
    # Also look for multiple customer IDs (comma-separated, "and", etc.)
    # Pattern: "customers 12345, 67890" or "customers 12345 and 67890"
    multi_patterns = [
        r'customers?\s+(?:#)?(\d+)(?:\s*[,and]+\s*(?:#)?(\d+))+',  # "customers 12345, 67890"
        r'customer\s+(?:#)?(\d+)\s+and\s+(?:#)?(\d+)',  # "customer 12345 and 67890"
        r'customers?\s+(?:#)?(\d+)\s*,\s*(?:#)?(\d+)',  # "customers 12345, 67890"
    ]
    
    for pattern in multi_patterns:
        matches = re.findall(pattern, question, re.IGNORECASE)
        for match in matches:
            if isinstance(match, tuple):
                customer_ids.extend([m for m in match if m])
            else:
                customer_ids.append(match)
    
    # Also try to extract numbers that appear after customer-related keywords
    # This catches cases like "what is customer 12345's CLV"
    context_patterns = [
        r'(?:customer|cust|id)\s+(?:#)?(\d+)',  # General pattern
    ]
    
    for pattern in context_patterns:
        matches = re.findall(pattern, question, re.IGNORECASE)
        customer_ids.extend(matches)
    
    # Remove duplicates while preserving order
    seen = set()
    unique_ids = []
    for cid in customer_ids:
        cid_clean = cid.strip()
        if cid_clean and cid_clean not in seen:
            seen.add(cid_clean)
            unique_ids.append(cid_clean)
    
    return unique_ids


def detect_customer_lookup_question(question: str, columns: List[str]) -> bool:
    """
    Detect if the question is asking about specific customer(s).
    
    Args:
        question: The user question
        columns: List of column names in the results
        
    Returns:
        True if question appears to be a customer lookup, False otherwise
    """
    if "customer_id" not in columns:
        return False
    
    question_lower = question.lower()
    
    # Keywords that suggest customer-specific lookup
    customer_lookup_keywords = [
        "customer",
        "cust",
        "for customer",
        "what is",
        "show me",
        "tell me about",
        "details for",
        "information about",
    ]
    
    # Check if question contains customer lookup keywords
    has_keyword = any(keyword in question_lower for keyword in customer_lookup_keywords)
    
    # Check if question contains customer ID patterns
    customer_ids = extract_customer_ids_from_question(question)
    has_customer_id = len(customer_ids) > 0
    
    # Also check for questions like "what is the CLV of customer X"
    has_specific_metric = any(metric in question_lower for metric in [
        "clv", "lifetime value", "churn risk", "churn probability", 
        "segment", "lifetime", "risk score"
    ])
    
    # Question is a customer lookup if:
    # 1. Has customer ID(s) explicitly mentioned, OR
    # 2. Has customer keyword + specific metric (e.g., "what is the CLV of customer 12345")
    return (has_customer_id) or (has_keyword and has_specific_metric and "customer" in question_lower)


def filter_rows_by_customer_ids(rows: List[Dict[str, Any]], customer_ids: List[str], customer_id_column: str = "customer_id") -> List[Dict[str, Any]]:
    """
    Filter rows to only include those matching the specified customer IDs.
    
    Args:
        rows: List of row dictionaries
        customer_ids: List of customer IDs to filter for (as strings)
        customer_id_column: Name of the customer ID column
        
    Returns:
        Filtered list of rows
    """
    if not customer_ids:
        return rows
    
    # Convert customer_ids to appropriate types (handle both string and numeric IDs)
    filtered_rows = []
    # Normalize customer IDs from question (remove leading zeros, handle as strings)
    normalized_customer_ids = {str(int(cid)) if cid.isdigit() else str(cid).strip() for cid in customer_ids}
    
    for row in rows:
        row_customer_id = row.get(customer_id_column)
        if row_customer_id is None:
            continue
        
        # Convert row customer ID to normalized string format
        # Handle both string and numeric types, remove leading zeros
        try:
            if isinstance(row_customer_id, (int, float)):
                row_id_normalized = str(int(row_customer_id))
            else:
                # Try to parse as number first, then use as string
                try:
                    row_id_normalized = str(int(float(str(row_customer_id).strip())))
                except (ValueError, TypeError):
                    row_id_normalized = str(row_customer_id).strip()
        except (ValueError, TypeError):
            row_id_normalized = str(row_customer_id).strip()
        
        # Check if normalized IDs match
        if row_id_normalized in normalized_customer_ids:
            filtered_rows.append(row)
    
    return filtered_rows


def result_to_text(
    question: str, 
    columns: List[str], 
    rows: List[Dict[str, Any]], 
    row_count: int,
    max_retries: int = 3,
    retry_delay: float = 1.0,
    large_result_threshold: int = 100,
    max_rows_for_llm: int = 100,
    is_segmentation: bool = False
) -> str:
    """
    Convert SQL query results into natural language text with resilience.
    Handles large result sets by providing statistical summaries.
    Automatically detects and filters for customer-specific lookups.
    
    Args:
        question: The original user question
        columns: List of column names
        rows: List of dictionaries representing the rows
        row_count: Number of rows returned
        max_retries: Maximum number of retry attempts
        retry_delay: Delay between retries in seconds
        large_result_threshold: Threshold for considering results "large"
        max_rows_for_llm: Maximum number of rows to include in LLM prompt
        is_segmentation: If True, provide segment-aware summaries
        
    Returns:
        Natural language explanation of the results
    """
    # Step 1: Detect if this is a customer lookup question and filter rows
    original_row_count = row_count
    filtered_rows = rows
    is_customer_lookup = False
    customer_ids_found = []
    
    if detect_customer_lookup_question(question, columns):
        customer_ids_found = extract_customer_ids_from_question(question)
        if customer_ids_found:
            is_customer_lookup = True
            filtered_rows = filter_rows_by_customer_ids(rows, customer_ids_found)
            
            # If no matching customers found, provide helpful message with context
            if not filtered_rows:
                # Check if we have any rows at all (to provide better context)
                if original_row_count == 0:
                    return f"No data found for customer(s): {', '.join(customer_ids_found)}. The query returned no results."
                
                # Provide context about why customer might not be found
                context_msg = f"No data found for customer(s): {', '.join(customer_ids_found)}."
                
                # Add helpful context based on the type of query
                if is_segmentation:
                    context_msg += " This customer may not be active at the cutoff date (2011-12-09) or may not meet the segmentation criteria. Segmentation only includes active customers."
                else:
                    context_msg += " Please verify the customer ID(s) are correct and that the customer exists in the dataset."
                
                # Show sample of available customer IDs to help user
                if original_row_count > 0 and original_row_count <= 100:
                    # If dataset is small, show available customer IDs
                    available_ids = [str(row.get("customer_id", "")) for row in rows[:10] if row.get("customer_id") is not None]
                    if available_ids:
                        context_msg += f" Sample of available customer IDs: {', '.join(available_ids[:5])}..."
                elif original_row_count > 0:
                    # For large datasets, just mention the count
                    context_msg += f" The dataset contains {original_row_count} customer records."
                
                return context_msg
            
            # Update row_count to reflect filtered results
            row_count = len(filtered_rows)
    
    # Use filtered_rows for all subsequent processing
    rows = filtered_rows
    
    # Add context about filtering if customer lookup was performed
    filter_context = ""
    if is_customer_lookup and customer_ids_found:
        if len(customer_ids_found) == 1:
            filter_context = f"\nNote: Results filtered for customer {customer_ids_found[0]} (showing {row_count} matching record(s) out of {original_row_count} total).\n"
        else:
            filter_context = f"\nNote: Results filtered for customers {', '.join(customer_ids_found)} (showing {row_count} matching record(s) out of {original_row_count} total).\n"
    
    # Format the results for the LLM
    if row_count == 0:
        results_text = "The query returned no results (empty result set)."
    elif is_segmentation and "segment" in columns:
        # Segment-aware logic for segmentation results
        results_text = f"The segmentation analysis returned {row_count} customer(s).\n\n"
        results_text += f"Columns: {', '.join(columns)}\n\n"
        
        # Group by segment
        segment_groups = {}
        for row in rows:
            segment = row.get("segment", "Unknown")
            if segment not in segment_groups:
                segment_groups[segment] = []
            segment_groups[segment].append(row)
        
        # Segment-level statistics
        results_text += f"Segment Distribution ({len(segment_groups)} segments):\n"
        segment_counts = {seg: len(segment_rows) for seg, segment_rows in segment_groups.items()}
        for segment, count in sorted(segment_counts.items(), key=lambda x: x[1], reverse=True):
            pct = (count / row_count) * 100
            results_text += f"  {segment}: {count} customers ({pct:.1f}%)\n"
        results_text += "\n"
        
        # Segment-level statistics for numeric columns
        numeric_cols = [col for col in columns if col not in ["customer_id", "segment", "risk_label", "life_bucket", "action_tag", "recommended_action"]]
        
        if numeric_cols:
            results_text += "Segment-level Statistics:\n"
            for segment, segment_rows in sorted(segment_groups.items(), key=lambda x: len(x[1]), reverse=True):
                results_text += f"\n  Segment: {segment} ({len(segment_rows)} customers)\n"
                
                for col in numeric_cols:
                    try:
                        values = [row.get(col) for row in segment_rows if row.get(col) is not None]
                        if values:
                            numeric_vals = []
                            for v in values:
                                try:
                                    if isinstance(v, (int, float)):
                                        numeric_vals.append(float(v))
                                except (ValueError, AttributeError):
                                    pass
                            
                            if numeric_vals:
                                results_text += f"    {col}: min={min(numeric_vals):.2f}, max={max(numeric_vals):.2f}, mean={sum(numeric_vals)/len(numeric_vals):.2f}\n"
                    except Exception:
                        pass
                
                # Show segment characteristics
                if "risk_label" in columns:
                    risk_labels = [row.get("risk_label") for row in segment_rows if row.get("risk_label")]
                    if risk_labels:
                        risk_dist = {}
                        for rl in risk_labels:
                            risk_dist[rl] = risk_dist.get(rl, 0) + 1
                        results_text += f"    Risk labels: {dict(risk_dist)}\n"
                
                if "life_bucket" in columns:
                    life_buckets = [row.get("life_bucket") for row in segment_rows if row.get("life_bucket")]
                    if life_buckets:
                        life_dist = {}
                        for lb in life_buckets:
                            life_dist[lb] = life_dist.get(lb, 0) + 1
                        results_text += f"    Life buckets: {dict(life_dist)}\n"
                
                if "recommended_action" in columns:
                    actions = [row.get("recommended_action") for row in segment_rows if row.get("recommended_action")]
                    if actions:
                        unique_actions = list(set(actions))
                        results_text += f"    Recommended actions: {', '.join(unique_actions[:3])}\n"
        
        results_text += f"\n\nTotal customers: {row_count}"
        results_text += f"\nTotal segments: {len(segment_groups)}"
        
        # Show sample customers from top segments
        results_text += f"\n\nSample customers (up to 5 per top segment):\n"
        top_segments = sorted(segment_groups.items(), key=lambda x: len(x[1]), reverse=True)[:3]
        sample_count = 0
        for segment, segment_rows in top_segments:
            if sample_count >= 15:  # Limit total samples
                break
            sample_rows = segment_rows[:5]
            results_text += f"\n  {segment} segment ({len(segment_rows)} total):\n"
            results_text += json.dumps(sample_rows, indent=4)
            sample_count += len(sample_rows)
    
    elif row_count > large_result_threshold or row_count > max_rows_for_llm:
        # For large results, provide statistical summary instead of raw data
        results_text = f"The query returned {row_count} row(s).\n\n"
        results_text += f"Columns: {', '.join(columns)}\n\n"
        
        # Calculate statistics for numeric columns using ALL rows (not just sample)
        numeric_stats = {}
        # Use all rows for statistics calculation, but limit sample for display
        sample_rows = rows[:min(20, len(rows))]  # Sample for display
        
        for col in columns:
            try:
                # Calculate stats on ALL rows for accuracy
                values = [row.get(col) for row in rows if row.get(col) is not None]
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
                            "count": len(numeric_vals),
                            "total": len(values)
                        }
            except Exception:
                pass
        
        if numeric_stats:
            results_text += "Statistical summary of numeric columns (calculated from all rows):\n"
            for col, stats in numeric_stats.items():
                results_text += f"  {col}: min={stats['min']:.2f}, max={stats['max']:.2f}, mean={stats['mean']:.2f}, count={stats['count']}\n"
            results_text += "\n"
        
        # Show sample rows (limited)
        results_text += f"Sample of first {len(sample_rows)} rows (out of {row_count} total):\n"
        results_text += json.dumps(sample_rows, indent=2)
        
        # Add summary statistics
        results_text += f"\n\nTotal rows: {row_count}"
        if numeric_stats:
            results_text += f"\nNumeric columns analyzed: {len(numeric_stats)}"
    else:
        # For smaller results, show all data (but still respect max_rows_for_llm)
        if row_count <= max_rows_for_llm:
            results_text = f"The query returned {row_count} row(s).\n\n"
            results_text += f"Columns: {', '.join(columns)}\n\n"
            results_text += "Results:\n"
            results_text += json.dumps(rows, indent=2)
        else:
            # Even if under threshold, respect max_rows_for_llm
            results_text = f"The query returned {row_count} row(s).\n\n"
            results_text += f"Columns: {', '.join(columns)}\n\n"
            results_text += f"Showing first {max_rows_for_llm} rows:\n"
            results_text += json.dumps(rows[:max_rows_for_llm], indent=2)
    
    prompt = f"""USER QUESTION:
{question}
{filter_context}
QUERY RESULTS:
{results_text}

Provide a natural language answer that directly addresses the user's question based on these results.
Be specific with numbers, dates, and findings. If there are no results, explain that clearly.
For large result sets, focus on key patterns, trends, and summary statistics rather than listing all individual rows.
If the results were filtered for specific customer(s), make sure to mention that in your answer.
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
