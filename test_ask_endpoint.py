# test_ask_endpoint.py - Test routing logic (SQL vs Analytics) without API calls

def _detect_required_tool(question: str):
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


def determine_system(question: str) -> str:
    """Determine which system should handle the question: 'analytics' or 'sql'"""
    required_tool = _detect_required_tool(question)
    
    if required_tool:
        return "analytics"
    else:
        # Check if it's clearly a SQL question
        question_lower = question.lower()
        sql_keywords = [
            "revenue by", "total revenue", "sum of revenue",
            "how many transactions", "count of transactions", "number of transactions",
            "transactions in", "transactions by", "transactions for",
            "list all", "show all", "all customers", "all transactions",
            "by country", "by month", "by date", "by product"
        ]
        if any(kw in question_lower for kw in sql_keywords):
            return "sql"
        # If no clear indicator, default to SQL (LLM will decide)
        return "sql"


def test_routing(question: str, expected_system: str = None):
    """Test the routing logic for a question"""
    system = determine_system(question)
    required_tool = _detect_required_tool(question)
    
    print(f"\n{'='*60}")
    print(f"Question: {question}")
    print(f"{'='*60}")
    print(f"  System: {system.upper()}")
    if required_tool:
        print(f"  Required Tool: {required_tool}")
    else:
        print(f"  Required Tool: None (LLM will decide)")
    
    if expected_system:
        status = "✓" if system == expected_system else "✗"
        print(f"  Expected: {expected_system.upper()} {status}")
    
    return system


if __name__ == "__main__":
    print("Testing routing logic (SQL vs Analytics)")
    print("=" * 60)
    
    # Analytics questions
    print("\n📊 ANALYTICS QUESTIONS (should route to analytics):")
    test_routing("What are the top 10 customers by CLV?", "analytics")
    test_routing("Show me the top customers by customer lifetime value", "analytics")
    test_routing("Who are the high-risk customers likely to churn?", "analytics")
    test_routing("What is the churn probability for customer 14646 in the next 90 days?", "analytics")
    test_routing("What is the expected lifetime for customer 14646?", "analytics")
    test_routing("Show me customer segments and recommended actions", "analytics")
    test_routing("What are the RFM metrics for customers?", "analytics")
    test_routing("Who is likely to churn?", "analytics")
    
    # SQL questions
    print("\n💾 SQL QUESTIONS (should route to SQL):")
    test_routing("What is the total revenue by country?", "sql")
    test_routing("How many transactions were there in December 2011?", "sql")
    test_routing("Revenue by month in 2011", "sql")
    test_routing("Show me all transactions", "sql")
    test_routing("List all customers", "sql")
    test_routing("Count of transactions by country", "sql")
    test_routing("Sum of revenue by product", "sql")
    
    # Edge cases
    print("\n❓ EDGE CASES:")
    test_routing("What are the top customers?", "sql")  # No CLV mention, should be SQL
    test_routing("Show me revenue", "sql")  # Ambiguous but should be SQL
    test_routing("Top 10 customers", "sql")  # No CLV/value mention, should be SQL
    
    print(f"\n{'='*60}")
    print("Routing tests completed!")
    print(f"{'='*60}")
