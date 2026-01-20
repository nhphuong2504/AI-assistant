# test_sql_routing.py - Test SQL questions with real API answers

import requests
import json
import pandas as pd

API_URL = "http://127.0.0.1:8000"


def test_sql_question(question: str, show_full_response: bool = False):
    """Test a SQL question with real API call and show the answer"""
    print(f"\n{'='*60}")
    print(f"Question: {question}")
    print(f"{'='*60}")
    
    try:
        response = requests.post(
            f"{API_URL}/ask",
            json={"question": question},
            timeout=120
        )
        
        if response.status_code != 200:
            print(f"❌ Error {response.status_code}: {response.text}")
            return False
        
        data = response.json()
        
        # Check routing
        used_tools = data.get("used_tools", [])
        if used_tools:
            # text_to_sql is the SQL tool, so it's correct
            if "text_to_sql" in used_tools:
                print(f"✓ Correctly routed to SQL (using text_to_sql tool)")
            else:
                # Other analytics tools shouldn't be used for SQL questions
                analytics_tools = [t for t in used_tools if t != "text_to_sql"]
                if analytics_tools:
                    print(f"⚠️  WARNING: Analytics tools were used: {analytics_tools}")
                    print(f"   This should be a SQL question!")
                else:
                    print(f"✓ Tools used: {used_tools}")
        else:
            print(f"✓ Correctly routed to SQL (no tools, direct SQL generation)")
        
        # Show SQL
        sql = data.get("sql", "")
        if sql:
            print(f"\n📝 Generated SQL:")
            print(f"{sql}")
        else:
            print(f"\n⚠️  No SQL generated")
        
        # Show answer
        answer = data.get("answer", "")
        if answer:
            print(f"\n💬 Answer:")
            print(f"{answer}")
        
        # Show results
        rows = data.get("rows", [])
        row_count = data.get("row_count", 0)
        
        if rows and len(rows) > 0:
            print(f"\n📊 Results ({row_count} rows):")
            df = pd.DataFrame(rows)
            # Show first 10 rows
            if len(df) > 10:
                print(df.head(10).to_string(index=False))
                print(f"\n... ({len(df) - 10} more rows)")
            else:
                print(df.to_string(index=False))
        elif row_count > 0:
            print(f"\n📊 Row count: {row_count} (data not shown)")
        else:
            print(f"\n📊 No results returned")
        
        # Show chart info if available
        chart = data.get("chart")
        if chart:
            print(f"\n📈 Chart suggestion: {chart.get('type')} - {chart.get('x')} vs {chart.get('y')}")
        
        # Show debug info if available
        debug_info = data.get("debug_info")
        if debug_info and show_full_response:
            print(f"\n🔍 Debug Info:")
            print(json.dumps(debug_info, indent=2))
        
        return True
        
    except requests.exceptions.ConnectionError:
        print("❌ Connection error: Is the server running?")
        print("   Start with: uvicorn app.main:app --reload --host 127.0.0.1 --port 8000")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False


if __name__ == "__main__":
    import sys
    
    print("Testing SQL Questions with Real API Answers")
    print("=" * 60)
    print("Make sure the server is running: uvicorn app.main:app --reload --host 127.0.0.1 --port 8000")
    print("=" * 60)
    
    # Check if user wants to test all or just one
    test_all = "--all" in sys.argv
    show_debug = "--debug" in sys.argv
    
    sql_questions = [
        # Revenue queries
        "What is the total revenue by country?",
        "Show me revenue by month in 2011",
        "Total revenue for each country",
        "Revenue breakdown by month",
        
        # Transaction counts
        "How many transactions were there in December 2011?",
        "Count of transactions by country",
        "Number of transactions per month",
        
        # List/Show queries
        "Show me all transactions from December 2011",
        "List top 10 customers by transaction count",
        
        # Aggregations
        "Average order value by country",
        "Customer count by country",
        "Product sales by month",
        
        # Date-based queries
        "Transactions in December 2011",
        "Sales in 2011",
        "Revenue by quarter in 2011",
        
        # Simple queries
        "What are the top 5 countries by revenue?",
        "Show me the best selling products"
    ]
    
    if test_all:
        print(f"\nTesting {len(sql_questions)} SQL questions...\n")
        for i, question in enumerate(sql_questions, 1):
            print(f"\n[{i}/{len(sql_questions)}]")
            test_sql_question(question, show_full_response=show_debug)
        
        print(f"\n{'='*60}")
        print("All tests completed!")
        print(f"{'='*60}")
    else:
        # Test just the first question, or let user specify
        if len(sys.argv) > 1 and not sys.argv[1].startswith("--"):
            # User provided a question
            question = " ".join([arg for arg in sys.argv[1:] if not arg.startswith("--")])
            test_sql_question(question, show_full_response=show_debug)
        else:
            # Test first question as example
            print("\nTesting first question as example...")
            print("Use --all to test all questions, or provide a question as argument")
            print("Use --debug to show full debug information\n")
            test_sql_question(sql_questions[0], show_full_response=show_debug)

