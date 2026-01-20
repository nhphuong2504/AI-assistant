# test_ask_endpoint.py - Test the /ask endpoint with analytics and SQL fallback

import requests
import json

API_URL = "http://127.0.0.1:8000"

def test_ask(question: str, description: str):
    """Test the /ask endpoint with a question"""
    print(f"\n{'='*60}")
    print(f"Test: {description}")
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
            return
        
        data = response.json()
        
        print(f"\n✓ Response received")
        print(f"  Answer: {data.get('answer', 'N/A')[:200]}...")
        print(f"  SQL: {data.get('sql', 'N/A')[:100]}...")
        print(f"  Used Tools: {data.get('used_tools', None)}")
        print(f"  Tool Calls Made: {data.get('debug_info', {}).get('tool_calls_made', 0) if data.get('debug_info') else 0}")
        print(f"  Row Count: {data.get('row_count', 0)}")
        
        if data.get('used_tools'):
            print(f"  ✓ Analytics tools were used: {data['used_tools']}")
        elif data.get('sql'):
            print(f"  ✓ SQL fallback was used")
        
    except requests.exceptions.ConnectionError:
        print("❌ Connection error: Is the server running?")
        print("   Start with: uvicorn app.main:app --reload --host 127.0.0.1 --port 8000")
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    print("Testing /ask endpoint")
    print("Make sure the server is running: uvicorn app.main:app --reload --host 127.0.0.1 --port 8000")
    
    # Test 1: Analytics question (should use tools)
    test_ask(
        "What are the top 10 customers by CLV?",
        "Analytics - CLV (should use get_clv tool)"
    )
    
    # Test 2: Analytics question (should use tools)
    test_ask(
        "Who are the high-risk customers likely to churn?",
        "Analytics - Risk Score (should use get_risk_score tool)"
    )
    
    # Test 3: Analytics question (should use tools)
    test_ask(
        "What is the churn probability for customer 14646 in the next 90 days?",
        "Analytics - Churn Probability (should use get_churn_probability tool)"
    )
    
    # Test 4: SQL question (should fallback to SQL)
    test_ask(
        "What is the total revenue by country?",
        "SQL Fallback - Raw data query (should use SQL)"
    )
    
    # Test 5: SQL question (should fallback to SQL)
    test_ask(
        "How many transactions were there in December 2011?",
        "SQL Fallback - Simple aggregate (should use SQL)"
    )
    
    # Test 6: Analytics question with customer filter
    test_ask(
        "What is the expected lifetime for customer 14646?",
        "Analytics - Expected Lifetime with customer filter"
    )
    
    print(f"\n{'='*60}")
    print("All tests completed!")
    print(f"{'='*60}")

