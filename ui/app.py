import streamlit as st
import pandas as pd
import requests

API_URL = st.sidebar.text_input("API URL", "http://127.0.0.1:8000")

#streamlit run ui/app.py

st.title("Retail Data Assistant — SQL Playground")

tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["Schema", "SQL", "Ask (LangChain)", "CLV", "Survival"]
)


with tab1:
    st.subheader("Database schema")
    if st.button("Load schema"):
        r = requests.get(f"{API_URL}/schema", timeout=30)
        if r.status_code != 200:
            st.error(r.text)
        else:
            schema = r.json()
            st.json(schema)

with tab2:
    st.subheader("Run a safe SQL query")

    sample = """SELECT
  country,
  COUNT(DISTINCT customer_id) AS customers,
  SUM(revenue) AS revenue
FROM transactions
GROUP BY country
ORDER BY revenue DESC
LIMIT 20
"""
    sql = st.text_area("SQL (SELECT-only)", value=sample, height=200)
    limit = st.slider("Default LIMIT (applied if query has none)", 1, 5000, 500)

    if st.button("Run query"):
        r = requests.post(f"{API_URL}/query", json={"sql": sql, "limit": limit}, timeout=60)
        if r.status_code != 200:
            st.error(r.text)
        else:
            payload = r.json()
            df = pd.DataFrame(payload["rows"])
            st.write(f"Rows returned: {payload['row_count']}")
            st.dataframe(df, use_container_width=True)

with tab3:
    st.subheader("Ask a question (LangChain Agent)")
    st.caption("Uses LangChain for multi-step reasoning and conversation memory. Supports complex queries that may require multiple tools.")

    # Memory settings
    col1, col2 = st.columns([3, 1])
    with col1:
        use_memory = st.checkbox("Use conversation memory", value=True, 
                                help="Enable to remember context across questions")
    with col2:
        if st.button("Clear Memory"):
            try:
                r = requests.post(f"{API_URL}/ask-langchain/clear-memory", timeout=10)
                if r.status_code == 200:
                    st.success("Memory cleared!")
                else:
                    st.error(f"Failed to clear memory: {r.text}")
            except Exception as e:
                st.error(f"Error clearing memory: {str(e)}")

    # Conversation history (if using memory)
    if 'langchain_history' not in st.session_state:
        st.session_state.langchain_history = []

    # Display conversation history
    if st.session_state.langchain_history and use_memory:
        with st.expander("Conversation History", expanded=False):
            for i, (q, a) in enumerate(st.session_state.langchain_history):
                st.markdown(f"**Q{i+1}:** {q}")
                st.markdown(f"**A{i+1}:** {a}")
                st.divider()

    # Question input
    q = st.text_input("Question", 
                     placeholder="e.g., 'What are the high-risk customers?' or 'Show me churn probability for top 5 high-risk customers'",
                     key="langchain_question")
    
    if st.button("Ask (LangChain)", type="primary"):
        if not q:
            st.warning("Please enter a question")
        else:
            with st.spinner("Processing with LangChain agent..."):
                try:
                    r = requests.post(
                        f"{API_URL}/ask-langchain", 
                        json={"question": q, "use_memory": use_memory}, 
                        timeout=300
                    )
                    if r.status_code != 200:
                        st.error(f"Error: {r.text}")
                    else:
                        payload = r.json()
                        
                        # Add to conversation history
                        if use_memory:
                            st.session_state.langchain_history.append((q, payload["answer"]))
                        
                        st.markdown("### Answer")
                        st.markdown(payload["answer"])
                        
                        # Show info about LangChain capabilities
                        with st.expander("ℹ️ About LangChain Mode"):
                            st.markdown("""
                            **LangChain Agent Features:**
                            - **Multi-step reasoning**: Can chain multiple tools together
                            - **Conversation memory**: Remembers context across questions
                            - **Natural language**: Returns conversational answers
                            - **Automatic tool selection**: Agent chooses the best tools
                            
                            **Example complex queries:**
                            - "Show me the churn probability for the top 5 high-risk customers"
                            - "What's the expected lifetime of customers in the High risk segment?"
                            - "Compare churn probabilities between UK and non-UK customers"
                            """)
                except requests.exceptions.Timeout:
                    st.error("Request timed out. The query may be too complex or the server is busy.")
                except Exception as e:
                    st.error(f"Error: {str(e)}")

with tab4:
    st.subheader("Customer Lifetime Value (BG/NBD + Gamma-Gamma)")

    cutoff = st.date_input("Cutoff date (calibration end)", value=pd.to_datetime("2011-12-09"))
    horizon = st.slider("CLV horizon (days)", 30, 365, 90)

    if st.button("Run CLV"):
        r = requests.post(
            f"{API_URL}/clv",
            json={"cutoff_date": str(cutoff), "horizon_days": int(horizon)},
            timeout=180,
        )
        if r.status_code != 200:
            st.error(r.text)
        else:
            payload = r.json()
            st.json(payload["summary"])

            df = pd.DataFrame(payload["top_customers"])
            st.dataframe(df, use_container_width=True)

with tab5:
    st.subheader("Survival Analysis")
    st.caption("Cutoff date is fixed at 2011-12-09 (inclusive). Inactivity days is fixed at 90 days.")

    st.markdown("## Kaplan–Meier (All customers)")

    if st.button("Run KM (All customers)"):
        r = requests.post(
            f"{API_URL}/survival/km?inactivity_days=90",
            timeout=180,
        )
        if r.status_code != 200:
            st.error(r.text)
        else:
            payload = r.json()
            st.write(f"Cutoff: {payload['cutoff_date']} | Inactivity days: {payload['inactivity_days']}")
            st.write(f"N customers: {payload['n_customers']} | Churn rate: {payload['churn_rate']:.3f}")

            # Display survival curve data
            plot_df = pd.DataFrame(payload["survival_curve"])
            st.markdown("### Survival Curve Data")
            st.dataframe(plot_df, use_container_width=True)

    st.markdown("## Customer Risk Scoring (Cox Model)")

    if st.button("Score Customers"):
        with st.spinner("Fitting Cox model and scoring customers..."):
            r = requests.post(
                f"{API_URL}/survival/score?inactivity_days=90",
                timeout=300,
            )
        if r.status_code != 200:
            st.error(r.text)
        else:
            payload = r.json()
            st.write(f"Cutoff: {payload['cutoff_date']} | Inactivity days: {payload['inactivity_days']}")
            st.write(f"N customers: {payload['n_customers']}")

            # Display summary
            summary = payload['summary']
            st.markdown("### Summary")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Customers", summary['n_customers'])
            with col2:
                st.metric("Mean Risk Score", f"{summary['risk_score_mean']:.3f}")
            with col3:
                st.metric("Max Risk Score", f"{summary['risk_score_max']:.3f}")

            # Risk bucket distribution
            st.markdown("### Risk Bucket Distribution")
            bucket_counts = summary['risk_bucket_counts']
            bucket_df = pd.DataFrame({
                'Risk Bucket': list(bucket_counts.keys()),
                'Count': list(bucket_counts.values())
            })
            st.dataframe(bucket_df, use_container_width=True)

            # Display scored customers
            df = pd.DataFrame(payload['scored_customers'])
            st.markdown("### Scored Customers")
            st.dataframe(df, use_container_width=True)

    st.markdown("## Churn Probability Prediction (Cox Model)")

    col1, col2 = st.columns(2)
    with col1:
        X_days = st.slider("Prediction horizon (days)", 7, 365, 90, help="Probability of churn in the next X days")
    
    if st.button("Predict Churn Probability"):
        with st.spinner("Fitting Cox model and predicting churn probabilities..."):
            r = requests.post(
                f"{API_URL}/survival/churn-probability?inactivity_days=90&X_days={X_days}",
                timeout=300,
            )
        if r.status_code != 200:
            st.error(r.text)
        else:
            payload = r.json()
            st.write(f"Cutoff: {payload['cutoff_date']} | Inactivity days: {payload['inactivity_days']} | Horizon: {payload['X_days']} days")
            st.write(f"N active customers: {payload['n_customers']}")

            # Display summary
            summary = payload['summary']
            st.markdown("### Summary")
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Customers", summary['n_customers'])
            with col2:
                st.metric("Mean Churn Prob", f"{summary['churn_probability_mean']:.3f}")
            with col3:
                st.metric("Median Churn Prob", f"{summary['churn_probability_median']:.3f}")
            with col4:
                st.metric("Max Churn Prob", f"{summary['churn_probability_max']:.3f}")

            # Display predictions
            df = pd.DataFrame(payload['predictions'])
            st.markdown("### Churn Probability Predictions")
            st.dataframe(df, use_container_width=True)

    st.markdown("## Expected Remaining Lifetime (Cox Model)")

    col1 = st.columns(1)[0]
    H_days = st.slider("Horizon (days)", 30, 1825, 365, help="Horizon for restricted expected remaining lifetime")
    
    if st.button("Compute Expected Lifetime"):
        with st.spinner("Fitting Cox model and computing expected remaining lifetime..."):
            r = requests.post(
                f"{API_URL}/survival/expected-lifetime?inactivity_days=90&H_days={H_days}",
                timeout=300,
            )
        if r.status_code != 200:
            st.error(r.text)
        else:
            payload = r.json()
            st.write(f"Cutoff: {payload['cutoff_date']} | Inactivity days: {payload['inactivity_days']} | Horizon: {payload['H_days']} days")
            st.write(f"N active customers: {payload['n_customers']}")

            # Display summary
            summary = payload['summary']
            st.markdown("### Summary")
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Customers", summary['n_customers'])
            with col2:
                st.metric("Mean Expected Lifetime", f"{summary['expected_lifetime_mean']:.2f} days")
            with col3:
                st.metric("Median Expected Lifetime", f"{summary['expected_lifetime_median']:.2f} days")
            with col4:
                st.metric("Max Expected Lifetime", f"{summary['expected_lifetime_max']:.2f} days")

            # Display expected lifetimes
            df = pd.DataFrame(payload['expected_lifetimes'])
            st.markdown("### Expected Remaining Lifetime Predictions")
            st.dataframe(df, use_container_width=True)

    st.markdown("## Customer Segmentation (Risk × Expected Lifetime)")

    if st.button("Build Segmentation Table"):
        with st.spinner("Building segmentation table..."):
            r = requests.post(
                f"{API_URL}/survival/segmentation?inactivity_days=90&H_days=365",
                timeout=300,
            )
        if r.status_code != 200:
            st.error(r.text)
        else:
            payload = r.json()
            st.write(f"Cutoff: {payload['cutoff_date']} | Inactivity days: {payload['inactivity_days']} | Horizon: {payload['H_days']} days")
            st.write(f"N active customers: {payload['n_customers']}")

            # Display cutoffs
            cutoffs = payload['cutoffs']
            st.markdown("### ERL Bucket Cutoffs")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("33rd Percentile (q33)", f"{cutoffs['q33']:.2f} days")
            with col2:
                st.metric("67th Percentile (q67)", f"{cutoffs['q67']:.2f} days")
            with col3:
                st.metric("Horizon (H_days)", f"{cutoffs['H_days']} days")

            # Display summary
            summary = payload['summary']
            st.markdown("### Summary")
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Total Customers", summary['n_customers'])
                st.metric("Mean ERL", f"{summary['erl_mean']:.2f} days")
            with col2:
                st.metric("Median ERL", f"{summary['erl_median']:.2f} days")

            # Segment distribution
            st.markdown("### Segment Distribution")
            df = pd.DataFrame(payload['segments'])
            
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**By Segment:**")
                segment_counts = pd.Series(summary['segment_counts']).sort_index()
                st.dataframe(segment_counts.reset_index().rename(columns={'index': 'Segment', 0: 'Count'}), use_container_width=True, hide_index=True)
            
            with col2:
                st.markdown("**By Action Tag:**")
                action_counts = pd.Series(summary['action_tag_counts']).sort_index()
                st.dataframe(action_counts.reset_index().rename(columns={'index': 'Action Tag', 0: 'Count'}), use_container_width=True, hide_index=True)

            # Display full segmentation table
            st.markdown("### Full Segmentation Table")
            st.dataframe(df, use_container_width=True)

            # Filter by segment
            if len(df) > 0:
                st.markdown("### Filter by Segment")
                selected_segment = st.selectbox("Select segment to view details", 
                                              options=sorted(df['segment'].unique()))
                segment_df = df[df['segment'] == selected_segment]
                st.write(f"**{len(segment_df)} customers in {selected_segment} segment**")
                st.dataframe(segment_df[['customer_id', 'risk_label', 't0', 'erl_365_days', 
                                       'life_bucket', 'action_tag', 'recommended_action']], 
                           use_container_width=True)


