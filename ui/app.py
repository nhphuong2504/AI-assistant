import streamlit as st
import pandas as pd
import requests
import plotly.express as px

API_URL = st.sidebar.text_input("API URL", "http://127.0.0.1:8000")

#streamlit run ui/app.py

st.title("Retail Data Assistant — SQL Playground")

tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["Schema", "SQL", "Ask", "CLV", "Survival"]
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

            if len(df.columns) >= 2 and len(df) > 0:
                st.markdown("### Quick chart")
                x = st.selectbox("X axis", options=list(df.columns), index=0)
                y = st.selectbox("Y axis", options=list(df.columns), index=1)
                chart_type = st.selectbox("Chart type", ["bar", "line", "scatter"], index=0)

                if chart_type == "bar":
                    fig = px.bar(df, x=x, y=y)
                elif chart_type == "line":
                    fig = px.line(df, x=x, y=y)
                else:
                    fig = px.scatter(df, x=x, y=y)

                st.plotly_chart(fig, use_container_width=True)

with tab3:
    st.subheader("Ask a question (Analytics Tools + SQL)")
    st.caption("The system will automatically use analytics tools (CLV, risk, churn, etc.) or SQL based on your question.")
    
    # Sample questions
    st.markdown("### Sample Questions")
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Analytics Questions:**")
        sample_analytics = [
            "What are the top 10 customers by CLV?",
            "Who are the high-risk customers likely to churn?",
            "What is the churn probability for customer 14646 in the next 90 days?",
            "What is the expected lifetime for customer 14646?",
            "Show me customer segments and recommended actions"
        ]
        for i, sample in enumerate(sample_analytics):
            if st.button(f"📊 {sample}", key=f"analytics_{i}", use_container_width=True):
                st.session_state.question = sample
    
    with col2:
        st.markdown("**SQL Questions:**")
        sample_sql = [
            "What is the total revenue by country?",
            "How many transactions were there in December 2011?",
            "Revenue by month in 2011",
            "Show me the top 5 countries by customer count"
        ]
        for i, sample in enumerate(sample_sql):
            if st.button(f"💾 {sample}", key=f"sql_{i}", use_container_width=True):
                st.session_state.question = sample
    
    # Question input
    default_q = st.session_state.get("question", "What are the top 10 customers by CLV?")
    q = st.text_input("Question", value=default_q)
    
    if st.button("Ask", type="primary"):
        with st.spinner("Processing your question..."):
            r = requests.post(f"{API_URL}/ask", json={"question": q}, timeout=180)
        
        if r.status_code != 200:
            st.error(r.text)
        else:
            payload = r.json()
            
            # Show which tools were used
            used_tools = payload.get("used_tools", [])
            debug_info = payload.get("debug_info", {})
            
            st.markdown("### Tool Usage")
            if used_tools:
                tool_badges = ", ".join([f"`{tool}`" for tool in used_tools])
                st.success(f"✅ Used analytics tools: {tool_badges}")
                if debug_info:
                    st.info(f"Tool calls made: {debug_info.get('tool_calls_made', len(used_tools))}")
            else:
                st.info("ℹ️ Used SQL fallback (no analytics tools)")
            
            # Show SQL if generated
            if payload.get("sql"):
                st.markdown("### Generated SQL")
                st.code(payload["sql"], language="sql")
            
            # Show answer
            st.markdown("### Answer")
            st.markdown(payload["answer"])
            
            # Show results if available
            if payload.get("rows") and len(payload["rows"]) > 0:
                df = pd.DataFrame(payload["rows"])
                st.markdown(f"### Result preview ({payload.get('row_count', len(df))} rows)")
                st.dataframe(df, use_container_width=True)
                
                # Show chart if available
                chart = payload.get("chart")
                if chart and len(df) > 0 and chart.get("x") in df.columns and chart.get("y") in df.columns:
                    st.markdown("### Suggested chart")
                    if chart["type"] == "bar":
                        fig = px.bar(df, x=chart["x"], y=chart["y"], title=chart.get("title"))
                    elif chart["type"] == "line":
                        fig = px.line(df, x=chart["x"], y=chart["y"], title=chart.get("title"))
                    else:
                        fig = px.scatter(df, x=chart["x"], y=chart["y"], title=chart.get("title"))
                    st.plotly_chart(fig, use_container_width=True)
            elif payload.get("row_count", 0) == 0 and not used_tools:
                st.warning("No results returned. This might be an analytics-only response.")
            
            # Debug info (expandable)
            if debug_info:
                with st.expander("🔍 Debug Information"):
                    st.json(debug_info)

with tab4:
    st.subheader("Customer Lifetime Value (BG/NBD + Gamma-Gamma)")
    st.caption("Cutoff date is fixed at 2011-12-09 (inclusive).")

    horizon = st.slider("CLV horizon (days)", 30, 365, 90)

    if st.button("Run CLV"):
        r = requests.post(
            f"{API_URL}/clv",
            json={"horizon_days": int(horizon)},
            timeout=180,
        )
        if r.status_code != 200:
            st.error(r.text)
        else:
            payload = r.json()
            st.json(payload["summary"])

            df = pd.DataFrame(payload["top_customers"])
            st.dataframe(df, use_container_width=True)

            if "clv" in df.columns:
                fig = px.bar(df, x="customer_id", y="clv", title="Top customers by predicted CLV")
                st.plotly_chart(fig, use_container_width=True)

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

            # Build a plotting dataframe
            plot_df = pd.DataFrame(payload["survival_curve"])

            fig = px.line(
                plot_df,
                x="time",
                y="survival",
                title="Kaplan–Meier Survival Curve (All customers)",
            )
            st.plotly_chart(fig, use_container_width=True)

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

            # Charts
            if len(df) > 0:
                st.markdown("### Risk Score Distribution")
                fig = px.histogram(df, x="risk_score", nbins=50, title="Distribution of Risk Scores")
                st.plotly_chart(fig, use_container_width=True)

                st.markdown("### Risk Percentile vs Risk Score")
                fig = px.scatter(df, x="risk_percentile", y="risk_score", 
                               color="risk_bucket", 
                               title="Risk Percentile vs Risk Score",
                               labels={"risk_percentile": "Risk Percentile", "risk_score": "Risk Score"})
                st.plotly_chart(fig, use_container_width=True)

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

            # Charts
            if len(df) > 0:
                st.markdown("### Churn Probability Distribution")
                fig = px.histogram(df, x="churn_probability", nbins=50, 
                                 title=f"Distribution of Churn Probabilities (next {X_days} days)")
                st.plotly_chart(fig, use_container_width=True)

                st.markdown("### Churn Probability vs Current Duration (t0)")
                fig = px.scatter(df, x="t0", y="churn_probability", 
                               title=f"Churn Probability vs Current Duration (next {X_days} days)",
                               labels={"t0": "Current Duration (days)", "churn_probability": "Churn Probability"})
                st.plotly_chart(fig, use_container_width=True)

                st.markdown("### Survival Probabilities")
                fig = px.scatter(df, x="survival_at_t0", y="survival_at_t0_plus_X",
                               title="Survival at t0 vs Survival at t0+X",
                               labels={"survival_at_t0": "Survival at t0", 
                                      "survival_at_t0_plus_X": "Survival at t0+X"})
                st.plotly_chart(fig, use_container_width=True)

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

            # Charts
            if len(df) > 0:
                st.markdown("### Expected Lifetime Distribution")
                fig = px.histogram(df, x="expected_remaining_life_days", nbins=50, 
                                 title=f"Distribution of Expected Remaining Lifetime (horizon: {H_days} days)")
                st.plotly_chart(fig, use_container_width=True)

                st.markdown("### Expected Lifetime vs Current Duration (t0)")
                fig = px.scatter(df, x="t0", y="expected_remaining_life_days", 
                               title=f"Expected Remaining Lifetime vs Current Duration (horizon: {H_days} days)",
                               labels={"t0": "Current Duration (days)", 
                                      "expected_remaining_life_days": "Expected Remaining Lifetime (days)"})
                st.plotly_chart(fig, use_container_width=True)

                # Show relationship: customers with longer tenure should generally have lower expected remaining lifetime
                # (they've already "used up" more of their lifetime)
                st.markdown("### Expected Lifetime by Current Duration Bins")
                bins = pd.cut(df['t0'], bins=10)
                df['t0_bin'] = bins
                df['t0_bin_center'] = df['t0_bin'].apply(lambda x: x.mid)
                bin_stats = df.groupby('t0_bin_center')['expected_remaining_life_days'].agg(['mean', 'std', 'count']).reset_index()
                fig = px.bar(bin_stats, x='t0_bin_center', y='mean',
                           error_y='std',
                           title="Mean Expected Remaining Lifetime by Current Duration",
                           labels={"t0_bin_center": "Current Duration (days)", 
                                  "mean": "Mean Expected Remaining Lifetime (days)"})
                st.plotly_chart(fig, use_container_width=True)

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

            # Charts
            if len(df) > 0:
                st.markdown("### Segment Distribution (Bar Chart)")
                segment_counts = df['segment'].value_counts().sort_index()
                fig = px.bar(x=segment_counts.index, y=segment_counts.values,
                           title="Number of Customers by Segment",
                           labels={"x": "Segment", "y": "Number of Customers"})
                st.plotly_chart(fig, use_container_width=True)

                st.markdown("### Risk Label vs Life Bucket (Heatmap)")
                pivot = pd.crosstab(df['risk_label'], df['life_bucket'])
                fig = px.imshow(pivot.values, 
                              labels=dict(x="Life Bucket", y="Risk Label", color="Count"),
                              x=pivot.columns,
                              y=pivot.index,
                              title="Customer Count by Risk Label × Life Bucket",
                              text_auto=True,
                              aspect="auto")
                st.plotly_chart(fig, use_container_width=True)

                st.markdown("### ERL Distribution by Segment")
                fig = px.box(df, x='segment', y='erl_365_days',
                           title="Expected Remaining Lifetime Distribution by Segment",
                           labels={"segment": "Segment", "erl_365_days": "Expected Remaining Lifetime (days)"})
                fig.update_xaxes(tickangle=45)
                st.plotly_chart(fig, use_container_width=True)

                # Filter by segment
                st.markdown("### Filter by Segment")
                selected_segment = st.selectbox("Select segment to view details", 
                                              options=sorted(df['segment'].unique()))
                segment_df = df[df['segment'] == selected_segment]
                st.write(f"**{len(segment_df)} customers in {selected_segment} segment**")
                st.dataframe(segment_df[['customer_id', 'risk_label', 't0', 'erl_365_days', 
                                       'life_bucket', 'action_tag', 'recommended_action']], 
                           use_container_width=True)


