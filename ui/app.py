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
    st.subheader("Ask a question (Text-to-SQL)")

    q = st.text_input("Question", "Revenue by month in 2011")
    if st.button("Ask"):
        r = requests.post(f"{API_URL}/ask", json={"question": q}, timeout=90)
        if r.status_code != 200:
            st.error(r.text)
        else:
            payload = r.json()
            st.markdown("### Generated SQL")
            st.code(payload["sql"], language="sql")

            st.markdown("### Answer")
            st.write(payload["answer"])

            df = pd.DataFrame(payload["rows"])
            st.markdown("### Result preview")
            st.dataframe(df, use_container_width=True)

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

with tab4:
    st.subheader("Customer Lifetime Value (BG/NBD + Gamma-Gamma)")

    cutoff = st.date_input("Cutoff date (calibration end)", value=pd.to_datetime("2011-09-30"))
    horizon = st.slider("CLV horizon (days)", 30, 365, 180)

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


