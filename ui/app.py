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


