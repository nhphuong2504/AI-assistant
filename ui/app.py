import streamlit as st
import pandas as pd
import requests
import plotly.express as px

API_URL = st.sidebar.text_input("API URL", "http://127.0.0.1:8000")

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

    st.markdown("---")
    st.markdown("## Kaplan–Meier (Stratified)")

    strat = st.selectbox(
        "Stratify KM curves by",
        options=[
            ("is_uk", "UK vs Non-UK"),
            ("orders_per_month", "Orders per month (low/medium/high)"),
            ("aov", "Average order value (low/medium/high)"),
            ("monetary_value", "Monetary value (repeat buyers only; low/medium/high)"),
        ],
        format_func=lambda x: x[1],
    )[0]

    if st.button("Run stratified KM"):
        r = requests.post(
            f"{API_URL}/survival/km_strat?stratify={strat}&inactivity_days=90",
            timeout=180,
        )
        if r.status_code != 200:
            st.error(r.text)
        else:
            payload = r.json()

            st.write(f"Cutoff: {payload['cutoff_date']} | Inactivity days: {payload['inactivity_days']}")

            # Show group stats
            st.markdown("### Group sizes & churn rates")
            grp_df = pd.DataFrame([
                {"group": g, "n": int(info["n"]), "churn_rate": info["churn_rate"]}
                for g, info in payload["groups"].items()
            ]).sort_values("group")
            st.dataframe(grp_df, use_container_width=True)

            # Build a plotting dataframe
            plot_rows = []
            for g, curve in payload["curves"].items():
                for pt in curve:
                    plot_rows.append({"group": g, "time": pt["time"], "survival": pt["survival"]})
            plot_df = pd.DataFrame(plot_rows)

            fig = px.line(
                plot_df,
                x="time",
                y="survival",
                color="group",
                title="Kaplan–Meier Survival Curves (Stratified)",
            )
            st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    st.markdown("## Cox Proportional Hazards")

    if st.button("Run Cox"):
        r = requests.post(
            f"{API_URL}/survival/cox?inactivity_days=90",
            timeout=180,
        )
        if r.status_code != 200:
            st.error(r.text)
        else:
            payload = r.json()
            st.write(f"Population: {payload['population']}")
            st.write(f"N customers: {payload['n_customers']}")
            st.write(f"Cutoff: {payload['cutoff_date']} | Inactivity days: {payload['inactivity_days']}")

            df_sum = pd.DataFrame(payload["summary"])
            # Make it readable
            df_sum = df_sum.sort_values("p")
            st.dataframe(df_sum, use_container_width=True)

    st.markdown("---")
    st.markdown("## Per-Customer Conditional Churn Probabilities")
    st.caption("Forward-looking probabilities: P(churn within H days | alive at cutoff)")

    horizon_days = st.number_input(
        "Prediction horizon (days)",
        min_value=1,
        max_value=3650,
        value=30,
        step=1,
        help="Number of days from cutoff to predict churn probability"
    )
    
    # Fixed segmentation thresholds (not user-configurable)
    prob_threshold_red = 0.7
    prob_threshold_amber_low = 0.4

    if st.button("Compute Churn Probabilities"):
        r = requests.post(
            f"{API_URL}/survival/churn_prob?inactivity_days=90&horizon_days={int(horizon_days)}&prob_threshold_red={prob_threshold_red}&prob_threshold_amber_low={prob_threshold_amber_low}",
            timeout=180,
        )
        if r.status_code != 200:
            st.error(r.text)
        else:
            payload = r.json()
            # Store results in session state so filters don't reset
            st.session_state['churn_prob_payload'] = payload
            st.session_state['churn_prob_horizon'] = int(horizon_days)

    # Display results if available in session state
    if 'churn_prob_payload' in st.session_state:
        payload = st.session_state['churn_prob_payload']
        horizon_days = st.session_state.get('churn_prob_horizon', 30)
        
        st.write(f"Cutoff: {payload['cutoff_date']} | Inactivity days: {payload['inactivity_days']}")
        st.write(f"N customers (alive at cutoff): {payload['n_customers']}")

        df = pd.DataFrame(payload["customers"])
        
        # Get the dynamic churn probability column name
        churn_col = f"churn_prob_cond_{int(horizon_days)}d"
        
        # Filtering and sorting options
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            min_churn_prob = st.slider(
                f"Min {int(horizon_days)}-day conditional churn probability",
                0.0, 1.0, 0.0, 0.01,
                help=f"Filter customers by minimum {int(horizon_days)}-day conditional churn probability"
            )
        with col2:
            segment_filter = st.selectbox(
                "Filter by segment",
                options=["All", "Red", "Amber", "Green"],
                index=0,
                help="Filter customers by risk segment"
            )
        with col3:
            max_customers = st.slider(
                "Max customers to display",
                10, 1000, 100, 10
            )
        with col4:
            sort_by = st.selectbox(
                "Sort by",
                options=["segment", "hazard_score", churn_col],
                index=1,
                help="hazard_score: continuous risk ranking (partial hazard). Higher = higher risk."
            )
        
        # Apply filters
        df_filtered = df[df[churn_col] >= min_churn_prob].copy()
        if segment_filter != "All":
            df_filtered = df_filtered[df_filtered["segment"] == segment_filter].copy()
        
        # Sort by selected column (handle segment specially)
        if sort_by == "segment":
            # Custom sort: Red > Amber > Green
            segment_order = {"Red": 0, "Amber": 1, "Green": 2}
            df_filtered = df_filtered.copy()
            df_filtered["_sort_segment"] = df_filtered["segment"].map(segment_order)
            df_filtered = df_filtered.sort_values("_sort_segment", ascending=True).drop("_sort_segment", axis=1)
        else:
            df_filtered = df_filtered.sort_values(sort_by, ascending=False)
        
        df_filtered = df_filtered.head(max_customers)
        
        st.markdown("### Customer Conditional Churn Probabilities")
        st.caption("hazard_score: Partial hazard for continuous risk ranking. Higher values indicate higher churn risk.")
        st.caption("Segmentation: Red = top 10% hazard_score & high prob (≥0.7); Amber = p70-p90 or medium prob (0.4-0.7); Green = rest")
        
        # Build format dict dynamically
        format_dict = {
            "hazard_score": "{:.3f}",
            churn_col: "{:.3f}",
            "orders_per_month": "{:.2f}",
            "aov": "{:.2f}",
            "tenure_days": "{:.0f}",
            "gap_days": "{:.0f}",
        }
        
        # Apply color styling to segment column
        def color_segment(val):
            if val == "Red":
                return "background-color: #ffcccc"
            elif val == "Amber":
                return "background-color: #fff4cc"
            elif val == "Green":
                return "background-color: #ccffcc"
            return ""
        
        styled_df = df_filtered.style.format(format_dict).applymap(
            color_segment, subset=["segment"]
        )
        
        st.dataframe(
            styled_df,
            use_container_width=True,
            height=400
        )
        
        if len(df_filtered) > 0:
            # Segment distribution
            st.markdown("### Segment Distribution")
            segment_counts = df_filtered["segment"].value_counts()
            fig_segment = px.bar(
                x=segment_counts.index,
                y=segment_counts.values,
                title="Customer Count by Segment",
                labels={"x": "Segment", "y": "Count"},
                color=segment_counts.index,
                color_discrete_map={"Red": "#ff6666", "Amber": "#ffcc66", "Green": "#66cc66"}
            )
            st.plotly_chart(fig_segment, use_container_width=True)
            
            st.markdown("### Hazard Score Distribution")
            fig_hazard = px.histogram(
                df_filtered,
                x="hazard_score",
                nbins=30,
                color="segment",
                title="Hazard Score Distribution by Segment",
                labels={"hazard_score": "Hazard Score (Partial Hazard)", "count": "Count"},
                color_discrete_map={"Red": "#ff6666", "Amber": "#ffcc66", "Green": "#66cc66"}
            )
            st.plotly_chart(fig_hazard, use_container_width=True)
            
            st.markdown(f"### {int(horizon_days)}-Day Conditional Churn Probability Distribution")
            fig_churn = px.histogram(
                df_filtered,
                x=churn_col,
                nbins=30,
                title=f"{int(horizon_days)}-Day Conditional Churn Probability Distribution",
                labels={churn_col: f"{int(horizon_days)}-Day Conditional Churn Probability", "count": "Count"}
            )
            st.plotly_chart(fig_churn, use_container_width=True)
            
            st.markdown("### Churn Risk vs Customer Attributes")
            col1, col2 = st.columns(2)
            
            with col1:
                x_attr = st.selectbox(
                    "X axis (customer attribute)",
                    options=["gap_days", "tenure_days", "orders_per_month", "aov"],
                    index=0
                )
            
            with col2:
                y_metric = st.selectbox(
                    "Y axis (risk metric)",
                    options=["hazard_score", churn_col],
                    index=0
                )
            
            y_label = f"{int(horizon_days)}-Day Conditional Churn Probability" if y_metric == churn_col else "Hazard Score (Partial Hazard)"
            fig_scatter = px.scatter(
                df_filtered,
                x=x_attr,
                y=y_metric,
                color="segment",
                hover_data=["customer_id"],
                title=f"{y_label} vs {x_attr.replace('_', ' ').title()}",
                labels={
                    y_metric: y_label,
                    x_attr: x_attr.replace("_", " ").title()
                },
                color_discrete_map={"Red": "#ff6666", "Amber": "#ffcc66", "Green": "#66cc66"}
            )
            st.plotly_chart(fig_scatter, use_container_width=True)

    st.markdown("---")
    st.markdown("## Customer Survival Curve Lookup")
    st.caption("Look up the conditional survival curve and expected remaining lifetime for a specific customer by ID")
    st.caption("Conditional survival: S(t₀+u)/S(t₀) where u is time from cutoff (conditional on being alive at cutoff)")

    lookup_customer_id = st.text_input(
        "Customer ID",
        value="",
        help="Enter customer ID to look up survival curve"
    )

    if st.button("Look Up Customer"):
        if not lookup_customer_id:
            st.error("Please enter a customer ID")
        else:
            r = requests.get(
                f"{API_URL}/survival/customer/{lookup_customer_id}?inactivity_days=90",
                timeout=180,
            )
            if r.status_code != 200:
                st.error(r.text)
            else:
                payload = r.json()
                
                if not payload["found"]:
                    st.error(f"Customer {lookup_customer_id}: {payload.get('error', 'Not found')}")
                else:
                    st.success(f"Customer {lookup_customer_id} found")
                    st.write(f"**Tenure at cutoff:** {payload['tenure_days']:.0f} days")
                    st.write(f"**Expected remaining lifetime:** {payload['expected_remaining_lifetime']:.1f} days ({payload['expected_remaining_lifetime']/30:.1f} months)")
                    
                    # Plot survival curve
                    if payload["survival_curve"]:
                        curve_df = pd.DataFrame(payload["survival_curve"])
                        
                        fig = px.line(
                            curve_df,
                            x="time",
                            y="survival",
                            title=f"Conditional Survival Curve for Customer {lookup_customer_id} (from cutoff onwards)",
                            labels={
                                "time": "Days from cutoff (u)",
                                "survival": "Conditional Survival Probability S(t₀+u)/S(t₀)"
                            },
                            markers=True
                        )
                        fig.add_hline(
                            y=0.5,
                            line_dash="dash",
                            line_color="gray",
                            annotation_text="50% survival",
                            annotation_position="right"
                        )
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Show survival curve data
                        st.markdown("### Conditional Survival Curve Data")
                        st.caption("Time: days from cutoff (u). Survival: conditional probability S(t₀+u)/S(t₀)")
                        st.dataframe(
                            curve_df.style.format({
                                "time": "{:.0f}",
                                "survival": "{:.4f}"
                            }),
                            use_container_width=True,
                            height=300
                        )

