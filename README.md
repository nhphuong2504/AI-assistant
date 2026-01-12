# Retail Data Assistant - CLV Model

## Project

A Customer Lifetime Value (CLV) prediction system using BG/NBD and Gamma-Gamma models for retail customer analytics.

## CLV Model Trustworthiness: Testing & Results Summary

### How We Validate CLV Outputs

We validate CLV model trustworthiness through three complementary approaches:

1. **Holdout Validation on Future Data**: Split data at 2011-09-30; train models on calibration period (≤2011-09-30), predict for holdout period (>2011-09-30), and compare predictions to actual outcomes
2. **Goodness of Fit**: Verify models reproduce calibration period patterns (BG/NBD frequency distribution, Gamma-Gamma AOV accuracy)
3. **Business Magnitude Sanity Checks**: Ensure CLV values are reasonable and rankings align with business logic (top customers should have high frequency/AOV)

### Test Results

#### Strong Trust Indicators ✅

- **Predictive Power**: Purchase correlation **0.82**, Revenue correlation **0.60** — model strongly predicts future behavior
- **Rank Ordering**: Top 100 predicted customers performed **165x better** ($8,909.90 vs $54.02 actual revenue) — rankings are highly accurate
- **Model Calibration**: Gamma-Gamma AOV error **1.9%** (predicted $408.75 vs actual $401.25) — excellent fit
- **Reasonable Magnitudes**: Median CLV **$690.93** within expected range ($201-$2,006) — values make business sense
- **Sensible Rankings**: Top 10% CLV customers have **10.5x higher frequency** (15.56 vs 1.48) and **8.4x higher AOV** ($1,064 vs $127) — aligns with business intuition

#### Areas of Concern ⚠️

- **Aggregate Underestimation**: Model predicts **29% fewer purchases** and **45% less revenue** than actual — conservative bias
- **Recommendation**: Use for **relative ranking/prioritization**, not absolute dollar forecasting

### Conclusion

**CLV outputs are trustworthy** for:
- ✅ Customer prioritization and segmentation  
- ✅ Identifying high-value customers for targeted marketing
- ✅ Relative comparisons (165x difference validated in holdout)

The model is well-calibrated with strong rank ordering and reasonable magnitudes. Conservative bias makes it better suited for prioritization than precise revenue forecasting.

## Running CLV Tests

To validate the CLV model:

```bash
./venv/bin/python test/test-clv.py
```

The test suite performs:
1. Holdout validation (predictions vs actuals on future data)
2. Goodness of fit checks (BG/NBD & Gamma-Gamma models)
3. Business magnitude sanity checks

## Project Structure

- `analytics/clv.py` - CLV model implementation (BG/NBD + Gamma-Gamma)
- `test/test-clv.py` - Comprehensive trustworthiness test suite
- `etl/load_online_retail.py` - Data loading and preprocessing
- `app/` - FastAPI application for queries and CLV predictions
- `ui/` - Streamlit dashboard interface


By Day 5 of this project, I already had a functioning CLV pipeline and a stable transactional data model, but something still felt incomplete. CLV could tell me how valuable customers might be, yet it said very little about why that value differed across customers or when it was likely to disappear. At that point, it became clear that predicting churn as a binary outcome would be premature. Before asking whether a customer churns, I needed to understand how long customers survive and what governs that survival. That realization led naturally to survival analysis.

I began the day by locking down modeling assumptions. The cutoff date was fixed at December 9, 2011, the last observed transaction date in the dataset, and it was treated as inclusive. I deliberately removed any configurability around this decision, both in the API and the UI, because allowing it to vary would introduce unnecessary ambiguity and risk data leakage. If a parameter is not meant to change, it should not exist. Churn was defined operationally as customer inactivity for a specified number of days, primarily testing 60-day and 90-day windows. This definition mirrors how churn is treated in real retail systems, where prolonged silence is effectively equivalent to exit.

With those assumptions in place, the core task of the day became building a single, auditable customer-level survival table. Each customer was represented by one row, frozen at the cutoff date, with properly censored durations. The table included time-to-event information alongside behavioral covariates such as days since last purchase, tenure, normalized purchase frequency, product diversity, average order value, and geography. This dataset became the backbone of everything that followed, ensuring that all models would be grounded in the same view of customer history.

Rather than immediately fitting a parametric or semi-parametric model, I started with Kaplan–Meier survival curves. This was an intentional choice. Kaplan–Meier requires no assumptions about functional form and allows the data to speak for itself. The baseline survival curve for all customers exhibited a steep early drop followed by a long tail, which aligned perfectly with retail intuition: many customers churn quickly, while a smaller group remains loyal for a long time. That initial validation gave confidence that the churn definition and censoring logic were sound.

The real insights emerged once I stratified the Kaplan–Meier curves. When customers were split by domestic versus international markets, the survival curves were nearly identical, and churn rates differed by less than one percentage point. This immediately suggested that geography was not a primary driver of churn. In contrast, stratifying by normalized purchase frequency produced dramatic separation. Customers with low orders per month churned almost immediately, medium-frequency customers showed moderate survival, and high-frequency customers barely churned at all. This single set of curves explained more about the business than any aggregate metric had up to that point. Engagement intensity, not spending, was clearly the dominant retention signal.

Stratifying by average order value and by monetary value among repeat buyers reinforced this conclusion. Higher-spending customers did churn less, but the effect was modest compared to frequency. Even among repeat buyers, monetary value mattered far less than engagement patterns. Spending more did not necessarily imply stronger loyalty; buying more often did.

Only after these patterns were clear did I move to Cox proportional hazards modeling. The Cox model allowed me to quantify the effects observed in the Kaplan–Meier curves while controlling for correlated variables. Initial attempts produced convergence warnings, which was not surprising given the strong collinearity between frequency and tenure and the near-deterministic nature of recency. Rather than forcing the model to fit, I addressed the underlying numerical issues by log-transforming heavy-tailed features, standardizing covariates, and introducing L2 penalization. With these adjustments, the model converged cleanly and produced stable estimates.

The results were remarkably coherent. Inactivity emerged as the dominant churn trigger, with even modest increases in time since last purchase dramatically increasing churn risk. Tenure had a strong protective effect, reflecting survivorship bias and accumulated loyalty. Normalized purchase frequency significantly reduced churn risk, while product diversity also played a meaningful role by increasing switching costs. In contrast, average order value and geography were statistically insignificant once behavior was accounted for. The model was effectively telling a simple story: churn is behavioral, not monetary.

Repeating the Cox analysis on repeat buyers sharpened this narrative further. Inactivity became even more predictive, while the effects of frequency and tenure moderated. Monetary value still failed to emerge as a dominant driver. Conditioning on repeat behavior changed the magnitude of effects but not their ordering, reinforcing the idea that engagement patterns govern customer lifetime far more than how much customers spend.

By the end of Day 5, the structure of churn in this business was clear. Recency and engagement intensity dominate churn risk, tenure reflects survivorship rather than active behavior, product diversity increases retention through switching costs, and monetary value is largely secondary once behavior is known. This understanding aligned perfectly with the CLV results and provided a causal backbone for everything that would follow.

Day 5 did not produce a flashy new model; it produced clarity. It prevented me from optimizing for the wrong signals and established a principled foundation for churn classification, uplift modeling, and causal analysis. Survival analysis forced a shift from asking whether churn could be predicted to understanding what governs customer lifetime. That shift made the entire system stronger.