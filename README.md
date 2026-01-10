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
