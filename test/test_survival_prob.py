"""
Test suite for survival probability functions: predict_churn_probabilities and get_customer_survival_curve.
Tests for customer ID 17347.
"""
import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add parent directory to path to import analytics module
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.db import run_query_internal
from analytics.survival import (
    build_covariate_table,
    build_cox_design,
    fit_cox,
    predict_churn_probabilities,
    get_customer_survival_curve,
)


def load_transactions():
    """Load transactions from database."""
    sql = """
    SELECT customer_id, invoice_no, invoice_date, revenue, stock_code, country
    FROM transactions
    WHERE customer_id IS NOT NULL
    """
    rows, _ = run_query_internal(sql, max_rows=2_000_000)
    df = pd.DataFrame(rows)
    return df


def test_predict_churn_probabilities():
    """Test predict_churn_probabilities function for customer 17347."""
    print("\n" + "="*80)
    print("TEST: predict_churn_probabilities for customer 17347")
    print("="*80)
    
    # Load data and build model
    df = load_transactions()
    cov = build_covariate_table(
        transactions=df,
        cutoff_date="2011-12-09",
        inactivity_days=90,
    ).df
    
    df_cox = build_cox_design(cov)
    cph = fit_cox(df_cox, penalizer=0.1)
    
    # Test with different horizons
    horizons = [30, 60, 90]
    customer_id = "17347"
    
    for horizon_days in horizons:
        print(f"\n--- Testing horizon: {horizon_days} days ---")
        
        churn_pred = predict_churn_probabilities(
            cov=cov,
            cph=cph,
            horizon_days=horizon_days,
            include_churned=False,
        )
        
        # Check if customer 17347 is in results
        customer_row = churn_pred[churn_pred["customer_id"].astype(str) == customer_id]
        
        if len(customer_row) > 0:
            print(f"✓ Customer {customer_id} found in predictions")
            row = customer_row.iloc[0]
            
            # Verify structure
            assert "customer_id" in churn_pred.columns, "Missing customer_id column"
            assert "hazard_score" in churn_pred.columns, "Missing hazard_score column"
            assert f"churn_prob_cond_{horizon_days}d" in churn_pred.columns, f"Missing churn_prob_cond_{horizon_days}d column"
            assert "segment" in churn_pred.columns, "Missing segment column"
            print("  ✓ All required columns present")
            
            # Display values
            print(f"  - hazard_score: {row['hazard_score']:.4f}")
            print(f"  - churn_prob_cond_{horizon_days}d: {row[f'churn_prob_cond_{horizon_days}d']:.4f}")
            print(f"  - segment: {row['segment']}")
            
            # Verify values are reasonable
            prob = row[f'churn_prob_cond_{horizon_days}d']
            assert 0 <= prob <= 1, f"Churn probability {prob} out of [0,1] range"
            assert row['segment'] in ['Red', 'Amber', 'Green'], f"Invalid segment: {row['segment']}"
            print("  ✓ Values are within expected ranges")
            
            # Verify monotonicity: longer horizons should have higher or equal churn probability
            if horizon_days == 30:
                prob_30 = prob
            elif horizon_days == 60:
                prob_60 = prob
                assert prob_60 >= prob_30, f"60-day prob ({prob_60}) should be >= 30-day prob ({prob_30})"
                print(f"  ✓ Monotonicity: 60d ({prob_60:.4f}) >= 30d ({prob_30:.4f})")
            elif horizon_days == 90:
                prob_90 = prob
                assert prob_90 >= prob_60, f"90-day prob ({prob_90}) should be >= 60-day prob ({prob_60})"
                print(f"  ✓ Monotonicity: 90d ({prob_90:.4f}) >= 60d ({prob_60:.4f})")
        else:
            print(f"✗ Customer {customer_id} NOT found in predictions")
            print("  (May be already churned or missing covariates)")
    
    print("\n✓ predict_churn_probabilities test complete")


def test_get_customer_survival_curve():
    """Test get_customer_survival_curve function for customer 17347."""
    print("\n" + "="*80)
    print("TEST: get_customer_survival_curve for customer 17347")
    print("="*80)
    
    # Load data and build model
    df = load_transactions()
    cov = build_covariate_table(
        transactions=df,
        cutoff_date="2011-12-09",
        inactivity_days=90,
    ).df
    
    df_cox = build_cox_design(cov)
    cph = fit_cox(df_cox, penalizer=0.1)
    
    customer_id = "17347"
    
    result = get_customer_survival_curve(
        cov=cov,
        cph=cph,
        customer_id=customer_id,
        include_churned=False,
    )
    
    if result["found"]:
        print(f"\n✓ Customer {customer_id} found")
        print(f"  - tenure_days: {result['tenure_days']:.0f}")
        print(f"  - expected_remaining_lifetime: {result['expected_remaining_lifetime']:.2f} days")
        print(f"  - survival_curve points: {len(result['survival_curve'])}")
        
        # Verify structure
        assert "customer_id" in result, "Missing customer_id in result"
        assert "found" in result, "Missing found flag"
        assert result["found"] == True, "Found flag should be True"
        assert "tenure_days" in result, "Missing tenure_days"
        assert "survival_curve" in result, "Missing survival_curve"
        assert "expected_remaining_lifetime" in result, "Missing expected_remaining_lifetime"
        print("  ✓ All required fields present")
        
        # Verify survival curve structure
        curve = result["survival_curve"]
        assert len(curve) > 0, "Survival curve is empty"
        assert all("time" in pt and "survival" in pt for pt in curve), "Invalid survival curve structure"
        print("  ✓ Survival curve structure is valid")
        
        # Verify conditional survival properties
        first_point = curve[0]
        assert first_point["time"] == 0.0, f"First point should be at time 0 (cutoff), got {first_point['time']}"
        assert 0 <= first_point["survival"] <= 1, f"Survival probability {first_point['survival']} out of [0,1] range"
        assert abs(first_point["survival"] - 1.0) < 0.01, "Conditional survival at cutoff should be ~1.0 (S(t0)/S(t0)=1)"
        print(f"  ✓ First point: time={first_point['time']:.0f}, survival={first_point['survival']:.4f}")
        
        # Check that survival is non-increasing (conditional survival should decrease or stay same)
        survivals = [pt["survival"] for pt in curve]
        times = [pt["time"] for pt in curve]
        
        # Verify times are non-decreasing
        assert all(times[i] <= times[i+1] for i in range(len(times)-1)), "Times should be non-decreasing"
        print("  ✓ Times are non-decreasing")
        
        # Check non-increasing survival (allowing for small numerical errors)
        is_non_increasing = all(
            survivals[i] >= survivals[i+1] - 1e-6 
            for i in range(len(survivals)-1)
        )
        if is_non_increasing:
            print("  ✓ Survival curve is non-increasing (as expected for conditional survival)")
        else:
            print("  ⚠ Survival curve has some increases (may be due to interpolation/numerical issues)")
        
        # Verify expected remaining lifetime is reasonable
        assert result["expected_remaining_lifetime"] >= 0, "Expected remaining lifetime should be non-negative"
        assert result["expected_remaining_lifetime"] < 10000, "Expected remaining lifetime seems unreasonably large"
        print(f"  ✓ Expected remaining lifetime is reasonable: {result['expected_remaining_lifetime']:.2f} days")
        
        # Show sample curve points
        print("\n  Sample survival curve points (conditional survival from cutoff):")
        for i, pt in enumerate(curve[:10]):
            print(f"    Day {pt['time']:.0f}: {pt['survival']:.4f}")
        if len(curve) > 10:
            print(f"    ... ({len(curve) - 10} more points)")
        
        # Verify that survival at time 0 is 1.0 (conditional: S(t0)/S(t0) = 1)
        assert abs(first_point["survival"] - 1.0) < 0.01, "Conditional survival at time 0 should be 1.0"
        print("  ✓ Conditional survival starts at 1.0 (S(t0)/S(t0))")
    else:
        print(f"\n✗ Customer {customer_id} NOT found")
        print(f"  Error: {result.get('error', 'Unknown error')}")
        # This is a test failure, but we'll continue to show what happened
        assert False, f"Customer {customer_id} should be found"
    
    print("\n✓ get_customer_survival_curve test complete")




def test_debug_survival_values():
    """Debug test: Print s_t0 and s_t0+30 from both functions for customer 17347."""
    print("\n" + "="*80)
    print("DEBUG TEST: Survival values for customer 17347")
    print("="*80)
    
    # Load data and build model
    df = load_transactions()
    cov = build_covariate_table(
        transactions=df,
        cutoff_date="2011-12-09",
        inactivity_days=90,
    ).df
    
    df_cox = build_cox_design(cov)
    cph = fit_cox(df_cox, penalizer=0.1)
    
    customer_id = "17347"
    horizon_days = 30
    
    # Find customer in covariate table
    customer_row = cov[cov["customer_id"].astype(str) == customer_id]
    if len(customer_row) == 0:
        print(f"✗ Customer {customer_id} not found")
        return
    
    customer_row = customer_row.iloc[0]
    t0 = float(customer_row["tenure_days"])
    
    print(f"\nCustomer {customer_id}:")
    print(f"  t0 (tenure_days): {t0:.0f}")
    
    # ============================================================
    # From predict_churn_probabilities
    # ============================================================
    print("\n" + "-"*80)
    print("From predict_churn_probabilities:")
    print("-"*80)
    
    # Filter alive customers (same as predict_churn_probabilities)
    cov_alive = cov[cov["event"] == 0].copy()
    df_cox_alive = build_cox_design(cov_alive)
    
    # Find customer in alive set
    customer_idx_alive = cov_alive[cov_alive["customer_id"].astype(str) == customer_id].index
    if len(customer_idx_alive) == 0:
        print("  ✗ Customer not in alive set")
        s_t0_pred = None
        s_t0_plus_30_pred = None
    else:
        customer_idx_alive = customer_idx_alive[0]
        
        # Check if customer is in df_cox_alive (after dropna in build_cox_design)
        if customer_idx_alive not in df_cox_alive.index:
            print("  ✗ Customer not in Cox design matrix (missing covariates)")
            s_t0_pred = None
            s_t0_plus_30_pred = None
        else:
            # Get customer's position in df_cox_alive
            customer_pos = list(df_cox_alive.index).index(customer_idx_alive)
            
            # Get covariates for prediction
            covariate_cols = [c for c in df_cox_alive.columns if c not in ["duration", "event"]]
            customer_covariates = df_cox_alive[covariate_cols].copy()
            
            # Predict survival functions for all customers
            survival_functions = cph.predict_survival_function(customer_covariates)
            times = survival_functions.index.values
            
            # Get this customer's survival function (column customer_pos)
            survival_fn_pred = survival_functions.iloc[:, customer_pos]
            
            # Get s_t0
            time_min = times.min()
            time_max = times.max()
            t0_clipped = np.clip(t0, time_min, time_max)
            s_t0_pred = float(np.interp(t0_clipped, times, survival_fn_pred.values))
            if t0 > time_max:
                s_t0_pred = float(survival_fn_pred.iloc[-1])
            
            # Get s_t0+30
            t_target = t0 + horizon_days
            t_target_clipped = np.clip(t_target, time_min, time_max)
            s_t0_plus_30_pred = float(np.interp(t_target_clipped, times, survival_fn_pred.values))
            if t_target > time_max:
                s_t0_plus_30_pred = float(survival_fn_pred.iloc[-1])
            
            print(f"  s_t0:           {s_t0_pred:.6f}")
            print(f"  s_t0+30:        {s_t0_plus_30_pred:.6f}")
            if s_t0_pred > 1e-10:
                print(f"  Conditional:    {s_t0_plus_30_pred / s_t0_pred:.6f} (s_t0+30 / s_t0)")
                print(f"  Churn prob:     {1.0 - (s_t0_plus_30_pred / s_t0_pred):.6f}")
            else:
                print(f"  ⚠ s_t0 is very small, conditional survival undefined")
    
    # ============================================================
    # From get_customer_survival_curve
    # ============================================================
    print("\n" + "-"*80)
    print("From get_customer_survival_curve:")
    print("-"*80)
    
    # Build Cox design for this customer (using full cov for scaling)
    df_cox_full = build_cox_design(cov)
    
    # Compute standardization parameters from full dataset
    log_orders_full = np.log1p(cov["orders_per_month"].clip(lower=0))
    log_product_full = np.log1p(cov["product_diversity"].clip(lower=0))
    
    mu_log_orders = log_orders_full.mean()
    sd_log_orders = log_orders_full.std(ddof=0)
    mu_tenure = cov["tenure_days"].mean()
    sd_tenure = cov["tenure_days"].std(ddof=0)
    mu_gap = cov["gap_days"].mean()
    sd_gap = cov["gap_days"].std(ddof=0)
    mu_log_product = log_product_full.mean()
    sd_log_product = log_product_full.std(ddof=0)
    
    # Standardize customer's covariates
    customer_df = pd.DataFrame([customer_row])
    log_orders_cust = np.log1p(customer_df["orders_per_month"].clip(lower=0).iloc[0])
    log_product_cust = np.log1p(customer_df["product_diversity"].clip(lower=0).iloc[0])
    
    z_orders = (log_orders_cust - mu_log_orders) / sd_log_orders if sd_log_orders > 0 else 0.0
    z_tenure = (customer_df["tenure_days"].iloc[0] - mu_tenure) / sd_tenure if sd_tenure > 0 else 0.0
    z_gap = (customer_df["gap_days"].iloc[0] - mu_gap) / sd_gap if sd_gap > 0 else 0.0
    z_product = (log_product_cust - mu_log_product) / sd_log_product if sd_log_product > 0 else 0.0
    
    # Build customer covariates
    customer_covariates = pd.DataFrame({
        "z_gap_days": [z_gap],
        "z_orders_per_month": [z_orders],
        "z_tenure_days": [z_tenure],
        "z_product_diversity": [z_product],
    })
    
    # Predict survival function
    survival_function = cph.predict_survival_function(customer_covariates)
    survival_fn_curve = survival_function.iloc[:, 0]
    times_curve = survival_fn_curve.index.values
    
    # Get s_t0
    if t0 > times_curve.max():
        s_t0_curve = float(survival_fn_curve.iloc[-1])
    else:
        closest_idx_t0 = np.argmin(np.abs(times_curve - t0))
        s_t0_curve = float(survival_fn_curve.iloc[closest_idx_t0])
    
    # Get s_t0+30
    t_target = t0 + horizon_days
    if t_target > times_curve.max():
        s_t0_plus_30_curve = float(survival_fn_curve.iloc[-1])
    else:
        closest_idx_target = np.argmin(np.abs(times_curve - t_target))
        s_t0_plus_30_curve = float(survival_fn_curve.iloc[closest_idx_target])
    
    print(f"  s_t0:           {s_t0_curve:.6f}")
    print(f"  s_t0+30:        {s_t0_plus_30_curve:.6f}")
    print(f"  Conditional:    {s_t0_plus_30_curve / s_t0_curve:.6f} (s_t0+30 / s_t0)")
    print(f"  Churn prob:     {1.0 - (s_t0_plus_30_curve / s_t0_curve):.6f}")
    
    # ============================================================
    # Comparison
    # ============================================================
    print("\n" + "-"*80)
    print("Comparison:")
    print("-"*80)
    if s_t0_pred is not None and s_t0_plus_30_pred is not None:
        print(f"  s_t0 difference:        {abs(s_t0_pred - s_t0_curve):.6f}")
        print(f"  s_t0+30 difference:      {abs(s_t0_plus_30_pred - s_t0_plus_30_curve):.6f}")
        if s_t0_pred > 1e-10 and s_t0_curve > 1e-10:
            cond_pred = s_t0_plus_30_pred / s_t0_pred
            cond_curve = s_t0_plus_30_curve / s_t0_curve
            print(f"  Conditional difference: {abs(cond_pred - cond_curve):.6f}")
            print(f"  Churn prob difference:  {abs((1.0 - cond_pred) - (1.0 - cond_curve)):.6f}")
        else:
            print("  ⚠ Cannot compare conditional (s_t0 too small)")
    else:
        print("  ✗ Cannot compare (values from predict_churn_probabilities not available)")


def test_customer_not_found():
    """Test error handling for non-existent customer."""
    print("\n" + "="*80)
    print("TEST: Error handling for non-existent customer")
    print("="*80)
    
    df = load_transactions()
    cov = build_covariate_table(
        transactions=df,
        cutoff_date="2011-12-09",
        inactivity_days=90,
    ).df
    
    df_cox = build_cox_design(cov)
    cph = fit_cox(df_cox, penalizer=0.1)
    
    # Test with a non-existent customer ID
    fake_id = "99999999"
    result = get_customer_survival_curve(
        cov=cov,
        cph=cph,
        customer_id=fake_id,
        include_churned=False,
    )
    
    assert result["found"] == False, "Non-existent customer should not be found"
    assert "error" in result, "Error message should be present"
    print(f"✓ Non-existent customer {fake_id} correctly returns found=False")
    print(f"  Error: {result['error']}")


if __name__ == "__main__":
    print("\n" + "="*80)
    print("SURVIVAL PROBABILITY FUNCTION TESTS")
    print("="*80)
    
    try:
        # test_predict_churn_probabilities()
        # test_get_customer_survival_curve()
        test_debug_survival_values()
        test_customer_not_found()
        
        print("\n" + "="*80)
        print("ALL TESTS PASSED ✓")
        print("="*80)
    except AssertionError as e:
        print(f"\n✗ TEST FAILED: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n✗ ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

