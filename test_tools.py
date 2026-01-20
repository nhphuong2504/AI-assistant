# test_tools.py
import warnings
import numpy as np
from app.tools import get_clv, get_risk_score, get_churn_probability, get_expected_lifetime, get_segmentation

# Suppress pandas/numpy log warnings (these are expected for edge cases with log transformations)
warnings.filterwarnings('ignore', category=RuntimeWarning, message='invalid value encountered in log')

def test_get_clv():
    """Test get_clv function"""
    print("Testing get_clv...")
    
    # Test 1: Basic call
    result = get_clv(horizon_days=180)
    assert result['cutoff_date'] == '2011-12-09'
    assert result['horizon_days'] == 180
    assert 'customers' in result
    assert 'summary' in result
    assert result['summary']['customers_total'] > 0
    print(f"✓ Basic test passed: {result['summary']['customers_total']} customers")
    
    # Test 2: With customer filter
    if result['customers']:
        test_id = str(result['customers'][0]['customer_id'])
        result_filtered = get_clv(horizon_days=180, customer_ids=[test_id])
        assert len(result_filtered['customers']) == 1
        # Handle type conversion (customer_id might be int or str)
        filtered_id = str(result_filtered['customers'][0]['customer_id'])
        assert filtered_id == test_id
        print(f"✓ Filter test passed: Found customer {test_id}")
    
    # Test 3: Validation - too low
    try:
        get_clv(horizon_days=10)  # Should fail (too low)
        print("✗ Validation test failed: Should have raised ValueError")
        assert False
    except ValueError as e:
        assert "30 and 365" in str(e)
        print(f"✓ Validation test (low) passed: {e}")
    
    # Test 4: Validation - too high
    try:
        get_clv(horizon_days=400)  # Should fail (too high)
        print("✗ Validation test failed: Should have raised ValueError")
        assert False
    except ValueError as e:
        assert "30 and 365" in str(e)
        print(f"✓ Validation test (high) passed: {e}")
    
    # Test 5: Check result structure
    assert 'clv_mean' in result['summary']
    assert 'clv_max' in result['summary']
    if result['customers']:
        assert 'clv' in result['customers'][0]
        assert 'frequency' in result['customers'][0]
        assert 'recency' in result['customers'][0]
    print("✓ Structure test passed")
    
    # Print sample results
    if result['customers']:
        print("\n📊 Sample CLV Results:")
        print(f"  Top customer ID: {result['customers'][0]['customer_id']}")
        print(f"  CLV: ${result['customers'][0]['clv']:.2f}")
        print(f"  Frequency: {result['customers'][0]['frequency']}")
        print(f"  Recency: {result['customers'][0]['recency']} days")
        print(f"  Predicted purchases: {result['customers'][0]['pred_purchases']:.2f}")
        print(f"  Predicted AOV: ${result['customers'][0]['pred_aov']:.2f}")
        print(f"  Summary - Mean CLV: ${result['summary']['clv_mean']:.2f}")
        print(f"  Summary - Max CLV: ${result['summary']['clv_max']:.2f}")
    
    print("get_clv tests completed!\n")

def test_get_risk_score():
    """Test get_risk_score function"""
    print("Testing get_risk_score...")
    
    result = get_risk_score()
    assert result['cutoff_date'] == '2011-12-09'
    assert result['inactivity_days'] == 90
    assert 'customers' in result
    assert 'summary' in result
    assert result['summary']['n_customers'] > 0
    print(f"✓ Basic test passed: {result['summary']['n_customers']} customers")
    
    # Test with customer filter
    if result['customers']:
        test_id = str(result['customers'][0]['customer_id'])
        result_filtered = get_risk_score(customer_ids=[test_id])
        assert len(result_filtered['customers']) == 1
        # Handle type conversion (customer_id might be int or str)
        filtered_id = str(result_filtered['customers'][0]['customer_id'])
        assert filtered_id == test_id
        print(f"✓ Filter test passed: Found customer {test_id}")
    
    # Test structure
    if result['customers']:
        customer = result['customers'][0]
        assert 'risk_score' in customer
        assert 'risk_rank' in customer
        assert 'risk_percentile' in customer
        assert 'risk_bucket' in customer
        assert customer['risk_bucket'] in ['High', 'Medium', 'Low']
    print("✓ Structure test passed")
    
    # Print sample results
    if result['customers']:
        print("\n📊 Sample Risk Score Results:")
        print(f"  Customer ID: {result['customers'][0]['customer_id']}")
        print(f"  Risk Score: {result['customers'][0]['risk_score']:.3f}")
        print(f"  Risk Rank: {result['customers'][0]['risk_rank']}")
        print(f"  Risk Percentile: {result['customers'][0]['risk_percentile']:.1f}%")
        print(f"  Risk Bucket: {result['customers'][0]['risk_bucket']}")
        print(f"  N Orders: {result['customers'][0]['n_orders']}")
        print(f"  Product Diversity: {result['customers'][0]['product_diversity']}")
        print(f"  Summary - Mean Risk Score: {result['summary']['risk_score_mean']:.3f}")
        print(f"  Summary - Risk Buckets: {result['summary']['risk_bucket_counts']}")
    
    print("get_risk_score tests completed!\n")

def test_get_churn_probability():
    """Test get_churn_probability function"""
    print("Testing get_churn_probability...")
    
    # Test 1: Basic call
    result = get_churn_probability(prediction_horizon_days=90)
    assert result['cutoff_date'] == '2011-12-09'
    assert result['inactivity_days'] == 90
    assert result['X_days'] == 90
    assert 'customers' in result
    assert 'summary' in result
    assert result['summary']['n_customers'] > 0
    print(f"✓ Basic test passed: {result['summary']['n_customers']} active customers")
    
    # Test 2: With customer filter
    if result['customers']:
        test_id = str(result['customers'][0]['customer_id'])
        result_filtered = get_churn_probability(prediction_horizon_days=90, customer_ids=[test_id])
        assert len(result_filtered['customers']) == 1
        # Handle type conversion (customer_id might be int or str)
        filtered_id = str(result_filtered['customers'][0]['customer_id'])
        assert filtered_id == test_id
        print(f"✓ Filter test passed: Found customer {test_id}")
    
    # Test 3: Validation - too low
    try:
        get_churn_probability(prediction_horizon_days=5)
        print("✗ Validation test failed: Should have raised ValueError")
        assert False
    except ValueError as e:
        assert "7 and 365" in str(e)
        print(f"✓ Validation test (low) passed: {e}")
    
    # Test 4: Validation - too high
    try:
        get_churn_probability(prediction_horizon_days=400)
        print("✗ Validation test failed: Should have raised ValueError")
        assert False
    except ValueError as e:
        assert "7 and 365" in str(e)
        print(f"✓ Validation test (high) passed: {e}")
    
    # Test 5: Check structure
    if result['customers']:
        customer = result['customers'][0]
        assert 'churn_probability' in customer
        assert 0 <= customer['churn_probability'] <= 1
        assert 'survival_at_t0' in customer
        assert 'survival_at_t0_plus_X' in customer
    print("✓ Structure test passed")
    
    # Print sample results
    if result['customers']:
        print("\n📊 Sample Churn Probability Results:")
        print(f"  Customer ID: {result['customers'][0]['customer_id']}")
        print(f"  Churn Probability (next {result['X_days']} days): {result['customers'][0]['churn_probability']:.3f} ({result['customers'][0]['churn_probability']*100:.1f}%)")
        print(f"  Current Duration (t0): {result['customers'][0]['t0']:.1f} days")
        print(f"  Survival at t0: {result['customers'][0]['survival_at_t0']:.3f}")
        print(f"  Survival at t0+{result['X_days']}: {result['customers'][0]['survival_at_t0_plus_X']:.3f}")
        print(f"  Summary - Mean Churn Prob: {result['summary']['churn_probability_mean']:.3f}")
        print(f"  Summary - Median Churn Prob: {result['summary']['churn_probability_median']:.3f}")
    
    print("get_churn_probability tests completed!\n")

def test_get_expected_lifetime():
    """Test get_expected_lifetime function"""
    print("Testing get_expected_lifetime...")
    
    result = get_expected_lifetime()
    assert result['cutoff_date'] == '2011-12-09'
    assert result['inactivity_days'] == 90
    assert result['H_days'] == 365
    assert 'customers' in result
    assert 'summary' in result
    assert result['summary']['n_customers'] > 0
    print(f"✓ Basic test passed: {result['summary']['n_customers']} active customers")
    
    # Test with customer filter
    if result['customers']:
        test_id = str(result['customers'][0]['customer_id'])
        result_filtered = get_expected_lifetime(customer_ids=[test_id])
        assert len(result_filtered['customers']) == 1
        # Handle type conversion (customer_id might be int or str)
        filtered_id = str(result_filtered['customers'][0]['customer_id'])
        assert filtered_id == test_id
        print(f"✓ Filter test passed: Found customer {test_id}")
    
    # Test structure
    if result['customers']:
        customer = result['customers'][0]
        assert 'expected_remaining_life_days' in customer
        assert 0 <= customer['expected_remaining_life_days'] <= 365
    print("✓ Structure test passed")
    
    # Print sample results
    if result['customers']:
        print("\n📊 Sample Expected Lifetime Results:")
        print(f"  Customer ID: {result['customers'][0]['customer_id']}")
        print(f"  Expected Remaining Lifetime: {result['customers'][0]['expected_remaining_life_days']:.1f} days")
        print(f"  Current Duration (t0): {result['customers'][0]['t0']:.1f} days")
        print(f"  Horizon (H_days): {result['H_days']} days")
        print(f"  Summary - Mean Expected Lifetime: {result['summary']['expected_lifetime_mean']:.1f} days")
        print(f"  Summary - Median Expected Lifetime: {result['summary']['expected_lifetime_median']:.1f} days")
        print(f"  Summary - Max Expected Lifetime: {result['summary']['expected_lifetime_max']:.1f} days")
    
    print("get_expected_lifetime tests completed!\n")

def test_get_segmentation():
    """Test get_segmentation function"""
    print("Testing get_segmentation...")
    
    result = get_segmentation()
    assert result['cutoff_date'] == '2011-12-09'
    assert result['inactivity_days'] == 90
    assert result['H_days'] == 365
    assert 'customers' in result
    assert 'cutoffs' in result
    assert 'summary' in result
    assert result['summary']['n_customers'] > 0
    print(f"✓ Basic test passed: {result['summary']['n_customers']} active customers")
    
    # Test with customer filter
    if result['customers']:
        test_id = str(result['customers'][0]['customer_id'])
        result_filtered = get_segmentation(customer_ids=[test_id])
        assert len(result_filtered['customers']) == 1
        # Handle type conversion (customer_id might be int or str)
        filtered_id = str(result_filtered['customers'][0]['customer_id'])
        assert filtered_id == test_id
        print(f"✓ Filter test passed: Found customer {test_id}")
    
    # Test structure
    if result['customers']:
        customer = result['customers'][0]
        assert 'segment' in customer
        assert 'action_tag' in customer
        assert 'recommended_action' in customer
        assert 'risk_label' in customer
        assert 'life_bucket' in customer
        assert customer['risk_label'] in ['High', 'Medium', 'Low']
        assert customer['life_bucket'] in ['Short', 'Medium', 'Long']
    print("✓ Structure test passed")
    
    # Test cutoffs
    assert 'q33' in result['cutoffs']
    assert 'q67' in result['cutoffs']
    assert result['cutoffs']['H_days'] == 365
    print("✓ Cutoffs test passed")
    
    # Print sample results
    if result['customers']:
        print("\n📊 Sample Segmentation Results:")
        customer = result['customers'][0]
        print(f"  Customer ID: {customer['customer_id']}")
        print(f"  Segment: {customer['segment']}")
        print(f"  Risk Label: {customer['risk_label']}")
        print(f"  Life Bucket: {customer['life_bucket']}")
        print(f"  Action Tag: {customer['action_tag']}")
        print(f"  Recommended Action: {customer['recommended_action']}")
        print(f"  Expected Remaining Lifetime: {customer['erl_365_days']:.1f} days")
        print(f"  Current Duration (t0): {customer['t0']:.1f} days")
        print(f"  Cutoffs - q33: {result['cutoffs']['q33']:.1f} days, q67: {result['cutoffs']['q67']:.1f} days")
        print(f"  Summary - Segment Counts: {dict(list(result['summary']['segment_counts'].items())[:5])}...")
        print(f"  Summary - Action Tag Counts: {dict(list(result['summary']['action_tag_counts'].items())[:5])}...")
    
    print("get_segmentation tests completed!\n")

if __name__ == "__main__":
    print("=" * 60)
    print("Running tool function tests...")
    print("=" * 60 + "\n")
    
    test_get_clv()
    test_get_risk_score()
    test_get_churn_probability()
    test_get_expected_lifetime()
    test_get_segmentation()
    
    print("=" * 60)
    print("All tests completed successfully! ✓")
    print("=" * 60)