import pandas as pd
import numpy as np
import os
from return_pipeline import ReturnFeatureExtractor

print("Loading data for leakage test...")
DATA_DIR = '../datathon-2026-round-1'
df_orders = pd.read_csv(os.path.join(DATA_DIR, 'orders.csv'), parse_dates=['order_date'])
df_order_items = pd.read_csv(os.path.join(DATA_DIR, 'order_items.csv'), low_memory=False)
df_returns = pd.read_csv(os.path.join(DATA_DIR, 'returns.csv'), parse_dates=['return_date'])
df_products = pd.read_csv(os.path.join(DATA_DIR, 'products.csv'))
df_customers = pd.read_csv(os.path.join(DATA_DIR, 'customers.csv'), parse_dates=['signup_date'])
df_reviews = pd.read_csv(os.path.join(DATA_DIR, 'reviews.csv'), parse_dates=['review_date'])
df_geography = pd.read_csv(os.path.join(DATA_DIR, 'geography.csv'))

# Filter a subset to run quickly
eligible_statuses = ['delivered', 'returned']
df_eligible = df_orders[df_orders['order_status'].isin(eligible_statuses)].copy()

# Sort chronologically
df_eligible = df_eligible.sort_values('order_date').reset_index(drop=True)

# Select a temporal split date
SPLIT_DATE = pd.Timestamp('2022-07-01')
df_train = df_eligible[df_eligible['order_date'] < SPLIT_DATE].copy()
df_test = df_eligible[df_eligible['order_date'] >= SPLIT_DATE].copy()

# We pick a specific test order to test leakage on.
# Let's say order_id = df_test.iloc[0]['order_id']
test_sample = df_test.iloc[[0]].copy()
test_order_date = test_sample.iloc[0]['order_date']
test_customer_id = test_sample.iloc[0]['customer_id']
print(f"Testing order_id: {test_sample.iloc[0]['order_id']}, order_date: {test_order_date}, customer_id: {test_customer_id}")

# 1. Transform with ORIGINAL df_returns and df_reviews
fe = ReturnFeatureExtractor(
    df_order_items=df_order_items,
    df_returns=df_returns,
    df_products=df_products,
    df_customers=df_customers,
    df_reviews=df_reviews,
    df_geography=df_geography
)
fe.fit(df_train)
features_orig = fe.transform(test_sample)

# Let's modify future returns of this customer/product that happen AFTER test_order_date
# and reviews that happen AFTER test_order_date
print("Injecting future fake return and review...")
df_returns_leaked = df_returns.copy()
# Add a return in the future (1 day after test_order_date) for this order/product
df_returns_leaked = pd.concat([
    df_returns_leaked,
    pd.DataFrame([{
        'return_id': 999999,
        'order_id': test_sample.iloc[0]['order_id'],
        'product_id': df_order_items[df_order_items['order_id'] == test_sample.iloc[0]['order_id']].iloc[0]['product_id'],
        'return_date': test_order_date + pd.Timedelta(days=1),
        'return_reason': 'Unsatisfied',
        'return_quantity': 1,
        'refund_amount': 10.0
    }])
], ignore_index=True)

df_reviews_leaked = df_reviews.copy()
# Add a review in the future (1 day after test_order_date)
df_reviews_leaked = pd.concat([
    df_reviews_leaked,
    pd.DataFrame([{
        'review_id': 999999,
        'order_id': test_sample.iloc[0]['order_id'],
        'customer_id': test_customer_id,
        'rating': 1,
        'review_date': test_order_date + pd.Timedelta(days=1)
    }])
], ignore_index=True)

# 2. Transform with modified (future-leaked) dataframes
fe_leaked = ReturnFeatureExtractor(
    df_order_items=df_order_items,
    df_returns=df_returns_leaked,
    df_products=df_products,
    df_customers=df_customers,
    df_reviews=df_reviews_leaked,
    df_geography=df_geography
)
fe_leaked.fit(df_train)
features_leaked = fe_leaked.transform(test_sample)

# Check if features are identical
diff_cols = []
for col in features_orig.columns:
    if col in ['order_date']:
        continue
    val_orig = features_orig[col].values[0]
    val_leak = features_leaked[col].values[0]
    # Check for difference, handling float precision and NaNs
    if pd.isna(val_orig) and pd.isna(val_leak):
        continue
    if val_orig != val_leak:
        diff_cols.append((col, val_orig, val_leak))

if len(diff_cols) == 0:
    print("\n✅ SUCCESS: No leakage detected! The features are identical despite future returns/reviews injection.")
else:
    print(f"\n❌ FAILURE: Leakage detected in columns:")
    for col, v1, v2 in diff_cols:
        print(f"  - {col}: Original={v1}, Leaked={v2}")
