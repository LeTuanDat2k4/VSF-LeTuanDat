import pandas as pd
import numpy as np
import os

DATA_DIR = '../datathon-2026-round-1'
df_orders = pd.read_csv(os.path.join(DATA_DIR, 'orders.csv'), parse_dates=['order_date'])
df_order_items = pd.read_csv(os.path.join(DATA_DIR, 'order_items.csv'), low_memory=False)
df_returns = pd.read_csv(os.path.join(DATA_DIR, 'returns.csv'), parse_dates=['return_date'])
df_products = pd.read_csv(os.path.join(DATA_DIR, 'products.csv'))
df_customers = pd.read_csv(os.path.join(DATA_DIR, 'customers.csv'), parse_dates=['signup_date'])
df_reviews = pd.read_csv(os.path.join(DATA_DIR, 'reviews.csv'), parse_dates=['review_date'])
df_geography = pd.read_csv(os.path.join(DATA_DIR, 'geography.csv'))

# Let's inspect the dates and check for the review date leakage
# For customer reviews:
print("--- Checking review leakage ---")
# Merge reviews with orders to get order_date and review_date
reviews_orders = df_reviews.merge(df_orders[['order_id', 'order_date', 'customer_id']], on=['order_id', 'customer_id'], how='inner')
print(f"Total reviews matched with orders: {len(reviews_orders)}")

# Check if review_date is always after order_date
reviews_orders['review_delay'] = (reviews_orders['review_date'] - reviews_orders['order_date']).dt.days
print(f"Review delay statistics (days):")
print(reviews_orders['review_delay'].describe())

# Now let's trace a specific customer who has multiple orders and reviews
# Let's see if there is any case where order_date of order B is between order_date of order A and review_date of order A.
# Sort by customer_id and order_date
reviews_orders = reviews_orders.sort_values(['customer_id', 'order_date'])
reviews_orders['prev_review_date'] = reviews_orders.groupby('customer_id')['review_date'].shift(1)
reviews_orders['prev_order_date'] = reviews_orders.groupby('customer_id')['order_date'].shift(1)

# A leak occurs if: the current order_date < prev_review_date
# Because that means when current order was placed, the previous review was not yet written!
leaked_reviews = reviews_orders[reviews_orders['order_date'] < reviews_orders['prev_review_date']]
print(f"Number of orders where the previous review was not yet written at order time: {len(leaked_reviews)}")
if len(leaked_reviews) > 0:
    print(leaked_reviews[['customer_id', 'prev_order_date', 'prev_review_date', 'order_date', 'review_date']].head(10))


print("\n--- Checking return leakage ---")
# For product return rate or customer return rate:
# Let's see if a return happened after a subsequent order, but is counted.
# We need to join returns to orders to get return_date.
orders_returns = df_orders.merge(df_returns[['order_id', 'return_date']].drop_duplicates(), on='order_id', how='left')
orders_returns = orders_returns.sort_values(['customer_id', 'order_date']).reset_index(drop=True)

# Let's trace customer returns
orders_returns['is_returned'] = orders_returns['return_date'].notna().astype(int)
orders_returns['cust_cum_returns_leaked'] = orders_returns.groupby('customer_id')['is_returned'].cumsum()
orders_returns['cust_cum_returns_leaked'] = orders_returns.groupby('customer_id')['cust_cum_returns_leaked'].shift(1).fillna(0)

# To calculate leakage-free cumulative returns:
# For each order, we should count how many of the customer's *previous* orders had a return_date < current order's order_date.
def compute_leakage_free_cust_returns(df):
    # For each row, count previous rows of same customer where return_date < order_date
    cust_returns_clean = []
    # We can do this efficiently by comparing all pairs or using a loop for active customers
    # But since it's a verification, let's write a simple loop for a subset or vectorized if possible.
    pass

# Let's find concrete examples where return_date of a previous order is after order_date of a subsequent order.
orders_returns['prev_return_date'] = orders_returns.groupby('customer_id')['return_date'].shift(1)
orders_returns['prev_order_date'] = orders_returns.groupby('customer_id')['order_date'].shift(1)

leaked_returns = orders_returns[orders_returns['order_date'] < orders_returns['prev_return_date']]
print(f"Number of orders where the previous order's return was not yet processed at order time: {len(leaked_returns)}")
if len(leaked_returns) > 0:
    print(leaked_returns[['customer_id', 'prev_order_date', 'prev_return_date', 'order_date', 'return_date']].head(10))
