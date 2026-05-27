import pandas as pd
import os

data_dir = r"c:\Users\PC\Documents\VSF\DataThon\datathon-2026-round-1"

# 1. Check products price vs cogs
df_prod = pd.read_csv(os.path.join(data_dir, "products.csv"))
negative_margin = df_prod[df_prod['price'] < df_prod['cogs']]
print("Products with price < cogs:", len(negative_margin))
if len(negative_margin) > 0:
    print(negative_margin.head())

# 2. Check discount vs total price in order_items.csv
df_items = pd.read_csv(os.path.join(data_dir, "order_items.csv"))
invalid_discount = df_items[df_items['discount_amount'] > (df_items['unit_price'] * df_items['quantity'])]
print("Order items with discount > total_price:", len(invalid_discount))

# 3. Check dates in shipments.csv
df_ship = pd.read_csv(os.path.join(data_dir, "shipments.csv"))
df_orders = pd.read_csv(os.path.join(data_dir, "orders.csv"))
df_merged = pd.merge(df_ship, df_orders, on='order_id')
invalid_dates = df_merged[df_merged['ship_date'] < df_merged['order_date']]
print("Shipments with ship_date < order_date:", len(invalid_dates))
invalid_delivery = df_merged[df_merged['delivery_date'] < df_merged['ship_date']]
print("Shipments with delivery_date < ship_date:", len(invalid_delivery))
