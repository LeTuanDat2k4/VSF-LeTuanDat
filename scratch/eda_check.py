import pandas as pd
import os

data_dir = r"c:\Users\PC\Documents\VSF\DataThon\datathon-2026-round-1"

# 1. Check regions in geography.csv
df_geo = pd.read_csv(os.path.join(data_dir, "geography.csv"))
print("Unique regions in geography.csv:", df_geo['region'].unique())

# 2. Check categories in products.csv
df_prod = pd.read_csv(os.path.join(data_dir, "products.csv"))
print("Unique categories in products.csv:", df_prod['category'].unique())

# 3. Check date range in sales.csv
df_sales = pd.read_csv(os.path.join(data_dir, "sales.csv"))
df_sales['Date'] = pd.to_datetime(df_sales['Date'])
print(f"Sales date range: {df_sales['Date'].min()} to {df_sales['Date'].max()}")
print("Are there missing days in sales.csv?", (df_sales['Date'].max() - df_sales['Date'].min()).days + 1 == len(df_sales))

# 4. Check orders and geography merge to see how regions map
df_orders = pd.read_csv(os.path.join(data_dir, "orders.csv"), nrows=1000)
df_merged = pd.merge(df_orders, df_geo, on='zip', how='inner')
print("Successfully merged orders with geography. Merged shape:", df_merged.shape)
