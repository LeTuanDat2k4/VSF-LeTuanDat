import pandas as pd
import os

data_dir = r"c:\Users\PC\Documents\VSF\DataThon\datathon-2026-round-1"

# Check duplicates in remaining large files
for name in ["order_items.csv", "orders.csv", "reviews.csv", "returns.csv", "shipments.csv", "inventory.csv"]:
    df = pd.read_csv(os.path.join(data_dir, name))
    dup_count = df.duplicated().sum()
    print(f"{name} duplicates: {dup_count}")

# Check the distribution of prices in products.csv
df_prod = pd.read_csv(os.path.join(data_dir, "products.csv"))
print("\nProducts price stats:")
print(df_prod['price'].describe())
print("\nTop 10 highest prices:")
print(df_prod.sort_values(by='price', ascending=False).head(10)[['product_id', 'product_name', 'price', 'cogs']])

# Check the distribution of quantity in order_items.csv
df_items = pd.read_csv(os.path.join(data_dir, "order_items.csv"))
print("\nOrder items quantity values:")
print(df_items['quantity'].value_counts())
print("\nOrder items discount_amount stats:")
print(df_items['discount_amount'].describe())
