import pandas as pd
import os

data_dir = r"c:\Users\PC\Documents\VSF\DataThon\datathon-2026-round-1"
df_promo = pd.read_csv(os.path.join(data_dir, "promotions.csv"))
df_promo.dropna(how='all', inplace=True)
print("Unique promo names:")
print(df_promo['promo_name'].unique()[:15])

print("\nPromo channel value counts:")
print(df_promo['promo_channel'].value_counts())

# Let's see some top promotions with their start/end dates
print("\nTop promotions dates:")
print(df_promo[['promo_id', 'promo_name', 'start_date', 'end_date']].head(10))
