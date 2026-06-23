import pandas as pd
from return_pipeline import build_pipeline
from sklearn.metrics import roc_auc_score
import os

print("Đang tải dữ liệu...")
DATA_DIR = '../datathon-2026-round-1'
df_orders = pd.read_csv(os.path.join(DATA_DIR, 'orders.csv'), parse_dates=['order_date'])
df_order_items = pd.read_csv(os.path.join(DATA_DIR, 'order_items.csv'), low_memory=False)
df_returns = pd.read_csv(os.path.join(DATA_DIR, 'returns.csv'), parse_dates=['return_date'])
df_products = pd.read_csv(os.path.join(DATA_DIR, 'products.csv'))
df_customers = pd.read_csv(os.path.join(DATA_DIR, 'customers.csv'), parse_dates=['signup_date'])
df_reviews = pd.read_csv(os.path.join(DATA_DIR, 'reviews.csv'), parse_dates=['review_date'])
df_geography = pd.read_csv(os.path.join(DATA_DIR, 'geography.csv'))

# Lọc df_orders theo điều kiện
eligible_statuses = ['delivered', 'returned']
df_eligible = df_orders[df_orders['order_status'].isin(eligible_statuses)].copy()

# Tạo nhãn is_returned cho df_eligible
returned_orders = df_returns['order_id'].unique()
y = df_eligible['order_id'].isin(returned_orders).astype(int)

# Khởi tạo Pipeline
print("Đang khởi tạo Pipeline...")
pipeline = build_pipeline(
    df_order_items=df_order_items,
    df_returns=df_returns,
    df_products=df_products,
    df_customers=df_customers,
    df_reviews=df_reviews,
    df_geography=df_geography,
    scale_pos_weight=14.34 # Có thể tính tự động bằng len(y==0) / len(y==1)
)

print("Đang chạy fit (Train mô hình)...")
# Trong thực tế, bạn sẽ chia train/test. Ở đây ta chạy test nhanh trên một mẫu 10,000 dòng để tiết kiệm bộ nhớ & thời gian
sample_idx = df_eligible.sample(n=10000, random_state=42).index
df_train_sample = df_eligible.loc[sample_idx]
y_train_sample = y.loc[sample_idx]

pipeline.fit(df_train_sample, y_train_sample)

print("Đang chạy dự đoán (Predict)...")
y_proba = pipeline.predict_proba(df_train_sample)[:, 1]

auc = roc_auc_score(y_train_sample, y_proba)
print(f"✅ Hoàn tất! AUC-ROC trên tập mẫu: {auc:.4f}")
