import pandas as pd
from return_pipeline import build_pipeline
from sklearn.metrics import roc_auc_score, classification_report
import os
import mlflow
import mlflow.sklearn

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

# Tạo nhãn is_returned
returned_orders = df_returns['order_id'].unique()
y = df_eligible['order_id'].isin(returned_orders).astype(int)

# ── Khởi tạo Pipeline ─────────────────────────────────────────────
print("\nĐang khởi tạo Pipeline...")
SPLIT_DATE = pd.Timestamp('2022-07-01')

# Tính scale_pos_weight từ tập train
train_mask_y = df_eligible['order_date'] < SPLIT_DATE
n_neg = (y[train_mask_y] == 0).sum()
n_pos = (y[train_mask_y] == 1).sum()
auto_scale = n_neg / max(n_pos, 1)

pipeline = build_pipeline(
    df_order_items=df_order_items,
    df_returns=df_returns,
    df_products=df_products,
    df_customers=df_customers,
    df_reviews=df_reviews,
    df_geography=df_geography,
    scale_pos_weight=auto_scale,
    early_stopping_rounds=50,
)

fe           = pipeline.named_steps['fe']
preprocessor = pipeline.named_steps['preprocessor']
classifier   = pipeline.named_steps['classifier']

# ── Bước 1: Feature Engineering trên TOÀN BỘ dữ liệu ─────────────
# Giữ nguyên lịch sử lũy kế (cumulative) xuyên suốt train → test,
# giống như notebook tính features trên toàn bộ rồi mới split.
print("Đang chạy Feature Engineering trên toàn bộ dữ liệu...")
X_all_fe = fe.fit_transform(df_eligible)
print(f"  → {X_all_fe.shape[0]:,} dòng × {X_all_fe.shape[1]} features (incl. order_id, order_date)")

# ── Bước 2: Split theo thời gian dựa trên output của FE ───────────
# QUAN TRỌNG: FE nội bộ sort_values + reset_index → thứ tự dòng thay đổi.
# Phải dùng order_date/order_id TỪ output FE để split chính xác.
train_mask = X_all_fe['order_date'] < SPLIT_DATE
test_mask  = X_all_fe['order_date'] >= SPLIT_DATE

# Tạo nhãn y dựa trên order_id từ FE output (đã re-order)
y_aligned = X_all_fe['order_id'].isin(returned_orders).astype(int)

y_train = y_aligned[train_mask]
y_test  = y_aligned[test_mask]

# Bỏ order_id, order_date trước khi đưa vào preprocessor
X_train_fe = X_all_fe[train_mask].drop(columns=['order_id', 'order_date'])
X_test_fe  = X_all_fe[test_mask].drop(columns=['order_id', 'order_date'])

print(f"\nTemporal Split tại {SPLIT_DATE.date()}:")
print(f"  Train: {len(X_train_fe):,} đơn  |  Return rate: {y_train.mean():.2%}")
print(f"  Test:  {len(X_test_fe):,} đơn  |  Return rate: {y_test.mean():.2%}")
print(f"  scale_pos_weight (tự động): {auto_scale:.2f}")

# ── Thiết lập MLflow ───────────────────────────────────────────────
mlflow.set_experiment("Return_Classification")

with mlflow.start_run():
    # ── Bước 3: Preprocessing (Impute + Encode) ───────────────────────
    print("\nĐang chạy Preprocessing...")
    X_train_pp = preprocessor.fit_transform(X_train_fe)
    X_test_pp  = preprocessor.transform(X_test_fe)
    
    # ── Bước 4: Train XGBoost với Early Stopping ──────────────────────
    print("Đang chạy fit với Early Stopping (patience=50)...")
    classifier.fit(
        X_train_pp, y_train,
        eval_set=[(X_train_pp, y_train), (X_test_pp, y_test)],
        verbose=100,
    )
    print(f"  → Dừng tại round: {classifier.best_iteration}")
    
    # ── Evaluate trên tập Test ─────────────────────────────────────────
    print("\nĐang chạy dự đoán trên tập Test...")
    y_test_proba = classifier.predict_proba(X_test_pp)[:, 1]
    
    auc_test = roc_auc_score(y_test, y_test_proba)
    print(f"\n{'='*50}")
    print(f"  AUC-ROC trên tập TEST: {auc_test:.4f}")
    print(f"{'='*50}")
    
    # Classification Report (threshold mặc định 0.5)
    y_test_pred = (y_test_proba >= 0.5).astype(int)
    print("\nClassification Report (threshold=0.5):")
    print(classification_report(y_test, y_test_pred, target_names=['Not Returned', 'Returned']))
    
    # ── Tham khảo: AUC trên tập Train ─────────────────────────────────
    y_train_proba = classifier.predict_proba(X_train_pp)[:, 1]
    auc_train = roc_auc_score(y_train, y_train_proba)
    print(f"(Tham khảo) AUC-ROC trên tập TRAIN: {auc_train:.4f}")
    print(f"Chênh lệch Train-Test: {auc_train - auc_test:.4f}")
    
    # Log parameters
    mlflow.log_params({
        "scale_pos_weight": auto_scale,
        "n_estimators": classifier.n_estimators,
        "learning_rate": classifier.learning_rate,
        "max_depth": classifier.max_depth,
        "min_child_weight": classifier.min_child_weight,
        "subsample": classifier.subsample,
        "colsample_bytree": classifier.colsample_bytree,
        "early_stopping_rounds": classifier.early_stopping_rounds,
        "split_date": str(SPLIT_DATE.date())
    })
    
    # Log metrics
    mlflow.log_metrics({
        "auc_train": auc_train,
        "auc_test": auc_test,
        "best_iteration": classifier.best_iteration if getattr(classifier, "best_iteration", None) is not None else classifier.n_estimators
    })
    
    # Log the entire pipeline
    print("Đang log pipeline vào MLflow...")
    mlflow.sklearn.log_model(
        sk_model=pipeline,
        artifact_path="model",
        code_paths=["return_pipeline.py"],
        serialization_format="cloudpickle"
    )
    print("Đã log pipeline thành công!")
    print(f"\n✅ Hoàn tất đánh giá Temporal Split với Early Stopping và log MLflow!")


