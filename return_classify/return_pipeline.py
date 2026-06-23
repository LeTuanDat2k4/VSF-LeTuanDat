import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OrdinalEncoder
from sklearn.impute import SimpleImputer
import xgboost as xgb

class ReturnFeatureExtractor(BaseEstimator, TransformerMixin):
    """
    Custom Transformer thực hiện toàn bộ quá trình Feature Engineering 
    từ file return_prediction_fe.ipynb.
    """
    def __init__(self, df_order_items, df_returns, df_products, df_customers, df_reviews, df_geography):
        # Nhận các bảng phụ làm tham số khởi tạo
        self.df_order_items = df_order_items
        self.df_returns = df_returns
        self.df_products = df_products
        self.df_customers = df_customers
        self.df_reviews = df_reviews
        self.df_geography = df_geography

    def fit(self, X, y=None):
        # Trong Pipeline chuẩn, fit dùng để tính toán các tham số từ tập train.
        # Ở đây, logic tính toán cumulative (leakage-safe) đang được thực hiện chung trên toàn tập dữ liệu
        # trong quá trình transform, nên ta chỉ cần pass. 
        # (Lưu ý: Để deploy thực tế online serving, bạn cần lưu trữ state của tập train tại đây).
        return self

    def transform(self, X):
        """
        X: df_orders (bảng đơn hàng gốc)
        """
        df_base = X.copy()
        
        # 1. Nhóm 1: Order Composition
        df_items = self.df_order_items.merge(self.df_products, on='product_id', how='left')
        df_items['line_total'] = df_items['quantity'] * df_items['unit_price'] - df_items['discount_amount']
        df_items['has_discount'] = (df_items['discount_amount'] > 0).astype(int)
        df_items['has_promo'] = df_items['promo_id'].notna().astype(int)
        
        size_map = {'S': 1, 'M': 2, 'L': 3, 'XL': 4}
        df_items['size_ordinal'] = df_items['size'].map(size_map)
        df_items['product_margin'] = (df_items['price'] - df_items['cogs']) / df_items['price'].clip(lower=1)
        
        order_agg = df_items.groupby('order_id').agg(
            n_products=('product_id', 'nunique'),
            n_items=('quantity', 'sum'),
            n_categories=('category', 'nunique'),
            n_segments=('segment', 'nunique'),
            n_sizes=('size', 'nunique'),
            n_colors=('color', 'nunique'),
            order_total_value=('line_total', 'sum'),
            avg_unit_price=('unit_price', 'mean'),
            std_unit_price=('unit_price', 'std'),
            total_discount=('discount_amount', 'sum'),
            avg_discount=('discount_amount', 'mean'),
            has_any_discount=('has_discount', 'max'),
            discount_item_ratio=('has_discount', 'mean'),
            has_any_promo=('has_promo', 'max'),
            promo_item_ratio=('has_promo', 'mean'),
            avg_size_ordinal=('size_ordinal', 'mean'),
            avg_margin=('product_margin', 'mean'),
            max_margin=('product_margin', 'max'),
            min_margin=('product_margin', 'min')
        ).reset_index()
        
        order_agg['std_unit_price'] = order_agg['std_unit_price'].fillna(0)
        
        price_range_df = df_items.groupby('order_id')['unit_price'].agg(price_max='max', price_min='min').reset_index()
        price_range_df['price_range'] = price_range_df['price_max'] - price_range_df['price_min']
        order_agg = order_agg.merge(price_range_df[['order_id', 'price_range']], on='order_id', how='left')
        
        order_agg['discount_pct'] = order_agg['total_discount'] / (order_agg['order_total_value'] + order_agg['total_discount']).clip(lower=1)
        
        df_base = df_base.merge(order_agg, on='order_id', how='left')
        
        # 2. Dominant Category & Segment
        dominant_cat = df_items.groupby('order_id')['category'].agg(lambda x: x.mode().iloc[0] if len(x) > 0 else 'unknown').reset_index(name='dominant_category')
        dominant_seg = df_items.groupby('order_id')['segment'].agg(lambda x: x.mode().iloc[0] if len(x) > 0 else 'unknown').reset_index(name='dominant_segment')
        df_base = df_base.merge(dominant_cat, on='order_id', how='left').merge(dominant_seg, on='order_id', how='left')
        
        # 3. Product Historical Return (Lưu ý: trong pipeline lý tưởng nên tính từ fit)
        # Tạo bảng: order_id -> order_date
        order_dates = df_base[['order_id', 'order_date']].copy()

        # Tạo bảng return history cho mỗi product
        items_with_date = self.df_order_items.merge(order_dates, on='order_id', how='inner')
        items_with_date = items_with_date.merge(
            self.df_returns[['order_id', 'product_id']].drop_duplicates().assign(was_returned=1),
            on=['order_id', 'product_id'], how='left'
        )
        items_with_date['was_returned'] = items_with_date['was_returned'].fillna(0).astype(int)
        items_with_date = items_with_date.sort_values('order_date')

        # Cumulative return rate per product (leakage-safe)
        items_with_date['prod_cum_returns'] = items_with_date.groupby('product_id')['was_returned'].cumsum()
        items_with_date['prod_cum_returns'] = items_with_date.groupby('product_id')['prod_cum_returns'].shift(1).fillna(0)
        items_with_date['prod_cum_total'] = items_with_date.groupby('product_id').cumcount()
        items_with_date['prod_hist_return_rate'] = items_with_date['prod_cum_returns'] / items_with_date['prod_cum_total'].clip(lower=1)

        # Aggregate product historical return rates lên cấp order
        order_prod_hist = items_with_date.groupby('order_id').agg(
            avg_prod_hist_return_rate=('prod_hist_return_rate', 'mean'),
            max_prod_hist_return_rate=('prod_hist_return_rate', 'max'),
            sum_prod_hist_returns=('prod_cum_returns', 'sum'),
        ).reset_index()

        df_base = df_base.merge(order_prod_hist, on='order_id', how='left')
        for col in ['avg_prod_hist_return_rate', 'max_prod_hist_return_rate', 'sum_prod_hist_returns']:
            df_base[col] = df_base[col].fillna(0)
        
        # 4. Customer Features
        df_base = df_base.merge(self.df_customers[['customer_id', 'signup_date', 'gender', 'age_group', 'acquisition_channel']], on='customer_id', how='left')
        df_base['customer_tenure_days'] = (df_base['order_date'] - df_base['signup_date']).dt.days.clip(lower=0)
        
        df_base = df_base.sort_values(['customer_id', 'order_date']).reset_index(drop=True)
        df_base['customer_order_number'] = df_base.groupby('customer_id').cumcount() + 1
        df_base['is_first_order'] = (df_base['customer_order_number'] == 1).astype(int)
        
        # Lấy is_returned từ df_returns để phục vụ cho các feature target-encoding dạng leakage-safe
        returned_orders = self.df_returns['order_id'].unique()
        df_base['is_returned'] = df_base['order_id'].isin(returned_orders).astype(int)

        # Customer historical return rate & count (Leakage-safe)
        df_base['cust_cum_returns'] = df_base.groupby('customer_id')['is_returned'].cumsum()
        df_base['cust_cum_returns'] = df_base.groupby('customer_id')['cust_cum_returns'].shift(1).fillna(0)
        df_base['cust_cum_orders'] = df_base.groupby('customer_id').cumcount()

        df_base['customer_return_rate'] = df_base['cust_cum_returns'] / df_base['cust_cum_orders'].clip(lower=1)
        df_base['customer_return_count'] = df_base['cust_cum_returns']
        df_base['customer_total_orders_before'] = df_base['cust_cum_orders']
        df_base.drop(columns=['cust_cum_returns', 'cust_cum_orders'], inplace=True)

        # Customer recency
        df_base = df_base.sort_values(['customer_id', 'order_date'])
        df_base['prev_order_date'] = df_base.groupby('customer_id')['order_date'].shift(1)
        df_base['customer_recency_days'] = (df_base['order_date'] - df_base['prev_order_date']).dt.days
        df_base['customer_recency_days'] = df_base['customer_recency_days'].fillna(-1)
        df_base.drop(columns=['prev_order_date'], inplace=True)

        # Customer average order value (historical)
        df_base = df_base.sort_values(['customer_id', 'order_date'])
        df_base['cust_cum_spending'] = df_base.groupby('customer_id')['order_total_value'].cumsum()
        df_base['cust_cum_spending'] = df_base.groupby('customer_id')['cust_cum_spending'].shift(1).fillna(0)
        df_base['cust_cum_cnt'] = df_base.groupby('customer_id').cumcount()
        df_base['customer_avg_order_value'] = df_base['cust_cum_spending'] / df_base['cust_cum_cnt'].clip(lower=1)
        df_base['customer_avg_order_value'] = df_base['customer_avg_order_value'].fillna(0)
        df_base.drop(columns=['cust_cum_spending', 'cust_cum_cnt'], inplace=True)

        # Channel features
        df_base = df_base.merge(self.df_geography[['zip', 'region']], on='zip', how='left')
        df_base['is_cod'] = (df_base['payment_method'] == 'cod').astype(int)

        # Temporal features
        df_base['order_month'] = df_base['order_date'].dt.month
        df_base['order_day_of_week'] = df_base['order_date'].dt.dayofweek
        df_base['order_day_of_year'] = df_base['order_date'].dt.dayofyear
        df_base['is_weekend'] = df_base['order_day_of_week'].isin([5, 6]).astype(int)
        df_base['order_quarter'] = df_base['order_date'].dt.quarter
        df_base['order_year'] = df_base['order_date'].dt.year

        df_base['month_sin'] = np.sin(2 * np.pi * df_base['order_month'] / 12)
        df_base['month_cos'] = np.cos(2 * np.pi * df_base['order_month'] / 12)
        df_base['dow_sin'] = np.sin(2 * np.pi * df_base['order_day_of_week'] / 7)
        df_base['dow_cos'] = np.cos(2 * np.pi * df_base['order_day_of_week'] / 7)

        # Customer review features (Leakage-safe)
        reviews_with_date = self.df_reviews.merge(
            df_base[['order_id', 'order_date', 'customer_id']],
            on=['order_id', 'customer_id'], how='inner'
        )
        reviews_with_date = reviews_with_date.sort_values(['customer_id', 'order_date'])

        reviews_with_date['cum_rating_sum'] = reviews_with_date.groupby('customer_id')['rating'].cumsum()
        reviews_with_date['cum_rating_sum'] = reviews_with_date.groupby('customer_id')['cum_rating_sum'].shift(1).fillna(0)
        reviews_with_date['cum_review_cnt'] = reviews_with_date.groupby('customer_id').cumcount()
        reviews_with_date['hist_avg_rating'] = reviews_with_date['cum_rating_sum'] / reviews_with_date['cum_review_cnt'].clip(lower=1)

        cust_review_hist = reviews_with_date.groupby(['order_id', 'customer_id']).agg(
            customer_avg_rating=('hist_avg_rating', 'first'),
            customer_review_count=('cum_review_cnt', 'first'),
        ).reset_index()

        df_base = df_base.merge(cust_review_hist, on=['order_id', 'customer_id'], how='left')
        df_base['customer_avg_rating'] = df_base['customer_avg_rating'].fillna(0)
        df_base['customer_review_count'] = df_base['customer_review_count'].fillna(0)
        df_base['has_reviewed'] = (df_base['customer_review_count'] > 0).astype(int)

        # Target-encoded features (Leakage-safe expanding mean)
        df_base = df_base.sort_values('order_date').reset_index(drop=True)

        df_base['cat_cum_returns'] = df_base.groupby('dominant_category')['is_returned'].cumsum()
        df_base['cat_cum_returns'] = df_base.groupby('dominant_category')['cat_cum_returns'].shift(1).fillna(0)
        df_base['cat_cum_total'] = df_base.groupby('dominant_category').cumcount()
        df_base['category_hist_return_rate'] = df_base['cat_cum_returns'] / df_base['cat_cum_total'].clip(lower=1)
        df_base.drop(columns=['cat_cum_returns', 'cat_cum_total'], inplace=True)

        df_base['reg_cum_returns'] = df_base.groupby('region')['is_returned'].cumsum()
        df_base['reg_cum_returns'] = df_base.groupby('region')['reg_cum_returns'].shift(1).fillna(0)
        df_base['reg_cum_total'] = df_base.groupby('region').cumcount()
        df_base['region_hist_return_rate'] = df_base['reg_cum_returns'] / df_base['reg_cum_total'].clip(lower=1)
        df_base.drop(columns=['reg_cum_returns', 'reg_cum_total'], inplace=True)

        df_base['pay_cum_returns'] = df_base.groupby('payment_method')['is_returned'].cumsum()
        df_base['pay_cum_returns'] = df_base.groupby('payment_method')['pay_cum_returns'].shift(1).fillna(0)
        df_base['pay_cum_total'] = df_base.groupby('payment_method').cumcount()
        df_base['payment_hist_return_rate'] = df_base['pay_cum_returns'] / df_base['pay_cum_total'].clip(lower=1)
        df_base.drop(columns=['pay_cum_returns', 'pay_cum_total'], inplace=True)

        # Nonlinear interaction features
        df_base['items_per_product'] = df_base['n_items'] / df_base['n_products'].clip(lower=1)
        df_base['discount_ratio'] = df_base['total_discount'] / (df_base['order_total_value'] + df_base['total_discount']).clip(lower=1)
        df_base['size_variety_ratio'] = df_base['n_sizes'] / df_base['n_products'].clip(lower=1)
        df_base['color_variety_ratio'] = df_base['n_colors'] / df_base['n_products'].clip(lower=1)
        df_base['avg_value_per_item'] = df_base['order_total_value'] / df_base['n_items'].clip(lower=1)
        df_base['customer_return_tendency'] = df_base['customer_return_rate'] * df_base['customer_order_number']

        # Trả về bộ Dataframe đã FE (chỉ lấy các cột feature mong muốn)
        numeric_features = [
            'n_products', 'n_items', 'n_categories', 'n_segments', 'n_sizes', 'n_colors',
            'order_total_value', 'avg_unit_price', 'std_unit_price',
            'total_discount', 'avg_discount', 'has_any_discount', 'discount_item_ratio', 'discount_pct',
            'has_any_promo', 'promo_item_ratio',
            'avg_size_ordinal', 'avg_margin', 'max_margin', 'min_margin', 'price_range',
            'avg_prod_hist_return_rate', 'max_prod_hist_return_rate', 'sum_prod_hist_returns',
            'is_cod', 'items_per_product', 'discount_ratio',
            'size_variety_ratio', 'color_variety_ratio',
            'avg_value_per_item', 'customer_return_tendency',
            'customer_tenure_days', 'customer_order_number', 'is_first_order',
            'customer_return_rate', 'customer_return_count', 'customer_total_orders_before',
            'customer_recency_days', 'customer_avg_order_value',
            'order_month', 'order_day_of_week', 'is_weekend', 'order_quarter', 'order_year',
            'month_sin', 'month_cos', 'dow_sin', 'dow_cos',
            'customer_avg_rating', 'customer_review_count', 'has_reviewed',
            'category_hist_return_rate', 'region_hist_return_rate', 'payment_hist_return_rate'
        ]
        categorical_features = [
            'payment_method', 'order_source', 'gender', 'age_group', 
            'acquisition_channel', 'region', 'dominant_category', 'dominant_segment'
        ]
        
        return df_base[numeric_features + categorical_features]


def build_pipeline(df_order_items, df_returns, df_products, df_customers, df_reviews, df_geography, scale_pos_weight):
    """
    Hàm xây dựng Full Sklearn Pipeline
    """
    # 1. Pipeline Feature Engineering
    fe_transformer = ReturnFeatureExtractor(
        df_order_items=df_order_items, 
        df_returns=df_returns, 
        df_products=df_products, 
        df_customers=df_customers, 
        df_reviews=df_reviews, 
        df_geography=df_geography
    )
    
    # 2. Pipeline Preprocessing (Tương đương LabelEncoder & FillNaN)
    # XGBoost xử lý Categorical tốt thông qua OrdinalEncoder
    numeric_features = [
        'n_products', 'n_items', 'n_categories', 'n_segments', 'n_sizes', 'n_colors',
        'order_total_value', 'avg_unit_price', 'std_unit_price',
        'total_discount', 'avg_discount', 'has_any_discount', 'discount_item_ratio', 'discount_pct',
        'has_any_promo', 'promo_item_ratio',
        'avg_size_ordinal', 'avg_margin', 'max_margin', 'min_margin', 'price_range',
        'avg_prod_hist_return_rate', 'max_prod_hist_return_rate', 'sum_prod_hist_returns',
        'is_cod', 'items_per_product', 'discount_ratio',
        'size_variety_ratio', 'color_variety_ratio',
        'avg_value_per_item', 'customer_return_tendency',
        'customer_tenure_days', 'customer_order_number', 'is_first_order',
        'customer_return_rate', 'customer_return_count', 'customer_total_orders_before',
        'customer_recency_days', 'customer_avg_order_value',
        'order_month', 'order_day_of_week', 'is_weekend', 'order_quarter', 'order_year',
        'month_sin', 'month_cos', 'dow_sin', 'dow_cos',
        'customer_avg_rating', 'customer_review_count', 'has_reviewed',
        'category_hist_return_rate', 'region_hist_return_rate', 'payment_hist_return_rate'
    ]
    categorical_features = [
        'payment_method', 'order_source', 'gender', 'age_group', 
        'acquisition_channel', 'region', 'dominant_category', 'dominant_segment'
    ]
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', SimpleImputer(strategy='constant', fill_value=0), numeric_features),
            ('cat', Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='constant', fill_value='unknown')),
                ('encoder', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1))
            ]), categorical_features)
        ],
        remainder='drop'
    )
    
    # 3. Model XGBoost (Tuned params)
    xgb_model = xgb.XGBClassifier(
        n_estimators=1000, 
        learning_rate=0.0174, 
        max_depth=6,
        min_child_weight=74, 
        subsample=0.673, 
        colsample_bytree=0.608,
        scale_pos_weight=scale_pos_weight, 
        eval_metric='auc',
        random_state=42, 
        n_jobs=-1,
        verbosity=0
    )
    
    # Gộp tất cả thành 1 Pipeline duy nhất
    full_pipeline = Pipeline(steps=[
        ('fe', fe_transformer),
        ('preprocessor', preprocessor),
        ('classifier', xgb_model)
    ])
    
    return full_pipeline

# Cách sử dụng:
if __name__ == "__main__":
    # 1. Load các bảng dữ liệu
    # df_orders = pd.read_csv('orders.csv')
    # df_items = ...
    
    # 2. Định nghĩa y
    # y = df_orders['is_returned']
    
    # 3. Build Pipeline
    # pipeline = build_pipeline(df_items, df_returns, df_products, df_customers, df_reviews, df_geography, scale_pos_weight=14.34)
    
    # 4. Fit & Predict
    # pipeline.fit(df_orders_train, y_train)
    # y_pred = pipeline.predict(df_orders_test)
    pass
