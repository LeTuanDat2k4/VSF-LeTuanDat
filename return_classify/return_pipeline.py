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
    từ file return_prediction_fe.ipynb một cách sạch sẽ, không bị Data Leakage.
    Đã tối ưu hóa hiệu năng bằng cách precompute các bảng lũy kế lịch sử trong fit(),
    giúp transform() chạy cực nhanh (mili-giây) khi inference trực tuyến.
    """
    def __init__(self, df_order_items, df_returns, df_products, df_customers, df_reviews, df_geography):
        # Nhận các bảng phụ làm tham số khởi tạo
        self.df_order_items = df_order_items.copy()
        self.df_returns = df_returns.copy()
        self.df_products = df_products.copy()
        self.df_customers = df_customers.copy()
        self.df_reviews = df_reviews.copy()
        self.df_geography = df_geography.copy()
        self.train_orders = None

        # Parse dates
        self.df_returns['return_date'] = pd.to_datetime(self.df_returns['return_date'])
        self.df_reviews['review_date'] = pd.to_datetime(self.df_reviews['review_date'])
        self.df_customers['signup_date'] = pd.to_datetime(self.df_customers['signup_date'])

    def fit(self, X, y=None):
        X_base = X.copy()
        X_base['order_date'] = pd.to_datetime(X_base['order_date'])
        self.train_orders = X_base[['order_id', 'order_date', 'customer_id', 'zip', 'payment_method', 'order_source']].copy()
        
        all_orders = self.train_orders
        
        # 1. Product Daily Statistics (Lũy kế số đơn và lượt trả của sản phẩm)
        all_prod_orders = self.df_order_items[['order_id', 'product_id']].merge(
            all_orders[['order_id', 'order_date']], on='order_id', how='inner'
        )
        all_prod_orders['order_date'] = pd.to_datetime(all_prod_orders['order_date'])
        
        prod_daily_orders = all_prod_orders.groupby(['product_id', 'order_date']).size().reset_index(name='daily_count')
        prod_daily_orders = prod_daily_orders.sort_values(['product_id', 'order_date'])
        prod_daily_orders['cum_orders'] = prod_daily_orders.groupby('product_id')['daily_count'].cumsum()
        prod_daily_orders['cum_orders_before'] = prod_daily_orders.groupby('product_id')['cum_orders'].shift(1).fillna(0)
        self.prod_daily_orders_ = prod_daily_orders[['product_id', 'order_date', 'cum_orders_before']].copy()
        
        prod_returns_df = self.df_returns[['product_id', 'return_date']].dropna().copy()
        prod_returns_df['return_date'] = pd.to_datetime(prod_returns_df['return_date'])
        prod_returns_daily = prod_returns_df.groupby(['product_id', 'return_date']).size().reset_index(name='daily_returns')
        prod_returns_daily = prod_returns_daily.sort_values(['product_id', 'return_date'])
        prod_returns_daily['cum_returns'] = prod_returns_daily.groupby('product_id')['daily_returns'].cumsum()
        self.prod_returns_daily_sorted_ = prod_returns_daily[['product_id', 'return_date', 'cum_returns']].copy()
        
        # 2. Customer Daily Statistics (Lũy kế số đơn hàng và chi tiêu)
        df_items_all = self.df_order_items.merge(self.df_products, on='product_id', how='left')
        df_items_all = df_items_all[df_items_all['order_id'].isin(all_orders['order_id'])].copy()
        df_items_all['line_total'] = df_items_all['quantity'] * df_items_all['unit_price'] - df_items_all['discount_amount']
        
        order_agg_fit = df_items_all.groupby('order_id').agg(
            order_total_value=('line_total', 'sum')
        ).reset_index()
        
        all_orders_agg = all_orders.merge(order_agg_fit, on='order_id', how='left')
        all_orders_agg['order_total_value'] = all_orders_agg['order_total_value'].fillna(0)
        all_orders_agg = all_orders_agg.sort_values('order_date')
        
        cust_daily = all_orders_agg.groupby(['customer_id', 'order_date']).agg(
            daily_count=('order_id', 'count'),
            daily_spending=('order_total_value', 'sum')
        ).reset_index()
        cust_daily = cust_daily.sort_values(['customer_id', 'order_date'])
        cust_daily['cum_orders'] = cust_daily.groupby('customer_id')['daily_count'].cumsum()
        cust_daily['cum_spending'] = cust_daily.groupby('customer_id')['daily_spending'].cumsum()
        cust_daily['cust_cum_orders_before'] = cust_daily.groupby('customer_id')['cum_orders'].shift(1).fillna(0)
        cust_daily['cust_cum_spending_before'] = cust_daily.groupby('customer_id')['cum_spending'].shift(1).fillna(0)
        cust_daily['prev_order_date'] = cust_daily.groupby('customer_id')['order_date'].shift(1)
        self.cust_daily_ = cust_daily[['customer_id', 'order_date', 'cust_cum_orders_before', 'cust_cum_spending_before', 'prev_order_date']].copy()
        
        cust_returns = self.df_returns[['order_id', 'return_date']].dropna().merge(
            all_orders[['order_id', 'customer_id']], on='order_id', how='inner'
        )
        cust_returns['return_date'] = pd.to_datetime(cust_returns['return_date'])
        cust_returns_daily = cust_returns.groupby(['customer_id', 'return_date']).size().reset_index(name='daily_returns')
        cust_returns_daily = cust_returns_daily.sort_values(['customer_id', 'return_date'])
        cust_returns_daily['cum_returns'] = cust_returns_daily.groupby('customer_id')['daily_returns'].cumsum()
        self.cust_returns_daily_sorted_ = cust_returns_daily[['customer_id', 'return_date', 'cum_returns']].copy()
        
        # 3. Customer Reviews Statistics (Lũy kế đánh giá)
        cust_reviews = self.df_reviews[['customer_id', 'review_date', 'rating']].dropna().copy()
        cust_reviews['review_date'] = pd.to_datetime(cust_reviews['review_date'])
        cust_reviews_daily = cust_reviews.groupby(['customer_id', 'review_date']).agg(
            daily_count=('rating', 'count'),
            daily_rating_sum=('rating', 'sum')
        ).reset_index()
        cust_reviews_daily = cust_reviews_daily.sort_values(['customer_id', 'review_date'])
        cust_reviews_daily['cum_reviews'] = cust_reviews_daily.groupby('customer_id')['daily_count'].cumsum()
        cust_reviews_daily['cum_rating_sum'] = cust_reviews_daily.groupby('customer_id')['daily_rating_sum'].cumsum()
        self.cust_reviews_daily_sorted_ = cust_reviews_daily[['customer_id', 'review_date', 'cum_reviews', 'cum_rating_sum']].copy()
        
        # 4. Target-encoded Category, Region, Payment Statistics
        dominant_cat_fit = df_items_all.groupby('order_id')['category'].agg(lambda x: x.mode().iloc[0] if len(x) > 0 else 'unknown').reset_index(name='dominant_category')
        all_orders_geo = all_orders.merge(self.df_geography[['zip', 'region']], on='zip', how='left')
        all_orders_geo = all_orders_geo.merge(dominant_cat_fit, on='order_id', how='left')
        all_order_metadata = all_orders_geo[['order_id', 'order_date', 'dominant_category', 'region', 'payment_method']].copy()
        
        # Category
        cat_daily_orders = all_order_metadata.groupby(['dominant_category', 'order_date']).size().reset_index(name='daily_count')
        cat_daily_orders = cat_daily_orders.sort_values(['dominant_category', 'order_date'])
        cat_daily_orders['cum_orders'] = cat_daily_orders.groupby('dominant_category')['daily_count'].cumsum()
        cat_daily_orders['cum_orders_before'] = cat_daily_orders.groupby('dominant_category')['cum_orders'].shift(1).fillna(0)
        self.cat_daily_orders_ = cat_daily_orders[['dominant_category', 'order_date', 'cum_orders_before']].copy()
        
        cat_returns = self.df_returns[['order_id', 'return_date']].dropna().merge(
            all_order_metadata[['order_id', 'dominant_category']], on='order_id', how='inner'
        )
        cat_returns['return_date'] = pd.to_datetime(cat_returns['return_date'])
        cat_returns_daily = cat_returns.groupby(['dominant_category', 'return_date']).size().reset_index(name='daily_returns')
        cat_returns_daily = cat_returns_daily.sort_values(['dominant_category', 'return_date'])
        cat_returns_daily['cum_returns'] = cat_returns_daily.groupby('dominant_category')['daily_returns'].cumsum()
        self.cat_returns_daily_sorted_ = cat_returns_daily[['dominant_category', 'return_date', 'cum_returns']].copy()
        
        # Region
        reg_daily_orders = all_order_metadata.groupby(['region', 'order_date']).size().reset_index(name='daily_count')
        reg_daily_orders = reg_daily_orders.sort_values(['region', 'order_date'])
        reg_daily_orders['cum_orders'] = reg_daily_orders.groupby('region')['daily_count'].cumsum()
        reg_daily_orders['cum_orders_before'] = reg_daily_orders.groupby('region')['cum_orders'].shift(1).fillna(0)
        self.reg_daily_orders_ = reg_daily_orders[['region', 'order_date', 'cum_orders_before']].copy()
        
        reg_returns = self.df_returns[['order_id', 'return_date']].dropna().merge(
            all_order_metadata[['order_id', 'region']], on='order_id', how='inner'
        )
        reg_returns['return_date'] = pd.to_datetime(reg_returns['return_date'])
        reg_returns_daily = reg_returns.groupby(['region', 'return_date']).size().reset_index(name='daily_returns')
        reg_returns_daily = reg_returns_daily.sort_values(['region', 'return_date'])
        reg_returns_daily['cum_returns'] = reg_returns_daily.groupby('region')['daily_returns'].cumsum()
        self.reg_returns_daily_sorted_ = reg_returns_daily[['region', 'return_date', 'cum_returns']].copy()
        
        # Payment
        pay_daily_orders = all_order_metadata.groupby(['payment_method', 'order_date']).size().reset_index(name='daily_count')
        pay_daily_orders = pay_daily_orders.sort_values(['payment_method', 'order_date'])
        pay_daily_orders['cum_orders'] = pay_daily_orders.groupby('payment_method')['daily_count'].cumsum()
        pay_daily_orders['cum_orders_before'] = pay_daily_orders.groupby('payment_method')['cum_orders'].shift(1).fillna(0)
        self.pay_daily_orders_ = pay_daily_orders[['payment_method', 'order_date', 'cum_orders_before']].copy()
        
        pay_returns = self.df_returns[['order_id', 'return_date']].dropna().merge(
            all_order_metadata[['order_id', 'payment_method']], on='order_id', how='inner'
        )
        pay_returns['return_date'] = pd.to_datetime(pay_returns['return_date'])
        pay_returns_daily = pay_returns.groupby(['payment_method', 'return_date']).size().reset_index(name='daily_returns')
        pay_returns_daily = pay_returns_daily.sort_values(['payment_method', 'return_date'])
        pay_returns_daily['cum_returns'] = pay_returns_daily.groupby('payment_method')['daily_returns'].cumsum()
        self.pay_returns_daily_sorted_ = pay_returns_daily[['payment_method', 'return_date', 'cum_returns']].copy()
        
        return self

    def transform(self, X):
        """
        X: df_orders (bảng đơn hàng cần dự đoán)
        """
        X_base = X.copy()
        X_base['order_date'] = pd.to_datetime(X_base['order_date'])
        
        # ── 1. Order Composition & Aggregates (Chỉ tính trên các đơn hàng của X) ──
        df_items_x = self.df_order_items[self.df_order_items['order_id'].isin(X_base['order_id'])].merge(self.df_products, on='product_id', how='left').copy()
        
        df_items_x['line_total'] = df_items_x['quantity'] * df_items_x['unit_price'] - df_items_x['discount_amount']
        df_items_x['has_discount'] = (df_items_x['discount_amount'] > 0).astype(int)
        df_items_x['has_promo'] = df_items_x['promo_id'].notna().astype(int)
        
        size_map = {'S': 1, 'M': 2, 'L': 3, 'XL': 4}
        df_items_x['size_ordinal'] = df_items_x['size'].map(size_map)
        df_items_x['product_margin'] = (df_items_x['price'] - df_items_x['cogs']) / df_items_x['price'].clip(lower=1)
        
        if len(df_items_x) > 0:
            order_agg = df_items_x.groupby('order_id').agg(
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
            
            price_range_df = df_items_x.groupby('order_id')['unit_price'].agg(price_max='max', price_min='min').reset_index()
            price_range_df['price_range'] = price_range_df['price_max'] - price_range_df['price_min']
            order_agg = order_agg.merge(price_range_df[['order_id', 'price_range']], on='order_id', how='left')
            
            order_agg['discount_pct'] = order_agg['total_discount'] / (order_agg['order_total_value'] + order_agg['total_discount']).clip(lower=1)
            
            dominant_cat = df_items_x.groupby('order_id')['category'].agg(lambda x: x.mode().iloc[0] if len(x) > 0 else 'unknown').reset_index(name='dominant_category')
            dominant_seg = df_items_x.groupby('order_id')['segment'].agg(lambda x: x.mode().iloc[0] if len(x) > 0 else 'unknown').reset_index(name='dominant_segment')
        else:
            cols = ['order_id', 'n_products', 'n_items', 'n_categories', 'n_segments', 'n_sizes', 'n_colors', 
                    'order_total_value', 'avg_unit_price', 'std_unit_price', 'total_discount', 'avg_discount', 
                    'has_any_discount', 'discount_item_ratio', 'has_any_promo', 'promo_item_ratio', 
                    'avg_size_ordinal', 'avg_margin', 'max_margin', 'min_margin', 'price_range', 'discount_pct']
            order_agg = pd.DataFrame(columns=cols)
            dominant_cat = pd.DataFrame(columns=['order_id', 'dominant_category'])
            dominant_seg = pd.DataFrame(columns=['order_id', 'dominant_segment'])
            
        df_base = X_base.copy()
        df_base = df_base.merge(order_agg, on='order_id', how='left')
        df_base = df_base.merge(dominant_cat, on='order_id', how='left')
        df_base = df_base.merge(dominant_seg, on='order_id', how='left')
        df_base = df_base.merge(self.df_geography[['zip', 'region']], on='zip', how='left')
        
        # Fill missing fields
        fill_cols = [
            'n_products', 'n_items', 'n_categories', 'n_segments', 'n_sizes', 'n_colors',
            'order_total_value', 'avg_unit_price', 'std_unit_price', 'total_discount', 'avg_discount',
            'has_any_discount', 'discount_item_ratio', 'has_any_promo', 'promo_item_ratio',
            'avg_size_ordinal', 'avg_margin', 'max_margin', 'min_margin', 'price_range', 'discount_pct'
        ]
        for c in fill_cols:
            df_base[c] = df_base[c].fillna(0)
        df_base['dominant_category'] = df_base['dominant_category'].fillna('unknown')
        df_base['dominant_segment'] = df_base['dominant_segment'].fillna('unknown')

        # ── 2. Product Historical Returns ─────────────────────────────────
        X_items = self.df_order_items[self.df_order_items['order_id'].isin(X_base['order_id'])][['order_id', 'product_id']].merge(
            X_base[['order_id', 'order_date']], on='order_id', how='inner'
        )
        X_items['order_date'] = pd.to_datetime(X_items['order_date'])
        
        if len(X_items) > 0:
            X_items = X_items.merge(
                self.prod_daily_orders_,
                on=['product_id', 'order_date'],
                how='left'
            )
            X_items['cum_orders_before'] = X_items['cum_orders_before'].fillna(0)
            
            X_items = X_items.sort_values('order_date')
            prod_returns_sorted = self.prod_returns_daily_sorted_.sort_values('return_date')
            X_items = pd.merge_asof(
                X_items,
                prod_returns_sorted,
                left_on='order_date',
                right_on='return_date',
                by='product_id',
                direction='backward',
                allow_exact_matches=False
            )
            X_items['cum_returns'] = X_items['cum_returns'].fillna(0)
            X_items['prod_hist_return_rate'] = X_items['cum_returns'] / X_items['cum_orders_before'].clip(lower=1)
            
            order_prod_hist = X_items.groupby('order_id').agg(
                avg_prod_hist_return_rate=('prod_hist_return_rate', 'mean'),
                max_prod_hist_return_rate=('prod_hist_return_rate', 'max'),
                sum_prod_hist_returns=('cum_returns', 'sum')
            ).reset_index()
        else:
            order_prod_hist = pd.DataFrame(columns=['order_id', 'avg_prod_hist_return_rate', 'max_prod_hist_return_rate', 'sum_prod_hist_returns'])

        # ── 3. Customer Features ──────────────────────────────────────────
        df_base = df_base.merge(
            self.df_customers[['customer_id', 'signup_date', 'gender', 'age_group', 'acquisition_channel']], 
            on='customer_id', how='left'
        )
        df_base['customer_tenure_days'] = (df_base['order_date'] - df_base['signup_date']).dt.days.clip(lower=0).fillna(0)
        
        df_base = df_base.sort_values('order_date')
        df_base = df_base.merge(
            self.cust_daily_,
            on=['customer_id', 'order_date'],
            how='left'
        )
        df_base['cust_cum_orders_before'] = df_base['cust_cum_orders_before'].fillna(0)
        df_base['cust_cum_spending_before'] = df_base['cust_cum_spending_before'].fillna(0)
        
        df_base['customer_order_number'] = df_base['cust_cum_orders_before'] + 1
        df_base['is_first_order'] = (df_base['customer_order_number'] == 1).astype(int)
        df_base['customer_recency_days'] = (df_base['order_date'] - df_base['prev_order_date']).dt.days.fillna(-1)
        df_base['customer_avg_order_value'] = (df_base['cust_cum_spending_before'] / df_base['cust_cum_orders_before'].clip(lower=1)).fillna(0)
        df_base.drop(columns=['cust_cum_spending_before', 'prev_order_date'], errors='ignore', inplace=True)
        
        cust_returns_sorted = self.cust_returns_daily_sorted_.sort_values('return_date')
        df_base = pd.merge_asof(
            df_base,
            cust_returns_sorted,
            left_on='order_date',
            right_on='return_date',
            by='customer_id',
            direction='backward',
            allow_exact_matches=False
        )
        df_base['cum_returns'] = df_base['cum_returns'].fillna(0)
        df_base['customer_return_rate'] = (df_base['cum_returns'] / df_base['cust_cum_orders_before'].clip(lower=1)).fillna(0)
        df_base['customer_return_count'] = df_base['cum_returns']
        df_base['customer_total_orders_before'] = df_base['cust_cum_orders_before']
        df_base.drop(columns=['return_date', 'cum_returns', 'cust_cum_orders_before'], errors='ignore', inplace=True)

        # ── 4. Customer Review Features ───────────────────────────────────
        cust_reviews_sorted = self.cust_reviews_daily_sorted_.sort_values('review_date')
        df_base = pd.merge_asof(
            df_base,
            cust_reviews_sorted,
            left_on='order_date',
            right_on='review_date',
            by='customer_id',
            direction='backward',
            allow_exact_matches=False
        )
        df_base['cum_reviews'] = df_base['cum_reviews'].fillna(0)
        df_base['cum_rating_sum'] = df_base['cum_rating_sum'].fillna(0)
        df_base['customer_avg_rating'] = (df_base['cum_rating_sum'] / df_base['cum_reviews'].clip(lower=1)).fillna(0)
        df_base['customer_review_count'] = df_base['cum_reviews']
        df_base['has_reviewed'] = (df_base['customer_review_count'] > 0).astype(int)
        df_base.drop(columns=['review_date', 'cum_reviews', 'cum_rating_sum'], errors='ignore', inplace=True)

        # ── 5. Target-encoded Features ────────────────────────────────────
        # Category
        df_base = df_base.merge(
            self.cat_daily_orders_,
            on=['dominant_category', 'order_date'],
            how='left'
        )
        df_base['cum_orders_before'] = df_base['cum_orders_before'].fillna(0)
        
        cat_returns_sorted = self.cat_returns_daily_sorted_.sort_values('return_date')
        df_base = pd.merge_asof(
            df_base,
            cat_returns_sorted,
            left_on='order_date',
            right_on='return_date',
            by='dominant_category',
            direction='backward',
            allow_exact_matches=False
        )
        df_base['cum_returns'] = df_base['cum_returns'].fillna(0)
        df_base['category_hist_return_rate'] = (df_base['cum_returns'] / df_base['cum_orders_before'].clip(lower=1)).fillna(0)
        df_base.drop(columns=['return_date', 'cum_returns', 'cum_orders_before'], errors='ignore', inplace=True)
        
        # Region
        df_base = df_base.merge(
            self.reg_daily_orders_,
            on=['region', 'order_date'],
            how='left'
        )
        df_base['cum_orders_before'] = df_base['cum_orders_before'].fillna(0)
        
        reg_returns_sorted = self.reg_returns_daily_sorted_.sort_values('return_date')
        df_base = pd.merge_asof(
            df_base,
            reg_returns_sorted,
            left_on='order_date',
            right_on='return_date',
            by='region',
            direction='backward',
            allow_exact_matches=False
        )
        df_base['cum_returns'] = df_base['cum_returns'].fillna(0)
        df_base['region_hist_return_rate'] = (df_base['cum_returns'] / df_base['cum_orders_before'].clip(lower=1)).fillna(0)
        df_base.drop(columns=['return_date', 'cum_returns', 'cum_orders_before'], errors='ignore', inplace=True)
        
        # Payment Method
        df_base = df_base.merge(
            self.pay_daily_orders_,
            on=['payment_method', 'order_date'],
            how='left'
        )
        df_base['cum_orders_before'] = df_base['cum_orders_before'].fillna(0)
        
        pay_returns_sorted = self.pay_returns_daily_sorted_.sort_values('return_date')
        df_base = pd.merge_asof(
            df_base,
            pay_returns_sorted,
            left_on='order_date',
            right_on='return_date',
            by='payment_method',
            direction='backward',
            allow_exact_matches=False
        )
        df_base['cum_returns'] = df_base['cum_returns'].fillna(0)
        df_base['payment_hist_return_rate'] = (df_base['cum_returns'] / df_base['cum_orders_before'].clip(lower=1)).fillna(0)
        df_base.drop(columns=['return_date', 'cum_returns', 'cum_orders_before'], errors='ignore', inplace=True)

        # ── 6. Other features & interactions ──────────────────────────────
        df_base = df_base.merge(order_prod_hist, on='order_id', how='left')
        for col in ['avg_prod_hist_return_rate', 'max_prod_hist_return_rate', 'sum_prod_hist_returns']:
            df_base[col] = df_base[col].fillna(0)

        df_base['is_cod'] = (df_base['payment_method'] == 'cod').astype(int)

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

        df_base['items_per_product'] = df_base['n_items'] / df_base['n_products'].clip(lower=1)
        df_base['discount_ratio'] = df_base['total_discount'] / (df_base['order_total_value'] + df_base['total_discount']).clip(lower=1)
        df_base['size_variety_ratio'] = df_base['n_sizes'] / df_base['n_products'].clip(lower=1)
        df_base['color_variety_ratio'] = df_base['n_colors'] / df_base['n_products'].clip(lower=1)
        df_base['avg_value_per_item'] = df_base['order_total_value'] / df_base['n_items'].clip(lower=1)
        df_base['customer_return_tendency'] = df_base['customer_return_rate'] * df_base['customer_order_number']

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
        
        return df_base[['order_id', 'order_date'] + numeric_features + categorical_features]


def build_pipeline(df_order_items, df_returns, df_products, df_customers, df_reviews, df_geography, scale_pos_weight, early_stopping_rounds=50):
    """
    Hàm xây dựng Full Sklearn Pipeline.
    
    Lưu ý: Khi sử dụng early_stopping, KHÔNG gọi pipeline.fit() trực tiếp.
    Thay vào đó, dùng các bước thủ công:
        fe = pipeline.named_steps['fe']
        preprocessor = pipeline.named_steps['preprocessor']
        classifier = pipeline.named_steps['classifier']
        
        X_train_fe = fe.fit_transform(df_train)
        X_train_pp = preprocessor.fit_transform(X_train_fe)
        
        X_val_fe = fe.transform(df_val)
        X_val_pp = preprocessor.transform(X_val_fe)
        
        classifier.fit(X_train_pp, y_train,
                        eval_set=[(X_train_pp, y_train), (X_val_pp, y_val)],
                        verbose=100)
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
    
    # 3. Model XGBoost (Tuned params + Early Stopping)
    xgb_model = xgb.XGBClassifier(
        n_estimators=1000, 
        learning_rate=0.0174, 
        max_depth=6,
        min_child_weight=74, 
        subsample=0.673, 
        colsample_bytree=0.608,
        scale_pos_weight=scale_pos_weight, 
        eval_metric='auc',
        early_stopping_rounds=early_stopping_rounds,
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
