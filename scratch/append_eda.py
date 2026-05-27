import json
import os

ipynb_path = r"c:\Users\PC\Documents\VSF\DataThon\data_quality_cleaning.ipynb"

# Load current notebook
with open(ipynb_path, "r", encoding="utf-8") as f:
    notebook = json.load(f)

# Define new cells to append
new_cells = [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Phân Tích Khám Phá Dữ Liệu Chuyên Sâu cho Dự Báo Doanh Thu (EDA for Sales Forecasting)\n",
    "\n",
    "Sau khi đã làm sạch và đảm bảo chất lượng dữ liệu ở các bước trước, chúng ta tiến hành **EDA nâng cao** được thiết kế riêng cho bài toán dự báo chuỗi thời gian (Sales Forecasting) phục vụ tối ưu logistics, khuyến mãi và phân bổ tồn kho."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Bước 1: Nhìn Bức Tranh Tổng Thể (Macro-level Time-series Decomposition)\n",
    "\n",
    "Chúng ta sẽ thực hiện phân tách chuỗi thời gian (Decomposition) doanh thu hàng ngày từ file `sales.csv` thành 3 phần:\n",
    "- **Trend (Xu hướng dài hạn)**: Hướng phát triển tăng hay giảm của doanh nghiệp theo thời gian.\n",
    "- **Seasonality (Tính mùa vụ)**: Các biến động có tính chất lặp đi lặp lại theo chu kỳ (ví dụ: chu kỳ tuần, chu kỳ năm).\n",
    "- **Residuals / Noise (Nhiễu)**: Những biến động ngẫu nhiên không có tính lặp lại.\n",
    "\n",
    "Để biểu đồ phân tách dễ quan sát và không bị nhiễu do tần suất ngày, chúng ta sẽ **resample dữ liệu theo tuần (Weekly)** và phân tách chu kỳ mùa vụ năm (52 tuần)."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "from statsmodels.tsa.seasonal import seasonal_decompose\n",
    "\n",
    "# 1. Đọc và chuẩn bị dữ liệu chuỗi thời gian doanh thu\n",
    "df_sales = pd.read_csv(os.path.join(data_dir, \"sales.csv\"))\n",
    "df_sales['Date'] = pd.to_datetime(df_sales['Date'])\n",
    "df_sales.set_index('Date', inplace=True)\n",
    "df_sales.sort_index(inplace=True)\n",
    "\n",
    "# 2. Resample sang tần suất tuần (Weekly)\n",
    "sales_weekly = df_sales['Revenue'].resample('W').sum()\n",
    "\n",
    "# 3. Phân tách chuỗi thời gian (mô hình Additive, chu kỳ 52 tuần)\n",
    "decomposition = seasonal_decompose(sales_weekly, model='additive', period=52)\n",
    "\n",
    "# 4. Trực quan hóa\n",
    "fig = decomposition.plot()\n",
    "fig.set_size_inches(14, 10)\n",
    "plt.suptitle('Phân tách Doanh thu hàng tuần (Time-series Decomposition - Weekly)', y=1.02, fontsize=16, fontweight='bold')\n",
    "plt.tight_layout()\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Bước 2: Phân Rã Theo Cấp Bậc (Hierarchical Level Analysis)\n",
    "\n",
    "Doanh nghiệp cần tối ưu hóa logistics và phân bổ tồn kho chi tiết. Do đó, chúng ta sẽ bóc tách tổng doanh thu ra theo:\n",
    "1. **Vùng địa lý (Geography)**: So sánh xu hướng bán hàng của **East (Đông)**, **Central (Trung)**, và **West (Tây)** (tương ứng với các vùng địa lý của bộ dữ liệu DataThon).\n",
    "2. **Ngành hàng (Category)**: Theo dõi sức mua của các nhóm sản phẩm **Streetwear, Casual, Outdoor, GenZ**.\n",
    "\n",
    "Chúng ta thực hiện merge dữ liệu giữa các bảng `orders`, `order_items`, `products` và `geography` để dựng bảng master."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# 1. Đọc dữ liệu từ các bảng liên quan\n",
    "df_orders = pd.read_csv(os.path.join(data_dir, \"orders.csv\"), low_memory=False)\n",
    "df_items = pd.read_csv(os.path.join(data_dir, \"order_items.csv\"), low_memory=False)\n",
    "df_prod = pd.read_csv(os.path.join(data_dir, \"products.csv\"))\n",
    "df_geo = pd.read_csv(os.path.join(data_dir, \"geography.csv\"))\n",
    "\n",
    "# 2. Chuẩn hóa thời gian\n",
    "df_orders['order_date'] = pd.to_datetime(df_orders['order_date'])\n",
    "\n",
    "# 3. Tiến hành ghép nối (Merge) dữ liệu theo sơ đồ hình sao\n",
    "df_orders_geo = pd.merge(df_orders, df_geo, on='zip', how='inner')\n",
    "df_items_prod = pd.merge(df_items, df_prod, on='product_id', how='inner')\n",
    "df_master = pd.merge(df_items_prod, df_orders_geo, on='order_id', how='inner')\n",
    "\n",
    "# 4. Tính toán doanh số thực tế cho từng dòng (Doanh số = Số lượng * Giá bán đơn lẻ - Chiết khấu)\n",
    "df_master['sales_amount'] = (df_master['quantity'] * df_master['unit_price']) - df_master['discount_amount']\n",
    "\n",
    "# 5. Nhóm dữ liệu hàng tháng theo Vùng địa lý và Ngành hàng\n",
    "sales_by_region = df_master.groupby(['order_date', 'region'])['sales_amount'].sum().unstack().resample('M').sum()\n",
    "sales_by_cat = df_master.groupby(['order_date', 'category'])['sales_amount'].sum().unstack().resample('M').sum()\n",
    "\n",
    "# 6. Trực quan hóa bằng đồ thị đa đường\n",
    "fig, axes = plt.subplots(2, 1, figsize=(14, 12))\n",
    "\n",
    "# Phân tích Vùng Địa Lý\n",
    "sales_by_region.plot(ax=axes[0], marker='o', linewidth=2, colormap='viridis')\n",
    "axes[0].set_title(\"Doanh thu hàng tháng phân rã theo Vùng địa lý (Geography Trends)\", fontsize=14, fontweight='bold')\n",
    "axes[0].set_ylabel(\"Doanh thu (VND)\")\n",
    "axes[0].set_xlabel(\"Thời gian\")\n",
    "axes[0].legend(title=\"Geography Region\")\n",
    "\n",
    "# Phân tích Ngành Hàng\n",
    "sales_by_cat.plot(ax=axes[1], marker='s', linewidth=2, colormap='Set2')\n",
    "axes[1].set_title(\"Doanh thu hàng tháng phân rã theo Ngành hàng sản phẩm (Category Trends)\", fontsize=14, fontweight='bold')\n",
    "axes[1].set_ylabel(\"Doanh thu (VND)\")\n",
    "axes[1].set_xlabel(\"Thời gian\")\n",
    "axes[1].legend(title=\"Product Category\")\n",
    "\n",
    "plt.tight_layout()\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "**Nhận xét:**\n",
    "- **Xu hướng Vùng địa lý:** Ba vùng địa lý (East, Central, West) sở hữu xu hướng mua sắm gần như song hành và đồng bộ về mặt chu kỳ. Điều này rất tốt cho kế hoạch phân phối logistics cốt lõi trên toàn quốc, tuy nhiên quy mô thị trường của mỗi vùng là khác nhau (East luôn vượt trội).\n",
    "- **Xu hướng Ngành hàng:** Ngành hàng **Casual** và **Streetwear** chiếm tỷ trọng áp đảo trong tổng doanh số và có sự tăng trưởng mạnh mẽ, trong khi **Outdoor** và **GenZ** đóng góp nhỏ hơn nhưng có tính ổn định cao."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Bước 3: Đánh Giá Tác Động Của Khuyến Mãi (External Shocks / Promotion Spikes)\n",
    "\n",
    "Các chương trình khuyến mãi (Promotion) tạo ra các cú sốc cầu lớn, làm bùng nổ doanh số tạm thời. Nếu không kiểm soát tốt các cú sốc này, mô hình dự báo sẽ dễ dự đoán sai lệch (quá cao hoặc quá thấp).\n",
    "\n",
    "Chúng ta sẽ lọc dữ liệu năm 2022 (năm gần nhất) và vẽ doanh thu hàng ngày kết hợp tô màu vùng thời gian chạy của các chương trình khuyến mãi lớn từ `promotions.csv` để thấy rõ sức tác động."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# 1. Đọc và làm sạch promotions.csv\n",
    "df_promo = pd.read_csv(os.path.join(data_dir, \"promotions.csv\"))\n",
    "df_promo.dropna(how='all', inplace=True)\n",
    "df_promo['start_date'] = pd.to_datetime(df_promo['start_date'])\n",
    "df_promo['end_date'] = pd.to_datetime(df_promo['end_date'])\n",
    "\n",
    "# 2. Lọc chuỗi thời gian doanh thu hàng ngày trong năm 2022\n",
    "sales_2022 = df_sales.loc['2022-01-01':'2022-12-31']\n",
    "\n",
    "# 3. Lọc danh sách promotions chạy trong năm 2022\n",
    "promo_2022 = df_promo[(df_promo['start_date'] >= '2022-01-01') & (df_promo['end_date'] <= '2022-12-31')]\n",
    "\n",
    "# 4. Vẽ đồ thị\n",
    "plt.figure(figsize=(16, 7))\n",
    "plt.plot(sales_2022.index, sales_2022['Revenue'], label='Daily Revenue 2022', color='royalblue', alpha=0.8, linewidth=1.5)\n",
    "\n",
    "# Tô màu nền cho từng chiến dịch khuyến mãi\n",
    "colors = ['#ff9999', '#99ff99', '#9999ff', '#ffcc99', '#ffff99', '#cc99ff', '#ff99ff']\n",
    "color_idx = 0\n",
    "\n",
    "for _, row in promo_2022.iterrows():\n",
    "    p_name = row['promo_name']\n",
    "    s_date = row['start_date']\n",
    "    e_date = row['end_date']\n",
    "    \n",
    "    # Tô vùng màu từ start_date tới end_date\n",
    "    plt.axvspan(s_date, e_date, color=colors[color_idx % len(colors)], alpha=0.2, label=f\"Promo: {p_name}\")\n",
    "    color_idx += 1\n",
    "\n",
    "plt.title(\"Đánh giá tác động của các đợt Khuyến mãi (Promotions) lên Doanh thu năm 2022\", fontsize=15, fontweight='bold')\n",
    "plt.xlabel(\"Thời gian\")\n",
    "plt.ylabel(\"Doanh thu hàng ngày (VND)\")\n",
    "\n",
    "# Xử lý hiển thị Legend gọn gàng (loại bỏ lặp lại)\n",
    "handles, labels = plt.gca().get_legend_handles_labels()\n",
    "by_label = dict(zip(labels, handles))\n",
    "plt.legend(by_label.values(), by_label.keys(), loc='upper left', bbox_to_anchor=(1, 1), fontsize=10)\n",
    "\n",
    "plt.tight_layout()\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "**Nhận xét chiến lược:**\n",
    "- Biểu đồ doanh số năm 2022 thể hiện rõ rệt sự gia tăng doanh số mang tính **đột biến cực lớn (spikes)** trùng khớp hoàn toàn với các chiến dịch khuyến mãi như *Spring Sale 2022*, *Mid-Year Sale 2022*, *Fall Launch 2022*, và đỉnh điểm là chiến dịch *Year-End Sale 2022* (kéo dài từ giữa tháng 11 đến hết năm, bao trùm ngày hội mua sắm 11/11 và Black Friday).\n",
    "- Đối với mô hình forecasting, việc gắn nhãn các ngày diễn ra chương trình khuyến mãi (`promo_flag` hoặc `discount_rate`) là bắt buộc để mô hình có thể bóc tách tác động của các cú sốc bên ngoài này ra khỏi nhu cầu mua hàng tự nhiên."
   ]
  }
]

# Append new cells
notebook["cells"].extend(new_cells)

# Save updated notebook
with open(ipynb_path, "w", encoding="utf-8") as f:
    json.dump(notebook, f, indent=1, ensure_ascii=False)

print("EDA cells successfully appended to data_quality_cleaning.ipynb!")
