**1. Hiểu Dữ Liệu Trước**
Trước khi phân tích sâu, hãy tạo một bảng tổng quan cho từng file:
```text
Tên bảng | Số dòng | Khóa chính | Mức dữ liệu | Cột thời gian | Dùng để trả lời câu hỏi gì?
```
Ví dụ:
```text
orders        -> mức đơn hàng
order_items   -> mức sản phẩm trong đơn
customers     -> mức khách hàng
products      -> mức sản phẩm
sales         -> mức ngày
inventory     -> mức sản phẩm-tháng
web_traffic   -> mức ngày/kênh traffic
```
Việc này rất quan trọng vì nếu bạn join sai mức dữ liệu, kết quả EDA sẽ sai.
**2. Chọn Một “Câu Chuyện” Chính**
Bạn không nên phân tích tất cả mọi thứ. Dataset này có nhiều hướng, nhưng report chỉ có giới hạn. Hãy chọn 1-2 hướng chính.
Một số hướng EDA hợp lý:
**Hướng A: Revenue & Profit**
Câu hỏi:
```text
Doanh thu tăng/giảm như thế nào theo thời gian?
Lợi nhuận gộp có tăng cùng doanh thu không?
Có giai đoạn nào revenue cao nhưng margin thấp không?
Yếu tố nào kéo revenue lên: số đơn, số lượng sản phẩm, giá, discount, hay product mix?
```
Chỉ số nên tính:
```text
Revenue
COGS
Gross profit = Revenue - COGS
Gross margin = Gross profit / Revenue
Number of orders
Average order value
Discount rate
```
Biểu đồ phù hợp:
```text
Line chart theo thời gian
Monthly revenue/profit trend
Revenue vs gross margin
Seasonality by month/day-of-week
```
**Hướng B: Customer Behavior**
Câu hỏi:
```text
Khách hàng nào có giá trị cao?
Khách hàng quay lại mua nhiều không?
Nhóm tuổi/giới tính/kênh acquisition nào có AOV hoặc repeat rate cao hơn?
Có dấu hiệu nhóm khách hàng nào đang kém hiệu quả không?
```
Chỉ số:
```text
Total spend per customer
Number of orders per customer
Average order value
Days between orders
Repeat purchase rate
```
Biểu đồ:
```text
Histogram số đơn mỗi khách
Boxplot AOV theo age_group/gender/channel
Cohort hoặc retention đơn giản
Top customer segments
```
**Hướng C: Product & Return**
Câu hỏi:
```text
Sản phẩm/category nào tạo nhiều revenue nhất?
Sản phẩm nào margin tốt nhất?
Sản phẩm nào bán chạy nhưng bị trả nhiều?
Lý do trả hàng có khác nhau theo category/size không?
```
Chỉ số:
```text
Revenue by category
Gross margin by category
Return rate = returned quantity / sold quantity
Refund amount
Average rating
```
Biểu đồ:
```text
Bar chart category revenue/profit
Scatter plot revenue vs return rate
Return reason by category
Rating vs return rate
```
**Hướng D: Promotion & Discount**
Câu hỏi:
```text
Khuyến mãi có thật sự làm tăng doanh thu không?
Khuyến mãi làm tăng profit hay chỉ tăng volume?
Promo nào đi kèm return rate cao?
Discount sâu có làm margin giảm mạnh không?
```
Chỉ số:
```text
Promo vs non-promo revenue
Discount amount
Discount rate
Gross margin
Return rate
```
Biểu đồ:
```text
Promo vs non-promo comparison
Discount rate vs gross margin
Revenue before/during/after promotion
```
**Hướng E: Inventory & Operations**
Câu hỏi:
```text
Có sản phẩm/category nào hay stockout không?
Stockout có liên quan tới mất doanh thu không?
Có overstock ở nhóm sản phẩm bán chậm không?
Tồn kho có đang hỗ trợ tốt cho demand không?
```
Chỉ số:
```text
Stockout days
Fill rate
Sell-through rate
Days of supply
Overstock flag
Units sold
```
Biểu đồ:
```text
Stockout trend by month
Sell-through by category
Inventory efficiency matrix
```

**Hướng F: Sales Forecasting (Dự báo Doanh thu)**
Đây là hướng EDA chuyên sâu phục vụ trực tiếp cho mô hình Time-Series.
Câu hỏi:
```text
Xu hướng (trend) dài hạn của doanh thu là gì?
Tính mùa vụ (seasonality) thể hiện ra sao (theo tháng, ngày trong tuần)?
Các sự kiện/ngày lễ và chương trình khuyến mãi tác động bao nhiêu % đến demand?
Mức độ biến động doanh thu giữa các khu vực (Geography) và ngành hàng (Category) như thế nào?
Tình trạng hết hàng (stockout) có làm doanh thu ghi nhận bị thấp hơn nhu cầu thực tế (censored demand) không?
```
Chỉ số & Feature cần tạo/theo dõi:
```text
Sales/Quantity Sold (Target variable)
Lag features (Sales của n ngày/tuần trước)
Rolling window features (Trung bình, độ lệch chuẩn sales 7 ngày/30 ngày qua)
Time features (DayOfWeek, Month, Quarter, IsHoliday, IsWeekend)
Promo features (HasPromo, Discount_Rate)
Hierarchy features (LocationID, CategoryID)
Stockout flag
```
Biểu đồ:
```text
Time-series line plot (kết hợp đánh dấu ngày lễ/promo)
Time-series Decomposition (tách Trend, Seasonality, Residual)
ACF/PACF plots (Phân tích tự tương quan)
Heatmap doanh thu theo DayOfWeek x Month
```