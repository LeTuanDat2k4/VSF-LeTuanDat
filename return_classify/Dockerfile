# Sử dụng Python image chính thức nhẹ
FROM python:3.12-slim

# Thiết lập thư mục làm việc trong container
WORKDIR /app

# Sao chép file requirements.txt trước để tận dụng cache của Docker
COPY requirements.txt .

# Cài đặt các thư viện phụ thuộc
RUN pip install --no-cache-dir -r requirements.txt

# Sao chép mã nguồn ứng dụng và cấu trúc pipeline
COPY app.py .
COPY return_pipeline.py .

# Tạo thư mục cho model và copy model từ mlruns sang
COPY model_artifact/ ./model_artifact/

# Thiết lập các biến môi trường
ENV PYTHONUNBUFFERED=1
ENV MODEL_PATH=/app/model_artifact
# URL Redis mặc định, có thể ghi đè lúc chạy container
ENV REDIS_URL=redis://default:UxHxBHpIF4iA2wffTzU20h50bbaiFrAv@suit-jeans-desire-88328.db.redis.io:12526

# Mở cổng 8000 cho FastAPI
EXPOSE 8000

# Lệnh khởi chạy ứng dụng
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
