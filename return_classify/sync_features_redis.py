import os
import mlflow
import mlflow.sklearn
import redis

# Cấu hình Redis URL
REDIS_URL = "redis://default:UxHxBHpIF4iA2wffTzU20h50bbaiFrAv@suit-jeans-desire-88328.db.redis.io:12526"

def main():
    print("🔄 Đang cấu hình MLflow tracking URI về database SQLite local...")
    db_path = os.path.abspath("mlflow.db").replace("\\", "/")
    mlflow.set_tracking_uri(f"sqlite:///{db_path}")
    
    experiment = mlflow.get_experiment_by_name("Return_Classification")
    if experiment is None:
        raise RuntimeError("MLflow experiment 'Return_Classification' không tìm thấy!")
        
    runs = mlflow.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["attributes.start_time DESC"],
        max_results=1,
    )
    if len(runs) == 0:
        raise RuntimeError("Không tìm thấy lượt chạy (run) nào trong MLflow!")
        
    run_id = runs.iloc[0]["run_id"]
    model_uri = f"runs:/{run_id}/model"
    print(f"📦 Đang tải mô hình từ run ID: {run_id}...")
    
    pipeline = mlflow.sklearn.load_model(model_uri)
    fe_step = pipeline.named_steps['fe']
    
    print("🔌 Kiểm tra kết nối Redis Cloud...")
    try:
        r = redis.from_url(REDIS_URL, decode_responses=True)
        if r.ping():
            print("✅ Kết nối tới Redis Cloud thành công!")
    except Exception as e:
        print(f"❌ Kết nối tới Redis Cloud thất bại: {e}")
        return

    # Thực hiện đẩy dữ liệu
    print("🚀 Bắt đầu đồng bộ đặc trưng...")
    fe_step.upload_to_redis(REDIS_URL)
    print("🎉 Quá trình đồng bộ hoàn tất!")

if __name__ == "__main__":
    main()
