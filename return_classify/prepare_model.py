import os
import shutil
import mlflow
import mlflow.artifacts

def main():
    print("🔍 Đang cấu hình MLflow tracking URI...")
    # Cấu hình tracking URI trỏ vào SQLite local
    db_path = os.path.abspath("mlflow.db").replace("\\", "/")
    mlflow.set_tracking_uri(f"sqlite:///{db_path}")
    
    experiment = mlflow.get_experiment_by_name("Return_Classification")
    if experiment is None:
        raise RuntimeError("Không tìm thấy experiment 'Return_Classification'")
        
    runs = mlflow.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["attributes.start_time DESC"],
        max_results=1,
    )
    if len(runs) == 0:
        raise RuntimeError("Không tìm thấy run nào!")
        
    run_id = runs.iloc[0]["run_id"]
    print(f"✅ Tìm thấy run ID mới nhất: {run_id}")
    
    model_uri = f"runs:/{run_id}/model"
    dst_dir = os.path.abspath("model_artifact")
    if os.path.exists(dst_dir):
        shutil.rmtree(dst_dir)
    os.makedirs(dst_dir, exist_ok=True)
    
    print(f"📦 Đang tải và sao chép model từ {model_uri} sang {dst_dir}...")
    # Sử dụng API chính thức của MLflow để tải artifact về local
    mlflow.artifacts.download_artifacts(artifact_uri=model_uri, dst_path=dst_dir)
    
    # MLflow tải về sẽ tạo thư mục con 'model' bên trong dst_dir (tức là model_artifact/model)
    # Ta di chuyển các file ra ngoài thư mục cha model_artifact để Dockerfile copy trực tiếp
    model_inside = os.path.join(dst_dir, "model")
    if os.path.exists(model_inside):
        for item in os.listdir(model_inside):
            shutil.move(os.path.join(model_inside, item), os.path.join(dst_dir, item))
        os.rmdir(model_inside)
        
    print("🎉 Sao chép mô hình thành công!")

if __name__ == "__main__":
    main()
