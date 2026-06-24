import mlflow
import mlflow.sklearn
import pandas as pd
import os

print("Searching for the latest MLflow run...")
experiment = mlflow.get_experiment_by_name("Return_Classification")
runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id], order_by=["attributes.start_time DESC"], max_results=1)

if len(runs) == 0:
    raise ValueError("No runs found in MLflow!")

run_id = runs.iloc[0]['run_id']
model_uri = f"runs:/{run_id}/model"
print(f"Loading model from URI: {model_uri}...")

# Load the model
model = mlflow.sklearn.load_model(model_uri)
print("Model loaded successfully!")
print(f"Model type: {type(model)}")

# Load a few test rows to run prediction
print("Loading a few rows of test data...")
DATA_DIR = '../datathon-2026-round-1'
df_orders = pd.read_csv(os.path.join(DATA_DIR, 'orders.csv'), parse_dates=['order_date']).head(5)

# Run predict/predict_proba
# Note: the loaded pipeline includes ReturnFeatureExtractor, so we pass df_orders directly!
print("Running prediction directly on raw orders...")
predictions = model.predict(df_orders)
probabilities = model.predict_proba(df_orders)[:, 1]

print("\nPredictions:")
for i, (pred, prob) in enumerate(zip(predictions, probabilities)):
    print(f"Row {i}: Order ID = {df_orders.iloc[i]['order_id']} | Predicted Class = {pred} | Probability = {prob:.4f}")

print("\n✅ Verification successful!")
