from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer

if __name__ == "__main__":
    # 1. Ingest data
    obj = DataIngestion()
    train_data, test_data = obj.initiate_data_ingestion()

    # 2. Transform data
    data_transformation = DataTransformation()
    X_train, X_test, y_train, y_test, _ = data_transformation.initiate_data_transformation(train_data, test_data)

    # 3. Train and evaluate models
    model_trainer = ModelTrainer()
    report, model_path = model_trainer.train_and_evaluate(X_train, X_test, y_train, y_test)

print("\n=== Model training report ===")


for model_name, metrics in report.items():
    print(f"{model_name:15s} R²: {metrics['r2']:.4f} | RMSE: {metrics['rmse']:.4f}")

# Step 5: Show best model metrics
best_model_name = max(report, key=lambda m: report[m]['r2'])
best_metrics = report[best_model_name]

print("\n===== BEST MODEL =====")
print(f"Model: {best_model_name}")
print(f"R² Score: {best_metrics['r2']:.4f}")
print(f"RMSE: {best_metrics['rmse']:.4f}")
print("======================\n")
