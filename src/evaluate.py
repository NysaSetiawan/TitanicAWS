import mlflow
from sklearn.metrics import accuracy_score, precision_score, recall_score, classification_report

mlflow.set_tracking_uri("sqlite:///mlflow.db")

def evaluate_pipeline(pipeline, X_train, y_train, X_test, y_test) -> dict:
    train_preds = pipeline.predict(X_train)
    test_preds = pipeline.predict(X_test)

    train_acc = accuracy_score(y_train, train_preds)
    test_acc = accuracy_score(y_test, test_preds)
    
    test_prec = precision_score(y_test, test_preds, average="binary", zero_division=0)
    test_rec = recall_score(y_test, test_preds, average="binary", zero_division=0)

    model_name = pipeline.steps[-1][0]

    with mlflow.start_run(run_name=f"Eval_{model_name}", nested=True):
        mlflow.log_param("model_type", model_name)
        mlflow.log_metric("train_accuracy", train_acc)
        mlflow.log_metric("test_accuracy", test_acc)
        mlflow.log_metric("test_precision", test_prec)
        mlflow.log_metric("test_recall", test_rec)

    return {
        "train_accuracy": train_acc,
        "test_accuracy": test_acc
    }

def print_comparison(results: dict) -> None:
    """Menampilkan tabel perbandingan hasil evaluasi antar model."""
    print("\nModel Comparison")
    print(f"{'Model':<20} | {'Train Accuracy':<15} | {'Test Accuracy':<15}")
    print("-" * 56)
    for name, metrics in results.items():
        print(f"{name:<20} | {metrics['train_accuracy']:<15.4f} | {metrics['test_accuracy']:<15.4f}")


def select_best(results: dict, metric: str = "test_accuracy") -> str:
    return max(results, key=lambda k: results[k][metric])


def print_classification_report(pipeline, X_test, y_test, class_names: list) -> None:
    preds = pipeline.predict(X_test)
    print(classification_report(y_test, preds, target_names=class_names))