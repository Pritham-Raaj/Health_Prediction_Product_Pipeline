"""
Train heart disease prediction model with MLflow experiment tracking
Run this script to train the model and save artifacts for the ML service
"""
import os
import sys
import json
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv
import snowflake.connector

# ML libraries
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix
)

# MLflow
import mlflow
import mlflow.sklearn
from mlflow.models.signature import infer_signature

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Load environment variables
env_path = Path(__file__).parent.parent / '.env'
load_dotenv(env_path)


class ModelTrainer:
    """Train and track ML models with MLflow"""

    def __init__(self, experiment_name="heart_disease_prediction"):
        self.experiment_name = experiment_name
        self.mlflow_tracking_uri = Path(__file__).parent.parent / "mlflow_tracking"
        self.mlflow_tracking_uri.mkdir(exist_ok=True)

        # Set MLflow tracking URI (use forward slashes for Windows compatibility)
        tracking_uri = str(self.mlflow_tracking_uri.absolute()).replace('\\', '/')
        mlflow.set_tracking_uri(f"file:///{tracking_uri}")
        mlflow.set_experiment(experiment_name)

        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.scaler = None
        self.feature_names = None

    def load_data_from_snowflake(self):
        """Load training data from Snowflake"""
        print("Connecting to Snowflake...")

        conn = snowflake.connector.connect(
            account=os.environ["SNOWFLAKE_ACCOUNT"],
            user=os.environ["SNOWFLAKE_USER"],
            password=os.environ["SNOWFLAKE_PASSWORD"],
            warehouse=os.environ["SNOWFLAKE_WAREHOUSE"],
            database=os.environ["SNOWFLAKE_DATABASE"],
            schema=os.environ["SNOWFLAKE_ANALYTICS_SCHEMA"],
            role=os.environ["SNOWFLAKE_ROLE"],
        )

        print("Loading features from Snowflake...")
        query = f"SELECT * FROM {os.environ['SNOWFLAKE_ANALYTICS_SCHEMA']}.heart_features"
        heart_features = pd.read_sql(query, conn)
        conn.close()

        print(f"Loaded {len(heart_features)} records")
        return heart_features

    def prepare_data(self, df):
        """Prepare data for training"""
        print("Preparing data...")

        # Split features and target
        X = df.drop(columns=["RESULT"])
        y = df["RESULT"]

        self.feature_names = X.columns.tolist()
        print(f"Features: {self.feature_names}")

        # Scale features
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)

        # Train/test split
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X_scaled, y, test_size=0.2, random_state=42, stratify=y
        )

        print(f"Training set: {len(self.X_train)} samples")
        print(f"Test set: {len(self.X_test)} samples")

    def train_baseline_models(self):
        """Train baseline models and compare"""
        print("\n" + "=" * 50)
        print("Training baseline models...")
        print("=" * 50)

        models = {
            'Logistic Regression': LogisticRegression(random_state=42),
            'Random Forest': RandomForestClassifier(random_state=42),
            'KNN': KNeighborsClassifier()
        }

        results = {}

        for name, model in models.items():
            with mlflow.start_run(run_name=f"{name}_baseline"):
                print(f"\nTraining {name}...")

                # Train model
                model.fit(self.X_train, self.y_train)

                # Make predictions
                y_pred = model.predict(self.X_test)
                y_pred_proba = model.predict_proba(self.X_test)[:, 1] if hasattr(model, 'predict_proba') else None

                # Calculate metrics
                metrics = self.calculate_metrics(y_pred, y_pred_proba)

                # Log parameters
                mlflow.log_params({
                    "model_type": name,
                    "random_state": 42 if hasattr(model, 'random_state') else None
                })

                # Log metrics
                mlflow.log_metrics(metrics)

                # Log model
                signature = infer_signature(self.X_train, model.predict(self.X_train))
                mlflow.sklearn.log_model(model, "model", signature=signature)

                results[name] = {
                    'model': model,
                    'metrics': metrics
                }

                print(f"{name} - Accuracy: {metrics['accuracy']:.4f}")

        return results

    def tune_best_model(self, best_model_name, model):
        """Hyperparameter tuning for the best model"""
        print(f"\n" + "=" * 50)
        print(f"Tuning {best_model_name}...")
        print("=" * 50)

        # Define parameter grids
        param_grids = {
            'Logistic Regression': {
                'C': np.logspace(-1, 1, 10),
                'solver': ['liblinear']
            },
            'Random Forest': {
                'n_estimators': [100, 200, 300],
                'max_depth': [None, 5, 10, 15],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            },
            'KNN': {
                'n_neighbors': range(1, 21)
            }
        }

        param_grid = param_grids.get(best_model_name, {})

        with mlflow.start_run(run_name=f"{best_model_name}_tuned"):
            # Perform randomized search
            random_search = RandomizedSearchCV(
                estimator=model,
                param_distributions=param_grid,
                n_iter=50,
                cv=5,
                verbose=1,
                random_state=42,
                n_jobs=-1
            )

            random_search.fit(self.X_train, self.y_train)

            # Get best model
            best_model = random_search.best_estimator_

            # Make predictions
            y_pred = best_model.predict(self.X_test)
            y_pred_proba = best_model.predict_proba(self.X_test)[:, 1]

            # Calculate metrics
            metrics = self.calculate_metrics(y_pred, y_pred_proba)

            # Log parameters
            mlflow.log_params({
                "model_type": best_model_name,
                **random_search.best_params_
            })

            # Log metrics
            mlflow.log_metrics(metrics)
            mlflow.log_metric("cv_best_score", random_search.best_score_)

            # Log classification report
            report = classification_report(self.y_test, y_pred)
            mlflow.log_text(report, "classification_report.txt")

            # Log confusion matrix
            cm = confusion_matrix(self.y_test, y_pred)
            mlflow.log_text(str(cm), "confusion_matrix.txt")

            # Log model
            signature = infer_signature(self.X_train, best_model.predict(self.X_train))
            mlflow.sklearn.log_model(
                best_model,
                "model",
                signature=signature,
                registered_model_name="heart_disease_predictor"
            )

            print(f"\nBest parameters: {random_search.best_params_}")
            print(f"Tuned {best_model_name} - Accuracy: {metrics['accuracy']:.4f}")

            return best_model, metrics

    def calculate_metrics(self, y_pred, y_pred_proba=None):
        """Calculate evaluation metrics"""
        metrics = {
            'accuracy': accuracy_score(self.y_test, y_pred),
            'precision': precision_score(self.y_test, y_pred, average='binary'),
            'recall': recall_score(self.y_test, y_pred, average='binary'),
            'f1_score': f1_score(self.y_test, y_pred, average='binary')
        }

        if y_pred_proba is not None:
            metrics['roc_auc'] = roc_auc_score(self.y_test, y_pred_proba)

        return metrics

    def save_model_artifacts(self, model, metrics):
        """Save model artifacts for ML service"""
        print("\n" + "=" * 50)
        print("Saving model artifacts...")
        print("=" * 50)

        models_dir = Path(__file__).parent.parent / "ml_service" / "models"
        models_dir.mkdir(exist_ok=True, parents=True)

        # Save model
        import joblib
        model_path = models_dir / "heart_disease_model.pkl"
        joblib.dump(model, model_path)
        print(f"Model saved to: {model_path}")

        # Save scaler
        scaler_path = models_dir / "scaler.pkl"
        joblib.dump(self.scaler, scaler_path)
        print(f"Scaler saved to: {scaler_path}")

        # Save feature names
        feature_names_path = models_dir / "feature_names.json"
        with open(feature_names_path, 'w') as f:
            json.dump(self.feature_names, f)
        print(f"Feature names saved to: {feature_names_path}")

        # Save metadata
        metadata = {
            "model_version": "1.0.0",
            "training_date": datetime.now().isoformat(),
            "metrics": metrics,
            "feature_names": self.feature_names,
            "training_samples": len(self.X_train),
            "test_samples": len(self.X_test)
        }

        metadata_path = models_dir / "model_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        print(f"Metadata saved to: {metadata_path}")

    def save_reference_data(self):
        """Save reference data for drift detection"""
        print("\n" + "=" * 50)
        print("Saving reference data for drift detection...")
        print("=" * 50)

        # Use test set as reference data
        reference_df = pd.DataFrame(
            self.X_test,
            columns=self.feature_names
        )

        monitoring_dir = Path(__file__).parent.parent / "ml_service" / "monitoring"
        monitoring_dir.mkdir(exist_ok=True, parents=True)

        reference_path = monitoring_dir / "reference_data.csv"
        reference_df.to_csv(reference_path, index=False)
        print(f"Reference data saved to: {reference_path}")

    def run(self):
        """Run complete training pipeline"""
        print("=" * 50)
        print("Heart Disease Prediction Model Training")
        print("=" * 50)

        # Load data
        df = self.load_data_from_snowflake()

        # Prepare data
        self.prepare_data(df)

        # Train baseline models
        results = self.train_baseline_models()

        # Find best model
        best_model_name = max(results, key=lambda k: results[k]['metrics']['accuracy'])
        best_model = results[best_model_name]['model']

        print(f"\nBest baseline model: {best_model_name}")

        # Tune best model
        tuned_model, metrics = self.tune_best_model(best_model_name, best_model)

        # Save artifacts
        self.save_model_artifacts(tuned_model, metrics)

        # Save reference data
        self.save_reference_data()

        print("\n" + "=" * 50)
        print("Training completed successfully!")
        print("=" * 50)
        print(f"\nFinal Model: {best_model_name}")
        print(f"Accuracy: {metrics['accuracy']:.4f}")
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall: {metrics['recall']:.4f}")
        print(f"F1 Score: {metrics['f1_score']:.4f}")
        print(f"ROC-AUC: {metrics.get('roc_auc', 'N/A')}")
        print(f"\nMLflow tracking URI: {mlflow.get_tracking_uri()}")
        print(f"View experiments: mlflow ui --backend-store-uri {self.mlflow_tracking_uri}")


if __name__ == "__main__":
    trainer = ModelTrainer()
    trainer.run()
