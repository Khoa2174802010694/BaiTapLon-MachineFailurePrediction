
import os
import json
import joblib
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)


# ==========================================
# 1. CONFIGURATION
# ==========================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_PATH = os.path.join(BASE_DIR, "data", "ai4i2020.csv")
MODEL_DIR = os.path.join(BASE_DIR, "models")
RESULT_DIR = os.path.join(BASE_DIR, "results")

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)

TARGET = "Machine failure"

RANDOM_STATE = 42


# ==========================================
# 2. LOAD DATA
# ==========================================

def load_data():
    print("\n[1] Loading dataset...")

    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(
            f"Dataset not found: {DATA_PATH}"
        )

    df = pd.read_csv(DATA_PATH)

    print(f"Dataset shape: {df.shape}")
    print(f"Missing values: {df.isnull().sum().sum()}")
    print(f"Duplicate rows: {df.duplicated().sum()}")

    return df


# ==========================================
# 3. PREPROCESSING
# ==========================================

def prepare_data(df):

    print("\n[2] Preparing data...")

    # Remove identifiers and failure-type labels
    columns_to_drop = [
        "UDI",
        "Product ID",
        "TWF",
        "HDF",
        "PWF",
        "OSF",
        "RNF"
    ]

    X = df.drop(
        columns=columns_to_drop + [TARGET],
        errors="ignore"
    )

    y = df[TARGET]

    # Identify categorical and numerical columns
    categorical_features = X.select_dtypes(
        include=["object", "category"]
    ).columns.tolist()

    numerical_features = X.select_dtypes(
        include=["number"]
    ).columns.tolist()

    # Preprocessing pipelines
    numerical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer([
        ("num", numerical_pipeline, numerical_features),
        ("cat", categorical_pipeline, categorical_features)
    ])

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=y
    )

    print(f"Training samples: {len(X_train)}")
    print(f"Testing samples: {len(X_test)}")
    print(f"Input features: {X.shape[1]}")

    return (
        X_train,
        X_test,
        y_train,
        y_test,
        preprocessor
    )


# ==========================================
# 4. DEFINE MODELS
# ==========================================

def get_models():

    return {
        "Logistic Regression": LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            random_state=RANDOM_STATE
        ),

        "KNN": KNeighborsClassifier(
            n_neighbors=5,
            weights="distance"
        ),

        "Decision Tree": DecisionTreeClassifier(
            max_depth=8,
            class_weight="balanced",
            random_state=RANDOM_STATE
        ),

        "Random Forest": RandomForestClassifier(
            n_estimators=200,
            max_depth=12,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1
        ),

        "SVM": SVC(
            kernel="rbf",
            C=1.0,
            probability=True,
            class_weight="balanced",
            random_state=RANDOM_STATE
        )
    }


# ==========================================
# 5. TRAIN AND EVALUATE
# ==========================================

def train_models(
    X_train,
    X_test,
    y_train,
    y_test,
    preprocessor
):

    print("\n[3] Training models...")

    results = []
    trained_models = {}

    models = get_models()

    for name, model in models.items():

        print(f"\nTraining: {name}")

        pipeline = Pipeline([
            ("preprocessor", preprocessor),
            ("classifier", model)
        ])

        pipeline.fit(X_train, y_train)

        y_pred = pipeline.predict(X_test)
        y_prob = pipeline.predict_proba(X_test)[:, 1]

        metrics = {
            "Model": name,
            "Accuracy": accuracy_score(y_test, y_pred),
            "Precision": precision_score(
                y_test, y_pred, zero_division=0
            ),
            "Recall": recall_score(
                y_test, y_pred, zero_division=0
            ),
            "F1-score": f1_score(
                y_test, y_pred, zero_division=0
            ),
            "ROC-AUC": roc_auc_score(y_test, y_prob)
        }

        results.append(metrics)
        trained_models[name] = pipeline

        print(
            f"Accuracy:  {metrics['Accuracy']:.4f}\n"
            f"Precision: {metrics['Precision']:.4f}\n"
            f"Recall:    {metrics['Recall']:.4f}\n"
            f"F1-score:  {metrics['F1-score']:.4f}\n"
            f"ROC-AUC:   {metrics['ROC-AUC']:.4f}"
        )

    return pd.DataFrame(results), trained_models


# ==========================================
# 6. SAVE RESULTS AND BEST MODEL
# ==========================================

def save_results(
    results,
    trained_models,
    X_test,
    y_test
):

    print("\n[4] Saving results...")

    results = results.sort_values(
        by="F1-score",
        ascending=False
    )

    results.to_csv(
        os.path.join(RESULT_DIR, "model_comparison.csv"),
        index=False
    )

    # Select best model based on F1-score
    best_name = results.iloc[0]["Model"]
    best_model = trained_models[best_name]

    joblib.dump(
        best_model,
        os.path.join(MODEL_DIR, "best_model.pkl")
    )

    # Save model metadata
    metadata = {
        "model_name": best_name,
        "target": TARGET,
        "input_features": X_test.columns.tolist(),
        "metrics": results.iloc[0].to_dict()
    }

    with open(
        os.path.join(MODEL_DIR, "metadata.json"),
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(metadata, f, ensure_ascii=False, indent=4)

    # Save best model predictions
    y_pred = best_model.predict(X_test)

    report = classification_report(
        y_test,
        y_pred,
        zero_division=0
    )

    with open(
        os.path.join(RESULT_DIR, "classification_report.txt"),
        "w",
        encoding="utf-8"
    ) as f:
        f.write(report)

    cm = confusion_matrix(y_test, y_pred)

    pd.DataFrame(
        cm,
        index=["Actual Normal", "Actual Failure"],
        columns=["Predicted Normal", "Predicted Failure"]
    ).to_csv(
        os.path.join(RESULT_DIR, "confusion_matrix.csv")
    )

    print(f"\nBest model: {best_name}")
    print("\nModel comparison:")
    print(results.to_string(index=False))

    print("\nSaved model and evaluation files.")


# ==========================================
# 7. MAIN
# ==========================================

def main():

    df = load_data()

    (
        X_train,
        X_test,
        y_train,
        y_test,
        preprocessor
    ) = prepare_data(df)

    results, trained_models = train_models(
        X_train,
        X_test,
        y_train,
        y_test,
        preprocessor
    )

    save_results(
        results,
        trained_models,
        X_test,
        y_test
    )

    print("\n================================")
    print("MODEL TRAINING COMPLETED!")
    print("================================")


if __name__ == "__main__":
    main()
