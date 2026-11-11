import os
import json
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
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
TEST_SIZE = 0.2
N_SPLITS = 5


# ==========================================
# 2. LOAD DATA
# ==========================================

def load_data():
    print("\n[1] Loading dataset...")

    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)

    if TARGET not in df.columns:
        raise ValueError(
            f"Target column '{TARGET}' was not found in the dataset."
        )

    print(f"Dataset shape: {df.shape}")
    print(f"Missing values: {df.isnull().sum().sum()}")
    print(f"Duplicate rows: {df.duplicated().sum()}")
    print("\nTarget distribution:")
    print(df[TARGET].value_counts().sort_index())
    print(f"Failure rate: {df[TARGET].mean() * 100:.2f}%")

    return df


# ==========================================
# 3. PREPROCESSING AND DATA SPLIT
# ==========================================

def prepare_data(df):
    print("\n[2] Preparing data...")

    # Remove identifiers and failure-mode labels.
    # These columns are not used as input features.
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

    categorical_features = X.select_dtypes(
        include=["object", "category"]
    ).columns.tolist()

    numerical_features = X.select_dtypes(
        include=["number"]
    ).columns.tolist()

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

    # Keep the test set untouched until final evaluation.
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y
    )

    print(f"Training samples: {len(X_train)}")
    print(f"Testing samples: {len(X_test)}")
    print(f"Input features: {X.shape[1]}")
    print(f"Training failure rate: {y_train.mean() * 100:.2f}%")
    print(f"Testing failure rate: {y_test.mean() * 100:.2f}%")

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
# 5. CROSS-VALIDATION, TRAINING AND EVALUATION
# ==========================================

def train_models(
    X_train,
    X_test,
    y_train,
    y_test,
    preprocessor
):
    print("\n[3] Cross-validation and model evaluation...")

    models = get_models()
    cv = StratifiedKFold(
        n_splits=N_SPLITS,
        shuffle=True,
        random_state=RANDOM_STATE
    )

    results = []
    trained_models = {}

    for name, model in models.items():
        print(f"\nProcessing: {name}")

        # Build a fresh pipeline for this model.
        pipeline = Pipeline([
            ("preprocessor", preprocessor),
            ("classifier", model)
        ])

        # Cross-validation is performed only on the training set.
        # The test set is not used to select the best model.
        cv_scores = cross_val_score(
            pipeline,
            X_train,
            y_train,
            cv=cv,
            scoring="f1",
            n_jobs=-1
        )

        cv_f1_mean = cv_scores.mean()
        cv_f1_std = cv_scores.std()

        print(
            f"Cross-validation F1: "
            f"{cv_f1_mean:.4f} ± {cv_f1_std:.4f}"
        )

        # Fit the model on the complete training set.
        pipeline.fit(X_train, y_train)

        # Evaluate on the untouched test set.
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
            "ROC-AUC": roc_auc_score(y_test, y_prob),
            "CV F1 mean": cv_f1_mean,
            "CV F1 std": cv_f1_std
        }

        results.append(metrics)
        trained_models[name] = pipeline

        print(
            f"Test Accuracy:  {metrics['Accuracy']:.4f}\n"
            f"Test Precision: {metrics['Precision']:.4f}\n"
            f"Test Recall:    {metrics['Recall']:.4f}\n"
            f"Test F1-score:  {metrics['F1-score']:.4f}\n"
            f"Test ROC-AUC:   {metrics['ROC-AUC']:.4f}"
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

    # Select the best model using mean cross-validation F1-score.
    # Test-set metrics are not used for model selection.
    results = results.sort_values(
        by="CV F1 mean",
        ascending=False
    ).reset_index(drop=True)

    results.to_csv(
        os.path.join(RESULT_DIR, "model_comparison.csv"),
        index=False
    )

    best_name = results.iloc[0]["Model"]
    best_model = trained_models[best_name]

    joblib.dump(
        best_model,
        os.path.join(MODEL_DIR, "best_model.pkl")
    )

    # Evaluate the selected model on the test set.
    y_pred = best_model.predict(X_test)

    best_row = results.iloc[0]

    metadata = {
        "model_name": best_name,
        "selection_criterion": "Mean F1-score from stratified 5-fold cross-validation",
        "cross_validation_folds": N_SPLITS,
        "target": TARGET,
        "input_features": X_test.columns.tolist(),
        "metrics": {
            key: (
                float(value)
                if isinstance(value, (int, float))
                else value
            )
            for key, value in best_row.to_dict().items()
        }
    }

    with open(
        os.path.join(MODEL_DIR, "metadata.json"),
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(metadata, f, ensure_ascii=False, indent=4)

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

    print(f"\nBest model selected by CV F1 mean: {best_name}")
    print(f"Mean CV F1-score: {best_row['CV F1 mean']:.4f}")
    print(f"CV F1 standard deviation: {best_row['CV F1 std']:.4f}")

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
