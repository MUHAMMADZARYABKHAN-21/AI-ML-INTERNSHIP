"""
Task 2: End-to-End ML Pipeline — Customer Churn Prediction
DevelopersHub Corporation – AI/ML Engineering Internship

Uses Scikit-learn Pipeline API with preprocessing, Logistic Regression,
Random Forest, GridSearchCV, and joblib export.
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib, os

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, StratifiedKFold
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score, classification_report,
    confusion_matrix, roc_curve, ConfusionMatrixDisplay,
)

# ─────────────────────────────────────────
#  1. Load Dataset
# ─────────────────────────────────────────
def load_telco_churn():
    """
    Load Telco Customer Churn dataset.
    Falls back to a synthetic equivalent if the file isn't present.
    Download from: https://www.kaggle.com/datasets/blastchar/telco-customer-churn
    """
    url = "https://raw.githubusercontent.com/dsrscientist/dataset1/master/Telecom-Churn.csv"
    local = "WA_Fn-UseC_-Telco-Customer-Churn.csv"

    if os.path.exists(local):
        df = pd.read_csv(local)
        print(f"Loaded local file: {local}  shape={df.shape}")
    else:
        try:
            df = pd.read_csv(url)
            print(f"Loaded from URL  shape={df.shape}")
        except Exception:
            print("Generating synthetic Telco-like dataset…")
            df = _synthetic_telco(7043)

    return df


def _synthetic_telco(n: int) -> pd.DataFrame:
    """Synthesise a Telco-style churn dataset."""
    rng = np.random.default_rng(42)

    df = pd.DataFrame({
        "customerID": [f"CUST-{i:05d}" for i in range(n)],
        "gender":          rng.choice(["Male", "Female"], n),
        "SeniorCitizen":   rng.integers(0, 2, n),
        "Partner":         rng.choice(["Yes", "No"], n),
        "Dependents":      rng.choice(["Yes", "No"], n),
        "tenure":          rng.integers(0, 72, n).astype(float),
        "PhoneService":    rng.choice(["Yes", "No"], n),
        "MultipleLines":   rng.choice(["Yes", "No", "No phone service"], n),
        "InternetService": rng.choice(["DSL", "Fiber optic", "No"], n),
        "OnlineSecurity":  rng.choice(["Yes", "No", "No internet service"], n),
        "TechSupport":     rng.choice(["Yes", "No", "No internet service"], n),
        "Contract":        rng.choice(["Month-to-month", "One year", "Two year"], n,
                                      p=[0.55, 0.25, 0.20]),
        "PaperlessBilling":rng.choice(["Yes", "No"], n),
        "PaymentMethod":   rng.choice(
            ["Electronic check", "Mailed check", "Bank transfer", "Credit card"], n),
        "MonthlyCharges":  rng.uniform(18, 118, n).round(2),
        "TotalCharges":    np.nan,   # will be derived + some NaN injected
        "Churn":           rng.choice(["Yes", "No"], n, p=[0.27, 0.73]),
    })
    df["TotalCharges"] = (df["tenure"] * df["MonthlyCharges"]).round(2)
    df.loc[rng.choice(n, 11, replace=False), "TotalCharges"] = np.nan
    return df


# ─────────────────────────────────────────
#  2. EDA
# ─────────────────────────────────────────
def eda(df: pd.DataFrame):
    print("\n" + "─"*60)
    print("EXPLORATORY DATA ANALYSIS")
    print("─"*60)
    print(f"Shape          : {df.shape}")
    print(f"Missing values :\n{df.isnull().sum()[df.isnull().sum() > 0]}")
    print(f"\nChurn rate     : {df['Churn'].value_counts(normalize=True).to_dict()}")

    fig, axes = plt.subplots(1, 3, figsize=(16, 4))

    # Churn distribution
    df["Churn"].value_counts().plot.bar(ax=axes[0], color=["#2ecc71","#e74c3c"], rot=0)
    axes[0].set_title("Churn Distribution"); axes[0].set_xlabel("")

    # Monthly charges by churn
    df.groupby("Churn")["MonthlyCharges"].plot.kde(ax=axes[1], legend=True)
    axes[1].set_title("Monthly Charges by Churn")

    # Tenure by churn
    df.groupby("Churn")["tenure"].plot.kde(ax=axes[2], legend=True)
    axes[2].set_title("Tenure by Churn")

    plt.suptitle("Telco Churn – EDA", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig("eda_plots.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("Saved: eda_plots.png")


# ─────────────────────────────────────────
#  3. Preprocessing
# ─────────────────────────────────────────
def preprocess(df: pd.DataFrame):
    # Drop non-predictive columns
    df = df.drop(columns=["customerID"], errors="ignore")

    # Fix TotalCharges: sometimes it's a string with spaces
    if df["TotalCharges"].dtype == object:
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

    # Binary target
    df["Churn"] = (df["Churn"] == "Yes").astype(int)

    X = df.drop(columns=["Churn"])
    y = df["Churn"]

    # Identify column types
    num_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    cat_cols = X.select_dtypes(include=["object"]).columns.tolist()

    print(f"\nNumerical features ({len(num_cols)}): {num_cols}")
    print(f"Categorical features ({len(cat_cols)}): {cat_cols}")
    return X, y, num_cols, cat_cols


# ─────────────────────────────────────────
#  4. Build pipelines
# ─────────────────────────────────────────
def build_pipeline(num_cols, cat_cols, classifier):
    num_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  StandardScaler()),
    ])
    cat_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("ohe",     OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    preprocessor = ColumnTransformer([
        ("num", num_transformer, num_cols),
        ("cat", cat_transformer, cat_cols),
    ])
    return Pipeline([
        ("preprocessor", preprocessor),
        ("classifier",   classifier),
    ])


# ─────────────────────────────────────────
#  5. Evaluate
# ─────────────────────────────────────────
def evaluate(name, pipeline, X_train, X_test, y_train, y_test):
    pipeline.fit(X_train, y_train)
    y_pred  = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    f1  = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)

    print(f"\n{'─'*40}")
    print(f"Model: {name}")
    print(f"  Accuracy : {acc:.4f}")
    print(f"  F1 Score : {f1:.4f}")
    print(f"  ROC-AUC  : {auc:.4f}")
    print(classification_report(y_test, y_pred, target_names=["No Churn","Churn"]))

    return {"name": name, "acc": acc, "f1": f1, "auc": auc,
            "y_pred": y_pred, "y_proba": y_proba, "pipeline": pipeline}


# ─────────────────────────────────────────
#  6. Visualisations
# ─────────────────────────────────────────
def plot_results(results, X_test, y_test):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # ROC curves
    for r in results:
        fpr, tpr, _ = roc_curve(y_test, r["y_proba"])
        axes[0].plot(fpr, tpr, label=f"{r['name']} (AUC={r['auc']:.3f})", linewidth=2)
    axes[0].plot([0,1],[0,1],"k--", alpha=0.4)
    axes[0].set_title("ROC Curves"); axes[0].set_xlabel("FPR"); axes[0].set_ylabel("TPR")
    axes[0].legend()

    # Bar comparison
    metrics = ["acc", "f1", "auc"]
    x       = np.arange(len(metrics))
    width   = 0.35
    for i, r in enumerate(results[:2]):
        axes[1].bar(x + i*width, [r[m] for m in metrics], width, label=r["name"])
    axes[1].set_xticks(x + width/2)
    axes[1].set_xticklabels(["Accuracy","F1","AUC"])
    axes[1].set_ylim(0.5, 1.0); axes[1].legend()
    axes[1].set_title("Model Comparison")

    # Confusion matrix – best model
    best = max(results, key=lambda r: r["auc"])
    ConfusionMatrixDisplay(
        confusion_matrix(y_test, best["y_pred"]),
        display_labels=["No Churn","Churn"],
    ).plot(ax=axes[2], colorbar=False)
    axes[2].set_title(f"Confusion Matrix – {best['name']}")

    plt.suptitle("Telco Churn – Model Evaluation", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig("model_evaluation.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("Saved: model_evaluation.png")


def plot_feature_importance(rf_pipeline, num_cols, cat_cols):
    rf      = rf_pipeline["classifier"]
    ohe     = rf_pipeline["preprocessor"].transformers_[1][1]["ohe"]
    cat_enc = list(ohe.get_feature_names_out(cat_cols))
    feat_names = num_cols + cat_enc

    importances = pd.Series(rf.feature_importances_, index=feat_names)
    top15 = importances.nlargest(15)

    plt.figure(figsize=(9, 5))
    top15.sort_values().plot.barh(color="#3498db")
    plt.title("Top 15 Feature Importances – Random Forest")
    plt.tight_layout()
    plt.savefig("feature_importance.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("Saved: feature_importance.png")


# ─────────────────────────────────────────
#  7. Main
# ─────────────────────────────────────────
def main():
    df = load_telco_churn()
    eda(df)

    X, y, num_cols, cat_cols = preprocess(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"\nTrain: {X_train.shape} | Test: {X_test.shape}")

    # ── Logistic Regression ───────────────────
    lr_pipe = build_pipeline(num_cols, cat_cols,
                             LogisticRegression(max_iter=1000, C=1.0))
    lr_res  = evaluate("Logistic Regression", lr_pipe, X_train, X_test, y_train, y_test)

    # ── Random Forest ─────────────────────────
    rf_pipe = build_pipeline(num_cols, cat_cols,
                             RandomForestClassifier(n_estimators=200, random_state=42))
    rf_res  = evaluate("Random Forest", rf_pipe, X_train, X_test, y_train, y_test)

    # ── GridSearchCV on Logistic Regression ───
    print("\n" + "─"*60)
    print("GridSearchCV – Logistic Regression hyperparameter tuning")
    param_grid = {
        "classifier__C":        [0.01, 0.1, 1, 10, 100],
        "classifier__penalty":  ["l1", "l2"],
        "classifier__solver":   ["liblinear"],
    }
    tuned_pipe = build_pipeline(num_cols, cat_cols, LogisticRegression(max_iter=1000))
    grid = GridSearchCV(tuned_pipe, param_grid, cv=StratifiedKFold(5),
                        scoring="roc_auc", n_jobs=-1, verbose=1)
    grid.fit(X_train, y_train)
    print(f"Best params: {grid.best_params_}")
    print(f"Best CV AUC: {grid.best_score_:.4f}")
    tuned_res = evaluate("Tuned LR (GridSearch)", grid.best_estimator_,
                         X_train, X_test, y_train, y_test)

    # ── Plots ─────────────────────────────────
    plot_results([lr_res, rf_res, tuned_res], X_test, y_test)
    plot_feature_importance(rf_pipe, num_cols, cat_cols)

    # ── Cross-validation ──────────────────────
    print("\n5-Fold Cross-Validation (Random Forest):")
    cv_scores = cross_val_score(rf_pipe, X, y, cv=5, scoring="roc_auc", n_jobs=-1)
    print(f"  AUC: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

    # ── Export pipelines ──────────────────────
    best_pipeline = max([lr_res, rf_res, tuned_res], key=lambda r: r["auc"])["pipeline"]
    joblib.dump(best_pipeline, "churn_pipeline.joblib")
    joblib.dump(rf_pipe,       "rf_pipeline.joblib")
    print("\nPipelines exported:")
    print("  churn_pipeline.joblib  (best overall)")
    print("  rf_pipeline.joblib     (Random Forest)")

    # Summary
    print("\n" + "═"*60)
    print("SUMMARY")
    print("═"*60)
    for r in [lr_res, rf_res, tuned_res]:
        print(f"  {r['name']:30s}  AUC={r['auc']:.4f}  F1={r['f1']:.4f}")


if __name__ == "__main__":
    main()
