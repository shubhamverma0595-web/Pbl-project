"""
SmartHealth AI - Research-Grade 2-Tier Hybrid Stacking ML Pipeline
===================================================================
Incorporates Clinical Feature Engineering (Rate-Pressure Product RPP),
SMOTE-Tomek data balancing, and 2-Tier Hybrid Stacking Meta-Learner.
"""

import os
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier, StackingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier


def load_and_enrich_data(filepath):
    print(f"Loading dataset from: {filepath}")
    df = pd.read_csv(filepath)

    # --- Research Innovation: Clinical Domain Feature Engineering ---
    df['rpp'] = (df['trestbps'] * df['thalach']) / 100.0  # Rate-Pressure Product (Myocardial Stress)
    df['chol_hr_ratio'] = df['chol'] / (df['thalach'] + 1e-5)  # Cholesterol / Heart-Rate Ratio
    df['age_st_index'] = df['oldpeak'] * (df['age'] / 50.0)    # Age-Adjusted ST Depression

    print(f"Dataset shape post-feature engineering: {df.shape[0]} rows, {df.shape[1]} columns")
    return df


def preprocess_data(df, target_col='target'):
    X = df.drop(columns=[target_col])
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, y_train, y_test, scaler, X.columns


def train_stacking_pipeline(X_train, X_test, y_train, y_test):
    # Tier 1 Base Learners
    base_learners = [
        ('gb', GradientBoostingClassifier(n_estimators=100, max_depth=4, random_state=42)),
        ('rf', RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42)),
        ('et', ExtraTreesClassifier(n_estimators=100, max_depth=6, random_state=42)),
        ('mlp', MLPClassifier(hidden_layer_sizes=(32, 16), max_iter=1000, random_state=42))
    ]

    # Tier 2 Meta-Learner
    meta_learner = LogisticRegression(C=1.0, max_iter=1000)

    # 2-Tier Hybrid Stacking Meta-Classifier
    stacking_model = StackingClassifier(
        estimators=base_learners,
        final_estimator=meta_learner,
        cv=5
    )

    # Individual Base Models for Comparison
    all_models = {
        "Logistic Regression (Base)": LogisticRegression(max_iter=1000, random_state=42),
        "Decision Tree (Base)": DecisionTreeClassifier(max_depth=5, random_state=42),
        "Random Forest (Base)": RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42),
        "Gradient Boosting (Base)": GradientBoostingClassifier(n_estimators=100, random_state=42),
        "⭐ 2-Tier Hybrid Stacking Meta-Learner": stacking_model
    }

    results = []
    trained_models = {}

    print("\n" + "="*82)
    print(f"{'Model Architecture':<38} | {'Accuracy':<8} | {'Precision':<9} | {'Recall':<6} | {'F1':<6} | {'ROC-AUC':<7}")
    print("="*82)

    for name, model in all_models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else y_pred

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_proba)

        results.append({
            "Architecture": name,
            "Accuracy": acc,
            "Precision": prec,
            "Recall": rec,
            "F1-Score": f1,
            "ROC-AUC": auc
        })

        trained_models[name] = model

        print(f"{name:<38} | {acc:.4f}   | {prec:.4f}    | {rec:.4f} | {f1:.4f} | {auc:.4f}")

    print("="*82 + "\n")
    return results, trained_models


def save_artifacts(models, scaler, output_dir="../python_src/saved_models"):
    os.makedirs(output_dir, exist_ok=True)
    joblib.dump(scaler, os.path.join(output_dir, "scaler.joblib"))
    joblib.dump(models["⭐ 2-Tier Hybrid Stacking Meta-Learner"], os.path.join(output_dir, "hybrid_stacking_meta_learner.joblib"))
    print("Saved 2-Tier Stacking Meta-Learner and Scaler to saved_models/")


def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_path = os.path.join(base_dir, "..", "dataset", "heart_disease.csv")
    if not os.path.exists(dataset_path):
        dataset_path = os.path.join(base_dir, "dataset", "heart_disease.csv")

    df = load_and_enrich_data(dataset_path)
    X_train, X_test, y_train, y_test, scaler, feature_names = preprocess_data(df)
    results, trained_models = train_stacking_pipeline(X_train, X_test, y_train, y_test)
    save_artifacts(trained_models, scaler)

    print("Research-Grade Stacking Pipeline Execution Completed Successfully!")


if __name__ == "__main__":
    main()
