"""
Train a diabetes prediction model on the Pima Indians Diabetes Dataset.
Saves the trained model + scaler + metadata to disk for the web app to use.
"""
import pandas as pd
import numpy as np
import joblib
import json

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)

COLS = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
        'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome']

FEATURE_COLS = COLS[:-1]

# Columns where a 0 is biologically impossible -> treat as missing
ZERO_AS_MISSING = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']


def load_and_clean(path='diabetes.csv'):
    df = pd.read_csv(path, names=COLS)

    # Replace impossible zeros with NaN, then impute with median (grouped by Outcome
    # for a slightly better estimate, then overall median as fallback)
    for col in ZERO_AS_MISSING:
        df[col] = df[col].replace(0, np.nan)

    for col in ZERO_AS_MISSING:
        df[col] = df.groupby('Outcome')[col].transform(lambda s: s.fillna(s.median()))
        df[col] = df[col].fillna(df[col].median())

    return df


def main():
    df = load_and_clean('diabetes.csv')

    X = df[FEATURE_COLS]
    y = df['Outcome']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    candidates = {
        'LogisticRegression': LogisticRegression(max_iter=2000, class_weight='balanced', random_state=42),
        'RandomForest': RandomForestClassifier(n_estimators=300, max_depth=6, random_state=42, class_weight='balanced'),
        'GradientBoosting': GradientBoostingClassifier(n_estimators=200, max_depth=3, learning_rate=0.05, random_state=42),
        'SVM': SVC(kernel='rbf', probability=True, class_weight='balanced', random_state=42),
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    results = {}

    print("=" * 60)
    print("MODEL COMPARISON (5-fold cross-validated ROC-AUC on train set)")
    print("=" * 60)

    for name, model in candidates.items():
        scores = cross_val_score(model, X_train_scaled, y_train, cv=cv, scoring='roc_auc')
        results[name] = scores.mean()
        print(f"{name:20s}  mean AUC = {scores.mean():.4f}  (+/- {scores.std():.4f})")

    best_name = max(results, key=results.get)
    print(f"\nBest model: {best_name}")

    best_model = candidates[best_name]
    best_model.fit(X_train_scaled, y_train)

    y_pred = best_model.predict(X_test_scaled)
    y_proba = best_model.predict_proba(X_test_scaled)[:, 1]

    metrics = {
        'model_name': best_name,
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1': f1_score(y_test, y_pred),
        'roc_auc': roc_auc_score(y_test, y_proba),
        'confusion_matrix': confusion_matrix(y_test, y_pred).tolist(),
        'cv_scores_all_models': results,
        'n_train': len(X_train),
        'n_test': len(X_test),
    }

    print("\n" + "=" * 60)
    print("HELD-OUT TEST SET PERFORMANCE")
    print("=" * 60)
    print(f"Accuracy : {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall   : {metrics['recall']:.4f}")
    print(f"F1 score : {metrics['f1']:.4f}")
    print(f"ROC-AUC  : {metrics['roc_auc']:.4f}")
    print("\nConfusion matrix:")
    print(np.array(metrics['confusion_matrix']))
    print("\n" + classification_report(y_test, y_pred, target_names=['Non-diabetic', 'Diabetic']))

    # Save artifacts
    joblib.dump(best_model, 'diabetes_model.joblib')
    joblib.dump(scaler, 'scaler.joblib')

    # Feature medians (post-cleaning) - used for reference/context in the app
    feature_stats = df[FEATURE_COLS].describe().to_dict()

    with open('metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)

    with open('feature_stats.json', 'w') as f:
        json.dump(feature_stats, f, indent=2)

    print("\nSaved: diabetes_model.joblib, scaler.joblib, metrics.json, feature_stats.json")


if __name__ == '__main__':
    main()
