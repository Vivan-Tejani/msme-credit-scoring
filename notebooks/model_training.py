import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'backend'))

import numpy as np
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, roc_auc_score, classification_report
from sklearn.linear_model import LogisticRegression
import xgboost as xgb
import lightgbm as lgb #type:ignore
import shap
import warnings
warnings.filterwarnings('ignore')

from app.services.data_generator import MSMEDataGenerator #type:ignore
from app.services.feature_engineering import FeatureEngineer #type:ignore


def generate_sample(seed, feature_engineer):
    gstin = f"27{seed:013d}Z{seed%10}"
    generator = MSMEDataGenerator(gstin, seed=seed)
    data = generator.generate_all()
    features = feature_engineer.compute_features(data)
    
    # Simulate default label based on key risk factors
    risk_score = 0
    if features.get('revenue_growth_3m', 0) < -30:
        risk_score += 2
    if features.get('customer_concentration_hhi', 0) > 0.7:
        risk_score += 2
    if features.get('working_capital_ratio', 1) < 0.5:
        risk_score += 2
    if features.get('business_age_months', 0) < 12:
        risk_score += 1
    if features.get('circular_flow_score', 0) > 0.4:
        risk_score += 3
    if features.get('mca_compliance_score', 0) < 0.3:
        risk_score += 1
    
    default_prob = min(0.9, risk_score / 10 + np.random.normal(0, 0.1))
    default = 1 if np.random.random() < default_prob else 0
    
    return features, default


def generate_dataset(n_samples=5000):
    print(f"Generating {n_samples} synthetic samples: ")
    feature_engineer = FeatureEngineer()
    
    samples = []
    labels = []
    for i in range(n_samples):
        f, l = generate_sample(i, feature_engineer)
        samples.append(f)
        labels.append(l)
        
        if (i + 1) % 1000 == 0:
            print(f"  Generated {i + 1}/{n_samples}")
    
    df = pd.DataFrame(samples)
    y = np.array(labels)
    
    print(f"\nDataset generated:")
    print(f"  Samples: {n_samples}")
    print(f"  Features: {df.shape[1]}")
    print(f"  Default rate: {y.mean():.2%}")
    
    return df, y


def train_xgboost(X_train, y_train, X_test, y_test):
    print("\n--- Training XGBoost ---")
    model = xgb.XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=3,
        random_state=42,
        use_label_encoder=False,
        eval_metric='logloss'
    )
    
    model.fit(X_train, y_train)
    
    pred = model.predict(X_test)
    prob = model.predict_proba(X_test)[:, 1]
    
    print(f"F1: {f1_score(y_test, pred):.4f}")
    print(f"AUC-ROC: {roc_auc_score(y_test, prob):.4f}")
    print(classification_report(y_test, pred))
    
    return model


def train_lightgbm(X_train, y_train, X_test, y_test):
    print("\n-- Training LightGBM --")
    model = lgb.LGBMClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        class_weight='balanced',
        random_state=42,
        verbose=-1
    )
    
    model.fit(X_train, y_train)
    
    pred = model.predict(X_test)
    prob = model.predict_proba(X_test)[:, 1]
    
    print(f"F1: {f1_score(y_test, pred):.4f}")
    print(f"AUC-ROC: {roc_auc_score(y_test, prob):.4f}")
    print(classification_report(y_test, pred))
    
    return model


def train_meta_learner(xgb_model, lgb_model, X_train, y_train, X_test, y_test):
    print("\n-- Training Meta-Learner --")
    
    xgb_train_prob = xgb_model.predict_proba(X_train)[:, 1].reshape(-1, 1)
    lgb_train_prob = lgb_model.predict_proba(X_train)[:, 1].reshape(-1, 1)
    meta_features = np.hstack([xgb_train_prob, lgb_train_prob])
    
    meta = LogisticRegression(random_state=42)
    meta.fit(meta_features, y_train)
    
    xgb_test_prob = xgb_model.predict_proba(X_test)[:, 1].reshape(-1, 1)
    lgb_test_prob = lgb_model.predict_proba(X_test)[:, 1].reshape(-1, 1)
    meta_test_features = np.hstack([xgb_test_prob, lgb_test_prob])
    
    meta_prob = meta.predict_proba(meta_test_features)[:, 1]
    meta_pred = meta.predict(meta_test_features)
    
    print(f"Ensemble F1: {f1_score(y_test, meta_pred):.4f}")
    print(f"Ensemble AUC-ROC: {roc_auc_score(y_test, meta_prob):.4f}")
    print(classification_report(y_test, meta_pred))
    
    return meta


def save_models(xgb_model, lgb_model, meta_learner):
    """Save trained models to disk."""
    model_dir = os.path.join(os.path.dirname(__file__), '..', 'backend', 'ml_models')
    os.makedirs(model_dir, exist_ok=True)
    
    with open(os.path.join(model_dir, 'xgboost_model.pkl'), 'wb') as f:
        pickle.dump(xgb_model, f)
    
    with open(os.path.join(model_dir, 'lightgbm_model.pkl'), 'wb') as f:
        pickle.dump(lgb_model, f)
    
    with open(os.path.join(model_dir, 'meta_learner.pkl'), 'wb') as f:
        pickle.dump(meta_learner, f)
    
    print(f"\nModels saved to {model_dir}")


def main():
    """Main training pipeline."""
    print("=" * 50)
    print("MSME Credit Scoring - Model Training")
    print("=" * 50)
    
    df, y = generate_dataset(n_samples=5000)
    
    # TTS
    X_train, X_test, y_train, y_test = train_test_split(
        df, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"\nTrain: {len(X_train)}, Test: {len(X_test)}")
    
    xgb_model = train_xgboost(X_train, y_train, X_test, y_test)
    lgb_model = train_lightgbm(X_train, y_train, X_test, y_test)
    meta_learner = train_meta_learner(xgb_model, lgb_model, X_train, y_train, X_test, y_test)
    
    # SHAP summary
    print("\n--- Generating SHAP Summary ---")
    explainer = shap.TreeExplainer(xgb_model)
    shap_values = explainer.shap_values(X_test.iloc[:100])
    
    model_dir = os.path.join(os.path.dirname(__file__), '..', 'backend', 'ml_models')
    os.makedirs(model_dir, exist_ok=True)
    shap.summary_plot(shap_values, X_test.iloc[:100], show=False)
    import matplotlib.pyplot as plt
    plt.savefig(os.path.join(model_dir, 'shap_summary.png'), bbox_inches='tight')
    print("SHAP summary saved")
    
    save_models(xgb_model, lgb_model, meta_learner)
    
    print("\n" + "=" * 50)
    print("Training complete!")
    print("=" * 50)


if __name__ == "__main__":
    main()