import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'backend'))

import numpy as np
import pandas as pd
import pickle
import multiprocessing as mp
from functools import partial

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import f1_score, roc_auc_score, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
import xgboost as xgb
import lightgbm as lgb
import shap
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

from app.services.data_generator import MSMEDataGenerator   # type: ignore
from app.services.feature_engineering import FeatureEngineer  # type: ignore


def _label_from_features(f: dict) -> int:

    wcr = f.get("working_capital_ratio", 1.0)
    wcr_risk = float(np.clip((1.4 - wcr) / (1.4 - 0.3), 0, 1))

    ir = f.get("inflow_ratio", 0.55)
    ir_risk = float(np.clip((0.62 - ir) / (0.62 - 0.28), 0, 1))

    mca = f.get("mca_compliance_score", 1.0)
    mca_risk = float(np.clip(1.0 - mca, 0, 1))

    nfr = f.get("not_filed_ratio", 0.0)
    nfr_risk = float(np.clip(nfr, 0, 1))

    rg = f.get("revenue_growth_3m", 0.0)
    rg_risk = float(np.clip((-rg + 70) / (70 + 100), 0, 1))

    hhi = f.get("customer_concentration_hhi", 0.1)
    hhi_risk = float(np.clip((hhi - 0.04) / (0.99 - 0.04), 0, 1))

    cfs = f.get("circular_flow_score", 0.0)
    cfs_risk = float(np.clip(cfs, 0, 1))

    # upi_counterparty_diversity: 1=safe(0), 0=risky(1)
    div = f.get("upi_counterparty_diversity", 0.5)
    div_risk = float(np.clip(1.0 - div, 0, 1))

    # Weighted sum of risks
    risk = (
        wcr_risk  * 0.25 +   # cash flow (strongest)
        ir_risk   * 0.20 +   # inflow ratio
        mca_risk  * 0.12 +   # compliance
        nfr_risk  * 0.12 +   # non-filing (severe)
        rg_risk   * 0.12 +   # revenue trend
        hhi_risk  * 0.08 +   # concentration
        cfs_risk  * 0.07 +   # circular flow
        div_risk  * 0.04     # UPI diversity
    )

    # Add small noise so the decision boundary isn't a hard cliff
    noise = np.random.normal(0, 0.04)
    default_prob = float(np.clip(risk + noise, 0.0, 0.97))

    return int(np.random.random() < default_prob)


# ---------------------------------------------------------------------------
# Sample generation (single process)
# ---------------------------------------------------------------------------

def _generate_one(seed: int) -> tuple:
    """Generate one (features_dict, label) pair. Must be top-level for pickling."""
    try:
        gstin = f"27{seed:013d}Z{seed % 10}"
        gen   = MSMEDataGenerator(gstin, seed=seed)
        data  = gen.generate_all()
        fe    = FeatureEngineer()
        feats = fe.compute_features(data)
        label = _label_from_features(feats)
        return feats, label
    except Exception as e:
        return None, None


# ---------------------------------------------------------------------------
# Dataset generation — uses multiprocessing on Kaggle/local
# ---------------------------------------------------------------------------

def generate_dataset(n_samples: int = 5000, n_workers: int = None):
    if n_workers is None:
        n_workers = max(1, mp.cpu_count() - 1)

    print(f"Generating {n_samples} samples using {n_workers} workers...")

    seeds = list(range(n_samples))

    with mp.Pool(processes=n_workers) as pool:
        results = []
        for i, res in enumerate(pool.imap(_generate_one, seeds, chunksize=50)):
            results.append(res)
            if (i + 1) % 500 == 0:
                print(f"  Generated {i + 1}/{n_samples}")

    samples = [f for f, l in results if f is not None]
    labels  = [l for f, l in results if f is not None]

    if len(samples) < n_samples * 0.95:
        print(f"  [WARN] Only {len(samples)}/{n_samples} samples generated successfully")

    df = pd.DataFrame(samples)
    y  = np.array(labels)

    print(f"\nDataset generated:")
    print(f"  Samples : {len(df)}")
    print(f"  Features: {df.shape[1]}")
    print(f"  Default rate: {y.mean():.2%}")

    return df, y


# ---------------------------------------------------------------------------
# Model training
# ---------------------------------------------------------------------------

def _make_xgb(scale_pos_weight: float) -> xgb.XGBClassifier:
    return xgb.XGBClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.03,
        subsample=0.75,
        colsample_bytree=0.75,
        min_child_weight=15,
        gamma=1.5,
        reg_alpha=2.0,
        reg_lambda=5.0,
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        eval_metric="logloss",
        n_jobs=-1,
    )


def _make_lgb(scale_pos_weight: float) -> lgb.LGBMClassifier:
    return lgb.LGBMClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.03,
        subsample=0.75,
        colsample_bytree=0.75,
        min_child_samples=30,
        reg_alpha=2.0,
        reg_lambda=5.0,
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        verbose=-1,
        n_jobs=-1,
    )


def train_xgboost(X_train, y_train, X_test, y_test):
    print("\n--- Training XGBoost ---")
    spw = float((y_train == 0).sum() / max((y_train == 1).sum(), 1))
    model = _make_xgb(spw)
    model.fit(X_train, y_train)

    for name, X, y in [("Train", X_train, y_train), ("Test", X_test, y_test)]:
        p = model.predict(X)
        pr = model.predict_proba(X)[:, 1]
        print(f"  [{name}] F1: {f1_score(y, p):.4f}  |  AUC-ROC: {roc_auc_score(y, pr):.4f}")

    print(classification_report(y_test, model.predict(X_test)))
    return model


def train_lightgbm(X_train, y_train, X_test, y_test):
    print("\n--- Training LightGBM ---")
    spw = float((y_train == 0).sum() / max((y_train == 1).sum(), 1))
    model = _make_lgb(spw)
    model.fit(X_train, y_train)

    for name, X, y in [("Train", X_train, y_train), ("Test", X_test, y_test)]:
        p = model.predict(X)
        pr = model.predict_proba(X)[:, 1]
        print(f"  [{name}] F1: {f1_score(y, p):.4f}  |  AUC-ROC: {roc_auc_score(y, pr):.4f}")

    print(classification_report(y_test, model.predict(X_test)))
    return model


def train_meta_learner(xgb_model, lgb_model, X_train, y_train, X_test, y_test):
    print("\n--- Training Meta-Learner (OOF stacking) ---")
    spw = float((y_train == 0).sum() / max((y_train == 1).sum(), 1))
    n_folds = 5
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

    oof_xgb = np.zeros(len(X_train))
    oof_lgb = np.zeros(len(X_train))

    for fold, (tr_idx, val_idx) in enumerate(skf.split(X_train, y_train), 1):
        Xtr, Xval = X_train[tr_idx], X_train[val_idx]
        ytr = y_train[tr_idx]

        fx = _make_xgb(spw)
        fx.fit(Xtr, ytr)
        oof_xgb[val_idx] = fx.predict_proba(Xval)[:, 1]

        fl = _make_lgb(spw)
        fl.fit(Xtr, ytr)
        oof_lgb[val_idx] = fl.predict_proba(Xval)[:, 1]

        print(f"  Fold {fold}/{n_folds} done")

    meta_train = np.column_stack([oof_xgb, oof_lgb])
    meta = LogisticRegression(C=1.0, class_weight="balanced", random_state=42, max_iter=500)
    meta.fit(meta_train, y_train)

    xgb_prob = xgb_model.predict_proba(X_test)[:, 1]
    lgb_prob = lgb_model.predict_proba(X_test)[:, 1]
    meta_test = np.column_stack([xgb_prob, lgb_prob])

    mp_ = meta.predict_proba(meta_test)[:, 1]
    mp_pred = meta.predict(meta_test)

    print(f"\n  Ensemble F1:      {f1_score(y_test, mp_pred):.4f}")
    print(f"  Ensemble AUC-ROC: {roc_auc_score(y_test, mp_):.4f}")
    print(classification_report(y_test, mp_pred))
    return meta


# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------

def save_models(xgb_model, lgb_model, meta_learner, feature_columns, imputer):
    model_dir = os.path.join(os.path.dirname(__file__), '..', 'backend', 'ml_models')
    os.makedirs(model_dir, exist_ok=True)

    for name, obj in [
        ("xgboost_model.pkl", xgb_model),
        ("lightgbm_model.pkl", lgb_model),
        ("meta_learner.pkl", meta_learner),
        ("feature_columns.pkl", feature_columns),
        ("imputer.pkl", imputer),
    ]:
        with open(os.path.join(model_dir, name), "wb") as f:
            pickle.dump(obj, f)

    print(f"\nAll models saved to: {model_dir}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 60)
    print("MSME Credit Scoring — Model Training")
    print("=" * 60)

    # Detect Kaggle environment and set workers accordingly
    on_kaggle = os.path.exists("/kaggle")
    n_workers = 2 if on_kaggle else max(1, mp.cpu_count() - 1)
    n_samples = 5000

    df, y = generate_dataset(n_samples=n_samples, n_workers=n_workers)

    feature_columns = df.columns.tolist()
    X_train_df, X_test_df, y_train, y_test = train_test_split(
        df, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"\nTrain: {len(X_train_df)}, Test: {len(X_test_df)}")
    print(f"Train default rate: {y_train.mean():.2%}")
    print(f"Test  default rate: {y_test.mean():.2%}")

    imputer = SimpleImputer(strategy="median")
    X_train = imputer.fit_transform(X_train_df)
    X_test  = imputer.transform(X_test_df)

    xgb_model  = train_xgboost(X_train, y_train, X_test, y_test)
    lgb_model  = train_lightgbm(X_train, y_train, X_test, y_test)
    meta_model = train_meta_learner(xgb_model, lgb_model, X_train, y_train, X_test, y_test)

    print("\n--- Generating SHAP Summary ---")
    explainer   = shap.TreeExplainer(xgb_model)
    shap_sample = X_test[:200]
    shap_values = explainer.shap_values(shap_sample)

    model_dir = os.path.join(os.path.dirname(__file__), '..', 'backend', 'ml_models')
    os.makedirs(model_dir, exist_ok=True)
    shap.summary_plot(shap_values, shap_sample, feature_names=feature_columns, show=False)
    plt.savefig(os.path.join(model_dir, "shap_summary.png"), bbox_inches="tight", dpi=150)
    plt.close()
    print("SHAP summary saved.")

    save_models(xgb_model, lgb_model, meta_model, feature_columns, imputer)

    print("\n" + "=" * 60)
    print("Training complete!")
    print("=" * 60)


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()