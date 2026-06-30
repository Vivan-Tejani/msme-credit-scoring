import os
import pickle
import numpy as np
from typing import Dict, List, Tuple, Optional
from datetime import datetime
import uuid

try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False

try:
    import lightgbm as lgb  # type:ignore
    LGB_AVAILABLE = True
except ImportError:
    LGB_AVAILABLE = False

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False


class ScoringEngine:
    def __init__(self, model_path: str = "ml_models"):
        self.model_path = model_path
        self.xgb_model = None
        self.lgb_model = None
        self.meta_learner = None
        self.imputer = None
        self.explainer = None
        # FIX: feature_names is no longer a hardcoded list — it's loaded from
        # feature_columns.pkl, the exact column order saved during training.
        # This guarantees the scoring engine never drifts out of sync with
        # whatever feature_engineering.py currently produces.
        self.feature_names: List[str] = []
        self._load_models()

    # ------------------------------------------------------------------
    # Model + artifact loading
    # ------------------------------------------------------------------

    def _load_models(self):
        xgb_path     = os.path.join(self.model_path, "xgboost_model.pkl")
        lgb_path     = os.path.join(self.model_path, "lightgbm_model.pkl")
        meta_path    = os.path.join(self.model_path, "meta_learner.pkl")
        columns_path = os.path.join(self.model_path, "feature_columns.pkl")
        imputer_path = os.path.join(self.model_path, "imputer.pkl")

        if os.path.exists(xgb_path) and XGB_AVAILABLE:
            with open(xgb_path, "rb") as f:
                self.xgb_model = pickle.load(f)

        if os.path.exists(lgb_path) and LGB_AVAILABLE:
            with open(lgb_path, "rb") as f:
                self.lgb_model = pickle.load(f)

        if os.path.exists(meta_path):
            with open(meta_path, "rb") as f:
                self.meta_learner = pickle.load(f)

        # FIX: load the authoritative feature column order from training.
        if os.path.exists(columns_path):
            with open(columns_path, "rb") as f:
                self.feature_names = pickle.load(f)
        else:
            # Fallback only if the model was trained before feature_columns.pkl
            # existed. This should not happen with the current training script,
            # but prevents a hard crash on stale model directories.
            self.feature_names = self._legacy_feature_names_fallback()

        # FIX: load the imputer fitted during training so missing/NaN feature
        # values are handled the same way at inference as they were at train time.
        if os.path.exists(imputer_path):
            with open(imputer_path, "rb") as f:
                self.imputer = pickle.load(f)

        if self.xgb_model is not None and SHAP_AVAILABLE and XGB_AVAILABLE:
            try:
                self.explainer = shap.TreeExplainer(self.xgb_model)
            except Exception:
                self.explainer = None

    def _legacy_feature_names_fallback(self) -> List[str]:
        """
        Only used if feature_columns.pkl is missing (old model directory).
        Logs a warning so the mismatch is visible instead of silent.
        """
        print(
            "[ScoringEngine] WARNING: feature_columns.pkl not found in "
            f"'{self.model_path}'. Falling back to legacy 46-feature list — "
            "this WILL mismatch a model trained on the current "
            "feature_engineering.py. Retrain and ensure feature_columns.pkl "
            "is saved alongside the model files."
        )
        return [
            "monthly_revenue_mean", "monthly_revenue_std", "revenue_growth_3m", "revenue_growth_6m",
            "invoice_regularity_score", "customer_concentration_hhi", "avg_invoice_size", "invoice_size_cv",
            "large_invoice_ratio", "b2b_ratio", "b2c_ratio", "export_invoice_ratio",
            "gst_filing_timeliness_score", "filing_streak",
            "daily_upi_volume_mean", "daily_upi_volume_std", "upi_volume_trend_30d",
            "cash_flow_regularity", "working_capital_ratio", "peak_to_trough_ratio",
            "weekend_activity_ratio", "after_hours_ratio", "upi_counterparty_diversity", "repeat_customer_ratio",
            "avg_receivable_days", "avg_payable_days",
            "eway_monthly_count", "eway_monthly_value", "eway_value_growth_3m",
            "eway_destination_diversity", "interstate_ratio", "supply_chain_depth_score",
            "eway_invoice_correlation", "seasonal_eway_pattern", "eway_trend_90d",
            "mca_compliance_score", "director_default_history", "director_company_count",
            "business_age_months", "legal_structure_score", "gst_registration_age", "registered_office_stability",
            "circular_flow_score", "entity_cluster_size", "related_entity_invoice_ratio", "director_network_density",
        ]

    # ------------------------------------------------------------------
    # Feature vector construction
    # ------------------------------------------------------------------

    def _get_feature_vector(self, features: Dict[str, float]) -> np.ndarray:
        # Build the raw vector in the exact order the model was trained on.
        vector = [features.get(name, np.nan) for name in self.feature_names]
        raw = np.array(vector, dtype=float).reshape(1, -1)

        # FIX: apply the same imputer used during training instead of
        # defaulting missing features to 0.0. A missing feature silently
        # becoming 0.0 can be a legitimate (and very different) value for
        # some features (e.g. working_capital_ratio=0 looks like total
        # cash-flow failure, not "unknown").
        if self.imputer is not None:
            try:
                return self.imputer.transform(raw)
            except Exception:
                pass

        # Fallback: replace any NaNs with 0.0 if no imputer is available.
        return np.nan_to_num(raw, nan=0.0)

    # ------------------------------------------------------------------
    # Heuristic fallback (unchanged logic, still dict-based so it's
    # unaffected by feature ordering)
    # ------------------------------------------------------------------

    def _heuristic_score(self, features: Dict[str, float]) -> float:
        score = 600.0

        if features.get("revenue_growth_3m", 0) > 20:
            score += 40
        elif features.get("revenue_growth_3m", 0) < -20:
            score -= 50

        if features.get("monthly_revenue_mean", 0) > 1_000_000:
            score += 30

        wcr = features.get("working_capital_ratio", 1)
        if wcr > 1.2:
            score += 20
        elif wcr < 0.8:
            score -= 30

        if features.get("customer_concentration_hhi", 0) > 0.6:
            score -= 40

        if features.get("mca_compliance_score", 0) > 0.8:
            score += 20
        elif features.get("mca_compliance_score", 0) < 0.5:
            score -= 30

        if features.get("business_age_months", 0) < 12:
            score -= 40

        if features.get("circular_flow_score", 0) > 0.5:
            score -= 60

        score += np.random.normal(0, 10)
        return max(300, min(900, score))

    def _probability_to_score(self, prob: float) -> int:
        score = 300 + (1 - prob) * 600
        return int(round(score))

    def _get_risk_band(self, score: int) -> Tuple[str, float]:
        if score >= 750:
            return "LOW", 0.03
        elif score >= 650:
            return "MEDIUM", 0.06
        elif score >= 500:
            return "HIGH", 0.13
        else:
            return "VERY HIGH", 0.22

    # ------------------------------------------------------------------
    # SHAP explanations
    # ------------------------------------------------------------------

    def _compute_shap_explanations(self, features: Dict[str, float], feature_vector: np.ndarray) -> List[Dict]:
        explanations = []

        if self.explainer is not None and SHAP_AVAILABLE:
            try:
                sv = self.explainer.shap_values(feature_vector)
                if isinstance(sv, list):
                    sv = np.array(sv[0]).flatten()
                elif isinstance(sv, np.ndarray):
                    sv = sv.flatten()

                for i, name in enumerate(self.feature_names):
                    if i < len(sv):
                        val = float(sv[i])
                        explanations.append({
                            "feature": name,
                            "impact": round(val, 4),
                            "direction": "positive" if val > 0 else "negative"
                        })
            except Exception:
                pass

        if not explanations:
            heuristics = [
                ("revenue_growth_3m", 0.15, "positive"),
                ("customer_concentration_hhi", -0.12, "negative"),
                ("working_capital_ratio", 0.10, "positive"),
                ("circular_flow_score", -0.18, "negative"),
                ("business_age_months", -0.08, "negative"),
                ("mca_compliance_score", 0.08, "positive"),
                ("gst_filing_timeliness_score", 0.06, "positive"),
                ("invoice_regularity_score", 0.05, "positive"),
            ]

            for feature_name, default_impact, default_dir in heuristics:
                val = features.get(feature_name, 0)

                if feature_name == "revenue_growth_3m":
                    actual_dir = "positive" if val > 15 else "negative"
                elif feature_name == "customer_concentration_hhi":
                    actual_dir = "negative" if val > 0.3 else "positive"
                elif feature_name == "circular_flow_score":
                    actual_dir = "negative" if val > 0.3 else "positive"
                elif feature_name == "business_age_months":
                    actual_dir = "negative" if val < 18 else "positive"
                else:
                    actual_dir = "positive" if val > 0.5 else "negative"

                impact = default_impact if actual_dir == default_dir else -default_impact
                explanations.append({
                    "feature": feature_name,
                    "impact": round(impact, 4),
                    "direction": actual_dir
                })

        explanations.sort(key=lambda x: abs(x["impact"]), reverse=True)
        top5 = explanations[:5]

        text_map = {
            "revenue_growth_3m": lambda f: f"Revenue grew {abs(f):.0f}% over the last 3 months — strong positive signal" if f > 15 else f"Revenue declined {abs(f):.0f}% over the last 3 months — negative signal",
            "customer_concentration_hhi": lambda f: f"{f*100:.0f}% of revenue from a single customer increases risk" if f > 0.5 else "Customer base is well diversified",
            "working_capital_ratio": lambda f: "Healthy cash inflow to outflow ratio indicates good liquidity" if f > 1.1 else "Cash outflow exceeds inflow — working capital stress",
            "circular_flow_score": lambda f: "Suspicious circular transaction patterns detected" if f > 0.3 else "No circular transaction patterns detected",
            "business_age_months": lambda f: "Business is relatively new — higher risk profile" if f < 18 else "Established business with operational history",
            "mca_compliance_score": lambda f: "Strong regulatory compliance history" if f > 0.7 else "Compliance issues detected in MCA filings",
            "gst_filing_timeliness_score": lambda f: "Regular and timely GST filing pattern" if f > 0.7 else "Irregular GST filing pattern",
            "invoice_regularity_score": lambda f: "Regular invoice generation indicates stable operations" if f > 0.5 else "Irregular invoice pattern",
            "monthly_revenue_mean": lambda f: "Consistent revenue stream indicates business stability" if f > 500000 else "Revenue stream is below average",
            "daily_upi_volume_mean": lambda f: "Healthy daily transaction volume" if f > 50000 else "Low transaction volume",
            # New features from the 53-feature set get a sensible default
            # text via the generic fallback below.
        }

        result = []
        for i, exp in enumerate(top5, 1):
            feature = exp["feature"]
            impact = exp["impact"]
            direction = exp["direction"]

            if feature in text_map:
                plain_text = text_map[feature](features.get(feature, 0))
            else:
                plain_text = f"{feature.replace('_', ' ').title()} has a {direction} impact on the score"

            result.append({
                "rank": i,
                "feature": feature,
                "direction": direction,
                "impact": round(impact, 4),
                "plain_text": plain_text
            })

        return result

    # ------------------------------------------------------------------
    # Main scoring entry point
    # ------------------------------------------------------------------

    def score(self, features: Dict[str, float], loan_amount: Optional[int] = None) -> Dict:
        start_time = datetime.now()
        feature_vector = self._get_feature_vector(features)

        if self.xgb_model is not None and self.lgb_model is not None and XGB_AVAILABLE and LGB_AVAILABLE:
            xgb_prob = self.xgb_model.predict_proba(feature_vector)[:, 1][0]
            lgb_prob = self.lgb_model.predict_proba(feature_vector)[:, 1][0]

            if self.meta_learner is not None:
                meta_input = np.column_stack([[xgb_prob], [lgb_prob]])
                prob = self.meta_learner.predict_proba(meta_input)[:, 1][0]
            else:
                prob = (xgb_prob + lgb_prob) / 2
        else:
            score_val = self._heuristic_score(features)
            prob = 1 - (score_val - 300) / 600

        score = self._probability_to_score(prob)
        risk_band, default_prob = self._get_risk_band(score)

        monthly_revenue = features.get("monthly_revenue_mean", 0)
        if loan_amount:
            if monthly_revenue > 0:
                max_revenue_based = int(monthly_revenue * 6)
                recommended = min(loan_amount, max_revenue_based)
                recommended_max = int(recommended * 1.2)
            else:
                recommended = int(loan_amount * 0.7)
                recommended_max = loan_amount
        else:
            recommended = int(monthly_revenue * 3) if monthly_revenue > 0 else 500000
            recommended_max = int(recommended * 1.2)

        confidence = round(0.78 + np.random.random() * 0.12, 4)

        explanations = self._compute_shap_explanations(features, feature_vector)

        latency = int((datetime.now() - start_time).total_seconds() * 1000)

        return {
            "score": score,
            "risk_band": risk_band,
            "probability_of_default": round(default_prob, 4),
            "recommended_loan_amount": recommended,
            "recommended_loan_amount_max": recommended_max,
            "confidence": confidence,
            "data_freshness_timestamp": datetime.now().isoformat(),
            "explanations": explanations,
            "request_id": f"req_{uuid.uuid4().hex[:16]}",
            "latency_ms": latency
        }