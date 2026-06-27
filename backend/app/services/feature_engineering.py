import numpy as np
import pandas as pd
from typing import Dict, List, Any
from datetime import datetime, timedelta
from collections import Counter
import math


class FeatureEngineer:
    def __init__(self):
        self.feature_names = []
    
    def compute_features(self, data: Dict[str, Any]) -> Dict[str, float]:
        invoices = data.get("gst_invoices", pd.DataFrame())
        upi = data.get("upi_transactions", pd.DataFrame())
        eway = data.get("eway_bills", pd.DataFrame())
        mca = data.get("mca_data", {})
        
        features = {}
        features.update(self._revenue_features(invoices))
        features.update(self._cash_flow_features(upi))
        features.update(self._supply_chain_features(invoices, eway))
        features.update(self._compliance_features(mca))
        features.update(self._fraud_signal_features(invoices, upi, mca))
        
        return features
    
    def _revenue_features(self, invoices: pd.DataFrame) -> Dict[str, float]:
        if invoices.empty:
            return {
                "monthly_revenue_mean": 0.0,
                "monthly_revenue_std": 0.0,
                "revenue_growth_3m": 0.0,
                "revenue_growth_6m": 0.0,
                "invoice_regularity_score": 0.0,
                "customer_concentration_hhi": 1.0,
                "avg_invoice_size": 0.0,
                "invoice_size_cv": 0.0,
                "large_invoice_ratio": 0.0,
                "b2b_ratio": 0.0,
                "b2c_ratio": 0.0,
                "export_invoice_ratio": 0.0,
                "gst_filing_timeliness_score": 0.0,
                "filing_streak": 0.0,
            }
        
        invoices["date"] = pd.to_datetime(invoices["date"])
        invoices = invoices.sort_values("date")
        
        monthly = invoices.groupby(invoices["date"].dt.to_period("M"))["total_amount"].sum().reset_index()
        monthly["date"] = monthly["date"].dt.to_timestamp()
        monthly = monthly.sort_values("date")
        
        monthly_revenue_mean = monthly["total_amount"].mean()
        monthly_revenue_std = monthly["total_amount"].std() if len(monthly) > 1 else 0.0
        
        if len(monthly) >= 4:
            last_3m = monthly["total_amount"].iloc[-3:].mean()
            prev_3m = monthly["total_amount"].iloc[-6:-3].mean()
            revenue_growth_3m = ((last_3m - prev_3m) / (prev_3m + 1)) * 100 if prev_3m > 0 else 0.0
        else:
            revenue_growth_3m = 0.0
        
        if len(monthly) >= 7:
            last_6m = monthly["total_amount"].iloc[-6:].mean()
            prev_6m = monthly["total_amount"].iloc[-12:-6].mean() if len(monthly) >= 12 else monthly["total_amount"].iloc[:6].mean()
            revenue_growth_6m = ((last_6m - prev_6m) / (prev_6m + 1)) * 100 if prev_6m > 0 else 0.0
        else:
            revenue_growth_6m = 0.0
        
        intervals = invoices["date"].diff().dt.days.dropna()
        if len(intervals) > 1 and intervals.sum() > 0:
            interval_counts = intervals.value_counts(normalize=True)
            entropy = -sum(p * math.log(p) for p in interval_counts if p > 0)
            max_entropy = math.log(len(interval_counts)) if len(interval_counts) > 1 else 1
            invoice_regularity_score = entropy / max_entropy if max_entropy > 0 else 0.0
        else:
            invoice_regularity_score = 0.0
        
        customer_totals = invoices.groupby("counterparty_gstin")["total_amount"].sum()
        total = customer_totals.sum()
        if total > 0:
            shares = customer_totals / total
            customer_concentration_hhi = (shares ** 2).sum()
        else:
            customer_concentration_hhi = 1.0
        
        avg_invoice_size = invoices["total_amount"].mean()
        invoice_size_cv = invoices["total_amount"].std() / avg_invoice_size if avg_invoice_size > 0 else 0.0
        large_invoice_ratio = (invoices["total_amount"] > 1_000_000).mean()
        
        type_counts = invoices["type"].value_counts(normalize=True)
        b2b_ratio = type_counts.get("b2b", 0.0)
        b2c_ratio = type_counts.get("b2c", 0.0)
        export_invoice_ratio = type_counts.get("export", 0.0)
        
        monthly_counts = invoices.groupby(invoices["date"].dt.to_period("M")).size()
        if len(monthly_counts) > 1:
            filing_streak = 0
            for count in reversed(monthly_counts.values):
                if count >= 5:
                    filing_streak += 1
                else:
                    break
            gst_filing_timeliness_score = min(1.0, filing_streak / 12.0)
        else:
            filing_streak = 0.0
            gst_filing_timeliness_score = 0.5
        
        return {
            "monthly_revenue_mean": round(monthly_revenue_mean, 2),
            "monthly_revenue_std": round(monthly_revenue_std, 2),
            "revenue_growth_3m": round(revenue_growth_3m, 2),
            "revenue_growth_6m": round(revenue_growth_6m, 2),
            "invoice_regularity_score": round(invoice_regularity_score, 4),
            "customer_concentration_hhi": round(customer_concentration_hhi, 4),
            "avg_invoice_size": round(avg_invoice_size, 2),
            "invoice_size_cv": round(invoice_size_cv, 4),
            "large_invoice_ratio": round(large_invoice_ratio, 4),
            "b2b_ratio": round(b2b_ratio, 4),
            "b2c_ratio": round(b2c_ratio, 4),
            "export_invoice_ratio": round(export_invoice_ratio, 4),
            "gst_filing_timeliness_score": round(gst_filing_timeliness_score, 4),
            "filing_streak": round(filing_streak, 1),
        }
    
    def _cash_flow_features(self, upi: pd.DataFrame) -> Dict[str, float]:
        if upi.empty:
            return {
                "daily_upi_volume_mean": 0.0,
                "daily_upi_volume_std": 0.0,
                "upi_volume_trend_30d": 0.0,
                "cash_flow_regularity": 0.0,
                "working_capital_ratio": 1.0,
                "peak_to_trough_ratio": 1.0,
                "weekend_activity_ratio": 0.0,
                "after_hours_ratio": 0.0,
                "upi_counterparty_diversity": 0.0,
                "repeat_customer_ratio": 0.0,
                "avg_receivable_days": 0.0,
                "avg_payable_days": 0.0,
            }
        
        upi["date"] = pd.to_datetime(upi["date"])
        upi = upi.sort_values("date")
        
        daily = upi.groupby("date")["amount"].sum().reset_index()
        daily = daily.sort_values("date")
        
        daily_upi_volume_mean = daily["amount"].mean()
        daily_upi_volume_std = daily["amount"].std() if len(daily) > 1 else 0.0
        
        if len(daily) >= 30:
            last_30 = daily["amount"].iloc[-30:].mean()
            prev_30 = daily["amount"].iloc[-60:-30].mean() if len(daily) >= 60 else daily["amount"].iloc[:30].mean()
            upi_volume_trend_30d = ((last_30 - prev_30) / (prev_30 + 1)) * 100 if prev_30 > 0 else 0.0
        else:
            upi_volume_trend_30d = 0.0
        
        if len(daily) > 1:
            autocorr = daily["amount"].autocorr(lag=1)
            cash_flow_regularity = autocorr if not pd.isna(autocorr) else 0.0
        else:
            cash_flow_regularity = 0.0
        
        credits = upi[upi["type"] == "credit"]["amount"].sum()
        debits = upi[upi["type"] == "debit"]["amount"].sum()
        working_capital_ratio = credits / (debits + 1)
        
        if len(daily) >= 30:
            rolling_30 = daily["amount"].rolling(window=30, min_periods=1).sum()
            peak_to_trough_ratio = rolling_30.max() / (rolling_30.min() + 1)
        else:
            peak_to_trough_ratio = daily["amount"].max() / (daily["amount"].min() + 1)
        
        upi["weekday"] = upi["date"].dt.weekday
        weekend_txns = upi[upi["weekday"] >= 5]
        weekend_activity_ratio = len(weekend_txns) / len(upi) if len(upi) > 0 else 0.0
        
        upi["hour"] = upi["time"].apply(lambda x: int(x.split(":")[0]) if isinstance(x, str) else 12)
        after_hours = upi[(upi["hour"] < 9) | (upi["hour"] > 18)]
        after_hours_ratio = len(after_hours) / len(upi) if len(upi) > 0 else 0.0
        
        counterparty_counts = upi["counterparty_vpa"].nunique()
        upi_counterparty_diversity = min(1.0, counterparty_counts / 50.0)
        
        repeat_counts = upi["counterparty_vpa"].value_counts()
        repeat_customers = (repeat_counts > 1).sum()
        repeat_customer_ratio = repeat_customers / len(repeat_counts) if len(repeat_counts) > 0 else 0.0
        
        avg_receivable_days = 30.0
        avg_payable_days = 25.0
        
        return {
            "daily_upi_volume_mean": round(daily_upi_volume_mean, 2),
            "daily_upi_volume_std": round(daily_upi_volume_std, 2),
            "upi_volume_trend_30d": round(upi_volume_trend_30d, 2),
            "cash_flow_regularity": round(cash_flow_regularity, 4),
            "working_capital_ratio": round(working_capital_ratio, 4),
            "peak_to_trough_ratio": round(peak_to_trough_ratio, 4),
            "weekend_activity_ratio": round(weekend_activity_ratio, 4),
            "after_hours_ratio": round(after_hours_ratio, 4),
            "upi_counterparty_diversity": round(upi_counterparty_diversity, 4),
            "repeat_customer_ratio": round(repeat_customer_ratio, 4),
            "avg_receivable_days": round(avg_receivable_days, 2),
            "avg_payable_days": round(avg_payable_days, 2),
        }
    
    def _supply_chain_features(self, invoices: pd.DataFrame, eway: pd.DataFrame) -> Dict[str, float]:
        if eway.empty:
            return {
                "eway_monthly_count": 0.0,
                "eway_monthly_value": 0.0,
                "eway_value_growth_3m": 0.0,
                "eway_destination_diversity": 0.0,
                "interstate_ratio": 0.0,
                "supply_chain_depth_score": 0.0,
                "eway_invoice_correlation": 0.0,
                "seasonal_eway_pattern": 0.0,
                "eway_trend_90d": 0.0,
            }
        
        eway["date"] = pd.to_datetime(eway["date"])
        
        monthly_eway = eway.groupby(eway["date"].dt.to_period("M")).agg(
            count=("eway_bill_id", "count"),
            value=("value", "sum")
        ).reset_index()
        monthly_eway["date"] = monthly_eway["date"].dt.to_timestamp()
        
        eway_monthly_count = monthly_eway["count"].mean()
        eway_monthly_value = monthly_eway["value"].mean()
        
        if len(monthly_eway) >= 4:
            last_3m = monthly_eway["value"].iloc[-3:].mean()
            prev_3m = monthly_eway["value"].iloc[-6:-3].mean() if len(monthly_eway) >= 6 else monthly_eway["value"].iloc[:3].mean()
            eway_value_growth_3m = ((last_3m - prev_3m) / (prev_3m + 1)) * 100 if prev_3m > 0 else 0.0
        else:
            eway_value_growth_3m = 0.0
        
        eway_destination_diversity = min(1.0, eway["to_state"].nunique() / 10.0)
        interstate_ratio = (eway["to_state"] != eway["from_state"]).mean()
        
        supply_chain_depth_score = min(1.0, eway["to_state"].nunique() / 15.0)
        
        if not invoices.empty and len(monthly_eway) > 1:
            invoices["date"] = pd.to_datetime(invoices["date"])
            monthly_inv = invoices.groupby(invoices["date"].dt.to_period("M"))["total_amount"].sum().reset_index()
            monthly_inv["date"] = monthly_inv["date"].dt.to_timestamp()
            
            merged = pd.merge(monthly_inv, monthly_eway, on="date", how="inner")
            if len(merged) > 1:
                eway_invoice_correlation = merged["total_amount"].corr(merged["value"])
                eway_invoice_correlation = eway_invoice_correlation if not pd.isna(eway_invoice_correlation) else 0.0
            else:
                eway_invoice_correlation = 0.0
        else:
            eway_invoice_correlation = 0.0
        
        if len(monthly_eway) >= 3:
            seasonal_eway_pattern = monthly_eway["value"].std() / (monthly_eway["value"].mean() + 1)
        else:
            seasonal_eway_pattern = 0.0
        
        if len(monthly_eway) >= 3:
            eway_trend_90d = ((monthly_eway["value"].iloc[-1] - monthly_eway["value"].iloc[0]) / (monthly_eway["value"].iloc[0] + 1)) * 100
        else:
            eway_trend_90d = 0.0
        
        return {
            "eway_monthly_count": round(eway_monthly_count, 2),
            "eway_monthly_value": round(eway_monthly_value, 2),
            "eway_value_growth_3m": round(eway_value_growth_3m, 2),
            "eway_destination_diversity": round(eway_destination_diversity, 4),
            "interstate_ratio": round(interstate_ratio, 4),
            "supply_chain_depth_score": round(supply_chain_depth_score, 4),
            "eway_invoice_correlation": round(eway_invoice_correlation, 4),
            "seasonal_eway_pattern": round(seasonal_eway_pattern, 4),
            "eway_trend_90d": round(eway_trend_90d, 2),
        }
    
    def _compliance_features(self, mca: Dict) -> Dict[str, float]:
        filings = mca.get("filings", [])
        directors = mca.get("directors", [])
        
        if filings:
            filed_count = sum(1 for f in filings if f.get("status") == "filed")
            delayed_count = sum(1 for f in filings if f.get("status") == "delayed")
            total = len(filings)
            mca_compliance_score = filed_count / total if total > 0 else 0.0
        else:
            mca_compliance_score = 0.0
            filed_count = 0
            delayed_count = 0
        
        director_default_history = 0.0
        director_company_count = sum(d.get("other_companies", 0) for d in directors) / max(len(directors), 1)
        
        business_age_months = mca.get("business_age_months", 0)
        
        structure_scores = {
            "proprietorship": 0.2,
            "partnership": 0.4,
            "llp": 0.6,
            "private_limited": 1.0,
        }
        legal_structure_score = structure_scores.get(mca.get("legal_structure", ""), 0.3)
        
        gst_registration_age = min(1.0, business_age_months / 120.0)
        registered_office_stability = max(0.0, 1.0 - (mca.get("registered_office_changes", 0) * 0.2))
        
        return {
            "mca_compliance_score": round(mca_compliance_score, 4),
            "director_default_history": round(director_default_history, 4),
            "director_company_count": round(director_company_count, 2),
            "business_age_months": round(business_age_months, 1),
            "legal_structure_score": round(legal_structure_score, 4),
            "gst_registration_age": round(gst_registration_age, 4),
            "registered_office_stability": round(registered_office_stability, 4),
        }
    
    def _fraud_signal_features(self, invoices: pd.DataFrame, upi: pd.DataFrame, mca: Dict) -> Dict[str, float]:
        if not invoices.empty:
            customer_totals = invoices.groupby("counterparty_gstin")["total_amount"].sum()
            top_customer_ratio = customer_totals.max() / (customer_totals.sum() + 1)
            circular_flow_score = top_customer_ratio
            
            entity_cluster_size = invoices["counterparty_gstin"].nunique()
            related_entity_invoice_ratio = top_customer_ratio
        else:
            circular_flow_score = 0.0
            entity_cluster_size = 0.0
            related_entity_invoice_ratio = 0.0
        
        directors = mca.get("directors", [])
        if directors:
            total_other_companies = sum(d.get("other_companies", 0) for d in directors)
            director_network_density = min(1.0, total_other_companies / (len(directors) * 5.0))
        else:
            director_network_density = 0.0
        
        return {
            "circular_flow_score": round(circular_flow_score, 4),
            "entity_cluster_size": round(entity_cluster_size, 1),
            "related_entity_invoice_ratio": round(related_entity_invoice_ratio, 4),
            "director_network_density": round(director_network_density, 4),
        }