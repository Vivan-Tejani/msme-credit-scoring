import numpy as np
import pandas as pd
from typing import Dict, Any
from datetime import datetime
import math

class FeatureEngineer:
    def __init__(self):
        self.feature_names = []

    def compute_features(self, data: Dict[str, Any]) -> Dict[str, float]:
        invoices = data.get("gst_invoices", pd.DataFrame())
        upi      = data.get("upi_transactions", pd.DataFrame())
        eway     = data.get("eway_bills", pd.DataFrame())
        mca      = data.get("mca_data", {})

        features = {}
        features.update(self._revenue_features(invoices))
        features.update(self._cash_flow_features(upi))
        features.update(self._supply_chain_features(invoices, eway))
        features.update(self._compliance_features(mca))
        features.update(self._fraud_signal_features(invoices, upi, mca))
        return features

    def _revenue_features(self, invoices: pd.DataFrame) -> Dict[str, float]:
        empty = {
            "monthly_revenue_mean": 0.0, "monthly_revenue_std": 0.0,
            "revenue_growth_3m": 0.0, "revenue_growth_6m": 0.0,
            "revenue_volatility": 0.0,
            "invoice_regularity_score": 0.0,
            "customer_concentration_hhi": 1.0,
            "top1_customer_share": 1.0,
            "avg_invoice_size": 0.0, "invoice_size_cv": 0.0,
            "large_invoice_ratio": 0.0,
            "b2b_ratio": 0.0, "b2c_ratio": 0.0, "export_invoice_ratio": 0.0,
            "gst_filing_timeliness_score": 0.0, "filing_streak": 0.0,
            "recent_invoice_count_3m": 0.0, "invoice_count_drop_ratio": 0.0,
        }
        if invoices.empty:
            return empty

        inv = invoices.copy()
        inv["date"] = pd.to_datetime(inv["date"])
        inv = inv.sort_values("date")

        monthly = (
            inv.groupby(inv["date"].dt.to_period("M"))["total_amount"]
            .sum().reset_index()
        )
        monthly["date"] = monthly["date"].dt.to_timestamp()
        monthly = monthly.sort_values("date").reset_index(drop=True)

        monthly_revenue_mean = float(monthly["total_amount"].mean())
        monthly_revenue_std  = float(monthly["total_amount"].std()) if len(monthly) > 1 else 0.0
        revenue_volatility   = monthly_revenue_std / (monthly_revenue_mean + 1)

        if len(monthly) >= 6:
            last3 = monthly["total_amount"].iloc[-3:].mean()
            prev3 = monthly["total_amount"].iloc[-6:-3].mean()
            revenue_growth_3m = ((last3 - prev3) / (prev3 + 1)) * 100
        else:
            revenue_growth_3m = 0.0

        if len(monthly) >= 12:
            last6 = monthly["total_amount"].iloc[-6:].mean()
            prev6 = monthly["total_amount"].iloc[-12:-6].mean()
            revenue_growth_6m = ((last6 - prev6) / (prev6 + 1)) * 100
        else:
            revenue_growth_6m = 0.0

        # Invoice regularity (entropy of inter-arrival intervals)
        intervals = inv["date"].diff().dt.days.dropna()
        if len(intervals) > 1 and intervals.sum() > 0:
            p = intervals.value_counts(normalize=True)
            entropy = -sum(v * math.log(v) for v in p if v > 0)
            max_e = math.log(len(p)) if len(p) > 1 else 1.0
            invoice_regularity_score = entropy / max_e if max_e > 0 else 0.0
        else:
            invoice_regularity_score = 0.0

        ctotals = inv.groupby("counterparty_gstin")["total_amount"].sum()
        total = ctotals.sum()
        if total > 0:
            shares = ctotals / total
            customer_concentration_hhi = float((shares ** 2).sum())
            top1_customer_share = float(shares.max())
        else:
            customer_concentration_hhi = 1.0
            top1_customer_share = 1.0

        avg_invoice_size = float(inv["total_amount"].mean())
        invoice_size_cv  = float(inv["total_amount"].std() / avg_invoice_size) if avg_invoice_size > 0 else 0.0
        large_invoice_ratio = float((inv["total_amount"] > 1_000_000).mean())

        tc = inv["type"].value_counts(normalize=True)
        b2b_ratio   = float(tc.get("b2b", 0.0))
        b2c_ratio   = float(tc.get("b2c", 0.0))
        export_ratio = float(tc.get("export", 0.0))

        monthly_counts = inv.groupby(inv["date"].dt.to_period("M")).size()
        filing_streak = 0
        for cnt in reversed(monthly_counts.values):
            if cnt >= 3:
                filing_streak += 1
            else:
                break
        gst_filing_timeliness_score = min(1.0, filing_streak / 12.0)

        recent_inv = inv[inv["date"] >= (inv["date"].max() - pd.Timedelta(days=90))]
        prior_inv  = inv[
            (inv["date"] < (inv["date"].max() - pd.Timedelta(days=90))) &
            (inv["date"] >= (inv["date"].max() - pd.Timedelta(days=180)))
        ]
        recent_count = len(recent_inv)
        prior_count  = len(prior_inv)
        invoice_count_drop_ratio = (
            (prior_count - recent_count) / (prior_count + 1)
        ) if prior_count > 0 else 0.0

        return {
            "monthly_revenue_mean": round(monthly_revenue_mean, 2),
            "monthly_revenue_std":  round(monthly_revenue_std, 2),
            "revenue_growth_3m":    round(revenue_growth_3m, 4),
            "revenue_growth_6m":    round(revenue_growth_6m, 4),
            "revenue_volatility":   round(revenue_volatility, 4),
            "invoice_regularity_score":   round(invoice_regularity_score, 4),
            "customer_concentration_hhi": round(customer_concentration_hhi, 4),
            "top1_customer_share":        round(top1_customer_share, 4),
            "avg_invoice_size":    round(avg_invoice_size, 2),
            "invoice_size_cv":     round(invoice_size_cv, 4),
            "large_invoice_ratio": round(large_invoice_ratio, 4),
            "b2b_ratio":           round(b2b_ratio, 4),
            "b2c_ratio":           round(b2c_ratio, 4),
            "export_invoice_ratio":round(export_ratio, 4),
            "gst_filing_timeliness_score": round(gst_filing_timeliness_score, 4),
            "filing_streak":               round(float(filing_streak), 1),
            "recent_invoice_count_3m":     round(float(recent_count), 1),
            "invoice_count_drop_ratio":    round(invoice_count_drop_ratio, 4),
        }

    def _cash_flow_features(self, upi: pd.DataFrame) -> Dict[str, float]:
        empty = {
            "daily_upi_volume_mean": 0.0, "daily_upi_volume_std": 0.0,
            "upi_volume_trend_30d": 0.0, "upi_volume_trend_60d": 0.0,
            "cash_flow_regularity": 0.0,
            "working_capital_ratio": 1.0, "working_capital_trend": 0.0,
            "inflow_ratio": 0.5,
            "peak_to_trough_ratio": 1.0,
            "weekend_activity_ratio": 0.0, "after_hours_ratio": 0.0,
            "upi_counterparty_diversity": 0.0,
            "repeat_counterparty_ratio": 0.0,
            "avg_receivable_days": 30.0, "avg_payable_days": 25.0,
        }
        if upi.empty:
            return empty

        u = upi.copy()
        u["date"] = pd.to_datetime(u["date"])
        u = u.sort_values("date")

        daily = u.groupby("date")["amount"].sum().reset_index().sort_values("date")

        daily_vol_mean = float(daily["amount"].mean())
        daily_vol_std  = float(daily["amount"].std()) if len(daily) > 1 else 0.0

        if len(daily) >= 30:
            last30 = daily["amount"].iloc[-30:].mean()
            prev30 = daily["amount"].iloc[-60:-30].mean() if len(daily) >= 60 else daily["amount"].iloc[:30].mean()
            upi_volume_trend_30d = ((last30 - prev30) / (prev30 + 1)) * 100
        else:
            upi_volume_trend_30d = 0.0

        if len(daily) >= 60:
            last60 = daily["amount"].iloc[-60:].mean()
            prev60 = daily["amount"].iloc[:-60].mean() if len(daily) > 60 else daily["amount"].mean()
            upi_volume_trend_60d = ((last60 - prev60) / (prev60 + 1)) * 100
        else:
            upi_volume_trend_60d = 0.0

        autocorr = daily["amount"].autocorr(lag=1) if len(daily) > 1 else 0.0
        cash_flow_regularity = float(autocorr) if not pd.isna(autocorr) else 0.0

        credits = float(u[u["type"] == "credit"]["amount"].sum())
        debits  = float(u[u["type"] == "debit"]["amount"].sum())
        working_capital_ratio = credits / (debits + 1)

        inflow_ratio = float((u["type"] == "credit").mean())

        mid = u["date"].median()
        early  = u[u["date"] <= mid]
        recent = u[u["date"] >  mid]
        early_wc  = early[early["type"]   == "credit"]["amount"].sum() / (early[early["type"]   == "debit"]["amount"].sum() + 1)
        recent_wc = recent[recent["type"] == "credit"]["amount"].sum() / (recent[recent["type"] == "debit"]["amount"].sum() + 1)
        working_capital_trend = float(recent_wc - early_wc)

        rolling = daily["amount"].rolling(30, min_periods=1).sum()
        peak_to_trough = float(rolling.max() / (rolling.min() + 1))

        u["weekday"] = u["date"].dt.weekday
        weekend_ratio = float((u["weekday"] >= 5).mean())

        u["hour"] = u["time"].apply(lambda x: int(x.split(":")[0]) if isinstance(x, str) else 12)
        after_hours_ratio = float(((u["hour"] < 9) | (u["hour"] > 18)).mean())

        n_unique_vpa = u["counterparty_vpa"].nunique()
        upi_counterparty_diversity = min(1.0, n_unique_vpa / 50.0)

        vpa_counts = u["counterparty_vpa"].value_counts()
        repeat_ratio = float((vpa_counts > 1).sum() / len(vpa_counts)) if len(vpa_counts) > 0 else 0.0

        credit_dates = u[u["type"] == "credit"]["date"].sort_values()
        debit_dates  = u[u["type"] == "debit"]["date"].sort_values()
        avg_recv = float(credit_dates.diff().dt.days.dropna().mean()) if len(credit_dates) > 1 else 30.0
        avg_pay  = float(debit_dates.diff().dt.days.dropna().mean())  if len(debit_dates)  > 1 else 25.0

        return {
            "daily_upi_volume_mean": round(daily_vol_mean, 2),
            "daily_upi_volume_std":  round(daily_vol_std, 2),
            "upi_volume_trend_30d":  round(upi_volume_trend_30d, 4),
            "upi_volume_trend_60d":  round(upi_volume_trend_60d, 4),
            "cash_flow_regularity":  round(cash_flow_regularity, 4),
            "working_capital_ratio": round(working_capital_ratio, 4),
            "working_capital_trend": round(working_capital_trend, 4),
            "inflow_ratio":          round(inflow_ratio, 4),
            "peak_to_trough_ratio":  round(peak_to_trough, 4),
            "weekend_activity_ratio":round(weekend_ratio, 4),
            "after_hours_ratio":     round(after_hours_ratio, 4),
            "upi_counterparty_diversity": round(upi_counterparty_diversity, 4),
            "repeat_counterparty_ratio":  round(repeat_ratio, 4),
            "avg_receivable_days":   round(avg_recv, 2),
            "avg_payable_days":      round(avg_pay, 2),
        }

    def _supply_chain_features(self, invoices: pd.DataFrame, eway: pd.DataFrame) -> Dict[str, float]:
        empty = {
            "eway_monthly_count": 0.0, "eway_monthly_value": 0.0,
            "eway_value_growth_3m": 0.0, "eway_destination_diversity": 0.0,
            "interstate_ratio": 0.0, "supply_chain_depth_score": 0.0,
            "eway_invoice_correlation": 0.0, "seasonal_eway_pattern": 0.0,
            "eway_trend_90d": 0.0,
        }
        if eway.empty:
            return empty

        e = eway.copy()
        e["date"] = pd.to_datetime(e["date"])

        me = (
            e.groupby(e["date"].dt.to_period("M"))
             .agg(count=("eway_bill_id", "count"), value=("value", "sum"))
             .reset_index()
        )
        me["date"] = me["date"].dt.to_timestamp()

        eway_monthly_count = float(me["count"].mean())
        eway_monthly_value = float(me["value"].mean())

        if len(me) >= 6:
            eway_value_growth_3m = ((me["value"].iloc[-3:].mean() - me["value"].iloc[-6:-3].mean()) / (me["value"].iloc[-6:-3].mean() + 1)) * 100
        else:
            eway_value_growth_3m = 0.0

        eway_dest_div = min(1.0, e["to_state"].nunique() / 10.0)
        interstate    = float((e["to_state"] != e["from_state"]).mean())
        depth_score   = min(1.0, e["to_state"].nunique() / 15.0)

        eway_inv_corr = 0.0
        if not invoices.empty and len(me) > 1:
            inv = invoices.copy()
            inv["date"] = pd.to_datetime(inv["date"])
            mi = inv.groupby(inv["date"].dt.to_period("M"))["total_amount"].sum().reset_index()
            mi["date"] = mi["date"].dt.to_timestamp()
            merged = pd.merge(mi, me, on="date", how="inner")
            if len(merged) > 1:
                c = merged["total_amount"].corr(merged["value"])
                eway_inv_corr = float(c) if not pd.isna(c) else 0.0

        seasonal_pattern = float(me["value"].std() / (me["value"].mean() + 1)) if len(me) >= 3 else 0.0
        eway_trend_90d   = float((me["value"].iloc[-1] - me["value"].iloc[0]) / (me["value"].iloc[0] + 1) * 100) if len(me) >= 3 else 0.0

        return {
            "eway_monthly_count":       round(eway_monthly_count, 2),
            "eway_monthly_value":       round(eway_monthly_value, 2),
            "eway_value_growth_3m":     round(eway_value_growth_3m, 4),
            "eway_destination_diversity": round(eway_dest_div, 4),
            "interstate_ratio":         round(interstate, 4),
            "supply_chain_depth_score": round(depth_score, 4),
            "eway_invoice_correlation": round(eway_inv_corr, 4),
            "seasonal_eway_pattern":    round(seasonal_pattern, 4),
            "eway_trend_90d":           round(eway_trend_90d, 4),
        }

    def _compliance_features(self, mca: Dict) -> Dict[str, float]:
        filings   = mca.get("filings", [])
        directors = mca.get("directors", [])

        if filings:
            total   = len(filings)
            filed   = sum(1 for f in filings if f.get("status") == "filed")
            delayed = sum(1 for f in filings if f.get("status") == "delayed")
            # not_filed = 0 score, delayed = 0.5 score, filed = 1.0 score
            mca_compliance_score = (filed * 1.0 + delayed * 0.5) / total
            not_filed_ratio      = sum(1 for f in filings if f.get("status") == "not_filed") / total
        else:
            mca_compliance_score = 0.0
            not_filed_ratio      = 0.0

        age = mca.get("business_age_months", 0)
        dir_count     = max(len(directors), 1)
        other_cos     = sum(d.get("other_companies", 0) for d in directors) / dir_count
        structure_map = {"proprietorship": 0.2, "partnership": 0.4, "llp": 0.6, "private_limited": 1.0}
        legal_score   = structure_map.get(mca.get("legal_structure", ""), 0.3)
        reg_age       = min(1.0, age / 120.0)
        office_stab   = max(0.0, 1.0 - mca.get("registered_office_changes", 0) * 0.25)

        return {
            "mca_compliance_score":      round(mca_compliance_score, 4),
            "not_filed_ratio":           round(not_filed_ratio, 4),
            "director_company_count":    round(other_cos, 2),
            "business_age_months":       round(float(age), 1),
            "legal_structure_score":     round(legal_score, 4),
            "gst_registration_age":      round(reg_age, 4),
            "registered_office_stability": round(office_stab, 4),
        }

    def _fraud_signal_features(self, invoices: pd.DataFrame, upi: pd.DataFrame, mca: Dict) -> Dict[str, float]:

        if not upi.empty:
            u = upi.copy()
            credit_vpa = set(u[u["type"] == "credit"]["counterparty_vpa"])
            debit_vpa  = set(u[u["type"] == "debit"]["counterparty_vpa"])
            overlap    = credit_vpa & debit_vpa
            total_vpa  = credit_vpa | debit_vpa
            circular_flow_score = len(overlap) / len(total_vpa) if total_vpa else 0.0
        else:
            circular_flow_score = 0.0

        if not invoices.empty:
            ctotals = invoices.groupby("counterparty_gstin")["total_amount"].sum()
            top_ratio    = float(ctotals.max() / (ctotals.sum() + 1))
            cluster_size = float(invoices["counterparty_gstin"].nunique())
        else:
            top_ratio    = 0.0
            cluster_size = 0.0

        directors = mca.get("directors", [])
        if directors:
            total_other   = sum(d.get("other_companies", 0) for d in directors)
            dir_net_dens  = min(1.0, total_other / (len(directors) * 5.0))
        else:
            dir_net_dens  = 0.0

        return {
            "circular_flow_score":        round(circular_flow_score, 4),
            "entity_cluster_size":        round(cluster_size, 1),
            "related_entity_invoice_ratio": round(top_ratio, 4),
            "director_network_density":   round(dir_net_dens, 4),
        }