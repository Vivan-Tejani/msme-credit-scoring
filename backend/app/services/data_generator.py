import random
import numpy as np
import pandas as pd
from typing import Dict, List, Any
from datetime import datetime, timedelta
from faker import Faker

faker = Faker("en_IN")


class MSMEDataGenerator:
    def __init__(self, gstin: str, seed: int = 42):
        self.gstin = gstin
        self.seed = seed
        self.rng = np.random.RandomState(seed)
        random.seed(seed)
        faker.seed_instance(seed)
        
        self.business_profile = self._generate_business_profile()
        self.counterparties = self._generate_counterparties()
        
    def _generate_business_profile(self) -> Dict:
        age_months = self.rng.randint(6, 120)
        base_monthly_revenue = self.rng.lognormal(13, 1.2)
        return {
            "age_months": age_months,
            "base_monthly_revenue": base_monthly_revenue,
            "customer_count": self.rng.randint(5, 200),
            "industry": random.choice(["manufacturing", "retail", "wholesale", "services", "trading"]),
            "seasonal_peak_month": self.rng.randint(0, 11),
            "seasonal_strength": self.rng.uniform(0.8, 1.5),
        }
    
    def _generate_counterparties(self) -> List[Dict]:
        count = min(self.business_profile["customer_count"], 50)
        counterparties = []
        for _ in range(count):
            cp_gstin = (
                f"{self.rng.randint(10, 99)}"
                f"{''.join(faker.random_uppercase_letter() for _ in range(5))}"
                f"{self.rng.randint(1000, 9999)}"
                f"{faker.random_uppercase_letter()}1Z"
                f"{faker.random_uppercase_letter()}{faker.random_digit()}"
            )
            counterparties.append({
                "gstin": cp_gstin,
                "name": faker.company(),
                "type": random.choice(["b2b", "b2c", "export"]),
                "weight": self.rng.exponential(1.0),
            })
        total_weight = sum(c["weight"] for c in counterparties)
        for c in counterparties:
            c["weight"] /= total_weight
        return counterparties
    
    def generate_gst_invoices(self, months: int = 12) -> pd.DataFrame:
        invoices = []
        end_date = datetime.now().replace(day=1)
        
        for m in range(months):
            month_start = end_date - timedelta(days=30 * m)
            days_in_month = 30
            
            seasonal_mult = 1.0 + (self.business_profile["seasonal_strength"] - 1.0) * np.cos(
                2 * np.pi * (month_start.month - self.business_profile["seasonal_peak_month"]) / 12
            )
            
            monthly_revenue = self.business_profile["base_monthly_revenue"] * seasonal_mult * self.rng.lognormal(0, 0.2)
            n_invoices = max(5, int(monthly_revenue / self.rng.uniform(50000, 500000)))
            
            for _ in range(n_invoices):
                counterparty = random.choices(self.counterparties, weights=[c["weight"] for c in self.counterparties])[0]
                invoice_date = month_start + timedelta(days=self.rng.randint(0, days_in_month - 1))
                
                amount = self.rng.lognormal(12, 0.8)
                if amount > 1_000_000:
                    amount = self.rng.uniform(1_000_000, 5_000_000)
                
                invoices.append({
                    "gstin": self.gstin,
                    "invoice_id": f"INV-{invoice_date.strftime('%Y%m')}-{self.rng.randint(1000, 9999)}",
                    "date": invoice_date.strftime("%Y-%m-%d"),
                    "counterparty_gstin": counterparty["gstin"],
                    "counterparty_name": counterparty["name"],
                    "amount": round(amount, 2),
                    "type": counterparty["type"],
                    "tax_amount": round(amount * 0.18, 2),
                    "total_amount": round(amount * 1.18, 2),
                })
        
        return pd.DataFrame(invoices)
    
    def generate_upi_transactions(self, days: int = 90) -> pd.DataFrame:
        transactions = []
        end_date = datetime.now()
        
        base_daily_txns = self.rng.randint(10, 500)
        base_daily_volume = self.business_profile["base_monthly_revenue"] / 30
        
        for d in range(days):
            tx_date = end_date - timedelta(days=d)
            day_mult = 1.0 if tx_date.weekday() < 5 else 0.4
            
            n_txns = max(1, int(base_daily_txns * day_mult * self.rng.lognormal(0, 0.3)))
            daily_target = base_daily_volume * day_mult * self.rng.lognormal(0, 0.25)
            
            for _ in range(n_txns):
                amount = self.rng.lognormal(8.5, 0.6)
                amount = min(max(amount, 500), 200_000)
                
                is_inflow = self.rng.random() < 0.55
                
                transactions.append({
                    "gstin": self.gstin,
                    "txn_id": f"UPI{tx_date.strftime('%Y%m%d')}{self.rng.randint(100000, 999999)}",
                    "date": tx_date.strftime("%Y-%m-%d"),
                    "time": f"{self.rng.randint(0, 23):02d}:{self.rng.randint(0, 59):02d}",
                    "amount": round(amount, 2),
                    "type": "credit" if is_inflow else "debit",
                    "counterparty_vpa": f"{''.join(faker.random_lowercase_letter() for _ in range(8))}@{random.choice(['ybl', 'okaxis', 'okhdfcbank', 'paytm'])}",
                    "counterparty_name": faker.name() if not is_inflow else faker.company(),
                    "narration": random.choice(["PAYMENT", "SETTLEMENT", "INVOICE", "REFUND"]),
                })
        
        return pd.DataFrame(transactions)
    
    def generate_eway_bills(self, months: int = 12) -> pd.DataFrame:
        invoices = self.generate_gst_invoices(months)
        eway_bills = []
        
        all_states = ["27", "29", "33", "07", "09", "19"]
        other_states = [s for s in all_states if s != self.gstin[:2]]
        
        if len(other_states) == 0:
            other_states = all_states
        
        for _, inv in invoices.iterrows():
            if self.rng.random() < 0.5:
                to_state = self.rng.choice(other_states)
                
                eway_bills.append({
                    "gstin": self.gstin,
                    "eway_bill_id": f"EB{inv['invoice_id'][3:]}",
                    "date": inv["date"],
                    "invoice_ref": inv["invoice_id"],
                    "value": round(inv["total_amount"] * self.rng.uniform(0.9, 1.0), 2),
                    "distance_km": self.rng.randint(10, 2000),
                    "from_state": self.gstin[:2],
                    "to_state": to_state,
                    "vehicle_type": random.choice(["road", "rail", "air"]),
                })
        
        return pd.DataFrame(eway_bills)
    
    def generate_mca_filings(self) -> Dict:
        age = self.business_profile["age_months"]
        filings = []
        
        for year in range(max(0, age // 12 - 5), age // 12 + 1):
            fy = datetime.now().year - (age // 12 - year)
            filings.append({
                "type": "annual_return",
                "fy": f"{fy}-{fy+1}",
                "date": f"{fy+1}-10-{self.rng.randint(1, 31)}",
                "status": random.choices(["filed", "filed", "filed", "delayed"], weights=[0.7, 0.15, 0.1, 0.05])[0],
            })
        
        directors = []
        for i in range(self.rng.randint(1, 4)):
            directors.append({
                "name": faker.name(),
                "din": f"{self.rng.randint(10000000, 99999999)}",
                "appointed_date": (datetime.now() - timedelta(days=self.rng.randint(180, age * 30))).strftime("%Y-%m-%d"),
                "other_companies": self.rng.randint(0, 5),
            })
        
        return {
            "gstin": self.gstin,
            "business_age_months": age,
            "legal_structure": random.choice(["proprietorship", "partnership", "private_limited", "llp"]),
            "registered_office_changes": self.rng.randint(0, 3),
            "charges_created": self.rng.randint(0, 5),
            "charges_satisfied": max(0, self.rng.randint(-2, 5)),
            "filings": filings,
            "directors": directors,
        }
    
    def generate_all(self) -> Dict[str, Any]:
        return {
            "gstin": self.gstin,
            "gst_invoices": self.generate_gst_invoices(),
            "upi_transactions": self.generate_upi_transactions(),
            "eway_bills": self.generate_eway_bills(),
            "mca_data": self.generate_mca_filings(),
        }