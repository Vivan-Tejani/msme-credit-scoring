import random
import numpy as np
import pandas as pd
from typing import Dict, List, Any
from datetime import datetime, timedelta
from faker import Faker

faker = Faker("en_IN")


class MSMEDataGenerator:
    """
    Generates synthetic MSME financial data where stress_level (0=healthy, 1=distressed)
    flows strongly into 15+ measurable signals across GST invoices, UPI transactions,
    e-way bills, and MCA filings.

    Design principle: stressed businesses don't just look "slightly worse" — they show
    clear, measurable deterioration that feature engineering can detect and models can learn.
    """

    def __init__(self, gstin: str, seed: int = 42):
        self.gstin = gstin
        self.seed = seed
        self.rng = np.random.RandomState(seed)
        random.seed(seed)
        faker.seed_instance(seed)

        self.business_profile = self._generate_business_profile()
        self.counterparties = self._generate_counterparties()
        self._cached_invoices: pd.DataFrame | None = None

    # ------------------------------------------------------------------
    # Business profile — stress drives everything downstream
    # ------------------------------------------------------------------

    def _generate_business_profile(self) -> Dict:
        age_months = self.rng.randint(6, 120)
        base_monthly_revenue = self.rng.lognormal(13, 1.0)

        # Beta(1.5, 4) → most businesses healthy, ~25% meaningfully stressed
        stress_level = float(np.clip(self.rng.beta(1.5, 4.0), 0.0, 1.0))

        # Stressed businesses have fewer customers (concentration goes up)
        if stress_level > 0.7:
            customer_count = self.rng.randint(3, 15)       # very few clients
        elif stress_level > 0.4:
            customer_count = self.rng.randint(10, 50)
        else:
            customer_count = self.rng.randint(30, 200)     # healthy = diverse

        return {
            "age_months": age_months,
            "base_monthly_revenue": base_monthly_revenue,
            "customer_count": customer_count,
            "industry": random.choice(
                ["manufacturing", "retail", "wholesale", "services", "trading"]
            ),
            "seasonal_peak_month": self.rng.randint(0, 11),
            "seasonal_strength": self.rng.uniform(0.85, 1.3),
            "stress_level": stress_level,
        }

    def _generate_counterparties(self) -> List[Dict]:
        stress = self.business_profile["stress_level"]
        count = min(self.business_profile["customer_count"], 50)
        count = max(count, 2)

        counterparties = []
        for i in range(count):
            cp_gstin = (
                f"{self.rng.randint(10, 99)}"
                f"{''.join(faker.random_uppercase_letter() for _ in range(5))}"
                f"{self.rng.randint(1000, 9999)}"
                f"{faker.random_uppercase_letter()}1Z"
                f"{faker.random_uppercase_letter()}{faker.random_digit()}"
            )

            # Stressed businesses: 1-2 customers dominate (high HHI)
            # Healthy businesses: weights spread evenly (low HHI)
            if stress > 0.6 and i == 0:
                weight = self.rng.uniform(15.0, 30.0)   # one giant customer
            elif stress > 0.6 and i == 1:
                weight = self.rng.uniform(5.0, 10.0)    # one secondary
            elif stress > 0.6:
                weight = self.rng.uniform(0.01, 0.3)    
            elif stress > 0.35:
                weight = self.rng.exponential(2.0)
            else:
                weight = self.rng.exponential(0.8)      

            counterparties.append({
                "gstin": cp_gstin,
                "name": faker.company(),
                "type": random.choice(["b2b", "b2c", "export"]),
                "weight": max(weight, 0.001),
            })

        total = sum(c["weight"] for c in counterparties)
        for c in counterparties:
            c["weight"] /= total
        return counterparties

    def generate_gst_invoices(self, months: int = 12) -> pd.DataFrame:
        if self._cached_invoices is not None:
            return self._cached_invoices

        stress = self.business_profile["stress_level"]
        invoices = []
        end_date = datetime.now().replace(day=1)

        for m in range(months):
            month_start = end_date - timedelta(days=30 * m)

            seasonal_mult = 1.0 + (
                self.business_profile["seasonal_strength"] - 1.0
            ) * np.cos(
                2 * np.pi
                * (month_start.month - self.business_profile["seasonal_peak_month"])
                / 12
            )

            if m < 3:
                stress_mult = 1.0 - stress * 0.70
            elif m < 6:
                stress_mult = 1.0 - stress * 0.40
            elif m < 9:
                stress_mult = 1.0 - stress * 0.15
            else:
                stress_mult = 1.0  

            monthly_revenue = (
                self.business_profile["base_monthly_revenue"]
                * seasonal_mult
                * stress_mult
                * self.rng.lognormal(0, 0.12)  
            )
            monthly_revenue = max(monthly_revenue, 10_000)

            n_invoices = max(2, int(monthly_revenue / self.rng.uniform(80_000, 600_000)))

            if stress > 0.5 and m < 3:
                n_invoices = max(1, int(n_invoices * (1.0 - stress * 0.6)))

            for _ in range(n_invoices):
                counterparty = random.choices(
                    self.counterparties,
                    weights=[c["weight"] for c in self.counterparties],
                )[0]
                invoice_date = month_start + timedelta(
                    days=self.rng.randint(0, 28)
                )
                amount = self.rng.lognormal(11.5, 0.9)
                amount = min(max(amount, 5_000), 5_000_000)

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

        self._cached_invoices = pd.DataFrame(invoices)
        return self._cached_invoices

    def generate_upi_transactions(self, days: int = 90) -> pd.DataFrame:
        stress = self.business_profile["stress_level"]
        transactions = []
        end_date = datetime.now()

        base_daily_txns = self.rng.randint(15, 300)

        inflow_prob = 0.62 - stress * 0.34

        if stress > 0.7:
            vpa_pool_size = self.rng.randint(2, 4)
        elif stress > 0.4:
            vpa_pool_size = self.rng.randint(5, 15)
        else:
            vpa_pool_size = self.rng.randint(25, 60)

        for d in range(days):
            tx_date = end_date - timedelta(days=d)
            day_mult = 1.0 if tx_date.weekday() < 5 else 0.35
            recency = d / days   
            volume_mult = 1.0 - stress * (1.0 - recency) * 0.55

            n_txns = max(
                1,
                int(base_daily_txns * day_mult * volume_mult * self.rng.lognormal(0, 0.25))
            )

            for _ in range(n_txns):
                amount = self.rng.lognormal(8.5, 0.7)
                amount = float(np.clip(amount, 500, 200_000))

                is_inflow = self.rng.random() < inflow_prob

                vpa_index = self.rng.randint(0, vpa_pool_size)
                suffix = random.choice(["ybl", "okaxis", "okhdfcbank", "paytm"])
                vpa = f"merchant{vpa_index:03d}@{suffix}"

                transactions.append({
                    "gstin": self.gstin,
                    "txn_id": f"UPI{tx_date.strftime('%Y%m%d')}{self.rng.randint(100000, 999999)}",
                    "date": tx_date.strftime("%Y-%m-%d"),
                    "time": f"{self.rng.randint(0, 23):02d}:{self.rng.randint(0, 59):02d}",
                    "amount": round(amount, 2),
                    "type": "credit" if is_inflow else "debit",
                    "counterparty_vpa": vpa,
                    "counterparty_name": faker.name() if not is_inflow else faker.company(),
                    "narration": random.choice(
                        ["PAYMENT", "SETTLEMENT", "INVOICE", "REFUND"]
                    ),
                })

        return pd.DataFrame(transactions)

    def generate_eway_bills(self, months: int = 12) -> pd.DataFrame:
        invoices = self.generate_gst_invoices(months)
        stress = self.business_profile["stress_level"]
        eway_bills = []

        all_states = ["27", "29", "33", "07", "09", "19", "24", "06"]
        gstin_state = self.gstin[:2]
        other_states = [s for s in all_states if s != gstin_state] or all_states

        eway_probability = 0.65 - stress * 0.40

        for _, inv in invoices.iterrows():
            if self.rng.random() < eway_probability:
                to_state = self.rng.choice(other_states)
                eway_bills.append({
                    "gstin": self.gstin,
                    "eway_bill_id": f"EB{inv['invoice_id'][3:]}",
                    "date": inv["date"],
                    "invoice_ref": inv["invoice_id"],
                    "value": round(float(inv["total_amount"]) * self.rng.uniform(0.88, 1.0), 2),
                    "distance_km": self.rng.randint(10, 2000),
                    "from_state": gstin_state,
                    "to_state": to_state,
                    "vehicle_type": random.choice(["road", "rail", "air"]),
                })

        return pd.DataFrame(eway_bills)

    def generate_mca_filings(self) -> Dict:
        age = self.business_profile["age_months"]
        stress = self.business_profile["stress_level"]
        filings = []

        n_years = max(1, age // 12)
        for i in range(min(n_years, 6)):
            fy = datetime.now().year - i

            # Clear stress-tiered filing compliance
            if stress > 0.75:
                choices, weights = ["filed", "delayed", "not_filed"], [0.20, 0.35, 0.45]
            elif stress > 0.55:
                choices, weights = ["filed", "delayed", "not_filed"], [0.40, 0.35, 0.25]
            elif stress > 0.35:
                choices, weights = ["filed", "delayed", "not_filed"], [0.65, 0.25, 0.10]
            elif stress > 0.15:
                choices, weights = ["filed", "delayed"],              [0.85, 0.15]
            else:
                choices, weights = ["filed", "delayed"],              [0.96, 0.04]

            filings.append({
                "type": "annual_return",
                "fy": f"{fy}-{fy + 1}",
                "date": f"{fy + 1}-10-{self.rng.randint(1, 28):02d}",
                "status": random.choices(choices, weights=weights)[0],
            })

        directors = []
        for _ in range(self.rng.randint(1, 4)):
            max_days = max(age * 30, 181)
            appointed_days = int(self.rng.randint(180, max_days))
            directors.append({
                "name": faker.name(),
                "din": str(self.rng.randint(10_000_000, 99_999_999)),
                "appointed_date": (
                    datetime.now() - timedelta(days=appointed_days)
                ).strftime("%Y-%m-%d"),
                "other_companies": int(self.rng.randint(0, 5)),
            })

        return {
            "gstin": self.gstin,
            "business_age_months": int(age),
            "legal_structure": random.choice(
                ["proprietorship", "partnership", "private_limited", "llp"]
            ),
            "registered_office_changes": int(self.rng.randint(0, 3)),
            "charges_created": int(self.rng.randint(0, 5)),
            "charges_satisfied": int(max(0, self.rng.randint(-2, 5))),
            "filings": filings,
            "directors": directors,
            "stress_level": stress,
        }

    def generate_all(self) -> Dict[str, Any]:
        return {
            "gstin": self.gstin,
            "gst_invoices": self.generate_gst_invoices(),
            "upi_transactions": self.generate_upi_transactions(),
            "eway_bills": self.generate_eway_bills(),
            "mca_data": self.generate_mca_filings(),
        }