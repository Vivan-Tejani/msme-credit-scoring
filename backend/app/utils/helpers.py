import re
from typing import Optional


def validate_gstin(gstin: str) -> bool:
    pattern = r"^[0-9]{2}[A-Z]{5}[0-9]{4}[A-Z]{1}[1-9A-Z]{1}Z[0-9A-Z]{1}$"
    return bool(re.match(pattern, gstin))


def get_risk_band(score: int) -> str:
    if score >= 750:
        return "LOW"
    elif score >= 650:
        return "MEDIUM"
    elif score >= 500:
        return "HIGH"
    else:
        return "VERY HIGH"


def format_currency(amount: float) -> str:
    if amount >= 1_00_00_000:
        return f"₹{amount/1_00_00_000:.2f} Cr"
    elif amount >= 1_00_000:
        return f"₹{amount/1_00_000:.2f} L"
    else:
        return f"₹{amount:,.0f}"