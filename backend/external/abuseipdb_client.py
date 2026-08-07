"""
abuseipdb_client.py
----------------------
Live IP-reputation lookup. Real device/IP risk signal -- not mocked.
Free tier: 1000 checks/day, no card required.
"""

import os
import requests
from dotenv import load_dotenv

load_dotenv()
API_KEY = os.getenv("ABUSEIPDB_API_KEY")
BASE_URL = "https://api.abuseipdb.com/api/v2/check"


def check_ip(ip_address: str, max_age_days: int = 90) -> dict:
    """
    Returns abuse confidence score (0-100) and metadata for an IP.
    On any failure (bad key, rate limit, network), returns a safe
    default rather than crashing the pipeline.
    """
    if not API_KEY:
        return {"ip": ip_address, "abuse_confidence_score": 0, "error": "no_api_key"}

    try:
        resp = requests.get(
            BASE_URL,
            headers={"Key": API_KEY, "Accept": "application/json"},
            params={"ipAddress": ip_address, "maxAgeInDays": max_age_days},
            timeout=5,
        )
        resp.raise_for_status()
        data = resp.json()["data"]
        return {
            "ip": ip_address,
            "abuse_confidence_score": data.get("abuseConfidenceScore", 0),
            "total_reports": data.get("totalReports", 0),
            "country_code": data.get("countryCode"),
            "is_tor": data.get("isTor", False),
        }
    except Exception as e:
        return {"ip": ip_address, "abuse_confidence_score": 0, "error": str(e)}


if __name__ == "__main__":
    # Quick test with a known-clean IP
    result = check_ip("8.8.8.8")
    print(result)