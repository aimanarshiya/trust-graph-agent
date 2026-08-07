"""
debounce_client.py
---------------------
Live disposable-email check. Flags throwaway signup emails --
a real identity-risk signal. Free tier, no API key required for
the basic disposable-check endpoint.
"""

import requests

BASE_URL = "https://disposable.debounce.io/"


def check_email(email: str) -> dict:
    """
    Returns whether the email domain is a known disposable/throwaway
    provider. Fails safe (assumes not disposable) on any error.
    """
    try:
        resp = requests.get(BASE_URL, params={"email": email}, timeout=5)
        resp.raise_for_status()
        data = resp.json()
        return {
            "email": email,
            "disposable": data.get("disposable") == "true",
        }
    except Exception as e:
        return {"email": email, "disposable": False, "error": str(e)}


if __name__ == "__main__":
    print(check_email("test@mailinator.com"))   # should flag disposable
    print(check_email("test@gmail.com"))         # should not flag