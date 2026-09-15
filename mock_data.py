"""Mock KYC / verification records backing the ``check_verification_status`` tool.

This stands in for a real KYC database. Keys are normalised full names
(lowercase, single-spaced). Unknown names return a not-found payload so the
agent can say it has no record instead of inventing one.
"""

from __future__ import annotations

from typing import Any

MOCK_CUSTOMERS: dict[str, dict[str, Any]] = {
    "priya sharma": {
        "customer_name": "Priya Sharma",
        "pan_verified": True,
        "bank_verified": True,
        "selfie_uploaded": False,
    },
    "rahul mehta": {
        "customer_name": "Rahul Mehta",
        "pan_verified": True,
        "bank_verified": False,
        "selfie_uploaded": True,
    },
    "anita desai": {
        "customer_name": "Anita Desai",
        "pan_verified": False,
        "bank_verified": False,
        "selfie_uploaded": False,
    },
    "john doe": {
        "customer_name": "John Doe",
        "pan_verified": True,
        "bank_verified": True,
        "selfie_uploaded": True,
    },
    "meera iyer": {
        "customer_name": "Meera Iyer",
        "pan_verified": True,
        "bank_verified": False,
        "selfie_uploaded": False,
    },
}

_NOT_FOUND: dict[str, Any] = {
    "found": False,
    "message": "No verification record found for that name.",
}


def _normalise(name: str) -> str:
    return " ".join((name or "").split()).lower()


def lookup_verification_status(customer_name: str) -> dict[str, Any]:
    """Return a mock verification record for ``customer_name``.

    Matching is done on the normalised full name first, then on a unique
    first-name match, so the tool still works when the customer only gives
    their first name (a common case in voice conversations).
    """
    key = _normalise(customer_name)
    if not key:
        return {**_NOT_FOUND, "customer_name": customer_name}

    record = MOCK_CUSTOMERS.get(key)
    if record is not None:
        return {"found": True, **record}

    first_name_matches = [
        value
        for stored_key, value in MOCK_CUSTOMERS.items()
        if stored_key.split()[0] == key.split()[0]
    ]
    if len(first_name_matches) == 1:
        return {"found": True, **first_name_matches[0]}

    return {**_NOT_FOUND, "customer_name": customer_name}
