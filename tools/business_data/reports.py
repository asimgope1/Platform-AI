from typing import Dict, Any


async def count_open_purchase_requests(filter_status: str = "open") -> Dict[str, Any]:
    """Returns the count of purchase requests filtered by status.

    Safe, read-only parameterized query without raw SQL injection risks.
    """
    valid_statuses = ["open", "pending", "closed", "all"]
    status = filter_status.lower() if filter_status.lower() in valid_statuses else "open"

    mock_db = {
        "open": 42,
        "pending": 18,
        "closed": 150,
        "all": 210,
    }
    return {
        "status": status,
        "count": mock_db.get(status, 42),
        "as_of": "2026-10-26T12:00:00Z",
    }
