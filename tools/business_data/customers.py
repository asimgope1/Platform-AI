from typing import Dict, Any


async def get_customer_info(customer_id: str) -> Dict[str, Any]:
    """Retrieves safe non-sensitive summary information about a customer account."""
    mock_customers = {
        "cust_1": {"name": "Acme Corp", "tier": "Enterprise", "active_seats": 250},
        "cust_2": {"name": "Starlight Labs", "tier": "Startup", "active_seats": 15},
    }
    return mock_customers.get(
        customer_id,
        {"customer_id": customer_id, "name": "General Business Partner", "tier": "Standard", "active_seats": 5},
    )
