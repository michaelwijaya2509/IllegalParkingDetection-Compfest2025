import random

def calculate_urgency(coordinate):
    # calculate urgency score based on POI around the area
    # Scrape news with bs4

    # placeholder data
    urgency_data = {
        "urgency_score": 60,
        "adjustment_reason": "No critical factors for adjustment.",
        "reasons": [
            "Proximity to hospital",
            "Proximity to school",
            "Secondary road nearby"
        ],
        "narrative": "The parking event occurred near a hospital and a school, increasing its urgency. The closest major road is a secondary road.",
        "recommended_actions": [
            "Issue citation",
            "Dispatch traffic control"
        ],
        "confidence": "medium",
        "base_breakdown": {}
    }
    return urgency_data
