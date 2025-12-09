from typing import Literal

NBA_LABELS = {
    "EMI_RESTRUCTURE": ["emi कम", "emi reduce", "कम emi", "emi kam"],
    "PTP": ["कर दूंगा", "payment", "भुगतान", "pay kar", "next month", "agale mahine"],
    "EXTENSION": ["अभी पैसे नहीं", "paise nahi", "extra time", "agale mahine", "aur time"],
    "NO_RESPONSE": ["not reachable", "reachable nahi", "disconnect", "switched off"],
    "COMPLAINT_SUPPORT": ["problem", "issue", "complaint", "शिकायत", "complain"],
    "LEGAL_WARNING": ["legal", "notice", "कानूनी", "court"],
}

DEFAULT_NBA = "CUSTOMER_NOT_SURE"

SENTIMENT_LABELS = {
    "NEGATIVE": ["naraz", "paise nahi", "problem", "gussa", "angry", "complaint", "tension"],
    "POSITIVE": ["kar dunga", "sure", "ok", "ho jayega", "thank you", "thik hai"],
}
DEFAULT_SENTIMENT = "NEUTRAL"

# Example intent space (you can refine later)
INTENT_CLASSES = [
    "WILLING_TO_PAY",
    "NEEDS_EXTENSION",
    "EMI_REDUCTION_REQUEST",
    "DISPUTE_OR_COMPLAINT",
    "NOT_INTERESTED",
    "UNKNOWN_INTENT",
]


def detect_nba_rules(text: str) -> str:
    t = text.lower()
    for label, keywords in NBA_LABELS.items():
        for k in keywords:
            if k.lower() in t:
                return label
    return DEFAULT_NBA


def detect_sentiment_rules(text: str) -> Literal["POSITIVE", "NEUTRAL", "NEGATIVE"]:
    t = text.lower()
    for k in SENTIMENT_LABELS["NEGATIVE"]:
        if k.lower() in t:
            return "NEGATIVE"
    for k in SENTIMENT_LABELS["POSITIVE"]:
        if k.lower() in t:
            return "POSITIVE"
    return DEFAULT_SENTIMENT


def detect_intent_rules(text: str) -> str:
    """Very basic bootstrap intent heuristics – good enough for first training."""
    t = text.lower()

    if "emi कम" in t or "emi kam" in t or "emi reduce" in t:
        return "EMI_REDUCTION_REQUEST"
    if "next month" in t or "agale mahine" in t or "baad me" in t:
        return "NEEDS_EXTENSION"
    if "complaint" in t or "शिकायत" in t or "problem" in t:
        return "DISPUTE_OR_COMPLAINT"
    if "nahi dunga" in t or "interest nahi" in t:
        return "NOT_INTERESTED"
    if "kar dunga" in t or "pay karunga" in t or "payment karunga" in t:
        return "WILLING_TO_PAY"

    return "UNKNOWN_INTENT"
