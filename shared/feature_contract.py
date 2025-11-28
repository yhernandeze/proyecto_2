FEATURE_COLUMNS = [
    "bed",
    "bath",
    "acre_lot",
    "house_size",
    "zip_code",
    "brokered_by",
    "street",
]


def get_feature_names():
    """Return the ordered list of feature columns used across DAGs, API, and UI."""
    return list(FEATURE_COLUMNS)
