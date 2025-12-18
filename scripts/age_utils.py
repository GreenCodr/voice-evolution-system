from datetime import datetime, date


def calculate_age_at_recording(dob: str, recording_time: str) -> float:
    """
    dob: 'YYYY-MM-DD'
    recording_time: ISO timestamp 'YYYY-MM-DDTHH:MM:SSZ'
    Returns age in years (float)
    """
    dob_date = datetime.strptime(dob, "%Y-%m-%d").date()
    rec_date = datetime.strptime(
        recording_time.replace("Z", ""),
        "%Y-%m-%dT%H:%M:%S"
    ).date()

    age_days = (rec_date - dob_date).days
    age_years = age_days / 365.25

    return round(age_years, 2)