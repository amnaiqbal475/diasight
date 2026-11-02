"""Fixed advice per grade. Urdu must be checked by a native speaker before the final demo."""
ADVICE = {
    0: ("No signs found. Check again in 12 months.",
        "کوئی علامات نہیں ملیں۔ 12 مہینے بعد دوبارہ چیک کروائیں۔"),
    1: ("Very early changes. Check again in 6 to 12 months and keep sugar controlled.",
        "بہت ابتدائی تبدیلیاں۔ 6 سے 12 مہینے میں دوبارہ چیک کروائیں اور شوگر کنٹرول رکھیں۔"),
    2: ("Please see an eye doctor within the next few weeks.",
        "براہ کرم اگلے چند ہفتوں میں آنکھوں کے ڈاکٹر سے ملیں۔"),
    3: ("Please see an eye doctor this week.",
        "مہربانی کر کے اس ہفتے آنکھوں کے ڈاکٹر سے ملیں۔"),
    4: ("Urgent: see an eye doctor as soon as possible.",
        "فوری طور پر آنکھوں کے ڈاکٹر سے ملیں۔"),
}
DISCLAIMER = "Screening tool, not a diagnosis. An eye doctor always makes the final decision."


def get(grade: int):
    return ADVICE[int(grade)]
