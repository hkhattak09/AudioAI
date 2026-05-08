"""Alignment utilities."""

def align_words(text: str, duration: float) -> list[dict]:
    """Naive word alignment: split duration evenly across words.

    Args:
        text: transcript string
        duration: total duration in seconds

    Returns:
        List of dicts with word, start, end.
    """
    words = text.split()
    if not words:
        return []

    word_duration = duration / len(words)
    results = []
    for i, word in enumerate(words):
        start = i * word_duration
        end = (i + 1) * word_duration
        results.append({"word": word, "start": start, "end": end})

    return results
