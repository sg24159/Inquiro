import re

from shared.models import RawResult


def jaccard_similarity(a: str, b: str) -> float:
    tokens_a = set(re.findall(r"\w+", a.lower()))
    tokens_b = set(re.findall(r"\w+", b.lower()))
    if not tokens_a or not tokens_b:
        return 0.0
    return len(tokens_a & tokens_b) / len(tokens_a | tokens_b)


def filter_noise(results: list[RawResult], threshold: float = 0.75) -> list[RawResult]:
    kept: list[RawResult] = []
    for r in results:
        if len(r.snippet.split()) < 10:
            continue
        is_duplicate = False
        for existing in kept:
            if jaccard_similarity(r.title, existing.title) > threshold:
                is_duplicate = True
                break
        if not is_duplicate:
            kept.append(r)
    return kept
