from collections.abc import Sequence


def hit_at_k(retrieved: Sequence[str], expected: str, k: int) -> bool:
    return expected in retrieved[:k]


def reciprocal_rank(retrieved: Sequence[str], expected: str) -> float:
    try:
        return 1.0 / (retrieved.index(expected) + 1)
    except ValueError:
        return 0.0


def mean_reciprocal_rank(rows: Sequence[tuple[Sequence[str], str]]) -> float:
    if not rows:
        return 0.0
    return sum(reciprocal_rank(retrieved, expected) for retrieved, expected in rows) / len(rows)


def paired_improvement(before: Sequence[bool], after: Sequence[bool]) -> dict[str, int]:
    if len(before) != len(after):
        raise ValueError("paired results must have equal length")
    return {
        "improved": sum(not old and new for old, new in zip(before, after, strict=True)),
        "regressed": sum(old and not new for old, new in zip(before, after, strict=True)),
        "unchanged": sum(old == new for old, new in zip(before, after, strict=True)),
    }
