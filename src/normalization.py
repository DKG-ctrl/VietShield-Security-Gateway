from __future__ import annotations

from dataclasses import asdict, dataclass
import re
import unicodedata


ZERO_WIDTH = {"\u200b", "\u200c", "\u200d", "\u2060", "\ufeff"}
EXOTIC_SPACES = {
    "\u00a0", "\u1680", "\u2000", "\u2001", "\u2002", "\u2003",
    "\u2004", "\u2005", "\u2006", "\u2007", "\u2008", "\u2009",
    "\u200a", "\u202f", "\u205f", "\u3000",
}


@dataclass(frozen=True)
class NormalizationResult:
    raw_text: str
    normalized_text: str
    comparison_text: str
    anomaly_score: float
    signals: list[str]

    def to_dict(self) -> dict:
        return asdict(self)


def _script_of(char: str) -> str | None:
    name = unicodedata.name(char, "")
    for script in ("LATIN", "CYRILLIC", "GREEK", "ARABIC", "HEBREW", "HIRAGANA", "KATAKANA", "HANGUL", "CJK"):
        if script in name:
            return script
    return None


def normalize_text(text: str) -> NormalizationResult:
    if not isinstance(text, str):
        raise TypeError("text must be a string")

    raw_text = text
    signals: list[str] = []
    zero_width_count = sum(char in ZERO_WIDTH for char in text)
    control_count = sum(
        unicodedata.category(char) in {"Cc", "Cf"}
        and char not in {"\n", "\r", "\t"}
        and char not in ZERO_WIDTH
        for char in text
    )
    exotic_space_count = sum(char in EXOTIC_SPACES for char in text)

    if zero_width_count:
        signals.append(f"zero_width_characters:{zero_width_count}")
    if control_count:
        signals.append(f"control_characters:{control_count}")
    if exotic_space_count:
        signals.append(f"unusual_whitespace:{exotic_space_count}")

    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = "".join(" " if c in EXOTIC_SPACES else c for c in text)
    text = "".join(c for c in text if c not in ZERO_WIDTH)
    text = "".join(
        c for c in text
        if unicodedata.category(c) not in {"Cc", "Cf"} or c in {"\n", "\t"}
    )
    # NFC preserves Vietnamese diacritics and canonical meaning; unlike NFKC it
    # does not broadly fold compatibility characters in the user-visible text.
    normalized = unicodedata.normalize("NFC", text)
    normalized = re.sub(r"[ \t]+", " ", normalized)
    normalized = re.sub(r" *\n *", "\n", normalized).strip()

    scripts = {_script_of(c) for c in normalized if c.isalpha()}
    scripts.discard(None)
    confusable_mix = "LATIN" in scripts and bool(scripts & {"CYRILLIC", "GREEK"})
    if confusable_mix:
        signals.append("mixed_latin_confusable_scripts")

    combining_count = sum(unicodedata.combining(c) != 0 for c in raw_text)
    if combining_count >= 4 and combining_count / max(len(raw_text), 1) > 0.08:
        signals.append(f"dense_combining_marks:{combining_count}")

    score = min(
        1.0,
        (0.25 if zero_width_count else 0.0)
        + (0.25 if control_count else 0.0)
        + (0.10 if exotic_space_count else 0.0)
        + (0.45 if confusable_mix else 0.0)
        + (0.20 if combining_count >= 4 else 0.0),
    )
    comparison = unicodedata.normalize("NFKC", normalized).casefold()
    return NormalizationResult(raw_text, normalized, comparison, score, signals)


def remove_unsafe_formatting(text: str) -> str:
    """Conservative cleanup used only for warn_sanitize responses."""
    return normalize_text(text).normalized_text
