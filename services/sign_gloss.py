"""Texte brut → gloss + séquence de mots (aperçu signes côté client)."""

from __future__ import annotations


def text_to_gloss_and_sequence(raw: str, max_words: int = 16) -> tuple[str, list[str]]:
    words = [w for w in raw.replace("\n", " ").split() if w.strip()]
    seq = words[:max_words]
    gloss = " · ".join(seq[:12]).upper()
    return gloss, seq
