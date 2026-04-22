from __future__ import annotations

from typing import Dict

NORMAL_RVC_MODE = "normal-rvc"
ALIGNED_SUNO_MODE = "aligned-suno"

LEGACY_NORMAL_TRAINING_ALIASES = {
    "classic-rvc-support",
    "classic rvc support",
    "classic rvc + suno audition",
    "classic-rvc",
    "classic rvc",
    "persona-v1",
    "persona v1",
    "persona-v1.1",
    "persona v1.1",
    "persona-v11",
    "persona v11",
    "persona-lyric-repair",
    "persona lyric repair",
    "lyric-repair",
    "pipa-full",
    "pipa-lite",
    "pipa-logic-only",
    "normal-rvc",
    "normal rvc",
    "normal",
    "rvc",
}

LEGACY_ALIGNED_TRAINING_ALIASES = {
    "aligned-suno",
    "aligned suno",
    "suno-aligned",
    "suno aligned",
    "aligned",
    "persona-aligned-pth",
    "persona aligned pth",
    "aligned-pth",
    "concert-remaster-paired",
    "concert remaster paired",
    "concert-remaster",
    "concert remaster",
}

LEGACY_CLASSIC_PACKAGE_MODES = {
    NORMAL_RVC_MODE,
    "classic-rvc-support",
    "classic-rvc",
}

LEGACY_DIRECT_ALIGNED_PACKAGE_MODES = {
    ALIGNED_SUNO_MODE,
    "persona-aligned-pth",
    "concert-remaster-paired",
}

LEGACY_GUIDED_PACKAGE_LABELS: Dict[str, str] = {
    "persona-v1": "Persona v1.0",
    "persona-v1.1": "Persona v1.1",
    "persona-lyric-repair": "Persona lyric repair",
    "concert-remaster-paired": "Concert remaster",
}


def normalize_public_training_mode(value: str, default: str = NORMAL_RVC_MODE) -> str:
    normalized = str(value or "").strip().lower()
    if normalized in LEGACY_ALIGNED_TRAINING_ALIASES:
        return ALIGNED_SUNO_MODE
    if normalized in LEGACY_NORMAL_TRAINING_ALIASES:
        return NORMAL_RVC_MODE
    return default


def is_normal_rvc_mode(value: str) -> bool:
    return str(value or "").strip().lower() == NORMAL_RVC_MODE


def is_aligned_suno_mode(value: str) -> bool:
    return str(value or "").strip().lower() == ALIGNED_SUNO_MODE


def is_classic_rvc_package_mode(value: str) -> bool:
    return str(value or "").strip().lower() in LEGACY_CLASSIC_PACKAGE_MODES


def is_direct_aligned_package_mode(value: str) -> bool:
    return str(value or "").strip().lower() in LEGACY_DIRECT_ALIGNED_PACKAGE_MODES


def public_mode_label(value: str) -> str:
    normalized = str(value or "").strip().lower()
    if normalized == ALIGNED_SUNO_MODE:
        return "Aligned SUNO to Target"
    return "Normal RVC"


def package_mode_label(value: str) -> str:
    normalized = str(value or "").strip().lower()
    if normalized in LEGACY_DIRECT_ALIGNED_PACKAGE_MODES:
        if normalized == "concert-remaster-paired":
            return "Concert remaster"
        return "Aligned SUNO to Target"
    if normalized in LEGACY_CLASSIC_PACKAGE_MODES:
        return "Normal RVC"
    return LEGACY_GUIDED_PACKAGE_LABELS.get(normalized, "Persona package")
