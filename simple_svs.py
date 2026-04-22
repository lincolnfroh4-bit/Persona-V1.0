from __future__ import annotations

from contextlib import nullcontext
import gc
import json
import math
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import librosa
import numpy as np
import soundfile as sf
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset

from simple_touchup import lyrics_to_words, normalize_lyrics


SAMPLE_RATE = 44100
HOP_LENGTH = 512
N_FFT = 2048
WIN_LENGTH = 2048
N_MELS = 128
F0_MIN = 65.0
F0_MAX = 1100.0
FEATURE_CHUNK_SECONDS = 12.0
VOICE_SIGNATURE_BANDS = 16
VOICE_SIGNATURE_DIM = (VOICE_SIGNATURE_BANDS * 3) + 5
PERSONA_V11_CONTENT_DIM = 256
DIRECT_GUIDED_RECIPE_MODES = {
    "aligned-suno",
    "persona-v1.1",
    "concert-remaster-paired",
}
DIRECT_GUIDED_TRAINING_MODES = {
    "suno-aligned-mapper-v1.1",
    "persona-paired-mapper-v1.1",
    "concert-paired-mapper-v1.1",
}

PHONE_TOKENS: Tuple[str, ...] = (
    "PAD",
    "SP",
    "UNK",
    "AAR",
    "AH",
    "AO",
    "AW",
    "AY",
    "B",
    "CH",
    "D",
    "EH",
    "ER",
    "EY",
    "F",
    "G",
    "HH",
    "IH",
    "IY",
    "JH",
    "K",
    "KS",
    "KW",
    "L",
    "M",
    "N",
    "NG",
    "OW",
    "OY",
    "P",
    "R",
    "S",
    "SH",
    "T",
    "TH",
    "UH",
    "UW",
    "V",
    "W",
    "Y",
    "Z",
)
PHONE_TO_ID = {token: index for index, token in enumerate(PHONE_TOKENS)}


def pronunciation_units(word: str) -> List[str]:
    normalized = normalize_lyrics(word).replace(" ", "")
    if not normalized:
        return []

    trigraphs = {
        "tch": "CH",
        "dge": "JH",
        "igh": "AY",
        "eau": "OW",
    }
    digraphs = {
        "ch": "CH",
        "sh": "SH",
        "th": "TH",
        "ph": "F",
        "ng": "NG",
        "ck": "K",
        "qu": "KW",
        "wh": "W",
        "wr": "R",
        "kn": "N",
        "gn": "N",
        "ee": "IY",
        "ea": "IY",
        "oo": "UW",
        "ou": "AW",
        "ow": "OW",
        "oi": "OY",
        "oy": "OY",
        "ai": "EY",
        "ay": "EY",
        "oa": "OW",
        "au": "AO",
        "aw": "AO",
        "er": "ER",
        "ir": "ER",
        "ur": "ER",
        "ar": "AAR",
        "or": "AO",
    }
    singles = {
        "a": "AH",
        "b": "B",
        "c": "K",
        "d": "D",
        "e": "EH",
        "f": "F",
        "g": "G",
        "h": "HH",
        "i": "IH",
        "j": "JH",
        "k": "K",
        "l": "L",
        "m": "M",
        "n": "N",
        "o": "OW",
        "p": "P",
        "q": "K",
        "r": "R",
        "s": "S",
        "t": "T",
        "u": "UH",
        "v": "V",
        "w": "W",
        "x": "KS",
        "y": "Y",
        "z": "Z",
        "'": "",
    }

    units: List[str] = []
    index = 0
    while index < len(normalized):
        tri = normalized[index : index + 3]
        if tri in trigraphs:
            units.append(trigraphs[tri])
            index += 3
            continue
        duo = normalized[index : index + 2]
        if duo in digraphs:
            units.append(digraphs[duo])
            index += 2
            continue
        unit = singles.get(normalized[index], normalized[index].upper())
        if unit:
            units.extend(chunk for chunk in unit.split(" ") if chunk)
        index += 1
    if units and units[-1] in {"AH", "EH", "IH", "UH"} and normalized.endswith("e"):
        units[-1] = "IY"
    return [token if token in PHONE_TO_ID else "UNK" for token in units]


def _align_1d(feature: np.ndarray, target_frames: int) -> np.ndarray:
    working = np.asarray(feature, dtype=np.float32).reshape(-1)
    if target_frames <= 0:
        return np.zeros(0, dtype=np.float32)
    if working.size == target_frames:
        return working.astype(np.float32, copy=False)
    if working.size == 0:
        return np.zeros(target_frames, dtype=np.float32)
    if working.size > target_frames:
        return working[:target_frames].astype(np.float32, copy=False)
    return np.pad(working, (0, target_frames - working.size)).astype(np.float32, copy=False)


def _align_2d(feature: np.ndarray, target_frames: int) -> np.ndarray:
    working = np.asarray(feature, dtype=np.float32)
    if working.ndim != 2:
        working = np.asarray(working, dtype=np.float32).reshape(-1, 1)
    if target_frames <= 0:
        return np.zeros((0, working.shape[1]), dtype=np.float32)
    if working.shape[0] == target_frames:
        return working.astype(np.float32, copy=False)
    if working.shape[0] == 0:
        return np.zeros((target_frames, working.shape[1]), dtype=np.float32)
    if working.shape[0] == 1:
        return np.repeat(working.astype(np.float32, copy=False), target_frames, axis=0)

    source_positions = np.linspace(0.0, 1.0, num=working.shape[0], dtype=np.float32)
    target_positions = np.linspace(0.0, 1.0, num=target_frames, dtype=np.float32)
    channels = [
        np.interp(target_positions, source_positions, working[:, channel]).astype(np.float32)
        for channel in range(working.shape[1])
    ]
    return np.stack(channels, axis=1)


def _default_frame_map(source_frames: int, target_frames: int) -> np.ndarray:
    if target_frames <= 0:
        return np.zeros(0, dtype=np.float32)
    if source_frames <= 1:
        return np.zeros(target_frames, dtype=np.float32)
    return np.linspace(0.0, float(source_frames - 1), num=target_frames, dtype=np.float32)


def _warp_1d_by_frame_map(feature: np.ndarray, frame_map: np.ndarray) -> np.ndarray:
    working = np.asarray(feature, dtype=np.float32).reshape(-1)
    target_frames = int(np.asarray(frame_map).shape[0])
    if target_frames <= 0:
        return np.zeros(0, dtype=np.float32)
    if working.size == 0:
        return np.zeros(target_frames, dtype=np.float32)
    if working.size == 1:
        return np.repeat(working.astype(np.float32, copy=False), target_frames)
    source_positions = np.arange(working.size, dtype=np.float32)
    clipped_map = np.clip(np.asarray(frame_map, dtype=np.float32), 0.0, float(max(working.size - 1, 0)))
    return np.interp(clipped_map, source_positions, working).astype(np.float32)


def _warp_2d_by_frame_map(feature: np.ndarray, frame_map: np.ndarray) -> np.ndarray:
    working = np.asarray(feature, dtype=np.float32)
    if working.ndim != 2:
        working = np.asarray(working, dtype=np.float32).reshape(-1, 1)
    target_frames = int(np.asarray(frame_map).shape[0])
    if target_frames <= 0:
        return np.zeros((0, working.shape[1]), dtype=np.float32)
    if working.shape[0] == 0:
        return np.zeros((target_frames, working.shape[1]), dtype=np.float32)
    if working.shape[0] == 1:
        return np.repeat(working.astype(np.float32, copy=False), target_frames, axis=0)
    clipped_map = np.clip(np.asarray(frame_map, dtype=np.float32), 0.0, float(max(working.shape[0] - 1, 0)))
    source_positions = np.arange(working.shape[0], dtype=np.float32)
    channels = [
        np.interp(clipped_map, source_positions, working[:, channel]).astype(np.float32)
        for channel in range(working.shape[1])
    ]
    return np.stack(channels, axis=1)


def _invert_monotonic_frame_map(frame_map: np.ndarray, source_frames: int) -> np.ndarray:
    if source_frames <= 0:
        return np.zeros(0, dtype=np.float32)
    target_frames = int(np.asarray(frame_map).shape[0])
    if target_frames <= 0:
        return np.zeros(source_frames, dtype=np.float32)
    source_positions = np.maximum.accumulate(np.asarray(frame_map, dtype=np.float32))
    source_positions = np.clip(source_positions, 0.0, float(max(source_frames - 1, 0)))
    if source_positions.size == 1:
        return np.repeat(np.float32(0.0), source_frames).astype(np.float32, copy=False)
    source_positions = source_positions + np.linspace(0.0, 1e-4, num=source_positions.size, dtype=np.float32)
    target_positions = np.arange(target_frames, dtype=np.float32)
    desired_sources = np.arange(source_frames, dtype=np.float32)
    return np.interp(
        desired_sources,
        source_positions,
        target_positions,
        left=float(target_positions[0]),
        right=float(target_positions[-1]),
    ).astype(np.float32)


def _warp_audio_by_frame_map(
    audio: np.ndarray,
    *,
    sample_rate: int,
    frame_map: np.ndarray,
    target_sample_count: int,
) -> np.ndarray:
    working = np.asarray(audio, dtype=np.float32).reshape(-1)
    if target_sample_count <= 0:
        return np.zeros(0, dtype=np.float32)
    if working.size == 0:
        return np.zeros(target_sample_count, dtype=np.float32)
    if working.size == 1:
        return np.repeat(working.astype(np.float32, copy=False), target_sample_count)
    target_frame_positions = np.arange(len(frame_map), dtype=np.float32)
    if target_frame_positions.size <= 1:
        return np.resize(working.astype(np.float32, copy=False), target_sample_count).astype(np.float32, copy=False)
    target_sample_positions = np.linspace(0.0, float(max(len(frame_map) - 1, 0)), num=target_sample_count, dtype=np.float32)
    guide_frame_positions = np.interp(
        target_sample_positions,
        target_frame_positions,
        np.asarray(frame_map, dtype=np.float32),
    )
    guide_sample_positions = np.clip(guide_frame_positions * float(HOP_LENGTH), 0.0, float(max(working.size - 1, 0)))
    source_positions = np.arange(working.size, dtype=np.float32)
    return np.interp(guide_sample_positions, source_positions, working).astype(np.float32)


def _normalize_similarity_score(score: object) -> float:
    try:
        value = float(score)
    except Exception:
        value = 0.0
    if value > 1.0:
        value /= 100.0
    return float(np.clip(value, 0.0, 1.0))


def _safe_std(value: np.ndarray | float, floor: float = 1e-4) -> np.ndarray | float:
    if isinstance(value, np.ndarray):
        return np.maximum(value, floor)
    return max(float(value), floor)


def _ensure_voice_signature_dim(signature: np.ndarray | Sequence[float] | None) -> np.ndarray:
    working = np.asarray(signature if signature is not None else [], dtype=np.float32).reshape(-1)
    if working.size == VOICE_SIGNATURE_DIM:
        return working.astype(np.float32, copy=False)
    if working.size <= 0:
        return np.zeros(VOICE_SIGNATURE_DIM, dtype=np.float32)
    if working.size > VOICE_SIGNATURE_DIM:
        return working[:VOICE_SIGNATURE_DIM].astype(np.float32, copy=False)
    return np.pad(working.astype(np.float32, copy=False), (0, VOICE_SIGNATURE_DIM - working.size))


def _compute_voice_signature_np(
    *,
    log_mel: np.ndarray,
    log_f0: np.ndarray,
    vuv: np.ndarray,
) -> np.ndarray:
    mel = np.asarray(log_mel, dtype=np.float32)
    if mel.ndim != 2 or mel.shape[0] <= 0:
        return np.zeros(VOICE_SIGNATURE_DIM, dtype=np.float32)
    if mel.shape[1] != N_MELS:
        mel = _align_2d(mel, int(mel.shape[0]))
        mel = np.pad(mel, ((0, 0), (0, max(0, N_MELS - mel.shape[1]))), mode="edge")[:, :N_MELS]
    pooled = mel.reshape(mel.shape[0], VOICE_SIGNATURE_BANDS, -1).mean(axis=2)
    mean = pooled.mean(axis=0)
    std = pooled.std(axis=0)
    delta = np.zeros_like(mean, dtype=np.float32)
    if pooled.shape[0] > 1:
        delta = np.mean(np.abs(np.diff(pooled, axis=0)), axis=0).astype(np.float32, copy=False)
    voiced = np.asarray(vuv, dtype=np.float32).reshape(-1) > 0.5
    log_f0_values = np.asarray(log_f0, dtype=np.float32).reshape(-1)
    voiced_values = log_f0_values[voiced]
    voiced_ratio = float(np.mean(voiced.astype(np.float32))) if voiced.size else 0.0
    f0_mean = float(np.mean(voiced_values)) if voiced_values.size else 0.0
    f0_std = float(np.std(voiced_values)) if voiced_values.size else 0.0
    mel_level = float(np.mean(mean))
    brightness = float(np.mean(mean[-4:]) - np.mean(mean[:4]))
    signature = np.concatenate(
        [
            mean.astype(np.float32, copy=False),
            std.astype(np.float32, copy=False),
            delta.astype(np.float32, copy=False),
            np.asarray([voiced_ratio, f0_mean, f0_std, mel_level, brightness], dtype=np.float32),
        ]
    )
    return _ensure_voice_signature_dim(signature)


def _compute_voice_signature_torch(
    *,
    mel: torch.Tensor,
    log_f0: torch.Tensor,
    vuv: torch.Tensor,
    lengths: torch.Tensor,
) -> torch.Tensor:
    batch_size, max_frames, mel_bins = mel.shape
    if mel_bins != N_MELS:
        raise ValueError(f"Voice signature expects {N_MELS} mel bins, got {mel_bins}.")
    time_mask = (
        torch.arange(max_frames, device=lengths.device).unsqueeze(0) < lengths.unsqueeze(1)
    ).to(dtype=mel.dtype)
    pooled = mel.reshape(batch_size, max_frames, VOICE_SIGNATURE_BANDS, -1).mean(dim=-1)
    time_mask_expanded = time_mask.unsqueeze(-1)
    valid_counts = time_mask_expanded.sum(dim=1).clamp_min(1.0)
    mean = (pooled * time_mask_expanded).sum(dim=1) / valid_counts
    centered = (pooled - mean.unsqueeze(1)) * time_mask_expanded
    std = torch.sqrt((centered.square().sum(dim=1) / valid_counts).clamp_min(1e-6))

    if max_frames > 1:
        delta = torch.abs(pooled[:, 1:] - pooled[:, :-1])
        delta_mask = (time_mask[:, 1:] * time_mask[:, :-1]).unsqueeze(-1)
        delta_counts = delta_mask.sum(dim=1).clamp_min(1.0)
        delta_mean = (delta * delta_mask).sum(dim=1) / delta_counts
    else:
        delta_mean = torch.zeros_like(mean)

    voiced = torch.clamp(vuv, 0.0, 1.0) * time_mask
    voiced_counts = voiced.sum(dim=1).clamp_min(1.0)
    voiced_ratio = (voiced.sum(dim=1) / time_mask.sum(dim=1).clamp_min(1.0)).unsqueeze(-1)
    f0_mean = ((log_f0 * voiced).sum(dim=1) / voiced_counts).unsqueeze(-1)
    f0_centered = (log_f0 - f0_mean.squeeze(-1).unsqueeze(-1)) * voiced
    f0_std = torch.sqrt((f0_centered.square().sum(dim=1) / voiced_counts).clamp_min(1e-6)).unsqueeze(-1)
    mel_level = mean.mean(dim=-1, keepdim=True)
    brightness = (mean[:, -4:].mean(dim=-1, keepdim=True) - mean[:, :4].mean(dim=-1, keepdim=True))
    return torch.cat([mean, std, delta_mean, voiced_ratio, f0_mean, f0_std, mel_level, brightness], dim=-1)


def _render_log_mel_to_audio(log_mel: np.ndarray) -> np.ndarray:
    mel = np.exp(np.asarray(log_mel, dtype=np.float32).T)
    waveform = librosa.feature.inverse.mel_to_audio(
        mel,
        sr=SAMPLE_RATE,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        win_length=WIN_LENGTH,
        power=2.0,
        n_iter=48,
        fmin=0.0,
        fmax=SAMPLE_RATE / 2.0,
    )
    waveform = np.asarray(waveform, dtype=np.float32)
    peak = float(np.max(np.abs(waveform))) if waveform.size else 0.0
    if peak > 0.995:
        waveform = waveform * np.float32(0.995 / peak)
    return waveform


def _slice_audio_for_frame_range(
    audio: np.ndarray,
    *,
    start_frame: int,
    end_frame: int,
) -> np.ndarray:
    working = np.asarray(audio, dtype=np.float32).reshape(-1)
    frame_count = max(0, int(end_frame) - int(start_frame))
    target_samples = max(1, frame_count * HOP_LENGTH)
    if working.size <= 0:
        return np.zeros(target_samples, dtype=np.float32)
    start_sample = max(0, int(start_frame) * HOP_LENGTH)
    end_sample = min(working.size, start_sample + target_samples)
    sliced = working[start_sample:end_sample].astype(np.float32, copy=False)
    if sliced.size >= target_samples:
        return sliced[:target_samples].astype(np.float32, copy=False)
    return np.pad(sliced, (0, target_samples - sliced.size)).astype(np.float32, copy=False)


def _mask_audio_by_lengths(audio: torch.Tensor, sample_lengths: torch.Tensor) -> torch.Tensor:
    max_samples = int(audio.shape[1])
    valid = torch.arange(max_samples, device=sample_lengths.device).unsqueeze(0) < sample_lengths.unsqueeze(1)
    return audio * valid.to(dtype=audio.dtype)


@dataclass
class GuidedSVSFeatureExample:
    sample_id: str
    lyrics: str
    n_frames: int
    duration_seconds: float
    aligned_word_count: int
    frame_phone_coverage: float
    feature_dir: str
    source_clip: str = ""
    conditioning_clip: str = ""
    anchor_word: str = ""
    anchor_units: List[str] | None = None
    slice_kind: str = "full-clip"
    conditioning_similarity: float = 1.0
    alignment_score: float = 1.0
    difficulty_score: float = 0.0

    def to_dict(self) -> Dict[str, object]:
        return {
            "id": self.sample_id,
            "lyrics": self.lyrics,
            "n_frames": self.n_frames,
            "duration_seconds": round(self.duration_seconds, 3),
            "aligned_word_count": self.aligned_word_count,
            "frame_phone_coverage": round(self.frame_phone_coverage, 4),
            "feature_dir": self.feature_dir,
            "source_clip": self.source_clip,
            "conditioning_clip": self.conditioning_clip,
            "anchor_word": self.anchor_word,
            "anchor_units": list(self.anchor_units or []),
            "slice_kind": self.slice_kind,
            "conditioning_similarity": round(float(self.conditioning_similarity), 4),
            "alignment_score": round(float(self.alignment_score), 4),
            "difficulty_score": round(float(self.difficulty_score), 4),
        }


class GuidedSVSDataset(Dataset):
    def __init__(
        self,
        *,
        entries: Sequence[Dict[str, object]],
        dataset_dir: Path,
        stats: Dict[str, object],
        max_frames: int = 900,
        random_crop: bool = True,
    ):
        self.entries = [dict(entry) for entry in entries]
        self.dataset_dir = Path(dataset_dir)
        self.features_dir = self.dataset_dir / "features"
        self.max_frames = max(120, int(max_frames))
        self.random_crop = bool(random_crop)
        self.mel_mean = np.asarray(stats.get("mel_mean", [0.0] * N_MELS), dtype=np.float32)
        self.mel_std = np.asarray(stats.get("mel_std", [1.0] * N_MELS), dtype=np.float32)
        self.log_f0_mean = float(stats.get("log_f0_mean", 0.0))
        self.log_f0_std = float(_safe_std(float(stats.get("log_f0_std", 1.0))))
        self.energy_mean = float(stats.get("energy_mean", 0.0))
        self.energy_std = float(_safe_std(float(stats.get("energy_std", 1.0))))
        self.global_voice_signature = _ensure_voice_signature_dim(
            stats.get("global_voice_signature", [0.0] * VOICE_SIGNATURE_DIM)
        )

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor | int]:
        entry = self.entries[index]
        sample_id = str(entry["id"])
        feature_dir = self.features_dir / sample_id
        mel = np.load(feature_dir / "mel.npy").astype(np.float32)
        guide_mel_path = feature_dir / "guide_mel.npy"
        guide_mel = (
            np.load(guide_mel_path).astype(np.float32)
            if guide_mel_path.exists()
            else mel.astype(np.float32, copy=True)
        )
        phone_ids = np.load(feature_dir / "phone_ids.npy").astype(np.int64)
        log_f0 = np.load(feature_dir / "log_f0.npy").astype(np.float32)
        target_log_f0_path = feature_dir / "target_log_f0.npy"
        target_log_f0 = (
            np.load(target_log_f0_path).astype(np.float32)
            if target_log_f0_path.exists()
            else log_f0.astype(np.float32, copy=True)
        )
        target_vuv_path = feature_dir / "target_vuv.npy"
        target_vuv = (
            np.load(target_vuv_path).astype(np.float32)
            if target_vuv_path.exists()
            else (target_log_f0 > 0.0).astype(np.float32)
        )
        vuv = np.load(feature_dir / "vuv.npy").astype(np.float32)
        energy = np.load(feature_dir / "energy.npy").astype(np.float32)
        lyric_mask_path = feature_dir / "lyric_mask.npy"
        lyric_mask = (
            np.load(lyric_mask_path).astype(np.float32)
            if lyric_mask_path.exists()
            else np.ones_like(phone_ids, dtype=np.float32)
        )
        total_frames = int(
            min(
                mel.shape[0],
                guide_mel.shape[0],
                phone_ids.shape[0],
                log_f0.shape[0],
                target_log_f0.shape[0],
                target_vuv.shape[0],
                vuv.shape[0],
                energy.shape[0],
                lyric_mask.shape[0],
            )
        )
        start = 0
        end = total_frames
        if total_frames > self.max_frames:
            if self.random_crop:
                start = random.randint(0, total_frames - self.max_frames)
            end = start + self.max_frames

        mel = mel[start:end]
        guide_mel = guide_mel[start:end]
        phone_ids = phone_ids[start:end]
        log_f0 = log_f0[start:end]
        target_log_f0 = target_log_f0[start:end]
        target_vuv = target_vuv[start:end]
        vuv = vuv[start:end]
        energy = energy[start:end]
        lyric_mask = lyric_mask[start:end]
        voiced = vuv > 0.5
        norm_log_f0 = np.zeros_like(log_f0, dtype=np.float32)
        if np.any(voiced):
            norm_log_f0[voiced] = (log_f0[voiced] - self.log_f0_mean) / self.log_f0_std
        target_voiced = target_log_f0 > 0.0
        norm_target_log_f0 = np.zeros_like(target_log_f0, dtype=np.float32)
        if np.any(target_voiced):
            norm_target_log_f0[target_voiced] = (
                (target_log_f0[target_voiced] - self.log_f0_mean) / self.log_f0_std
            )
        norm_energy = (energy - self.energy_mean) / self.energy_std
        norm_mel = (mel - self.mel_mean[np.newaxis, :]) / self.mel_std[np.newaxis, :]
        norm_guide_mel = (guide_mel - self.mel_mean[np.newaxis, :]) / self.mel_std[np.newaxis, :]
        target_voice_signature = _compute_voice_signature_np(
            log_mel=norm_mel,
            log_f0=norm_target_log_f0,
            vuv=target_vuv,
        )

        return {
            "mel": torch.from_numpy(norm_mel.astype(np.float32, copy=False)),
            "guide_mel": torch.from_numpy(norm_guide_mel.astype(np.float32, copy=False)),
            "phone_ids": torch.from_numpy(phone_ids.astype(np.int64, copy=False)),
            "log_f0": torch.from_numpy(norm_log_f0.astype(np.float32, copy=False)),
            "target_log_f0": torch.from_numpy(norm_target_log_f0.astype(np.float32, copy=False)),
            "target_vuv": torch.from_numpy(target_vuv.astype(np.float32, copy=False)),
            "vuv": torch.from_numpy(vuv.astype(np.float32, copy=False)),
            "energy": torch.from_numpy(norm_energy.astype(np.float32, copy=False)),
            "lyric_mask": torch.from_numpy(lyric_mask.astype(np.float32, copy=False)),
            "voice_prototype": torch.from_numpy(self.global_voice_signature.astype(np.float32, copy=False)),
            "target_voice_signature": torch.from_numpy(target_voice_signature.astype(np.float32, copy=False)),
            "length": int(norm_mel.shape[0]),
        }


def collate_guided_svs(batch: Sequence[Dict[str, torch.Tensor | int]]) -> Dict[str, torch.Tensor]:
    lengths = torch.tensor([int(item["length"]) for item in batch], dtype=torch.long)
    mel = pad_sequence([item["mel"] for item in batch], batch_first=True, padding_value=0.0)
    guide_mel = pad_sequence([item["guide_mel"] for item in batch], batch_first=True, padding_value=0.0)
    phone_ids = pad_sequence(
        [item["phone_ids"] for item in batch],
        batch_first=True,
        padding_value=PHONE_TO_ID["PAD"],
    )
    log_f0 = pad_sequence([item["log_f0"] for item in batch], batch_first=True, padding_value=0.0)
    target_log_f0 = pad_sequence([item["target_log_f0"] for item in batch], batch_first=True, padding_value=0.0)
    target_vuv = pad_sequence([item["target_vuv"] for item in batch], batch_first=True, padding_value=0.0)
    vuv = pad_sequence([item["vuv"] for item in batch], batch_first=True, padding_value=0.0)
    energy = pad_sequence([item["energy"] for item in batch], batch_first=True, padding_value=0.0)
    lyric_mask = pad_sequence([item["lyric_mask"] for item in batch], batch_first=True, padding_value=0.0)
    voice_prototype = torch.stack([item["voice_prototype"] for item in batch], dim=0)
    target_voice_signature = torch.stack([item["target_voice_signature"] for item in batch], dim=0)
    return {
        "mel": mel,
        "guide_mel": guide_mel,
        "phone_ids": phone_ids,
        "log_f0": log_f0,
        "target_log_f0": target_log_f0,
        "target_vuv": target_vuv,
        "vuv": vuv,
        "energy": energy,
        "lyric_mask": lyric_mask,
        "voice_prototype": voice_prototype,
        "target_voice_signature": target_voice_signature,
        "lengths": lengths,
    }


class PersonaMapperNARDataset(Dataset):
    def __init__(
        self,
        *,
        entries: Sequence[Dict[str, object]],
        dataset_dir: Path,
        stats: Dict[str, object],
        max_frames: int = 900,
        random_crop: bool = True,
    ):
        self.entries = [dict(entry) for entry in entries]
        self.dataset_dir = Path(dataset_dir)
        self.features_dir = self.dataset_dir / "features"
        self.max_frames = max(96, int(max_frames))
        self.random_crop = bool(random_crop)
        self.mel_mean = np.asarray(stats.get("mel_mean", [0.0] * N_MELS), dtype=np.float32)
        self.mel_std = np.asarray(stats.get("mel_std", [1.0] * N_MELS), dtype=np.float32)
        self.log_f0_mean = float(stats.get("log_f0_mean", 0.0))
        self.log_f0_std = float(_safe_std(float(stats.get("log_f0_std", 1.0))))
        self.energy_mean = float(stats.get("energy_mean", 0.0))
        self.energy_std = float(_safe_std(float(stats.get("energy_std", 1.0))))
        content_mean = np.asarray(
            stats.get("nar_content_mean", [0.0] * int(stats.get("nar_content_dim", PERSONA_V11_CONTENT_DIM))),
            dtype=np.float32,
        )
        content_std = np.asarray(
            stats.get("nar_content_std", [1.0] * int(stats.get("nar_content_dim", PERSONA_V11_CONTENT_DIM))),
            dtype=np.float32,
        )
        self.content_mean = content_mean
        self.content_std = np.maximum(content_std, 1e-4)
        self.global_voice_signature = _ensure_voice_signature_dim(
            stats.get("global_voice_signature", [0.0] * VOICE_SIGNATURE_DIM)
        )
        self.use_voice_prototype_conditioning = bool(stats.get("use_voice_prototype_conditioning", False))

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor | int]:
        entry = self.entries[index]
        sample_id = str(entry["id"])
        feature_dir = self.features_dir / sample_id
        mel = np.load(feature_dir / "mel.npy").astype(np.float32)
        guide_mel = np.load(feature_dir / "guide_mel.npy").astype(np.float32)
        content = np.load(feature_dir / "nar_content.npy").astype(np.float32)
        beat_phase = np.load(feature_dir / "beat_phase.npy").astype(np.float32)
        phone_ids = np.load(feature_dir / "phone_ids.npy").astype(np.int64)
        log_f0 = np.load(feature_dir / "log_f0.npy").astype(np.float32)
        target_log_f0 = (
            np.load(feature_dir / "target_log_f0.npy").astype(np.float32)
            if (feature_dir / "target_log_f0.npy").exists()
            else log_f0.astype(np.float32, copy=True)
        )
        target_vuv = (
            np.load(feature_dir / "target_vuv.npy").astype(np.float32)
            if (feature_dir / "target_vuv.npy").exists()
            else (target_log_f0 > 0.0).astype(np.float32)
        )
        vuv = np.load(feature_dir / "vuv.npy").astype(np.float32)
        energy = np.load(feature_dir / "energy.npy").astype(np.float32)
        lyric_mask = (
            np.load(feature_dir / "lyric_mask.npy").astype(np.float32)
            if (feature_dir / "lyric_mask.npy").exists()
            else np.ones_like(phone_ids, dtype=np.float32)
        )
        if self.content_mean.shape[0] != content.shape[1]:
            local_content_mean = np.zeros(content.shape[1], dtype=np.float32)
            local_content_std = np.ones(content.shape[1], dtype=np.float32)
            copy_dim = min(int(self.content_mean.shape[0]), int(content.shape[1]))
            if copy_dim > 0:
                local_content_mean[:copy_dim] = self.content_mean[:copy_dim]
                local_content_std[:copy_dim] = self.content_std[:copy_dim]
        else:
            local_content_mean = self.content_mean
            local_content_std = self.content_std
        total_frames = int(
            min(
                mel.shape[0],
                guide_mel.shape[0],
                content.shape[0],
                beat_phase.shape[0],
                phone_ids.shape[0],
                log_f0.shape[0],
                target_log_f0.shape[0],
                target_vuv.shape[0],
                vuv.shape[0],
                energy.shape[0],
                lyric_mask.shape[0],
            )
        )
        start = 0
        end = total_frames
        if total_frames > self.max_frames:
            if self.random_crop:
                start = random.randint(0, total_frames - self.max_frames)
            end = start + self.max_frames

        mel = mel[start:end]
        guide_mel = guide_mel[start:end]
        content = content[start:end]
        beat_phase = beat_phase[start:end]
        phone_ids = phone_ids[start:end]
        log_f0 = log_f0[start:end]
        target_log_f0 = target_log_f0[start:end]
        target_vuv = target_vuv[start:end]
        vuv = vuv[start:end]
        energy = energy[start:end]
        lyric_mask = lyric_mask[start:end]

        norm_mel = (mel - self.mel_mean[np.newaxis, :]) / self.mel_std[np.newaxis, :]
        norm_guide_mel = (guide_mel - self.mel_mean[np.newaxis, :]) / self.mel_std[np.newaxis, :]
        norm_content = (content - local_content_mean[np.newaxis, :]) / np.maximum(local_content_std[np.newaxis, :], 1e-4)
        norm_log_f0 = np.zeros_like(log_f0, dtype=np.float32)
        voiced = vuv > 0.5
        if np.any(voiced):
            norm_log_f0[voiced] = (log_f0[voiced] - self.log_f0_mean) / self.log_f0_std
        norm_target_log_f0 = np.zeros_like(target_log_f0, dtype=np.float32)
        target_voiced = target_vuv > 0.5
        if np.any(target_voiced):
            norm_target_log_f0[target_voiced] = (
                (target_log_f0[target_voiced] - self.log_f0_mean) / self.log_f0_std
            )
        norm_energy = (energy - self.energy_mean) / self.energy_std
        target_voice_signature = _compute_voice_signature_np(
            log_mel=norm_mel,
            log_f0=norm_target_log_f0,
            vuv=target_vuv,
        )

        return {
            "mel": torch.from_numpy(norm_mel.astype(np.float32, copy=False)),
            "guide_mel": torch.from_numpy(norm_guide_mel.astype(np.float32, copy=False)),
            "content": torch.from_numpy(norm_content.astype(np.float32, copy=False)),
            "beat_phase": torch.from_numpy(beat_phase.astype(np.float32, copy=False)).unsqueeze(-1),
            "phone_ids": torch.from_numpy(phone_ids.astype(np.int64, copy=False)),
            "log_f0": torch.from_numpy(norm_log_f0.astype(np.float32, copy=False)),
            "target_log_f0": torch.from_numpy(norm_target_log_f0.astype(np.float32, copy=False)),
            "target_vuv": torch.from_numpy(target_vuv.astype(np.float32, copy=False)),
            "vuv": torch.from_numpy(vuv.astype(np.float32, copy=False)),
            "energy": torch.from_numpy(norm_energy.astype(np.float32, copy=False)),
            "lyric_mask": torch.from_numpy(lyric_mask.astype(np.float32, copy=False)),
            "voice_prototype": torch.from_numpy(
                (
                    self.global_voice_signature
                    if self.use_voice_prototype_conditioning
                    else np.zeros(VOICE_SIGNATURE_DIM, dtype=np.float32)
                ).astype(np.float32, copy=False)
            ),
            "target_voice_signature": torch.from_numpy(target_voice_signature.astype(np.float32, copy=False)),
            "length": int(norm_mel.shape[0]),
        }


def collate_persona_mapper_nar(batch: Sequence[Dict[str, torch.Tensor | int]]) -> Dict[str, torch.Tensor]:
    lengths = torch.tensor([int(item["length"]) for item in batch], dtype=torch.long)
    return {
        "mel": pad_sequence([item["mel"] for item in batch], batch_first=True, padding_value=0.0),
        "guide_mel": pad_sequence([item["guide_mel"] for item in batch], batch_first=True, padding_value=0.0),
        "content": pad_sequence([item["content"] for item in batch], batch_first=True, padding_value=0.0),
        "beat_phase": pad_sequence([item["beat_phase"] for item in batch], batch_first=True, padding_value=0.0),
        "phone_ids": pad_sequence(
            [item["phone_ids"] for item in batch],
            batch_first=True,
            padding_value=PHONE_TO_ID["PAD"],
        ),
        "log_f0": pad_sequence([item["log_f0"] for item in batch], batch_first=True, padding_value=0.0),
        "target_log_f0": pad_sequence(
            [item["target_log_f0"] for item in batch], batch_first=True, padding_value=0.0
        ),
        "target_vuv": pad_sequence([item["target_vuv"] for item in batch], batch_first=True, padding_value=0.0),
        "vuv": pad_sequence([item["vuv"] for item in batch], batch_first=True, padding_value=0.0),
        "energy": pad_sequence([item["energy"] for item in batch], batch_first=True, padding_value=0.0),
        "lyric_mask": pad_sequence([item["lyric_mask"] for item in batch], batch_first=True, padding_value=0.0),
        "voice_prototype": torch.stack([item["voice_prototype"] for item in batch], dim=0),
        "target_voice_signature": torch.stack([item["target_voice_signature"] for item in batch], dim=0),
        "lengths": lengths,
    }


class VocoderSliceDataset(Dataset):
    def __init__(
        self,
        *,
        entries: Sequence[Dict[str, object]],
        dataset_dir: Path,
        stats: Dict[str, object],
        max_frames: int = 256,
        random_crop: bool = True,
    ):
        self.entries = [dict(entry) for entry in entries]
        self.dataset_dir = Path(dataset_dir)
        self.features_dir = self.dataset_dir / "features"
        self.max_frames = max(64, int(max_frames))
        self.random_crop = bool(random_crop)
        self.mel_mean = np.asarray(stats.get("mel_mean", [0.0] * N_MELS), dtype=np.float32)
        self.mel_std = np.asarray(stats.get("mel_std", [1.0] * N_MELS), dtype=np.float32)

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor | int]:
        entry = self.entries[index]
        sample_id = str(entry["id"])
        feature_dir = self.features_dir / sample_id
        mel = np.load(feature_dir / "mel.npy").astype(np.float32)
        audio_path = feature_dir / "target_audio.npy"
        if not audio_path.exists():
            raise RuntimeError(f"Vocoder slice {sample_id} is missing target_audio.npy.")
        audio = np.load(audio_path).astype(np.float32).reshape(-1)
        total_frames = int(mel.shape[0])
        start = 0
        end = total_frames
        if total_frames > self.max_frames:
            if self.random_crop:
                start = random.randint(0, total_frames - self.max_frames)
            end = start + self.max_frames
        mel = mel[start:end].astype(np.float32, copy=False)
        sample_start = start * HOP_LENGTH
        sample_end = sample_start + (mel.shape[0] * HOP_LENGTH)
        audio = audio[sample_start:min(sample_end, audio.shape[0])].astype(np.float32, copy=False)
        expected_samples = max(1, mel.shape[0] * HOP_LENGTH)
        if audio.shape[0] < expected_samples:
            audio = np.pad(audio, (0, expected_samples - audio.shape[0])).astype(np.float32, copy=False)
        elif audio.shape[0] > expected_samples:
            audio = audio[:expected_samples].astype(np.float32, copy=False)
        norm_mel = (mel - self.mel_mean[np.newaxis, :]) / self.mel_std[np.newaxis, :]
        return {
            "mel": torch.from_numpy(norm_mel.astype(np.float32, copy=False)),
            "audio": torch.from_numpy(audio.astype(np.float32, copy=False)),
            "mel_length": int(norm_mel.shape[0]),
            "sample_length": int(audio.shape[0]),
        }


def collate_vocoder_slices(batch: Sequence[Dict[str, torch.Tensor | int]]) -> Dict[str, torch.Tensor]:
    mel_lengths = torch.tensor([int(item["mel_length"]) for item in batch], dtype=torch.long)
    sample_lengths = torch.tensor([int(item["sample_length"]) for item in batch], dtype=torch.long)
    mel = pad_sequence([item["mel"] for item in batch], batch_first=True, padding_value=0.0)
    audio = pad_sequence([item["audio"] for item in batch], batch_first=True, padding_value=0.0)
    return {
        "mel": mel,
        "audio": audio,
        "mel_lengths": mel_lengths,
        "sample_lengths": sample_lengths,
    }


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 4096):
        super().__init__()
        self.d_model = int(max(d_model, 2))
        self.register_buffer("pe", self._build_encoding(max_len), persistent=False)

    def _build_encoding(self, max_len: int) -> torch.Tensor:
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, self.d_model, 2) * (-math.log(10000.0) / max(self.d_model, 2)))
        pe = torch.zeros(max_len, self.d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe.unsqueeze(0)

    def _ensure_capacity(self, length: int) -> None:
        if length <= self.pe.shape[1]:
            return
        new_length = max(length, int(self.pe.shape[1]) * 2)
        self.pe = self._build_encoding(new_length)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        length = x.shape[1]
        self._ensure_capacity(length)
        return x + self.pe[:, :length, :].to(dtype=x.dtype, device=x.device)


class ContextualVoiceProfile(nn.Module):
    def __init__(
        self,
        d_model: int,
        *,
        pitch_bins: int = 8,
        energy_bins: int = 4,
        voicing_bins: int = 2,
    ) -> None:
        super().__init__()
        self.pitch_bins = max(2, int(pitch_bins))
        self.energy_bins = max(2, int(energy_bins))
        self.voicing_bins = max(2, int(voicing_bins))
        self.pitch_prototypes = nn.Embedding(self.pitch_bins, d_model)
        self.energy_prototypes = nn.Embedding(self.energy_bins, d_model)
        self.voicing_prototypes = nn.Embedding(self.voicing_bins, d_model)
        self.fuse = nn.Sequential(
            nn.LayerNorm(d_model * 3),
            nn.Linear(d_model * 3, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model),
        )

    def _quantize_normalized(self, values: torch.Tensor, bins: int) -> torch.Tensor:
        clipped = torch.clamp(values.float(), -3.0, 3.0)
        scaled = (clipped + 3.0) / 6.0
        return torch.clamp(torch.round(scaled * float(bins - 1)).long(), 0, bins - 1)

    def forward(
        self,
        *,
        log_f0: torch.Tensor,
        energy: torch.Tensor,
        vuv: torch.Tensor,
        lyric_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        pitch_ids = self._quantize_normalized(log_f0, self.pitch_bins)
        energy_ids = self._quantize_normalized(energy, self.energy_bins)
        voicing_ids = torch.clamp((vuv > 0.5).long(), 0, self.voicing_bins - 1)
        contextual = self.fuse(
            torch.cat(
                [
                    self.pitch_prototypes(pitch_ids),
                    self.energy_prototypes(energy_ids),
                    self.voicing_prototypes(voicing_ids),
                ],
                dim=-1,
            )
        )
        if lyric_mask is not None:
            contextual = contextual * (0.18 + (0.82 * lyric_mask.unsqueeze(-1).to(dtype=contextual.dtype)))
        return contextual


class PersonaMapperNAR(nn.Module):
    def __init__(
        self,
        *,
        content_dim: int = PERSONA_V11_CONTENT_DIM,
        d_model: int = 384,
        n_heads: int = 4,
        n_layers: int = 6,
        dropout: float = 0.1,
        n_mels: int = N_MELS,
        vocab_size: int = len(PHONE_TOKENS),
        voice_signature_dim: int = VOICE_SIGNATURE_DIM,
        use_guide_mel_conditioning: bool = False,
        use_voice_prototype_conditioning: bool = True,
        guide_condition_scale: float = 0.55,
    ) -> None:
        super().__init__()
        self.content_dim = max(32, int(content_dim))
        self.voice_signature_dim = max(1, int(voice_signature_dim))
        self.n_mels = int(n_mels)
        self.use_guide_mel_conditioning = bool(use_guide_mel_conditioning)
        self.use_voice_prototype_conditioning = bool(use_voice_prototype_conditioning)
        self.guide_condition_scale = float(max(0.0, guide_condition_scale))
        input_dim = self.content_dim + 2 + self.voice_signature_dim
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
        )
        if self.use_guide_mel_conditioning:
            self.guide_proj = nn.Sequential(
                nn.Linear(self.n_mels, d_model),
                nn.LayerNorm(d_model),
                nn.GELU(),
            )
            self.guide_gate = nn.Sequential(
                nn.Linear(d_model * 2, d_model),
                nn.Sigmoid(),
            )
        else:
            self.guide_proj = None
            self.guide_gate = None
        self.positional = PositionalEncoding(d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=max(1, int(n_layers)))
        self.output_norm = nn.LayerNorm(d_model)
        self.mel_proj = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, self.n_mels),
        )
        self.f0_proj = nn.Linear(d_model, 1)
        self.vuv_proj = nn.Linear(d_model, 1)
        self.phone_proj = nn.Linear(d_model, int(vocab_size))

    def forward(
        self,
        *,
        content: torch.Tensor,
        log_f0: torch.Tensor,
        beat_phase: torch.Tensor,
        lengths: torch.Tensor,
        voice_prototype: torch.Tensor,
        guide_mel: Optional[torch.Tensor] = None,
        return_aux: bool = False,
    ) -> Dict[str, torch.Tensor] | torch.Tensor:
        max_frames = int(content.shape[1])
        if self.use_voice_prototype_conditioning:
            repeated_voice = voice_prototype.unsqueeze(1).expand(-1, max_frames, -1)
        else:
            repeated_voice = torch.zeros(
                content.shape[0],
                max_frames,
                self.voice_signature_dim,
                device=content.device,
                dtype=content.dtype,
            )
        x = torch.cat([content, log_f0, beat_phase, repeated_voice], dim=-1)
        x = self.input_proj(x)
        if self.use_guide_mel_conditioning and guide_mel is not None and self.guide_proj is not None:
            guide_context = self.guide_proj(guide_mel.to(dtype=x.dtype))
            if self.guide_gate is not None:
                guide_gate = self.guide_gate(torch.cat([x, guide_context], dim=-1))
                guide_context = guide_gate * guide_context
            x = x + (self.guide_condition_scale * guide_context)
        x = self.positional(x)
        padding_mask = torch.arange(max_frames, device=lengths.device).unsqueeze(0) >= lengths.unsqueeze(1)
        x = self.transformer(x, src_key_padding_mask=padding_mask)
        x = self.output_norm(x)
        mel = self.mel_proj(x)
        target_log_f0 = self.f0_proj(x).squeeze(-1)
        target_vuv_logits = self.vuv_proj(x).squeeze(-1)
        phone_logits = self.phone_proj(x)
        outputs = {
            "mel": mel,
            "target_log_f0": target_log_f0,
            "target_vuv_logits": target_vuv_logits,
            "phone_logits": phone_logits,
        }
        return outputs if return_aux else mel


class FrameConditionedMelRegenerator(nn.Module):
    def __init__(
        self,
        *,
        vocab_size: int = len(PHONE_TOKENS),
        d_model: int = 192,
        n_heads: int = 4,
        n_layers: int = 6,
        dropout: float = 0.1,
        n_mels: int = N_MELS,
        voice_signature_dim: int = VOICE_SIGNATURE_DIM,
    ):
        super().__init__()
        self.phone_emb = nn.Embedding(vocab_size, d_model)
        self.f0_proj = nn.Sequential(nn.Linear(1, d_model), nn.SiLU(), nn.Linear(d_model, d_model))
        self.energy_proj = nn.Sequential(nn.Linear(1, d_model), nn.SiLU(), nn.Linear(d_model, d_model))
        self.vuv_proj = nn.Sequential(nn.Linear(1, d_model), nn.SiLU(), nn.Linear(d_model, d_model))
        self.voice_style_proj = nn.Sequential(
            nn.Linear(voice_signature_dim, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model),
        )
        self.positional = PositionalEncoding(d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.output_proj = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, n_mels),
        )
        self.pitch_proj = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, 1),
        )
        self.voicing_proj = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, 1),
        )
        self.phone_proj = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, vocab_size),
        )

    def forward(
        self,
        *,
        phone_ids: torch.Tensor,
        log_f0: torch.Tensor,
        vuv: torch.Tensor,
        energy: torch.Tensor,
        lengths: torch.Tensor,
        voice_prototype: Optional[torch.Tensor] = None,
        return_aux: bool = False,
    ) -> torch.Tensor | Dict[str, torch.Tensor]:
        x = self.phone_emb(phone_ids)
        x = x + self.f0_proj(log_f0.unsqueeze(-1))
        x = x + self.vuv_proj(vuv.unsqueeze(-1))
        x = x + self.energy_proj(energy.unsqueeze(-1))
        if voice_prototype is None:
            voice_prototype = torch.zeros(
                phone_ids.shape[0],
                VOICE_SIGNATURE_DIM,
                device=phone_ids.device,
                dtype=log_f0.dtype,
            )
        x = x + self.voice_style_proj(voice_prototype.to(dtype=x.dtype)).unsqueeze(1)
        x = self.positional(x)
        max_frames = int(phone_ids.shape[1])
        key_padding_mask = (
            torch.arange(max_frames, device=lengths.device).unsqueeze(0) >= lengths.unsqueeze(1)
        )
        x = self.encoder(x, src_key_padding_mask=key_padding_mask)
        mel = self.output_proj(x)
        if return_aux:
            return {
                "mel": mel,
                "target_log_f0": self.pitch_proj(x).squeeze(-1),
                "target_vuv_logits": self.voicing_proj(x).squeeze(-1),
                "phone_logits": self.phone_proj(x),
            }
        return mel


class GuideConditionedMelRegenerator(nn.Module):
    def __init__(
        self,
        *,
        vocab_size: int = len(PHONE_TOKENS),
        d_model: int = 256,
        n_heads: int = 4,
        n_layers: int = 8,
        dropout: float = 0.1,
        n_mels: int = N_MELS,
        voice_signature_dim: int = VOICE_SIGNATURE_DIM,
        guide_residual_refinement: bool = False,
        guide_residual_scale: float = 1.0,
        off_lyric_guide_floor: float = 0.04,
        stability_refine_enabled: bool = False,
        stability_refine_scale: float = 0.2,
        guide_mix_floor: float = 0.04,
        guide_mix_ceiling: float = 0.45,
        contextual_voice_scale: float = 0.28,
    ):
        super().__init__()
        self.guide_residual_refinement = bool(guide_residual_refinement)
        self.guide_residual_scale = float(max(0.05, guide_residual_scale))
        self.off_lyric_guide_floor = float(np.clip(off_lyric_guide_floor, 0.0, 0.35))
        self.stability_refine_enabled = bool(stability_refine_enabled)
        self.stability_refine_scale = float(np.clip(stability_refine_scale, 0.04, 0.65))
        self.guide_mix_floor = float(np.clip(guide_mix_floor, 0.0, 0.4))
        self.guide_mix_ceiling = float(
            np.clip(max(self.guide_mix_floor + 0.05, guide_mix_ceiling), self.guide_mix_floor + 0.05, 0.85)
        )
        self.contextual_voice_scale = float(np.clip(contextual_voice_scale, 0.0, 0.8))
        self.phone_emb = nn.Embedding(vocab_size, d_model)
        self.f0_proj = nn.Sequential(nn.Linear(1, d_model), nn.SiLU(), nn.Linear(d_model, d_model))
        self.energy_proj = nn.Sequential(nn.Linear(1, d_model), nn.SiLU(), nn.Linear(d_model, d_model))
        self.vuv_proj = nn.Sequential(nn.Linear(1, d_model), nn.SiLU(), nn.Linear(d_model, d_model))
        self.guide_mel_proj = nn.Sequential(
            nn.Linear(n_mels, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )
        self.voice_style_proj = nn.Sequential(
            nn.Linear(voice_signature_dim, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model),
        )
        self.contextual_voice_profile = ContextualVoiceProfile(d_model)
        self.cond_fuse = nn.Sequential(
            nn.LayerNorm(d_model * 2),
            nn.Linear(d_model * 2, d_model),
            nn.GELU(),
        )
        self.positional = PositionalEncoding(d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.output_proj = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, n_mels),
        )
        self.residual_proj = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, n_mels),
            nn.Tanh(),
        )
        self.blend_gate = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, n_mels),
            nn.Sigmoid(),
        )
        self.stability_temporal = nn.Sequential(
            nn.Conv1d(d_model, d_model, kernel_size=5, padding=2),
            nn.GELU(),
            nn.Conv1d(d_model, d_model, kernel_size=7, padding=3),
            nn.GELU(),
        )
        self.stability_residual = nn.Conv1d(d_model, n_mels, kernel_size=1)
        self.stability_gate = nn.Conv1d(d_model, n_mels, kernel_size=1)
        self.pitch_proj = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, 1),
        )
        self.voicing_proj = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, 1),
        )
        self.phone_proj = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, vocab_size),
        )
        self._initialize_residual_refinement_path()
        self._initialize_stability_refinement_path()

    def _initialize_residual_refinement_path(self) -> None:
        residual_linear = self.residual_proj[3]
        gate_linear = self.blend_gate[3]
        if isinstance(residual_linear, nn.Linear):
            nn.init.zeros_(residual_linear.weight)
            nn.init.zeros_(residual_linear.bias)
        if isinstance(gate_linear, nn.Linear):
            nn.init.zeros_(gate_linear.weight)
            nn.init.constant_(gate_linear.bias, -1.2)

    def _initialize_stability_refinement_path(self) -> None:
        nn.init.zeros_(self.stability_residual.weight)
        if self.stability_residual.bias is not None:
            nn.init.zeros_(self.stability_residual.bias)
        nn.init.zeros_(self.stability_gate.weight)
        if self.stability_gate.bias is not None:
            nn.init.constant_(self.stability_gate.bias, -2.0)

    def forward(
        self,
        *,
        guide_mel: torch.Tensor,
        phone_ids: torch.Tensor,
        log_f0: torch.Tensor,
        vuv: torch.Tensor,
        energy: torch.Tensor,
        lengths: torch.Tensor,
        voice_prototype: Optional[torch.Tensor] = None,
        lyric_mask: Optional[torch.Tensor] = None,
        return_aux: bool = False,
    ) -> torch.Tensor | Dict[str, torch.Tensor]:
        content = self.phone_emb(phone_ids)
        content = content + self.f0_proj(log_f0.unsqueeze(-1))
        content = content + self.vuv_proj(vuv.unsqueeze(-1))
        content = content + self.energy_proj(energy.unsqueeze(-1))
        if lyric_mask is None:
            lyric_gate = (
                (phone_ids != PHONE_TO_ID["SP"]) & (phone_ids != PHONE_TO_ID["PAD"])
            ).to(dtype=guide_mel.dtype).unsqueeze(-1)
        else:
            lyric_gate = lyric_mask.to(dtype=guide_mel.dtype).unsqueeze(-1)
        gated_guide_mel = guide_mel * (0.08 + (0.92 * lyric_gate))
        guide = self.guide_mel_proj(gated_guide_mel)
        x = self.cond_fuse(torch.cat([content, guide], dim=-1))
        if voice_prototype is None:
            voice_prototype = torch.zeros(
                phone_ids.shape[0],
                VOICE_SIGNATURE_DIM,
                device=phone_ids.device,
                dtype=guide_mel.dtype,
            )
        x = x + self.voice_style_proj(voice_prototype.to(dtype=x.dtype)).unsqueeze(1)
        if self.contextual_voice_scale > 0.0:
            contextual_voice = self.contextual_voice_profile(
                log_f0=log_f0,
                energy=energy,
                vuv=vuv,
                lyric_mask=lyric_mask,
            ).to(dtype=x.dtype)
            x = x + (self.contextual_voice_scale * contextual_voice)
        x = self.positional(x)
        max_frames = int(phone_ids.shape[1])
        key_padding_mask = (
            torch.arange(max_frames, device=lengths.device).unsqueeze(0) >= lengths.unsqueeze(1)
        )
        x = self.encoder(x, src_key_padding_mask=key_padding_mask)
        if self.guide_residual_refinement:
            guide_keep = guide_mel * (
                self.off_lyric_guide_floor
                + ((1.0 - self.off_lyric_guide_floor) * lyric_gate)
            )
            predicted_full = self.output_proj(x) + (self.guide_residual_scale * self.residual_proj(x))
            guide_mix = self.guide_mix_floor + (
                (self.guide_mix_ceiling - self.guide_mix_floor) * self.blend_gate(x)
            )
            guide_mix = guide_mix * (0.12 + (0.88 * lyric_gate.to(dtype=guide_keep.dtype)))
            mel = (guide_mix * guide_keep) + ((1.0 - guide_mix) * predicted_full)
        else:
            mel = self.output_proj(x)
        stability_delta = torch.zeros_like(mel)
        stability_gate = torch.zeros_like(mel)
        if self.stability_refine_enabled:
            temporal_features = self.stability_temporal(x.transpose(1, 2))
            stability_delta = torch.tanh(self.stability_residual(temporal_features)).transpose(1, 2)
            stability_gate = torch.sigmoid(self.stability_gate(temporal_features)).transpose(1, 2)
            stability_mask = 0.06 + (0.94 * lyric_gate.to(dtype=mel.dtype))
            mel = mel + (self.stability_refine_scale * stability_mask * stability_gate * stability_delta)
        if return_aux:
            return {
                "mel": mel,
                "target_log_f0": self.pitch_proj(x).squeeze(-1),
                "target_vuv_logits": self.voicing_proj(x).squeeze(-1),
                "phone_logits": self.phone_proj(x),
                "stability_delta": stability_delta,
                "stability_gate": stability_gate,
            }
        return mel


class GuideConditionedMelRegeneratorV11(GuideConditionedMelRegenerator):
    def __init__(
        self,
        *,
        vocab_size: int = len(PHONE_TOKENS),
        d_model: int = 256,
        n_heads: int = 4,
        n_layers: int = 8,
        dropout: float = 0.1,
        n_mels: int = N_MELS,
        voice_signature_dim: int = VOICE_SIGNATURE_DIM,
        guide_residual_scale: float = 0.72,
        off_lyric_guide_floor: float = 0.02,
        stability_refine_enabled: bool = False,
        stability_refine_scale: float = 0.12,
        contextual_voice_scale: float = 0.34,
        guide_context_layers: int = 2,
        guide_context_scale: float = 0.42,
        guide_delta_scale: float = 0.55,
        coherence_refine_scale: float = 0.22,
    ) -> None:
        super().__init__(
            vocab_size=vocab_size,
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers,
            dropout=dropout,
            n_mels=n_mels,
            voice_signature_dim=voice_signature_dim,
            guide_residual_refinement=False,
            guide_residual_scale=guide_residual_scale,
            off_lyric_guide_floor=off_lyric_guide_floor,
            stability_refine_enabled=stability_refine_enabled,
            stability_refine_scale=stability_refine_scale,
            guide_mix_floor=0.0,
            guide_mix_ceiling=0.0,
            contextual_voice_scale=contextual_voice_scale,
        )
        self.guide_context_scale = float(np.clip(guide_context_scale, 0.0, 1.2))
        self.guide_delta_scale = float(np.clip(guide_delta_scale, 0.0, 1.5))
        self.coherence_refine_scale = float(np.clip(coherence_refine_scale, 0.0, 0.8))
        context_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.guide_context_encoder = nn.TransformerEncoder(
            context_layer,
            num_layers=max(1, int(guide_context_layers)),
        )
        self.guide_cross_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.guide_cross_fuse = nn.Sequential(
            nn.LayerNorm(d_model * 2),
            nn.Linear(d_model * 2, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
            nn.Sigmoid(),
        )
        self.guide_delta_proj = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, n_mels),
            nn.Tanh(),
        )
        self.guide_delta_gate = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, n_mels),
            nn.Sigmoid(),
        )
        self.coherence_temporal = nn.Sequential(
            nn.Conv1d(d_model, d_model, kernel_size=5, padding=2),
            nn.GELU(),
            nn.Conv1d(d_model, d_model, kernel_size=9, padding=4),
            nn.GELU(),
        )
        self.coherence_proj = nn.Conv1d(d_model, n_mels, kernel_size=1)
        self.coherence_gate = nn.Conv1d(d_model, n_mels, kernel_size=1)
        self._initialize_v11_paths()

    def _initialize_v11_paths(self) -> None:
        delta_linear = self.guide_delta_proj[3]
        delta_gate_linear = self.guide_delta_gate[3]
        if isinstance(delta_linear, nn.Linear):
            nn.init.zeros_(delta_linear.weight)
            nn.init.zeros_(delta_linear.bias)
        if isinstance(delta_gate_linear, nn.Linear):
            nn.init.zeros_(delta_gate_linear.weight)
            nn.init.constant_(delta_gate_linear.bias, -1.6)
        nn.init.zeros_(self.coherence_proj.weight)
        if self.coherence_proj.bias is not None:
            nn.init.zeros_(self.coherence_proj.bias)
        nn.init.zeros_(self.coherence_gate.weight)
        if self.coherence_gate.bias is not None:
            nn.init.constant_(self.coherence_gate.bias, -1.8)

    def forward(
        self,
        *,
        guide_mel: torch.Tensor,
        phone_ids: torch.Tensor,
        log_f0: torch.Tensor,
        vuv: torch.Tensor,
        energy: torch.Tensor,
        lengths: torch.Tensor,
        voice_prototype: Optional[torch.Tensor] = None,
        lyric_mask: Optional[torch.Tensor] = None,
        return_aux: bool = False,
    ) -> torch.Tensor | Dict[str, torch.Tensor]:
        content = self.phone_emb(phone_ids)
        content = content + self.f0_proj(log_f0.unsqueeze(-1))
        content = content + self.vuv_proj(vuv.unsqueeze(-1))
        content = content + self.energy_proj(energy.unsqueeze(-1))
        if lyric_mask is None:
            lyric_gate = (
                (phone_ids != PHONE_TO_ID["SP"]) & (phone_ids != PHONE_TO_ID["PAD"])
            ).to(dtype=guide_mel.dtype).unsqueeze(-1)
        else:
            lyric_gate = lyric_mask.to(dtype=guide_mel.dtype).unsqueeze(-1)
        guide_context_input = guide_mel * (self.off_lyric_guide_floor + ((1.0 - self.off_lyric_guide_floor) * lyric_gate))
        guide = self.guide_mel_proj(guide_context_input)
        x = self.cond_fuse(torch.cat([content, guide], dim=-1))
        if voice_prototype is None:
            voice_prototype = torch.zeros(
                phone_ids.shape[0],
                VOICE_SIGNATURE_DIM,
                device=phone_ids.device,
                dtype=guide_mel.dtype,
            )
        x = x + self.voice_style_proj(voice_prototype.to(dtype=x.dtype)).unsqueeze(1)
        if self.contextual_voice_scale > 0.0:
            contextual_voice = self.contextual_voice_profile(
                log_f0=log_f0,
                energy=energy,
                vuv=vuv,
                lyric_mask=lyric_mask,
            ).to(dtype=x.dtype)
            x = x + (self.contextual_voice_scale * contextual_voice)
        x = self.positional(x)
        guide_context = self.positional(guide)
        max_frames = int(phone_ids.shape[1])
        key_padding_mask = (
            torch.arange(max_frames, device=lengths.device).unsqueeze(0) >= lengths.unsqueeze(1)
        )
        content_latent = self.encoder(x, src_key_padding_mask=key_padding_mask)
        guide_latent = self.guide_context_encoder(guide_context, src_key_padding_mask=key_padding_mask)
        attended_guide, _ = self.guide_cross_attn(
            query=content_latent,
            key=guide_latent,
            value=guide_latent,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        cross_gate = self.guide_cross_fuse(torch.cat([content_latent, attended_guide], dim=-1))
        fused = content_latent + (self.guide_context_scale * cross_gate * attended_guide)
        base_mel = self.output_proj(fused)
        detail_delta = self.guide_residual_scale * self.residual_proj(fused)
        guide_delta_gate = self.guide_delta_gate(attended_guide)
        guide_delta_gate = guide_delta_gate * (0.08 + (0.92 * lyric_gate.to(dtype=guide_delta_gate.dtype)))
        guide_delta = self.guide_delta_scale * self.guide_delta_proj(attended_guide)
        temporal_features = self.coherence_temporal(fused.transpose(1, 2))
        coherence_delta = torch.tanh(self.coherence_proj(temporal_features)).transpose(1, 2)
        coherence_gate = torch.sigmoid(self.coherence_gate(temporal_features)).transpose(1, 2)
        coherence_gate = coherence_gate * (0.10 + (0.90 * lyric_gate.to(dtype=coherence_gate.dtype)))
        mel = base_mel + detail_delta + (guide_delta_gate * guide_delta)
        mel = mel + (self.coherence_refine_scale * coherence_gate * coherence_delta)
        stability_delta = torch.zeros_like(mel)
        stability_gate = torch.zeros_like(mel)
        if self.stability_refine_enabled:
            stability_features = self.stability_temporal(fused.transpose(1, 2))
            stability_delta = torch.tanh(self.stability_residual(stability_features)).transpose(1, 2)
            stability_gate = torch.sigmoid(self.stability_gate(stability_features)).transpose(1, 2)
            stability_mask = 0.04 + (0.96 * lyric_gate.to(dtype=mel.dtype))
            mel = mel + (self.stability_refine_scale * stability_mask * stability_gate * stability_delta)
        if return_aux:
            return {
                "mel": mel,
                "guide_delta": guide_delta,
                "guide_delta_gate": guide_delta_gate,
                "coherence_delta": coherence_delta,
                "coherence_gate": coherence_gate,
                "target_log_f0": self.pitch_proj(fused).squeeze(-1),
                "target_vuv_logits": self.voicing_proj(fused).squeeze(-1),
                "phone_logits": self.phone_proj(fused),
                "stability_delta": stability_delta,
                "stability_gate": stability_gate,
            }
        return mel


class PersonaPostProcessRefiner(nn.Module):
    def __init__(
        self,
        *,
        vocab_size: int = len(PHONE_TOKENS),
        d_model: int = 192,
        n_heads: int = 4,
        n_layers: int = 6,
        dropout: float = 0.1,
        n_mels: int = N_MELS,
        voice_signature_dim: int = VOICE_SIGNATURE_DIM,
    ):
        super().__init__()
        self.phone_emb = nn.Embedding(vocab_size, d_model)
        self.f0_proj = nn.Sequential(nn.Linear(1, d_model), nn.SiLU(), nn.Linear(d_model, d_model))
        self.energy_proj = nn.Sequential(nn.Linear(1, d_model), nn.SiLU(), nn.Linear(d_model, d_model))
        self.vuv_proj = nn.Sequential(nn.Linear(1, d_model), nn.SiLU(), nn.Linear(d_model, d_model))
        self.guide_mel_proj = nn.Sequential(
            nn.Linear(n_mels, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )
        self.coarse_mel_proj = nn.Sequential(
            nn.Linear(n_mels, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )
        self.voice_style_proj = nn.Sequential(
            nn.Linear(voice_signature_dim, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model),
        )
        self.cond_fuse = nn.Sequential(
            nn.LayerNorm(d_model * 3),
            nn.Linear(d_model * 3, d_model),
            nn.GELU(),
        )
        self.positional = PositionalEncoding(d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.residual_proj = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, n_mels),
            nn.Tanh(),
        )
        self.blend_gate = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, n_mels),
            nn.Sigmoid(),
        )

    def forward(
        self,
        *,
        coarse_mel: torch.Tensor,
        guide_mel: torch.Tensor,
        phone_ids: torch.Tensor,
        log_f0: torch.Tensor,
        vuv: torch.Tensor,
        energy: torch.Tensor,
        lengths: torch.Tensor,
        voice_prototype: Optional[torch.Tensor] = None,
        lyric_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        content = self.phone_emb(phone_ids)
        content = content + self.f0_proj(log_f0.unsqueeze(-1))
        content = content + self.vuv_proj(vuv.unsqueeze(-1))
        content = content + self.energy_proj(energy.unsqueeze(-1))
        if lyric_mask is None:
            lyric_gate = (
                (phone_ids != PHONE_TO_ID["SP"]) & (phone_ids != PHONE_TO_ID["PAD"])
            ).to(dtype=guide_mel.dtype).unsqueeze(-1)
        else:
            lyric_gate = lyric_mask.to(dtype=guide_mel.dtype).unsqueeze(-1)
        guided_reference = guide_mel * (0.12 + (0.88 * lyric_gate))
        guide = self.guide_mel_proj(guided_reference)
        coarse = self.coarse_mel_proj(coarse_mel)
        x = self.cond_fuse(torch.cat([content, guide, coarse], dim=-1))
        if voice_prototype is None:
            voice_prototype = torch.zeros(
                phone_ids.shape[0],
                VOICE_SIGNATURE_DIM,
                device=phone_ids.device,
                dtype=guide_mel.dtype,
            )
        x = x + self.voice_style_proj(voice_prototype.to(dtype=x.dtype)).unsqueeze(1)
        x = self.positional(x)
        max_frames = int(phone_ids.shape[1])
        key_padding_mask = (
            torch.arange(max_frames, device=lengths.device).unsqueeze(0) >= lengths.unsqueeze(1)
        )
        x = self.encoder(x, src_key_padding_mask=key_padding_mask)
        gate = self.blend_gate(x) * (0.22 + (0.78 * lyric_gate.to(dtype=x.dtype)))
        residual = 0.35 * self.residual_proj(x)
        return coarse_mel + (gate * residual)


class VocoderResidualBlock(nn.Module):
    def __init__(self, channels: int, dilation: int):
        super().__init__()
        padding = dilation * 3
        self.block = nn.Sequential(
            nn.LeakyReLU(0.1),
            nn.Conv1d(channels, channels, kernel_size=7, padding=padding, dilation=dilation),
            nn.LeakyReLU(0.1),
            nn.Conv1d(channels, channels, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(x)


class PersonaNeuralVocoder(nn.Module):
    def __init__(
        self,
        *,
        n_mels: int = N_MELS,
        base_channels: int = 384,
        upsample_rates: Sequence[int] = (8, 8, 8),
    ):
        super().__init__()
        channels = int(base_channels)
        self.input_proj = nn.Conv1d(n_mels, channels, kernel_size=7, padding=3)
        self.upsamplers = nn.ModuleList()
        self.resblocks = nn.ModuleList()
        current_channels = channels
        for rate in upsample_rates:
            next_channels = max(64, current_channels // 2)
            self.upsamplers.append(
                nn.ConvTranspose1d(
                    current_channels,
                    next_channels,
                    kernel_size=rate * 2,
                    stride=rate,
                    padding=rate // 2,
                )
            )
            self.resblocks.append(
                nn.ModuleList(
                    [
                        VocoderResidualBlock(next_channels, dilation=1),
                        VocoderResidualBlock(next_channels, dilation=3),
                        VocoderResidualBlock(next_channels, dilation=5),
                    ]
                )
            )
            current_channels = next_channels
        self.output_proj = nn.Sequential(
            nn.LeakyReLU(0.1),
            nn.Conv1d(current_channels, current_channels, kernel_size=7, padding=3),
            nn.LeakyReLU(0.1),
            nn.Conv1d(current_channels, 1, kernel_size=7, padding=3),
            nn.Tanh(),
        )

    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        x = mel.transpose(1, 2)
        x = self.input_proj(x)
        for upsampler, residual_stack in zip(self.upsamplers, self.resblocks):
            x = upsampler(F.leaky_relu(x, negative_slope=0.1))
            residual_outputs = [block(x) for block in residual_stack]
            x = sum(residual_outputs) / float(len(residual_outputs))
        audio = self.output_proj(x).squeeze(1)
        target_samples = mel.shape[1] * HOP_LENGTH
        if audio.shape[1] > target_samples:
            audio = audio[:, :target_samples]
        elif audio.shape[1] < target_samples:
            audio = F.pad(audio, (0, target_samples - audio.shape[1]))
        return audio


class WavePatchDiscriminator(nn.Module):
    def __init__(self, channels: int = 32):
        super().__init__()
        channel_plan = [1, channels, channels * 2, channels * 4, channels * 8, channels * 8]
        self.layers = nn.ModuleList(
            [
                nn.Conv1d(channel_plan[0], channel_plan[1], kernel_size=15, stride=1, padding=7),
                nn.Conv1d(channel_plan[1], channel_plan[2], kernel_size=41, stride=2, padding=20, groups=4),
                nn.Conv1d(channel_plan[2], channel_plan[3], kernel_size=41, stride=2, padding=20, groups=8),
                nn.Conv1d(channel_plan[3], channel_plan[4], kernel_size=41, stride=4, padding=20, groups=16),
                nn.Conv1d(channel_plan[4], channel_plan[5], kernel_size=41, stride=4, padding=20, groups=16),
            ]
        )
        self.output_proj = nn.Conv1d(channel_plan[-1], 1, kernel_size=5, stride=1, padding=2)

    def forward(self, audio: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        x = audio.unsqueeze(1)
        features: List[torch.Tensor] = []
        for layer in self.layers:
            x = F.leaky_relu(layer(x), negative_slope=0.1)
            features.append(x)
        logits = self.output_proj(x)
        return logits, features


class MultiScaleWaveDiscriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.discriminators = nn.ModuleList(
            [
                WavePatchDiscriminator(32),
                WavePatchDiscriminator(24),
                WavePatchDiscriminator(16),
            ]
        )
        self.pool = nn.AvgPool1d(kernel_size=4, stride=2, padding=1)

    def forward(self, audio: torch.Tensor) -> List[Tuple[torch.Tensor, List[torch.Tensor]]]:
        outputs: List[Tuple[torch.Tensor, List[torch.Tensor]]] = []
        working = audio
        for index, discriminator in enumerate(self.discriminators):
            outputs.append(discriminator(working))
            if index < len(self.discriminators) - 1:
                working = self.pool(working.unsqueeze(1)).squeeze(1)
        return outputs


class GuidedSVSManager:
    def __init__(self, repo_root: Path):
        self.repo_root = Path(repo_root)
        self._inference_bundle_cache: Dict[str, Dict[str, object]] = {}
        self._content_extractor_cache: Dict[str, object] = {}
        self._rmvpe_model_cache: object | None = None

    def _load_json_if_exists(self, path_value: str | Path | None) -> Dict[str, object]:
        if not path_value:
            return {}
        candidate = Path(str(path_value))
        if not candidate.exists():
            return {}
        try:
            loaded = json.loads(candidate.read_text(encoding="utf-8"))
        except Exception:
            return {}
        return loaded if isinstance(loaded, dict) else {}

    def _resolve_bundle_artifact_path(
        self,
        *,
        checkpoint_path: Path,
        raw_value: str | Path | None,
        manifest_path: Path | None = None,
        training_report_path: Path | None = None,
    ) -> Path | None:
        value = str(raw_value or "").strip()
        candidates: List[Path] = []
        if value:
            raw_path = Path(value).expanduser()
            candidates.append(raw_path)
            if not raw_path.is_absolute():
                candidates.append(checkpoint_path.parent / raw_path)
            for depth in (1, 2, 3):
                if len(raw_path.parts) >= depth:
                    tail = Path(*raw_path.parts[-depth:])
                    candidates.append(checkpoint_path.parent / tail)
                    candidates.append(checkpoint_path.parent.parent / tail)
                    if checkpoint_path.parent.parent.parent != checkpoint_path.parent.parent:
                        candidates.append(checkpoint_path.parent.parent.parent / tail)
                    if manifest_path is not None:
                        candidates.append(manifest_path.parent / tail)
                    if training_report_path is not None:
                        candidates.append(training_report_path.parent / tail)
        seen: set[str] = set()
        for candidate in candidates:
            candidate_key = str(candidate)
            if candidate_key in seen:
                continue
            seen.add(candidate_key)
            if candidate.exists():
                return candidate
        return None

    def _build_fallback_inference_stats(
        self,
        *,
        config: Dict[str, object],
        training_report_path: Path | None = None,
    ) -> Dict[str, object]:
        report_payload = self._load_json_if_exists(training_report_path)
        voice_signature_dim = int(
            config.get(
                "voice_signature_dim",
                report_payload.get("voice_signature_dim", VOICE_SIGNATURE_DIM),
            )
            or VOICE_SIGNATURE_DIM
        )
        voice_signature = np.zeros(max(1, voice_signature_dim), dtype=np.float32)
        return {
            "n_mels": int(config.get("n_mels", N_MELS) or N_MELS),
            "voice_signature_dim": voice_signature_dim,
            "mel_mean": [0.0] * N_MELS,
            "mel_std": [1.0] * N_MELS,
            "log_f0_mean": 0.0,
            "log_f0_std": 1.0,
            "energy_mean": 0.0,
            "energy_std": 1.0,
            "global_voice_signature": voice_signature.tolist(),
            "sample_count": int(report_payload.get("sample_count", 0) or 0),
            "fallback_generated": True,
        }

    def _build_uniform_phrase_word_scores(
        self,
        *,
        phrase_text: str,
        duration_seconds: float,
    ) -> List[Dict[str, object]]:
        words = lyrics_to_words(phrase_text)
        if not words:
            return []
        total_duration = max(float(duration_seconds), 0.05)
        weights = np.asarray(
            [max(1, len(pronunciation_units(word))) for word in words],
            dtype=np.float32,
        )
        weights_sum = float(np.sum(weights)) if weights.size else 0.0
        if weights_sum <= 0.0:
            weights = np.ones(len(words), dtype=np.float32)
            weights_sum = float(len(words))
        cumulative = np.concatenate(
            [
                np.array([0.0], dtype=np.float32),
                np.cumsum(weights / weights_sum, dtype=np.float32),
            ]
        )
        scores: List[Dict[str, object]] = []
        for word_index, word in enumerate(words):
            start = float(cumulative[word_index] * total_duration)
            end = float(cumulative[word_index + 1] * total_duration)
            if end <= start:
                end = start + max(total_duration / float(max(len(words), 1)), 0.04)
            scores.append(
                {
                    "index": word_index,
                    "word": word,
                    "start": start,
                    "end": min(total_duration, end),
                }
            )
        return scores

    def _normalize_phrase_word_scores(
        self,
        *,
        phrase_text: str,
        phrase_word_scores: Optional[Sequence[Dict[str, object]]],
        duration_seconds: float,
        prefer_content_alignment: bool = False,
    ) -> List[Dict[str, object]]:
        words = lyrics_to_words(phrase_text)
        if not words:
            return []
        uniform_scores = self._build_uniform_phrase_word_scores(
            phrase_text=phrase_text,
            duration_seconds=duration_seconds,
        )
        usable = [
            dict(entry)
            for entry in (phrase_word_scores or [])
            if str(entry.get("word", "")).strip()
        ]
        if len(usable) != len(words):
            return uniform_scores

        starts = [float(entry.get("start", 0.0)) for entry in usable]
        ends = [max(float(entry.get("end", 0.0)), float(entry.get("start", 0.0)) + 1e-3) for entry in usable]
        offset = min(starts) if starts else 0.0
        total_duration = max(float(duration_seconds), 0.05)
        normalized: List[Dict[str, object]] = []
        monotonic_score = 1.0
        positive_duration_score = 0.0
        previous_end = 0.0
        for word_index, word in enumerate(words):
            raw_start = max(0.0, float(usable[word_index].get("start", 0.0)) - offset)
            raw_end = max(raw_start + 0.01, float(usable[word_index].get("end", raw_start + 0.01)) - offset)
            if raw_start + 1e-4 < previous_end:
                monotonic_score -= 1.0 / float(max(len(words), 1))
            if raw_end > raw_start + 0.015:
                positive_duration_score += 1.0
            previous_end = max(previous_end, raw_end)
            normalized.append(
                {
                    "index": word_index,
                    "word": word,
                    "start": min(total_duration, raw_start),
                    "end": min(total_duration, raw_end),
                }
            )
        if any(float(entry["end"]) <= float(entry["start"]) for entry in normalized):
            return uniform_scores

        last_end = max(float(entry["end"]) for entry in normalized) if normalized else 0.0
        coverage_ratio = float(np.clip(last_end / total_duration, 0.0, 1.0))
        monotonic_score = float(np.clip(monotonic_score, 0.0, 1.0))
        duration_score = float(np.clip(positive_duration_score / float(max(len(words), 1)), 0.0, 1.0))
        timing_reliability = float(np.clip((0.45 * monotonic_score) + (0.35 * duration_score) + (0.20 * coverage_ratio), 0.0, 1.0))
        if prefer_content_alignment:
            timing_reliability *= 0.2

        blended: List[Dict[str, object]] = []
        for raw_entry, uniform_entry in zip(normalized, uniform_scores):
            start = (
                (timing_reliability * float(raw_entry["start"]))
                + ((1.0 - timing_reliability) * float(uniform_entry["start"]))
            )
            end = (
                (timing_reliability * float(raw_entry["end"]))
                + ((1.0 - timing_reliability) * float(uniform_entry["end"]))
            )
            end = max(start + 0.01, end)
            blended.append(
                {
                    "index": int(raw_entry["index"]),
                    "word": str(raw_entry["word"]),
                    "start": min(total_duration, start),
                    "end": min(total_duration, end),
                }
            )
        return blended

    def _resolve_guided_stats_path(
        self,
        *,
        checkpoint_path: Path,
        manifest_path: Path | None = None,
        training_report_path: Path | None = None,
    ) -> Path:
        manifest = self._load_json_if_exists(manifest_path)
        manifest_training_report = self._resolve_bundle_artifact_path(
            checkpoint_path=checkpoint_path,
            raw_value=manifest.get("training_report_path", "") or "",
            manifest_path=manifest_path,
            training_report_path=training_report_path,
        )
        if training_report_path is None and manifest_training_report is not None and manifest_training_report.exists():
            training_report_path = manifest_training_report

        candidates: List[Path] = []
        manifest_stats = str(manifest.get("guided_regeneration_stats_path", "") or "").strip()
        if manifest_stats:
            resolved_manifest_stats = self._resolve_bundle_artifact_path(
                checkpoint_path=checkpoint_path,
                raw_value=manifest_stats,
                manifest_path=manifest_path,
                training_report_path=training_report_path,
            )
            if resolved_manifest_stats is not None:
                candidates.append(resolved_manifest_stats)
        if training_report_path is not None:
            report_payload = self._load_json_if_exists(training_report_path)
            dataset_stats_path = str(
                ((report_payload.get("guided_regeneration_dataset", {}) or {}).get("stats_path", "")) or ""
            ).strip()
            if dataset_stats_path:
                resolved_dataset_stats = self._resolve_bundle_artifact_path(
                    checkpoint_path=checkpoint_path,
                    raw_value=dataset_stats_path,
                    manifest_path=manifest_path,
                    training_report_path=training_report_path,
                )
                if resolved_dataset_stats is not None:
                    candidates.append(resolved_dataset_stats)
        candidates.extend(
            [
                checkpoint_path.parent / "guided_regeneration_stats.json",
                checkpoint_path.parent.parent / "_guided_svs_dataset" / "stats.json",
                checkpoint_path.parent.parent.parent / "_guided_svs_dataset" / "stats.json",
            ]
        )
        for candidate in candidates:
            if candidate.exists():
                return candidate
        raise RuntimeError(
            "Could not locate the pronunciation regenerator stats file for blueprint synthesis."
        )

    def _resolve_vocoder_checkpoint_path(
        self,
        *,
        checkpoint_path: Path,
        manifest_path: Path | None = None,
        training_report_path: Path | None = None,
    ) -> Path | None:
        manifest = self._load_json_if_exists(manifest_path)
        manifest_vocoder = str(manifest.get("guided_vocoder_path", "") or "").strip()
        candidates: List[Path] = []
        if manifest_vocoder:
            candidates.append(Path(manifest_vocoder).expanduser())
        if training_report_path is not None:
            report_payload = self._load_json_if_exists(training_report_path)
            vocoder_path = str((report_payload.get("guided_vocoder", {}) or {}).get("checkpoint_path", "") or "").strip()
            if vocoder_path:
                candidates.append(Path(vocoder_path).expanduser())
        candidates.extend(
            [
                checkpoint_path.parent / "guided_vocoder_best.pt",
                checkpoint_path.parent / "guided_vocoder_latest.pt",
            ]
        )
        for candidate in candidates:
            if candidate.exists():
                return candidate
        return None

    def _resolve_post_process_checkpoint_path(
        self,
        *,
        checkpoint_path: Path,
        manifest_path: Path | None = None,
        training_report_path: Path | None = None,
    ) -> Path | None:
        manifest = self._load_json_if_exists(manifest_path)
        candidates: List[Path] = []
        for manifest_key in ("guided_post_process_latest_path", "guided_post_process_path"):
            manifest_post_process = str(manifest.get(manifest_key, "") or "").strip()
            if manifest_post_process:
                candidates.append(Path(manifest_post_process).expanduser())
        if training_report_path is not None:
            report_payload = self._load_json_if_exists(training_report_path)
            post_process_payload = dict(report_payload.get("guided_post_process", {}) or {})
            for key in ("checkpoint_path", "latest_checkpoint_path"):
                candidate_path = str(post_process_payload.get(key, "") or "").strip()
                if candidate_path:
                    candidates.append(Path(candidate_path).expanduser())
        candidates.extend(
            [
                checkpoint_path.parent / "guided_post_process_best.pt",
                checkpoint_path.parent / "guided_post_process_latest.pt",
            ]
        )
        for candidate in candidates:
            if candidate.exists():
                return candidate
        return None

    def _load_inference_bundle(
        self,
        *,
        checkpoint_path: Path,
        config_path: Path | None = None,
        manifest_path: Path | None = None,
        training_report_path: Path | None = None,
    ) -> Dict[str, object]:
        cache_key = json.dumps(
            {
                "checkpoint_path": str(checkpoint_path.resolve()),
                "config_path": str(config_path.resolve()) if config_path is not None and config_path.exists() else "",
                "manifest_path": str(manifest_path.resolve()) if manifest_path is not None and manifest_path.exists() else "",
                "training_report_path": (
                    str(training_report_path.resolve())
                    if training_report_path is not None and training_report_path.exists()
                    else ""
                ),
            },
            sort_keys=True,
        )
        cached = self._inference_bundle_cache.get(cache_key)
        if cached is not None:
            return cached

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model, checkpoint_config = self._load_model_from_checkpoint(checkpoint_path, device=device)
        config = dict(checkpoint_config)
        if config_path is not None and config_path.exists():
            config.update(self._load_json_if_exists(config_path))
        stats_path_text = ""
        try:
            stats_path = self._resolve_guided_stats_path(
                checkpoint_path=checkpoint_path,
                manifest_path=manifest_path,
                training_report_path=training_report_path,
            )
            stats = self._load_json_if_exists(stats_path)
            stats_path_text = str(stats_path)
        except RuntimeError:
            stats = {}
        if not stats:
            stats = self._build_fallback_inference_stats(
                config=config,
                training_report_path=training_report_path,
            )
            stats_path_text = "<generated-fallback-stats>"
        vocoder_checkpoint_path = self._resolve_vocoder_checkpoint_path(
            checkpoint_path=checkpoint_path,
            manifest_path=manifest_path,
            training_report_path=training_report_path,
        )
        vocoder = None
        vocoder_config: Dict[str, object] = {}
        if vocoder_checkpoint_path is not None:
            try:
                vocoder, vocoder_config = self._load_vocoder_from_checkpoint(
                    vocoder_checkpoint_path,
                    device=device,
                )
            except Exception:
                vocoder = None
                vocoder_config = {}
        post_process_checkpoint_path = self._resolve_post_process_checkpoint_path(
            checkpoint_path=checkpoint_path,
            manifest_path=manifest_path,
            training_report_path=training_report_path,
        )
        post_process = None
        post_process_config: Dict[str, object] = {}
        if post_process_checkpoint_path is not None:
            try:
                post_process, post_process_config = self._load_post_process_from_checkpoint(
                    post_process_checkpoint_path,
                    device=device,
                )
            except Exception:
                post_process = None
                post_process_config = {}

        bundle = {
            "model": model,
            "post_process": post_process,
            "post_process_config": post_process_config,
            "vocoder": vocoder,
            "vocoder_config": vocoder_config,
            "device": device,
            "config": config,
            "stats": stats,
            "voice_prototype": _ensure_voice_signature_dim(stats.get("global_voice_signature", [])),
            "stats_path": stats_path_text,
            "checkpoint_path": str(checkpoint_path),
            "post_process_checkpoint_path": str(post_process_checkpoint_path) if post_process_checkpoint_path is not None else "",
            "vocoder_checkpoint_path": str(vocoder_checkpoint_path) if vocoder_checkpoint_path is not None else "",
        }
        self._inference_bundle_cache[cache_key] = bundle
        return bundle

    def _render_audio_from_bundle(
        self,
        *,
        normalized_mel: np.ndarray,
        bundle: Dict[str, object],
    ) -> Tuple[np.ndarray, str]:
        vocoder = bundle.get("vocoder")
        device = bundle.get("device", torch.device("cpu"))
        if isinstance(vocoder, PersonaNeuralVocoder):
            mel_tensor = torch.from_numpy(np.asarray(normalized_mel, dtype=np.float32)).unsqueeze(0).to(device)
            with torch.no_grad():
                generated_audio = vocoder(mel_tensor).squeeze(0).detach().cpu().numpy().astype(np.float32, copy=False)
            peak = float(np.max(np.abs(generated_audio))) if generated_audio.size else 0.0
            if peak > 0.995:
                generated_audio = generated_audio * np.float32(0.995 / peak)
            return generated_audio, "persona-neural-vocoder-v1"
        predicted_log_mel = self._denormalize_log_mel(normalized_mel, dict(bundle.get("stats", {})))
        return _render_log_mel_to_audio(predicted_log_mel), "griffinlim_preview_only"

    def _extract_f0(
        self,
        mono_audio: np.ndarray,
        sample_rate: int,
        frame_count: int,
        *,
        progress_callback: Optional[Callable[[float, str, str], None]] = None,
        cancel_event: Optional[object] = None,
    ) -> np.ndarray:
        mono = np.asarray(mono_audio, dtype=np.float32).reshape(-1)
        if mono.size <= 1:
            return np.zeros(frame_count, dtype=np.float32)
        chunk_samples = max(int(round(sample_rate * FEATURE_CHUNK_SECONDS)), HOP_LENGTH * 8)
        chunk_starts = list(range(0, mono.size, chunk_samples))
        total_chunks = max(1, len(chunk_starts))

        def check_cancel() -> None:
            if cancel_event is not None and bool(getattr(cancel_event, "is_set", lambda: False)()):
                raise InterruptedError("Training stopped by user.")

        def report_chunk_progress(chunk_index: int, total: int, engine_name: str, start_sample: int, end_sample: int) -> None:
            if progress_callback is None:
                return
            progress_callback(
                chunk_index / float(max(total, 1)),
                f"Extracting guided pitch chunk {chunk_index}/{total}...",
                (
                    f"{engine_name} | "
                    f"{start_sample / float(max(sample_rate, 1)):.1f}s-"
                    f"{end_sample / float(max(sample_rate, 1)):.1f}s"
                ),
            )

        try:
            if self._rmvpe_model_cache is None:
                from rmvpe import RMVPE

                rmvpe_candidates = [
                    self.repo_root / "rmvpe.pt",
                    self.repo_root / "assets" / "rmvpe.pt",
                    self.repo_root / "weights" / "rmvpe.pt",
                ]
                rmvpe_path = next((path for path in rmvpe_candidates if path.exists()), None)
                if rmvpe_path is not None:
                    rmvpe_device = "cuda:0" if torch.cuda.is_available() else "cpu"
                    self._rmvpe_model_cache = RMVPE(str(rmvpe_path), is_half=False, device=rmvpe_device)
            if self._rmvpe_model_cache is not None:
                rmvpe_chunk_samples = max(int(round(sample_rate * min(FEATURE_CHUNK_SECONDS, 6.0))), HOP_LENGTH * 8)
                rmvpe_chunk_starts = list(range(0, mono.size, rmvpe_chunk_samples))
                rmvpe_total_chunks = max(1, len(rmvpe_chunk_starts))
                rmvpe_overlap_samples = max(int(round(sample_rate * 0.35)), HOP_LENGTH * 6)
                collected: List[np.ndarray] = []
                for chunk_index, start_sample in enumerate(rmvpe_chunk_starts, start=1):
                    check_cancel()
                    end_sample = min(mono.size, start_sample + rmvpe_chunk_samples)
                    padded_start = max(0, start_sample - rmvpe_overlap_samples)
                    padded_end = min(mono.size, end_sample + rmvpe_overlap_samples)
                    report_chunk_progress(chunk_index, rmvpe_total_chunks, "rmvpe", start_sample, end_sample)
                    chunk = np.asarray(mono[padded_start:padded_end], dtype=np.float32)
                    f0 = np.asarray(
                        self._rmvpe_model_cache.infer_from_audio(chunk, thred=0.03),
                        dtype=np.float32,
                    ).reshape(-1)
                    padded_frame_count = max(1, int(round((padded_end - padded_start) / float(HOP_LENGTH))))
                    f0 = _align_1d(f0, padded_frame_count).astype(np.float32, copy=False)
                    trim_start = max(0, int(round((start_sample - padded_start) / float(HOP_LENGTH))))
                    core_frame_count = max(1, int(round((end_sample - start_sample) / float(HOP_LENGTH))))
                    trim_end = min(f0.shape[0], trim_start + core_frame_count)
                    core_f0 = f0[trim_start:trim_end]
                    if core_f0.shape[0] != core_frame_count:
                        core_f0 = _align_1d(core_f0, core_frame_count).astype(np.float32, copy=False)
                    collected.append(core_f0.astype(np.float32, copy=False))
                    if torch.cuda.is_available() and str(getattr(self._rmvpe_model_cache, "device", "")).startswith("cuda"):
                        torch.cuda.empty_cache()
                if not collected:
                    return np.zeros(frame_count, dtype=np.float32)
                aligned_f0 = _align_1d(np.concatenate(collected).astype(np.float32, copy=False), frame_count).astype(
                    np.float32,
                    copy=False,
                )
                return self._smooth_f0_contour(aligned_f0)
        except Exception:
            pass

        try:
            import torchcrepe

            device = "cuda" if torch.cuda.is_available() else "cpu"
            collected: List[np.ndarray] = []
            for chunk_index, start_sample in enumerate(chunk_starts, start=1):
                check_cancel()
                end_sample = min(mono.size, start_sample + chunk_samples)
                chunk = np.asarray(mono[start_sample:end_sample], dtype=np.float32)
                tensor = torch.from_numpy(chunk).unsqueeze(0).to(device=device, dtype=torch.float32)
                with torch.no_grad():
                    pitch, periodicity = torchcrepe.predict(
                        tensor,
                        sample_rate,
                        hop_length=HOP_LENGTH,
                        fmin=F0_MIN,
                        fmax=F0_MAX,
                        model="full" if torch.cuda.is_available() else "tiny",
                        return_periodicity=True,
                        batch_size=1024,
                        device=device,
                    )
                frequency = pitch.squeeze(0).detach().cpu().numpy().astype(np.float32)
                confidence = periodicity.squeeze(0).detach().cpu().numpy().astype(np.float32)
                frequency[confidence < 0.35] = 0.0
                collected.append(frequency.reshape(-1))
                report_chunk_progress(chunk_index, total_chunks, "torchcrepe", start_sample, end_sample)
                del tensor, pitch, periodicity
                if device == "cuda":
                    torch.cuda.empty_cache()
            if not collected:
                return np.zeros(frame_count, dtype=np.float32)
            return self._smooth_f0_contour(
                _align_1d(np.concatenate(collected).astype(np.float32, copy=False), frame_count)
            )
        except Exception:
            collected = []
            for chunk_index, start_sample in enumerate(chunk_starts, start=1):
                check_cancel()
                end_sample = min(mono.size, start_sample + chunk_samples)
                chunk = np.asarray(mono[start_sample:end_sample], dtype=np.float32)
                f0, _, _ = librosa.pyin(
                    chunk,
                    sr=sample_rate,
                    fmin=F0_MIN,
                    fmax=F0_MAX,
                    frame_length=N_FFT,
                    hop_length=HOP_LENGTH,
                )
                f0 = np.nan_to_num(np.asarray(f0, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
                collected.append(f0.reshape(-1))
                report_chunk_progress(chunk_index, total_chunks, "pyin fallback", start_sample, end_sample)
            if not collected:
                return np.zeros(frame_count, dtype=np.float32)
            return self._smooth_f0_contour(
                _align_1d(np.concatenate(collected).astype(np.float32, copy=False), frame_count)
            )

    def _build_word_boundaries(
        self,
        *,
        lyrics: str,
        word_scores: List[Dict[str, object]],
        frame_count: int,
        sample_rate: int,
    ) -> List[Dict[str, object]]:
        words = lyrics_to_words(lyrics)
        if frame_count <= 0 or not words:
            return []

        timed_entries = {
            int(entry.get("index", -1)): dict(entry)
            for entry in word_scores
            if int(entry.get("index", -1)) >= 0
        }
        total_words = max(len(words), 1)
        clip_duration_seconds = (frame_count * HOP_LENGTH) / float(max(sample_rate, 1))
        boundaries: List[Dict[str, object]] = []

        for word_index, word in enumerate(words):
            units = pronunciation_units(word) or ["SP"]
            entry = timed_entries.get(word_index, {})
            start_seconds = float(entry.get("start", 0.0))
            end_seconds = float(entry.get("end", 0.0))
            start_frame = int(round((start_seconds * sample_rate) / HOP_LENGTH))
            end_frame = int(round((end_seconds * sample_rate) / HOP_LENGTH))
            if end_frame <= start_frame:
                start_frame = int(round((word_index / float(total_words)) * frame_count))
                end_frame = int(round(((word_index + 1) / float(total_words)) * frame_count))
            start_frame = max(0, min(frame_count - 1, start_frame))
            end_frame = max(start_frame + 1, min(frame_count, end_frame))
            unit_step = (end_frame - start_frame) / float(max(len(units), 1))
            unit_segments: List[Dict[str, object]] = []
            for unit_index, unit in enumerate(units):
                unit_start = start_frame + int(round(unit_index * unit_step))
                unit_end = start_frame + int(round((unit_index + 1) * unit_step))
                unit_start = max(start_frame, min(end_frame - 1, unit_start))
                unit_end = max(unit_start + 1, min(end_frame, unit_end))
                unit_segments.append(
                    {
                        "unit_index": unit_index,
                        "unit": unit if unit in PHONE_TO_ID else "UNK",
                        "start_frame": unit_start,
                        "end_frame": unit_end,
                        "start_seconds": unit_start * HOP_LENGTH / float(max(sample_rate, 1)),
                        "end_seconds": unit_end * HOP_LENGTH / float(max(sample_rate, 1)),
                    }
                )
            boundaries.append(
                {
                    "index": word_index,
                    "word": word,
                    "units": list(units),
                    "start_frame": start_frame,
                    "end_frame": end_frame,
                    "start_seconds": start_frame * HOP_LENGTH / float(max(sample_rate, 1)),
                    "end_seconds": min(
                        clip_duration_seconds,
                        end_frame * HOP_LENGTH / float(max(sample_rate, 1)),
                    ),
                    "unit_segments": unit_segments,
                }
            )
        return boundaries

    def _build_phone_ids(
        self,
        *,
        lyrics: str,
        word_scores: List[Dict[str, object]],
        frame_count: int,
        sample_rate: int,
    ) -> Tuple[np.ndarray, float]:
        phone_ids = np.full(frame_count, PHONE_TO_ID["SP"], dtype=np.int64)
        if frame_count <= 0:
            return phone_ids, 0.0
        covered_frames = np.zeros(frame_count, dtype=np.float32)

        for word_boundary in self._build_word_boundaries(
            lyrics=lyrics,
            word_scores=word_scores,
            frame_count=frame_count,
            sample_rate=sample_rate,
        ):
            for unit_segment in word_boundary.get("unit_segments", []):
                unit_start = int(unit_segment.get("start_frame", 0))
                unit_end = int(unit_segment.get("end_frame", 0))
                unit_name = str(unit_segment.get("unit", "UNK"))
                if unit_end <= unit_start:
                    continue
                phone_ids[unit_start:unit_end] = PHONE_TO_ID.get(unit_name, PHONE_TO_ID["UNK"])
                covered_frames[unit_start:unit_end] = 1.0

        return phone_ids, float(np.mean(covered_frames)) if covered_frames.size else 0.0

    def _build_subword_anchor_spans(self, unit_count: int) -> List[Tuple[int, int]]:
        if unit_count <= 0:
            return []
        if unit_count == 1:
            return [(0, 0)]
        if unit_count == 2:
            return [(0, 1)]
        if unit_count == 3:
            return [(0, 2)]
        if unit_count == 4:
            return [(0, 2), (1, 3)]
        middle_start = max(1, min(unit_count - 3, (unit_count // 2) - 1))
        spans = [
            (0, 2),
            (middle_start, min(unit_count - 1, middle_start + 2)),
            (max(0, unit_count - 3), unit_count - 1),
        ]
        deduped: List[Tuple[int, int]] = []
        seen = set()
        for span in spans:
            if span not in seen:
                seen.add(span)
                deduped.append(span)
        return deduped

    def _save_training_slice(
        self,
        *,
        feature_dir: Path,
        target_log_mel: np.ndarray,
        target_audio: np.ndarray,
        guide_log_mel: np.ndarray,
        f0: np.ndarray,
        log_f0: np.ndarray,
        target_log_f0: np.ndarray,
        target_vuv: np.ndarray,
        vuv: np.ndarray,
        energy: np.ndarray,
        phone_ids: np.ndarray,
        lyric_mask: np.ndarray,
        target_voice_signature: np.ndarray,
        nar_content: Optional[np.ndarray] = None,
        beat_phase: Optional[np.ndarray] = None,
    ) -> None:
        feature_dir.mkdir(parents=True, exist_ok=True)
        np.save(feature_dir / "mel.npy", np.asarray(target_log_mel, dtype=np.float32))
        np.save(feature_dir / "target_audio.npy", np.asarray(target_audio, dtype=np.float32).reshape(-1))
        np.save(feature_dir / "guide_mel.npy", np.asarray(guide_log_mel, dtype=np.float32))
        np.save(feature_dir / "f0.npy", np.asarray(f0, dtype=np.float32))
        np.save(feature_dir / "log_f0.npy", np.asarray(log_f0, dtype=np.float32))
        np.save(feature_dir / "target_log_f0.npy", np.asarray(target_log_f0, dtype=np.float32))
        np.save(feature_dir / "target_vuv.npy", np.asarray(target_vuv, dtype=np.float32))
        np.save(feature_dir / "vuv.npy", np.asarray(vuv, dtype=np.float32))
        np.save(feature_dir / "energy.npy", np.asarray(energy, dtype=np.float32))
        np.save(feature_dir / "phone_ids.npy", np.asarray(phone_ids, dtype=np.int64))
        np.save(feature_dir / "lyric_mask.npy", np.asarray(lyric_mask, dtype=np.float32))
        np.save(
            feature_dir / "target_voice_signature.npy",
            _ensure_voice_signature_dim(target_voice_signature).astype(np.float32, copy=False),
        )
        if nar_content is not None:
            np.save(feature_dir / "nar_content.npy", np.asarray(nar_content, dtype=np.float32))
        if beat_phase is not None:
            np.save(feature_dir / "beat_phase.npy", np.asarray(beat_phase, dtype=np.float32).reshape(-1))

    def _extract_log_mel(self, mono_audio: np.ndarray, sample_rate: int) -> np.ndarray:
        mel = librosa.feature.melspectrogram(
            y=np.asarray(mono_audio, dtype=np.float32),
            sr=sample_rate,
            n_fft=N_FFT,
            hop_length=HOP_LENGTH,
            win_length=WIN_LENGTH,
            n_mels=N_MELS,
            power=2.0,
        )
        return np.log(np.maximum(mel, 1e-5)).T.astype(np.float32, copy=False)

    def _smooth_f0_contour(self, f0_contour: np.ndarray) -> np.ndarray:
        f0 = np.asarray(f0_contour, dtype=np.float32).reshape(-1)
        if f0.size <= 2:
            return f0.astype(np.float32, copy=False)
        voiced_mask = f0 > 50.0
        if not np.any(voiced_mask):
            return np.zeros_like(f0, dtype=np.float32)
        indices = np.arange(f0.shape[0], dtype=np.float32)
        interpolated = np.interp(indices, indices[voiced_mask], f0[voiced_mask]).astype(np.float32)
        kernel = 3
        padded = np.pad(interpolated, (kernel // 2, kernel // 2), mode="edge")
        median_smoothed = np.empty_like(interpolated)
        for idx in range(interpolated.shape[0]):
            median_smoothed[idx] = float(np.median(padded[idx : idx + kernel]))
        smooth_kernel = np.asarray([0.2, 0.6, 0.2], dtype=np.float32)
        smoothed = np.convolve(median_smoothed, smooth_kernel, mode="same").astype(np.float32, copy=False)
        smoothed[~voiced_mask] = 0.0
        return smoothed

    def _get_content_extractor(self) -> Dict[str, object]:
        cached = getattr(self, "_content_extractor_cache", None)
        if isinstance(cached, dict) and cached:
            return cached

        hubert_path_candidates = [
            self.repo_root / "hubert_base.pt",
            self.repo_root / "pretrained" / "hubert_base.pt",
        ]
        hubert_path = next((path for path in hubert_path_candidates if path.exists()), None)
        if hubert_path is not None:
            try:
                from fairseq import checkpoint_utils

                models, saved_cfg, _task = checkpoint_utils.load_model_ensemble_and_task([str(hubert_path)], suffix="")
                model = models[0]
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                model = model.to(device)
                if device.type == "cuda":
                    model = model.half()
                model.eval()
                extractor = {
                    "backend": "hubert-base-v1",
                    "device": device,
                    "model": model,
                    "saved_cfg": saved_cfg,
                    "content_dim": PERSONA_V11_CONTENT_DIM,
                }
                self._content_extractor_cache = extractor
                return extractor
            except Exception:
                pass

        onnx_path = self.repo_root / "pretrained" / "vec-768-layer-12.onnx"
        if onnx_path.exists():
            try:
                from lib.infer_pack.onnx_inference import ContentVec

                extractor = {
                    "backend": "contentvec-onnx",
                    "device": "cuda" if torch.cuda.is_available() else "cpu",
                    "model": ContentVec(str(onnx_path), "cuda" if torch.cuda.is_available() else "cpu"),
                    "content_dim": 768,
                }
                self._content_extractor_cache = extractor
                return extractor
            except Exception:
                pass

        extractor = {
            "backend": "mfcc-fallback",
            "device": "cpu",
            "model": None,
            "content_dim": PERSONA_V11_CONTENT_DIM,
        }
        self._content_extractor_cache = extractor
        return extractor

    def _extract_content_features(
        self,
        mono_audio: np.ndarray,
        sample_rate: int,
        frame_count: int,
    ) -> np.ndarray:
        mono = np.asarray(mono_audio, dtype=np.float32).reshape(-1)
        if mono.size <= 1:
            return np.zeros((frame_count, PERSONA_V11_CONTENT_DIM), dtype=np.float32)
        extractor = self._get_content_extractor()
        target_sr = 16000
        resampled = librosa.resample(mono, orig_sr=sample_rate, target_sr=target_sr).astype(np.float32, copy=False)
        backend = str(extractor.get("backend", "mfcc-fallback"))
        try:
            if backend == "hubert-base-v1":
                feats = torch.from_numpy(resampled).float()
                saved_cfg = extractor["saved_cfg"]
                if bool(getattr(saved_cfg.task, "normalize", False)):
                    with torch.no_grad():
                        feats = F.layer_norm(feats, feats.shape)
                feats = feats.view(1, -1)
                padding_mask = torch.BoolTensor(feats.shape).fill_(False)
                device = extractor["device"]
                model = extractor["model"]
                with torch.no_grad():
                    inputs = {
                        "source": feats.half().to(device) if device.type == "cuda" else feats.to(device),
                        "padding_mask": padding_mask.to(device),
                        "output_layer": 9,
                    }
                    logits = model.extract_features(**inputs)
                    projected = model.final_proj(logits[0]).squeeze(0).float().cpu().numpy()
                return _align_2d(np.asarray(projected, dtype=np.float32), frame_count).astype(np.float32, copy=False)
            if backend == "contentvec-onnx":
                model = extractor["model"]
                content = np.asarray(model.forward(resampled), dtype=np.float32)
                if content.ndim == 3:
                    content = content[0].T
                elif content.ndim == 2 and content.shape[0] < content.shape[1]:
                    content = content.T
                return _align_2d(content.astype(np.float32, copy=False), frame_count).astype(np.float32, copy=False)
        except Exception:
            pass

        mfcc = librosa.feature.mfcc(
            y=resampled,
            sr=target_sr,
            n_mfcc=64,
            n_fft=1024,
            hop_length=320,
        ).T.astype(np.float32, copy=False)
        delta = librosa.feature.delta(mfcc.T).T.astype(np.float32, copy=False)
        delta2 = librosa.feature.delta(mfcc.T, order=2).T.astype(np.float32, copy=False)
        fallback = np.concatenate([mfcc, delta, delta2], axis=1)
        if fallback.shape[1] < PERSONA_V11_CONTENT_DIM:
            repeat_factor = int(math.ceil(PERSONA_V11_CONTENT_DIM / float(max(fallback.shape[1], 1))))
            fallback = np.tile(fallback, (1, repeat_factor))
        fallback = fallback[:, :PERSONA_V11_CONTENT_DIM].astype(np.float32, copy=False)
        return _align_2d(fallback, frame_count).astype(np.float32, copy=False)

    def _compute_beat_phase(
        self,
        mono_audio: np.ndarray,
        sample_rate: int,
        frame_count: int,
    ) -> np.ndarray:
        mono = np.asarray(mono_audio, dtype=np.float32).reshape(-1)
        if mono.size <= 1 or frame_count <= 0:
            return np.zeros(max(frame_count, 1), dtype=np.float32)
        try:
            tempo, beat_frames = librosa.beat.beat_track(
                y=mono,
                sr=sample_rate,
                hop_length=HOP_LENGTH,
                units="frames",
            )
            beat_frames = np.asarray(beat_frames, dtype=np.int64).reshape(-1)
        except Exception:
            tempo = 0.0
            beat_frames = np.zeros(0, dtype=np.int64)
        if beat_frames.size < 2:
            beats_per_second = max(float(tempo) / 60.0, 1.0)
            period_frames = max(8, int(round(sample_rate / float(HOP_LENGTH) / beats_per_second)))
            phase = (np.arange(frame_count, dtype=np.float32) % float(period_frames)) / float(period_frames)
            return phase.astype(np.float32, copy=False)
        phase = np.zeros(frame_count, dtype=np.float32)
        beat_frames = np.clip(beat_frames, 0, max(frame_count - 1, 0))
        beat_frames = np.unique(beat_frames)
        if beat_frames[0] > 0:
            beat_frames = np.concatenate([np.array([0], dtype=np.int64), beat_frames])
        if beat_frames[-1] < frame_count - 1:
            median_step = int(np.median(np.diff(beat_frames))) if beat_frames.size > 1 else max(8, frame_count // 8)
            beat_frames = np.concatenate([beat_frames, np.array([min(frame_count - 1, beat_frames[-1] + max(median_step, 1))], dtype=np.int64)])
        for start, end in zip(beat_frames[:-1], beat_frames[1:]):
            if end <= start:
                continue
            phase[start:end] = np.linspace(0.0, 1.0, end - start, endpoint=False, dtype=np.float32)
        if beat_frames[-1] < frame_count:
            tail = frame_count - beat_frames[-1]
            phase[beat_frames[-1]:] = np.linspace(0.0, 1.0, tail, endpoint=False, dtype=np.float32)
        return phase.astype(np.float32, copy=False)

    def _compute_dtw_alignment(
        self,
        *,
        guide_log_mel: np.ndarray,
        target_log_mel: np.ndarray,
    ) -> Tuple[np.ndarray, float, Dict[str, object]]:
        guide_frames = int(np.asarray(guide_log_mel).shape[0])
        target_frames = int(np.asarray(target_log_mel).shape[0])
        if guide_frames <= 1 or target_frames <= 1:
            frame_map = _default_frame_map(guide_frames, target_frames)
            return frame_map, 0.0, {
                "mode": "linear-fallback",
                "guide_frames": guide_frames,
                "target_frames": target_frames,
                "normalized_cost": None,
            }
        try:
            guide_features = librosa.util.normalize(
                np.asarray(guide_log_mel, dtype=np.float32).T,
                norm=2,
                axis=0,
            )
            target_features = librosa.util.normalize(
                np.asarray(target_log_mel, dtype=np.float32).T,
                norm=2,
                axis=0,
            )
            cost_matrix, warping_path = librosa.sequence.dtw(
                X=guide_features,
                Y=target_features,
                metric="cosine",
            )
            path = np.asarray(warping_path[::-1], dtype=np.int64)
            if path.ndim != 2 or path.shape[1] != 2:
                raise RuntimeError("Unexpected DTW path shape.")
            target_axis = path[:, 1].astype(np.int64, copy=False)
            guide_axis = path[:, 0].astype(np.float32, copy=False)
            aggregated_targets: List[int] = []
            aggregated_guides: List[float] = []
            for target_frame in np.unique(target_axis):
                target_mask = target_axis == target_frame
                aggregated_targets.append(int(target_frame))
                aggregated_guides.append(float(np.mean(guide_axis[target_mask])))
            frame_map = np.interp(
                np.arange(target_frames, dtype=np.float32),
                np.asarray(aggregated_targets, dtype=np.float32),
                np.asarray(aggregated_guides, dtype=np.float32),
                left=float(aggregated_guides[0]),
                right=float(aggregated_guides[-1]),
            ).astype(np.float32)
            frame_map = np.maximum.accumulate(np.clip(frame_map, 0.0, float(max(guide_frames - 1, 0))))
            normalized_cost = float(cost_matrix[-1, -1] / float(max(path.shape[0], 1)))
            alignment_score = float(np.clip(1.0 / (1.0 + normalized_cost), 0.0, 1.0))
            return frame_map, alignment_score, {
                "mode": "dtw-cosine-mel",
                "guide_frames": guide_frames,
                "target_frames": target_frames,
                "path_length": int(path.shape[0]),
                "normalized_cost": round(normalized_cost, 6),
            }
        except Exception:
            frame_map = _default_frame_map(guide_frames, target_frames)
            return frame_map, 0.0, {
                "mode": "linear-fallback",
                "guide_frames": guide_frames,
                "target_frames": target_frames,
                "normalized_cost": None,
            }

    def _warp_word_scores_to_target_timeline(
        self,
        *,
        word_scores: Sequence[Dict[str, object]],
        source_to_target_frame_map: np.ndarray,
        sample_rate: int,
        target_frame_count: int,
    ) -> List[Dict[str, object]]:
        if not word_scores or len(source_to_target_frame_map) <= 0:
            return []
        warped_scores: List[Dict[str, object]] = []
        for entry in word_scores:
            word_index = int(entry.get("index", -1))
            if word_index < 0:
                continue
            start_seconds = float(entry.get("start", 0.0))
            end_seconds = max(start_seconds + 0.01, float(entry.get("end", start_seconds + 0.01)))
            start_frame = int(np.clip(round((start_seconds * sample_rate) / HOP_LENGTH), 0, max(len(source_to_target_frame_map) - 1, 0)))
            end_frame = int(np.clip(round((end_seconds * sample_rate) / HOP_LENGTH), 0, max(len(source_to_target_frame_map) - 1, 0)))
            warped_start_frame = float(source_to_target_frame_map[start_frame]) if len(source_to_target_frame_map) else 0.0
            warped_end_frame = float(source_to_target_frame_map[end_frame]) if len(source_to_target_frame_map) else warped_start_frame + 1.0
            if warped_end_frame <= warped_start_frame:
                warped_end_frame = warped_start_frame + 1.0
            warped_scores.append(
                {
                    "index": word_index,
                    "word": str(entry.get("word", "")),
                    "start": float(np.clip((warped_start_frame * HOP_LENGTH) / float(max(sample_rate, 1)), 0.0, (target_frame_count * HOP_LENGTH) / float(max(sample_rate, 1)))),
                    "end": float(np.clip((warped_end_frame * HOP_LENGTH) / float(max(sample_rate, 1)), 0.0, (target_frame_count * HOP_LENGTH) / float(max(sample_rate, 1)))),
                }
            )
        return warped_scores

    def _compute_difficulty_score(
        self,
        *,
        conditioning_similarity: float,
        alignment_score: float,
        phone_coverage: float,
    ) -> float:
        return float(
            np.clip(
                1.0 - (
                    (0.5 * _normalize_similarity_score(conditioning_similarity))
                    + (0.35 * float(np.clip(alignment_score, 0.0, 1.0)))
                    + (0.15 * float(np.clip(phone_coverage, 0.0, 1.0)))
                ),
                0.0,
                1.0,
            )
        )

    def _build_proxy_activity_tokens(
        self,
        *,
        primary_f0: np.ndarray,
        primary_energy: np.ndarray,
        secondary_f0: Optional[np.ndarray] = None,
        secondary_energy: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        primary_f0 = np.asarray(primary_f0, dtype=np.float32).reshape(-1)
        primary_energy = np.asarray(primary_energy, dtype=np.float32).reshape(-1)
        frame_count = int(min(primary_f0.shape[0], primary_energy.shape[0]))
        if frame_count <= 0:
            return (
                np.zeros((0,), dtype=np.int64),
                np.zeros((0,), dtype=np.float32),
            )
        secondary_f0_array = (
            np.asarray(secondary_f0, dtype=np.float32).reshape(-1)
            if secondary_f0 is not None
            else np.zeros((frame_count,), dtype=np.float32)
        )
        secondary_energy_array = (
            np.asarray(secondary_energy, dtype=np.float32).reshape(-1)
            if secondary_energy is not None
            else np.zeros((frame_count,), dtype=np.float32)
        )
        secondary_f0_array = _align_1d(secondary_f0_array, frame_count)
        secondary_energy_array = _align_1d(secondary_energy_array, frame_count)
        combined_energy = np.maximum(primary_energy[:frame_count], secondary_energy_array[:frame_count])
        voiced_mask = (primary_f0[:frame_count] > 0.0) | (secondary_f0_array[:frame_count] > 0.0)
        positive_energy = combined_energy[combined_energy > 1e-6]
        if positive_energy.size:
            energy_floor = float(np.percentile(positive_energy, 35))
        else:
            energy_floor = 0.0
        active_mask = voiced_mask | (combined_energy >= max(energy_floor, 0.015))
        if active_mask.size >= 3:
            active_mask = np.convolve(
                active_mask.astype(np.float32),
                np.ones(3, dtype=np.float32) / 3.0,
                mode="same",
            ) >= 0.34
        phone_ids = np.full(frame_count, PHONE_TO_ID["SP"], dtype=np.int64)
        phone_ids[active_mask] = PHONE_TO_ID["UNK"]
        return phone_ids, active_mask.astype(np.float32, copy=False)

    def build_identity_training_examples(
        self,
        *,
        sample_id_prefix: str,
        source_name: str,
        audio: np.ndarray,
        sample_rate: int,
        output_dir: Path,
        progress_callback: Optional[Callable[[float, str, str], None]] = None,
        cancel_event: Optional[object] = None,
    ) -> List[GuidedSVSFeatureExample]:
        def report(progress: float, message: str, detail: str = "") -> None:
            if progress_callback is not None:
                progress_callback(float(np.clip(progress, 0.0, 1.0)), message, detail)

        def check_cancel() -> None:
            if cancel_event is not None and bool(getattr(cancel_event, "is_set", lambda: False)()):
                raise InterruptedError("Training stopped by user.")

        mono_audio = np.asarray(audio, dtype=np.float32)
        if mono_audio.ndim == 2:
            mono_audio = mono_audio.mean(axis=1)
        mono_audio = np.asarray(mono_audio, dtype=np.float32).reshape(-1)
        duration_seconds = float(len(mono_audio) / max(sample_rate, 1))
        report(
            0.03,
            f"Preparing base-voice identity windows for {source_name}...",
            f"Audio length {duration_seconds:.1f}s",
        )

        check_cancel()
        log_mel = self._extract_log_mel(mono_audio, sample_rate)
        frame_count = int(log_mel.shape[0])
        report(
            0.18,
            f"Guide mel ready for {source_name}.",
            f"Frames {frame_count} | starting pitch extraction",
        )

        check_cancel()
        f0 = self._extract_f0(
            mono_audio,
            sample_rate,
            frame_count,
            progress_callback=lambda chunk_progress, message, detail: report(
                0.2 + (chunk_progress * 0.42),
                message,
                detail,
            ),
            cancel_event=cancel_event,
        )
        energy = librosa.feature.rms(
            y=mono_audio,
            frame_length=N_FFT,
            hop_length=HOP_LENGTH,
            center=True,
        ).squeeze()
        energy = _align_1d(energy, frame_count)
        log_f0 = np.where(f0 > 0.0, np.log(np.maximum(f0, 1.0)), 0.0).astype(np.float32)
        vuv = (f0 > 0.0).astype(np.float32)
        phone_ids, lyric_mask = self._build_proxy_activity_tokens(
            primary_f0=f0,
            primary_energy=energy,
        )
        guide_content = self._extract_content_features(mono_audio, sample_rate, frame_count)
        guide_beat_phase = self._compute_beat_phase(mono_audio, sample_rate, frame_count)

        feature_root = output_dir / "features"
        feature_root.mkdir(parents=True, exist_ok=True)
        window_frames = max(224, int(round((4.5 * sample_rate) / HOP_LENGTH)))
        step_frames = max(96, int(round(window_frames * 0.5)))
        starts = list(range(0, max(frame_count - window_frames, 0) + 1, step_frames))
        if not starts or starts[-1] != max(0, frame_count - window_frames):
            starts.append(max(0, frame_count - window_frames))
        starts = sorted(set(starts))

        sample_entries: List[GuidedSVSFeatureExample] = []
        total_windows = max(1, len(starts))
        for window_index, start_frame in enumerate(starts, start=1):
            check_cancel()
            end_frame = min(frame_count, start_frame + window_frames)
            if end_frame - start_frame < 8:
                continue
            sample_id = f"{sample_id_prefix}_base_{window_index:04d}"
            feature_dir = feature_root / sample_id
            local_target = log_mel[start_frame:end_frame].astype(np.float32, copy=False)
            local_f0 = f0[start_frame:end_frame].astype(np.float32, copy=False)
            local_log_f0 = log_f0[start_frame:end_frame].astype(np.float32, copy=False)
            local_vuv = vuv[start_frame:end_frame].astype(np.float32, copy=False)
            local_energy = energy[start_frame:end_frame].astype(np.float32, copy=False)
            local_phone_ids = phone_ids[start_frame:end_frame].astype(np.int64, copy=False)
            local_lyric_mask = lyric_mask[start_frame:end_frame].astype(np.float32, copy=False)
            local_content = guide_content[start_frame:end_frame].astype(np.float32, copy=False)
            local_beat_phase = guide_beat_phase[start_frame:end_frame].astype(np.float32, copy=False)
            self._save_training_slice(
                feature_dir=feature_dir,
                target_log_mel=local_target,
                target_audio=_slice_audio_for_frame_range(
                    mono_audio,
                    start_frame=start_frame,
                    end_frame=end_frame,
                ),
                guide_log_mel=local_target,
                f0=local_f0,
                log_f0=local_log_f0,
                target_log_f0=local_log_f0,
                target_vuv=local_vuv,
                vuv=local_vuv,
                energy=local_energy,
                phone_ids=local_phone_ids,
                lyric_mask=local_lyric_mask,
                target_voice_signature=_compute_voice_signature_np(
                    log_mel=local_target,
                    log_f0=local_log_f0,
                    vuv=local_vuv,
                ),
                nar_content=local_content,
                beat_phase=local_beat_phase,
            )
            sample_entries.append(
                GuidedSVSFeatureExample(
                    sample_id=sample_id,
                    lyrics="",
                    n_frames=int(local_target.shape[0]),
                    duration_seconds=float((end_frame - start_frame) * HOP_LENGTH / float(max(sample_rate, 1))),
                    aligned_word_count=0,
                    frame_phone_coverage=float(np.mean(local_lyric_mask > 0.5)) if local_lyric_mask.size else 0.0,
                    feature_dir=sample_id,
                    source_clip=source_name,
                    conditioning_clip=source_name,
                    slice_kind="base-identity-window",
                    conditioning_similarity=1.0,
                    alignment_score=1.0,
                    difficulty_score=0.0,
                )
            )
            report(
                0.66 + ((window_index / float(total_windows)) * 0.34),
                f"Saving base-voice window {window_index}/{total_windows} for {source_name}...",
                f"Frames {local_target.shape[0]} | self-conditioned identity learning",
            )

        if not sample_entries:
            raise RuntimeError("Base-voice training windows could not be prepared from the uploaded clip.")
        report(
            1.0,
            f"Finished base-voice identity windows for {source_name}.",
            f"Saved {len(sample_entries)} identity windows",
        )
        return sample_entries

    def build_truth_aligned_examples(
        self,
        *,
        sample_id_prefix: str,
        source_name: str,
        conditioning_name: str,
        guide_audio: np.ndarray,
        target_audio: np.ndarray,
        sample_rate: int,
        output_dir: Path,
        detail_windows: Optional[Sequence[Dict[str, object]]] = None,
        alignment_tolerance: str = "balanced",
        progress_callback: Optional[Callable[[float, str, str], None]] = None,
        cancel_event: Optional[object] = None,
    ) -> List[GuidedSVSFeatureExample]:
        def report(progress: float, message: str, detail: str = "") -> None:
            if progress_callback is not None:
                progress_callback(float(np.clip(progress, 0.0, 1.0)), message, detail)

        def check_cancel() -> None:
            if cancel_event is not None and bool(getattr(cancel_event, "is_set", lambda: False)()):
                raise InterruptedError("Training stopped by user.")

        guide_mono = np.asarray(guide_audio, dtype=np.float32)
        if guide_mono.ndim == 2:
            guide_mono = guide_mono.mean(axis=1)
        target_mono = np.asarray(target_audio, dtype=np.float32)
        if target_mono.ndim == 2:
            target_mono = target_mono.mean(axis=1)
        guide_mono = np.asarray(guide_mono, dtype=np.float32).reshape(-1)
        target_mono = np.asarray(target_mono, dtype=np.float32).reshape(-1)
        if min(int(guide_mono.shape[0]), int(target_mono.shape[0])) < int(sample_rate * 0.30):
            raise RuntimeError("Aligned source/target pair is too short to build truth windows.")
        duration_seconds = float(target_mono.shape[0] / float(max(sample_rate, 1)))
        report(
            0.03,
            f"Preparing DTW-aligned windows for {conditioning_name} -> {source_name}...",
            f"Guide {guide_mono.shape[0] / float(max(sample_rate, 1)):.1f}s | target {duration_seconds:.1f}s",
        )

        check_cancel()
        target_log_mel = self._extract_log_mel(target_mono, sample_rate)
        raw_guide_log_mel = self._extract_log_mel(guide_mono, sample_rate)
        dtw_frame_map, dtw_alignment_score, dtw_metadata = self._compute_dtw_alignment(
            guide_log_mel=raw_guide_log_mel,
            target_log_mel=target_log_mel,
        )
        guide_log_mel = _warp_2d_by_frame_map(raw_guide_log_mel, dtw_frame_map)
        warped_guide_audio = _warp_audio_by_frame_map(
            guide_mono,
            sample_rate=sample_rate,
            frame_map=dtw_frame_map,
            target_sample_count=len(target_mono),
        )
        frame_count = int(target_log_mel.shape[0])
        if frame_count < 8:
            raise RuntimeError("Aligned source/target pair did not produce enough frames.")
        guide_content = self._extract_content_features(warped_guide_audio, sample_rate, frame_count)
        guide_beat_phase = self._compute_beat_phase(warped_guide_audio, sample_rate, frame_count)
        report(
            0.16,
            f"DTW-aligned frames ready for {conditioning_name}.",
            (
                f"Frames {frame_count} | DTW {dtw_metadata.get('mode', 'unknown')} | "
                f"alignment {dtw_alignment_score:.3f}"
            ),
        )

        check_cancel()
        guide_f0 = self._extract_f0(
            warped_guide_audio,
            sample_rate,
            frame_count,
            progress_callback=lambda chunk_progress, message, detail: report(
                0.18 + (chunk_progress * 0.28),
                message,
                detail,
            ),
            cancel_event=cancel_event,
        )
        target_f0 = self._extract_f0(
            target_mono,
            sample_rate,
            frame_count,
            progress_callback=lambda chunk_progress, message, detail: report(
                0.48 + (chunk_progress * 0.16),
                "Extracting target pitch contour...",
                detail,
            ),
            cancel_event=cancel_event,
        )
        guide_energy = librosa.feature.rms(
            y=warped_guide_audio,
            frame_length=N_FFT,
            hop_length=HOP_LENGTH,
            center=True,
        ).squeeze()
        guide_energy = _align_1d(guide_energy, frame_count)
        target_energy = librosa.feature.rms(
            y=target_mono,
            frame_length=N_FFT,
            hop_length=HOP_LENGTH,
            center=True,
        ).squeeze()
        target_energy = _align_1d(target_energy, frame_count)
        guide_log_f0 = np.where(guide_f0 > 0.0, np.log(np.maximum(guide_f0, 1.0)), 0.0).astype(np.float32)
        target_log_f0 = np.where(target_f0 > 0.0, np.log(np.maximum(target_f0, 1.0)), 0.0).astype(np.float32)
        guide_vuv = (guide_f0 > 0.0).astype(np.float32)
        target_vuv = (target_f0 > 0.0).astype(np.float32)
        phone_ids, lyric_mask = self._build_proxy_activity_tokens(
            primary_f0=target_f0,
            primary_energy=target_energy,
            secondary_f0=guide_f0,
            secondary_energy=guide_energy,
        )
        guide_signature = _compute_voice_signature_np(
            log_mel=guide_log_mel,
            log_f0=guide_log_f0,
            vuv=guide_vuv,
        )
        target_signature = _compute_voice_signature_np(
            log_mel=target_log_mel,
            log_f0=target_log_f0,
            vuv=target_vuv,
        )
        conditioning_similarity = float(
            np.clip(
                np.dot(guide_signature, target_signature)
                / (
                    (np.linalg.norm(guide_signature) * np.linalg.norm(target_signature))
                    + 1e-6
                ),
                0.0,
                1.0,
            )
        )

        feature_root = output_dir / "features"
        feature_root.mkdir(parents=True, exist_ok=True)
        normalized_tolerance = str(alignment_tolerance or "balanced").strip().lower()
        window_seconds = {
            "forgiving": 2.6,
            "balanced": 2.3,
            "strict": 2.0,
        }.get(normalized_tolerance, 2.3)
        window_frames = max(128, int(round((window_seconds * sample_rate) / HOP_LENGTH)))
        step_frames = max(56, int(round(window_frames * 0.45)))
        starts = list(range(0, max(frame_count - window_frames, 0) + 1, step_frames))
        if not starts or starts[-1] != max(0, frame_count - window_frames):
            starts.append(max(0, frame_count - window_frames))
        starts = sorted(set(starts))
        minimum_active_ratio = {
            "forgiving": 0.08,
            "balanced": 0.12,
            "strict": 0.18,
        }.get(normalized_tolerance, 0.12)
        sample_entries: List[GuidedSVSFeatureExample] = []
        total_windows = max(1, len(starts))

        def save_window(
            *,
            start_frame: int,
            end_frame: int,
            sample_id: str,
            slice_kind: str,
            local_alignment_score: float,
        ) -> None:
            local_target = target_log_mel[start_frame:end_frame].astype(np.float32, copy=False)
            local_length = int(local_target.shape[0])
            if local_length < 8:
                return
            local_guide = guide_log_mel[start_frame:end_frame].astype(np.float32, copy=False)
            local_f0 = guide_f0[start_frame:end_frame].astype(np.float32, copy=False)
            local_log_f0 = guide_log_f0[start_frame:end_frame].astype(np.float32, copy=False)
            local_target_log_f0 = target_log_f0[start_frame:end_frame].astype(np.float32, copy=False)
            local_target_vuv = target_vuv[start_frame:end_frame].astype(np.float32, copy=False)
            local_vuv = guide_vuv[start_frame:end_frame].astype(np.float32, copy=False)
            local_energy = guide_energy[start_frame:end_frame].astype(np.float32, copy=False)
            local_content = guide_content[start_frame:end_frame].astype(np.float32, copy=False)
            local_beat_phase = guide_beat_phase[start_frame:end_frame].astype(np.float32, copy=False)
            local_phone_ids = phone_ids[start_frame:end_frame].astype(np.int64, copy=False)
            local_lyric_mask = lyric_mask[start_frame:end_frame].astype(np.float32, copy=False)
            local_coverage = float(np.mean(local_lyric_mask > 0.5)) if local_lyric_mask.size else 0.0
            if local_coverage < minimum_active_ratio:
                return
            feature_dir = feature_root / sample_id
            self._save_training_slice(
                feature_dir=feature_dir,
                target_log_mel=local_target,
                target_audio=_slice_audio_for_frame_range(
                    target_mono,
                    start_frame=start_frame,
                    end_frame=end_frame,
                ),
                guide_log_mel=local_guide,
                f0=local_f0,
                log_f0=local_log_f0,
                target_log_f0=local_target_log_f0,
                target_vuv=local_target_vuv,
                vuv=local_vuv,
                energy=local_energy,
                phone_ids=local_phone_ids,
                lyric_mask=local_lyric_mask,
                target_voice_signature=_compute_voice_signature_np(
                    log_mel=local_target,
                    log_f0=local_target_log_f0,
                    vuv=local_target_vuv,
                ),
                nar_content=local_content,
                beat_phase=local_beat_phase,
            )
            sample_entries.append(
                GuidedSVSFeatureExample(
                    sample_id=sample_id,
                    lyrics="",
                    n_frames=local_length,
                    duration_seconds=float(local_length * HOP_LENGTH / float(max(sample_rate, 1))),
                    aligned_word_count=0,
                    frame_phone_coverage=local_coverage,
                    feature_dir=sample_id,
                    source_clip=source_name,
                    conditioning_clip=conditioning_name,
                    slice_kind=slice_kind,
                    conditioning_similarity=conditioning_similarity,
                    alignment_score=float(np.clip(local_alignment_score, 0.0, 1.0)),
                    difficulty_score=self._compute_difficulty_score(
                        conditioning_similarity=conditioning_similarity,
                        alignment_score=local_alignment_score,
                        phone_coverage=local_coverage,
                    ),
                )
            )

        for window_index, start_frame in enumerate(starts, start=1):
            check_cancel()
            end_frame = min(frame_count, start_frame + window_frames)
            local_mask = lyric_mask[start_frame:end_frame]
            local_alignment_score = float(
                np.clip(
                    (0.72 * dtw_alignment_score)
                    + (0.28 * float(np.mean(local_mask > 0.5) if local_mask.size else 0.0)),
                    0.0,
                    1.0,
                )
            )
            save_window(
                start_frame=start_frame,
                end_frame=end_frame,
                sample_id=f"{sample_id_prefix}_truth_{window_index:04d}",
                slice_kind="paired-truth-window",
                local_alignment_score=local_alignment_score,
            )
            report(
                0.68 + ((window_index / float(total_windows)) * 0.32),
                f"Saving aligned window {window_index}/{total_windows} for {conditioning_name}...",
                f"Frames {max(0, end_frame - start_frame)} | activity-guided paired mapping",
            )

        detail_items = list(detail_windows or [])
        for detail_index, detail_window in enumerate(detail_items, start=1):
            repeat_count = max(1, int(detail_window.get("repeat_count", 1) or 1))
            start_sample = int(detail_window.get("start_sample", 0) or 0)
            end_sample = int(detail_window.get("end_sample", start_sample) or start_sample)
            if end_sample <= start_sample:
                continue
            pad_frames = max(8, int(round((0.10 * sample_rate) / HOP_LENGTH)))
            start_frame = max(0, int(round(start_sample / float(HOP_LENGTH))) - pad_frames)
            end_frame = min(frame_count, int(round(end_sample / float(HOP_LENGTH))) + pad_frames)
            base_score = float(
                np.clip(
                    (0.84 * dtw_alignment_score) + (0.16 * float(detail_window.get("score", 0.0) or 0.0)),
                    0.0,
                    1.0,
                )
            )
            for repeat_index in range(1, repeat_count + 1):
                save_window(
                    start_frame=start_frame,
                    end_frame=end_frame,
                    sample_id=f"{sample_id_prefix}_detail_{detail_index:02d}_r{repeat_index:02d}",
                    slice_kind="paired-detail-window",
                    local_alignment_score=base_score,
                )
        if not sample_entries:
            raise RuntimeError("Aligned source/target pair did not yield any truth windows.")
        report(
            1.0,
            f"Finished DTW-aligned windows for {conditioning_name}.",
            f"Saved {len(sample_entries)} aligned windows",
        )
        return sample_entries

    def build_paired_training_examples(
        self,
        *,
        sample_id_prefix: str,
        source_name: str,
        conditioning_name: str,
        guide_audio: np.ndarray,
        target_audio: np.ndarray,
        sample_rate: int,
        lyrics: str,
        target_word_scores: List[Dict[str, object]],
        guide_word_scores: Optional[List[Dict[str, object]]] = None,
        guide_similarity_score: float = 0.0,
        output_dir: Path,
        progress_callback: Optional[Callable[[float, str, str], None]] = None,
        cancel_event: Optional[object] = None,
    ) -> List[GuidedSVSFeatureExample]:
        def report(progress: float, message: str, detail: str = "") -> None:
            if progress_callback is not None:
                progress_callback(float(np.clip(progress, 0.0, 1.0)), message, detail)

        def check_cancel() -> None:
            if cancel_event is not None and bool(getattr(cancel_event, "is_set", lambda: False)()):
                raise InterruptedError("Training stopped by user.")

        guide_mono = np.asarray(guide_audio, dtype=np.float32)
        if guide_mono.ndim == 2:
            guide_mono = guide_mono.mean(axis=1)
        target_mono = np.asarray(target_audio, dtype=np.float32)
        if target_mono.ndim == 2:
            target_mono = target_mono.mean(axis=1)
        guide_mono = np.asarray(guide_mono, dtype=np.float32).reshape(-1)
        target_mono = np.asarray(target_mono, dtype=np.float32).reshape(-1)

        report(
            0.03,
            f"Preparing paired persona windows for {conditioning_name} -> {source_name}...",
            (
                f"Guide {len(guide_mono) / float(max(sample_rate, 1)):.1f}s | "
                f"target {len(target_mono) / float(max(sample_rate, 1)):.1f}s"
            ),
        )

        check_cancel()
        target_log_mel = self._extract_log_mel(target_mono, sample_rate)
        target_frame_count = int(target_log_mel.shape[0])
        raw_guide_log_mel = self._extract_log_mel(guide_mono, sample_rate)
        dtw_frame_map, dtw_alignment_score, dtw_metadata = self._compute_dtw_alignment(
            guide_log_mel=raw_guide_log_mel,
            target_log_mel=target_log_mel,
        )
        guide_log_mel = _warp_2d_by_frame_map(raw_guide_log_mel, dtw_frame_map)
        warped_guide_audio = _warp_audio_by_frame_map(
            guide_mono,
            sample_rate=sample_rate,
            frame_map=dtw_frame_map,
            target_sample_count=len(target_mono),
        )
        report(
            0.18,
            f"Guide/target frames aligned for {conditioning_name}.",
            (
                f"Target frames {target_frame_count} | "
                f"DTW {dtw_metadata.get('mode', 'unknown')} | "
                f"alignment {dtw_alignment_score:.3f}"
            ),
        )

        check_cancel()
        guide_f0 = self._extract_f0(
            warped_guide_audio,
            sample_rate,
            target_frame_count,
            progress_callback=lambda chunk_progress, message, detail: report(
                0.2 + (chunk_progress * 0.4),
                message,
                detail,
            ),
            cancel_event=cancel_event,
        )
        target_f0 = self._extract_f0(
            target_mono,
            sample_rate,
            target_frame_count,
            progress_callback=lambda chunk_progress, message, detail: report(
                0.6 + (chunk_progress * 0.08),
                "Extracting target pitch contour...",
                detail,
            ),
            cancel_event=cancel_event,
        )
        guide_energy = librosa.feature.rms(
            y=warped_guide_audio,
            frame_length=N_FFT,
            hop_length=HOP_LENGTH,
            center=True,
        ).squeeze()
        guide_energy = _align_1d(guide_energy, target_frame_count)
        guide_content = self._extract_content_features(warped_guide_audio, sample_rate, target_frame_count)
        guide_beat_phase = self._compute_beat_phase(warped_guide_audio, sample_rate, target_frame_count)
        guide_log_f0 = np.where(guide_f0 > 0.0, np.log(np.maximum(guide_f0, 1.0)), 0.0).astype(np.float32)
        target_log_f0 = np.where(target_f0 > 0.0, np.log(np.maximum(target_f0, 1.0)), 0.0).astype(np.float32)
        guide_vuv = (guide_f0 > 0.0).astype(np.float32)
        target_vuv = (target_f0 > 0.0).astype(np.float32)

        target_duration = float(len(target_mono) / float(max(sample_rate, 1)))
        guide_duration = float(len(guide_mono) / float(max(sample_rate, 1)))
        normalized_target_scores = self._normalize_phrase_word_scores(
            phrase_text=lyrics,
            phrase_word_scores=target_word_scores,
            duration_seconds=target_duration,
            prefer_content_alignment=True,
        )
        normalized_guide_scores = self._normalize_phrase_word_scores(
            phrase_text=lyrics,
            phrase_word_scores=guide_word_scores or target_word_scores,
            duration_seconds=guide_duration,
            prefer_content_alignment=True,
        )
        warped_guide_scores = self._warp_word_scores_to_target_timeline(
            word_scores=normalized_guide_scores,
            source_to_target_frame_map=_invert_monotonic_frame_map(dtw_frame_map, raw_guide_log_mel.shape[0]),
            sample_rate=sample_rate,
            target_frame_count=target_frame_count,
        )
        target_phone_ids, _ = self._build_phone_ids(
            lyrics=lyrics,
            word_scores=normalized_target_scores,
            frame_count=target_frame_count,
            sample_rate=sample_rate,
        )
        target_boundaries = self._build_word_boundaries(
            lyrics=lyrics,
            word_scores=normalized_target_scores,
            frame_count=target_frame_count,
            sample_rate=sample_rate,
        )
        guide_boundaries = self._build_word_boundaries(
            lyrics=lyrics,
            word_scores=warped_guide_scores or normalized_target_scores,
            frame_count=target_frame_count,
            sample_rate=sample_rate,
        )
        guide_boundary_lookup = {
            int(entry.get("index", -1)): dict(entry)
            for entry in guide_boundaries
            if int(entry.get("index", -1)) >= 0
        }

        slice_plans: List[Dict[str, object]] = []
        for word_boundary in target_boundaries:
            unit_segments = list(word_boundary.get("unit_segments", []))
            if not unit_segments:
                continue
            for anchor_index, (anchor_start, anchor_end) in enumerate(self._build_subword_anchor_spans(len(unit_segments))):
                context_start = max(0, anchor_start - 1)
                context_end = min(len(unit_segments) - 1, anchor_end + 1)
                slice_plans.append(
                    {
                        "word_boundary": word_boundary,
                        "anchor_index": anchor_index,
                        "anchor_start": anchor_start,
                        "anchor_end": anchor_end,
                        "context_start": context_start,
                        "context_end": context_end,
                    }
                )

        if not slice_plans:
            raise RuntimeError("No paired persona windows could be prepared from the aligned song pair.")

        feature_root = output_dir / "features"
        feature_root.mkdir(parents=True, exist_ok=True)
        pad_frames = max(4, int(round((0.12 * sample_rate) / HOP_LENGTH)))
        sample_entries: List[GuidedSVSFeatureExample] = []
        total_plans = max(1, len(slice_plans))

        for plan_index, plan in enumerate(slice_plans, start=1):
            check_cancel()
            word_boundary = dict(plan["word_boundary"])
            word_index = int(word_boundary.get("index", plan_index - 1))
            unit_segments = list(word_boundary.get("unit_segments", []))
            context_word_start = max(0, word_index - 1)
            context_word_end = min(len(target_boundaries) - 1, word_index + 1)
            target_context_start = dict(target_boundaries[context_word_start])
            target_context_end = dict(target_boundaries[context_word_end])
            target_start = max(0, int(target_context_start.get("start_frame", 0)) - pad_frames)
            target_end = min(
                target_frame_count,
                int(target_context_end.get("end_frame", target_frame_count)) + pad_frames,
            )
            if target_end - target_start < 4:
                continue

            guide_start_boundary = guide_boundary_lookup.get(context_word_start, {})
            guide_end_boundary = guide_boundary_lookup.get(context_word_end, guide_start_boundary)
            guide_start_frame = int(guide_start_boundary.get("start_frame", target_start))
            guide_end_frame = int(guide_end_boundary.get("end_frame", target_end))
            if guide_end_frame <= guide_start_frame:
                guide_start_frame = target_start
                guide_end_frame = target_end
            guide_start = max(0, int(guide_start_frame) - pad_frames)
            guide_end = min(guide_log_mel.shape[0], int(guide_end_frame) + pad_frames)
            if guide_end - guide_start < 4:
                guide_start = target_start
                guide_end = target_end

            sample_id = (
                f"{sample_id_prefix}_w{int(word_boundary.get('index', plan_index - 1)):04d}_"
                f"p{int(plan['anchor_index']):02d}"
            )
            feature_dir = feature_root / sample_id
            local_target = target_log_mel[target_start:target_end].astype(np.float32, copy=False)
            local_length = int(local_target.shape[0])
            local_guide_mel = _align_2d(guide_log_mel[guide_start:guide_end], local_length).astype(np.float32, copy=False)
            local_f0 = _align_1d(guide_f0[guide_start:guide_end], local_length).astype(np.float32, copy=False)
            local_log_f0 = _align_1d(guide_log_f0[guide_start:guide_end], local_length).astype(np.float32, copy=False)
            local_target_log_f0 = target_log_f0[target_start:target_end].astype(np.float32, copy=False)
            local_target_vuv = target_vuv[target_start:target_end].astype(np.float32, copy=False)
            local_vuv = _align_1d(guide_vuv[guide_start:guide_end], local_length).astype(np.float32, copy=False)
            local_energy = _align_1d(guide_energy[guide_start:guide_end], local_length).astype(np.float32, copy=False)
            local_content = _align_2d(guide_content[guide_start:guide_end], local_length).astype(np.float32, copy=False)
            local_beat_phase = _align_1d(guide_beat_phase[guide_start:guide_end], local_length).astype(np.float32, copy=False)
            local_phone_ids = target_phone_ids[target_start:target_end].astype(np.int64, copy=False)
            local_lyric_mask = (local_phone_ids != PHONE_TO_ID["SP"]).astype(np.float32, copy=False)
            local_coverage = float(np.mean(local_phone_ids != PHONE_TO_ID["SP"])) if local_phone_ids.size else 0.0
            local_difficulty = self._compute_difficulty_score(
                conditioning_similarity=guide_similarity_score,
                alignment_score=dtw_alignment_score,
                phone_coverage=local_coverage,
            )
            anchor_units = [
                str(unit_segments[idx].get("unit", "UNK"))
                for idx in range(int(plan["anchor_start"]), int(plan["anchor_end"]) + 1)
            ]
            if local_lyric_mask.size:
                guide_gate = (0.08 + (0.92 * local_lyric_mask[:, np.newaxis])).astype(np.float32, copy=False)
                local_guide_mel = (local_guide_mel * guide_gate).astype(np.float32, copy=False)

            self._save_training_slice(
                feature_dir=feature_dir,
                target_log_mel=local_target,
                target_audio=_slice_audio_for_frame_range(
                    target_mono,
                    start_frame=target_start,
                    end_frame=target_end,
                ),
                guide_log_mel=local_guide_mel,
                f0=local_f0,
                log_f0=local_log_f0,
                target_log_f0=local_target_log_f0,
                target_vuv=local_target_vuv,
                vuv=local_vuv,
                energy=local_energy,
                phone_ids=local_phone_ids,
                lyric_mask=local_lyric_mask,
                target_voice_signature=_compute_voice_signature_np(
                    log_mel=local_target,
                    log_f0=local_target_log_f0,
                    vuv=local_target_vuv,
                ),
                nar_content=local_content,
                beat_phase=local_beat_phase,
            )
            sample_entries.append(
                GuidedSVSFeatureExample(
                    sample_id=sample_id,
                    lyrics=str(word_boundary.get("word", "")),
                    n_frames=local_length,
                    duration_seconds=float(local_length * HOP_LENGTH / float(max(sample_rate, 1))),
                    aligned_word_count=1,
                    frame_phone_coverage=local_coverage,
                    feature_dir=sample_id,
                    source_clip=source_name,
                    conditioning_clip=conditioning_name,
                    anchor_word=str(word_boundary.get("word", "")),
                    anchor_units=anchor_units,
                    slice_kind="paired-persona-window",
                    conditioning_similarity=_normalize_similarity_score(guide_similarity_score),
                    alignment_score=dtw_alignment_score,
                    difficulty_score=local_difficulty,
                )
            )
            report(
                0.7 + ((plan_index / float(total_plans)) * 0.3),
                f"Saving paired persona window {plan_index}/{total_plans} for {conditioning_name}...",
                (
                    f"{word_boundary.get('word', '')} | "
                    f"guide {conditioning_name} -> target {source_name} | "
                    f"frames {local_length} | "
                    f"align {dtw_alignment_score:.3f}"
                ),
            )

        if not sample_entries:
            raise RuntimeError("Paired persona slicing finished without any usable windows.")
        report(
            1.0,
            f"Finished paired persona windows for {conditioning_name}.",
            f"Saved {len(sample_entries)} paired training slices",
        )
        return sample_entries

    def build_pronunciation_training_examples(
        self,
        *,
        sample_id_prefix: str,
        source_name: str,
        audio: np.ndarray,
        sample_rate: int,
        lyrics: str,
        word_scores: List[Dict[str, object]],
        output_dir: Path,
        progress_callback: Optional[Callable[[float, str, str], None]] = None,
        cancel_event: Optional[object] = None,
    ) -> List[GuidedSVSFeatureExample]:
        def report(progress: float, message: str, detail: str = "") -> None:
            if progress_callback is not None:
                progress_callback(float(np.clip(progress, 0.0, 1.0)), message, detail)

        def check_cancel() -> None:
            if cancel_event is not None and bool(getattr(cancel_event, "is_set", lambda: False)()):
                raise InterruptedError("Training stopped by user.")

        mono_audio = np.asarray(audio, dtype=np.float32)
        if mono_audio.ndim == 2:
            mono_audio = mono_audio.mean(axis=1)
        mono_audio = np.asarray(mono_audio, dtype=np.float32).reshape(-1)
        duration_seconds = float(len(mono_audio) / max(sample_rate, 1))
        report(
            0.02,
            f"Preparing pronunciation windows for {source_name}...",
            f"Audio length {duration_seconds:.1f}s",
        )

        check_cancel()
        mel = librosa.feature.melspectrogram(
            y=mono_audio,
            sr=sample_rate,
            n_fft=N_FFT,
            hop_length=HOP_LENGTH,
            win_length=WIN_LENGTH,
            n_mels=N_MELS,
            power=2.0,
        )
        log_mel = np.log(np.maximum(mel, 1e-5)).T.astype(np.float32, copy=False)
        frame_count = int(log_mel.shape[0])
        content_features = self._extract_content_features(mono_audio, sample_rate, frame_count)
        beat_phase = self._compute_beat_phase(mono_audio, sample_rate, frame_count)
        report(
            0.16,
            f"Mel frames ready for {source_name}.",
            f"Frames {frame_count} | starting pitch extraction",
        )

        check_cancel()
        f0 = self._extract_f0(
            mono_audio,
            sample_rate,
            frame_count,
            progress_callback=lambda chunk_progress, message, detail: report(
                0.18 + (chunk_progress * 0.44),
                message,
                detail,
            ),
            cancel_event=cancel_event,
        )
        report(
            0.64,
            f"Pitch extraction finished for {source_name}.",
            "Building energy contour",
        )

        check_cancel()
        energy = librosa.feature.rms(
            y=mono_audio,
            frame_length=N_FFT,
            hop_length=HOP_LENGTH,
            center=True,
        ).squeeze()
        energy = _align_1d(energy, frame_count)
        log_f0 = np.where(f0 > 0.0, np.log(np.maximum(f0, 1.0)), 0.0).astype(np.float32)
        vuv = (f0 > 0.0).astype(np.float32)
        report(
            0.7,
            f"Energy contour ready for {source_name}.",
            "Mapping aligned pronunciation spans",
        )

        check_cancel()
        word_boundaries = self._build_word_boundaries(
            lyrics=lyrics,
            word_scores=word_scores,
            frame_count=frame_count,
            sample_rate=sample_rate,
        )
        phone_ids, _ = self._build_phone_ids(
            lyrics=lyrics,
            word_scores=word_scores,
            frame_count=frame_count,
            sample_rate=sample_rate,
        )

        slice_plans: List[Dict[str, object]] = []
        for word_boundary in word_boundaries:
            unit_segments = list(word_boundary.get("unit_segments", []))
            if not unit_segments:
                continue
            anchor_spans = self._build_subword_anchor_spans(len(unit_segments))
            for anchor_index, (anchor_start, anchor_end) in enumerate(anchor_spans):
                context_start = max(0, anchor_start - 1)
                context_end = min(len(unit_segments) - 1, anchor_end + 1)
                slice_plans.append(
                    {
                        "word_boundary": word_boundary,
                        "anchor_index": anchor_index,
                        "anchor_start": anchor_start,
                        "anchor_end": anchor_end,
                        "context_start": context_start,
                        "context_end": context_end,
                    }
                )

        if not slice_plans:
            raise RuntimeError("No pronunciation windows could be prepared from the aligned dataset.")

        report(
            0.76,
            f"Pronunciation spans ready for {source_name}.",
            f"Windows {len(slice_plans)}",
        )
        pad_frames = max(3, int(round((0.08 * sample_rate) / HOP_LENGTH)))
        feature_root = output_dir / "features"
        feature_root.mkdir(parents=True, exist_ok=True)
        sample_entries: List[GuidedSVSFeatureExample] = []
        total_plans = max(1, len(slice_plans))

        for plan_index, plan in enumerate(slice_plans, start=1):
            check_cancel()
            word_boundary = dict(plan["word_boundary"])
            unit_segments = list(word_boundary.get("unit_segments", []))
            context_start_segment = unit_segments[int(plan["context_start"])]
            context_end_segment = unit_segments[int(plan["context_end"])]
            slice_start = max(0, int(context_start_segment["start_frame"]) - pad_frames)
            slice_end = min(frame_count, int(context_end_segment["end_frame"]) + pad_frames)
            if slice_end - slice_start < 4:
                continue

            sample_id = (
                f"{sample_id_prefix}_w{int(word_boundary.get('index', plan_index - 1)):04d}_"
                f"p{int(plan['anchor_index']):02d}"
            )
            feature_dir = feature_root / sample_id
            feature_dir.mkdir(parents=True, exist_ok=True)
            local_log_mel = log_mel[slice_start:slice_end].astype(np.float32, copy=False)
            local_f0 = f0[slice_start:slice_end].astype(np.float32, copy=False)
            local_log_f0 = log_f0[slice_start:slice_end].astype(np.float32, copy=False)
            local_vuv = vuv[slice_start:slice_end].astype(np.float32, copy=False)
            local_energy = energy[slice_start:slice_end].astype(np.float32, copy=False)
            local_content = content_features[slice_start:slice_end].astype(np.float32, copy=False)
            local_beat_phase = beat_phase[slice_start:slice_end].astype(np.float32, copy=False)
            local_phone_ids = phone_ids[slice_start:slice_end].astype(np.int64, copy=False)
            local_coverage = float(np.mean(local_phone_ids != PHONE_TO_ID["SP"])) if local_phone_ids.size else 0.0
            anchor_units = [
                str(unit_segments[idx].get("unit", "UNK"))
                for idx in range(int(plan["anchor_start"]), int(plan["anchor_end"]) + 1)
            ]

            self._save_training_slice(
                feature_dir=feature_dir,
                target_log_mel=local_log_mel,
                target_audio=_slice_audio_for_frame_range(
                    mono_audio,
                    start_frame=slice_start,
                    end_frame=slice_end,
                ),
                guide_log_mel=local_log_mel,
                f0=local_f0,
                log_f0=local_log_f0,
                target_log_f0=local_log_f0,
                target_vuv=local_vuv,
                vuv=local_vuv,
                energy=local_energy,
                phone_ids=local_phone_ids,
                lyric_mask=(local_phone_ids != PHONE_TO_ID["SP"]).astype(np.float32, copy=False),
                target_voice_signature=_compute_voice_signature_np(
                    log_mel=local_log_mel,
                    log_f0=local_log_f0,
                    vuv=local_vuv,
                ),
                nar_content=local_content,
                beat_phase=local_beat_phase,
            )

            sample_entries.append(
                GuidedSVSFeatureExample(
                    sample_id=sample_id,
                    lyrics=str(word_boundary.get("word", "")),
                    n_frames=int(local_log_mel.shape[0]),
                    duration_seconds=float((slice_end - slice_start) * HOP_LENGTH / float(max(sample_rate, 1))),
                    aligned_word_count=1,
                    frame_phone_coverage=local_coverage,
                    feature_dir=sample_id,
                    source_clip=source_name,
                    conditioning_clip=source_name,
                    anchor_word=str(word_boundary.get("word", "")),
                    anchor_units=anchor_units,
                    slice_kind="pronunciation-window",
                )
            )

            report(
                0.78 + ((plan_index / float(total_plans)) * 0.22),
                f"Saving pronunciation window {plan_index}/{total_plans} for {source_name}...",
                (
                    f"{word_boundary.get('word', '')} | "
                    f"units {' '.join(anchor_units)} | "
                    f"frames {local_log_mel.shape[0]}"
                ),
            )

        if not sample_entries:
            raise RuntimeError("Pronunciation-window extraction finished without any usable samples.")
        report(
            1.0,
            f"Pronunciation windows finished for {source_name}.",
            f"Saved {len(sample_entries)} local training slices",
        )
        return sample_entries

    def build_training_example(
        self,
        *,
        sample_id: str,
        audio: np.ndarray,
        sample_rate: int,
        lyrics: str,
        word_scores: List[Dict[str, object]],
        output_dir: Path,
        progress_callback: Optional[Callable[[float, str, str], None]] = None,
        cancel_event: Optional[object] = None,
    ) -> GuidedSVSFeatureExample:
        def report(progress: float, message: str, detail: str = "") -> None:
            if progress_callback is not None:
                progress_callback(float(np.clip(progress, 0.0, 1.0)), message, detail)

        def check_cancel() -> None:
            if cancel_event is not None and bool(getattr(cancel_event, "is_set", lambda: False)()):
                raise InterruptedError("Training stopped by user.")

        mono_audio = np.asarray(audio, dtype=np.float32)
        if mono_audio.ndim == 2:
            mono_audio = mono_audio.mean(axis=1)
        mono_audio = np.asarray(mono_audio, dtype=np.float32).reshape(-1)
        feature_dir = output_dir / "features" / sample_id
        feature_dir.mkdir(parents=True, exist_ok=True)
        duration_seconds = float(len(mono_audio) / max(sample_rate, 1))
        report(
            0.02,
            f"Preparing guided regeneration features for {sample_id}...",
            f"Audio length {duration_seconds:.1f}s",
        )

        check_cancel()
        mel = librosa.feature.melspectrogram(
            y=mono_audio,
            sr=sample_rate,
            n_fft=N_FFT,
            hop_length=HOP_LENGTH,
            win_length=WIN_LENGTH,
            n_mels=N_MELS,
            power=2.0,
        )
        log_mel = np.log(np.maximum(mel, 1e-5)).T.astype(np.float32, copy=False)
        frame_count = int(log_mel.shape[0])
        content_features = self._extract_content_features(mono_audio, sample_rate, frame_count)
        beat_phase = self._compute_beat_phase(mono_audio, sample_rate, frame_count)
        report(
            0.18,
            f"Mel frames ready for {sample_id}.",
            f"Frames {frame_count} | starting pitch extraction",
        )
        check_cancel()
        f0 = self._extract_f0(
            mono_audio,
            sample_rate,
            frame_count,
            progress_callback=lambda chunk_progress, message, detail: report(
                0.2 + (chunk_progress * 0.56),
                message,
                detail,
            ),
            cancel_event=cancel_event,
        )
        report(
            0.78,
            f"Pitch extraction finished for {sample_id}.",
            f"Frames {frame_count} | building energy contour",
        )
        check_cancel()
        energy = librosa.feature.rms(
            y=mono_audio,
            frame_length=N_FFT,
            hop_length=HOP_LENGTH,
            center=True,
        ).squeeze()
        energy = _align_1d(energy, frame_count)
        log_f0 = np.where(f0 > 0.0, np.log(np.maximum(f0, 1.0)), 0.0).astype(np.float32)
        vuv = (f0 > 0.0).astype(np.float32)
        report(
            0.86,
            f"Energy contour ready for {sample_id}.",
            "Mapping aligned pronunciation units onto frames",
        )
        check_cancel()
        phone_ids, coverage = self._build_phone_ids(
            lyrics=lyrics,
            word_scores=word_scores,
            frame_count=frame_count,
            sample_rate=sample_rate,
        )

        report(
            0.94,
            f"Saving guided regeneration tensors for {sample_id}.",
            f"Phone coverage {coverage * 100.0:.1f}%",
        )
        check_cancel()
        self._save_training_slice(
            feature_dir=feature_dir,
            target_log_mel=log_mel.astype(np.float32, copy=False),
            target_audio=_slice_audio_for_frame_range(
                mono_audio,
                start_frame=0,
                end_frame=frame_count,
            ),
            guide_log_mel=log_mel.astype(np.float32, copy=False),
            f0=f0.astype(np.float32, copy=False),
            log_f0=log_f0.astype(np.float32, copy=False),
            target_log_f0=log_f0.astype(np.float32, copy=False),
            target_vuv=vuv.astype(np.float32, copy=False),
            vuv=vuv.astype(np.float32, copy=False),
            energy=energy.astype(np.float32, copy=False),
            phone_ids=phone_ids.astype(np.int64, copy=False),
            lyric_mask=(phone_ids != PHONE_TO_ID["SP"]).astype(np.float32, copy=False),
            target_voice_signature=_compute_voice_signature_np(
                log_mel=log_mel.astype(np.float32, copy=False),
                log_f0=log_f0.astype(np.float32, copy=False),
                vuv=vuv.astype(np.float32, copy=False),
            ),
            nar_content=content_features.astype(np.float32, copy=False),
            beat_phase=beat_phase.astype(np.float32, copy=False),
        )
        report(
            1.0,
            f"Guided regeneration features finished for {sample_id}.",
            f"Frames {frame_count} | phone coverage {coverage * 100.0:.1f}%",
        )

        aligned_word_count = sum(
            1
            for entry in word_scores
            if float(entry.get("end", 0.0)) > float(entry.get("start", 0.0))
        )
        return GuidedSVSFeatureExample(
            sample_id=sample_id,
            lyrics=normalize_lyrics(lyrics),
            n_frames=frame_count,
            duration_seconds=float(len(mono_audio) / max(sample_rate, 1)),
            aligned_word_count=int(aligned_word_count),
            frame_phone_coverage=coverage,
            feature_dir=sample_id,
            conditioning_clip=sample_id,
        )

    def finalize_training_dataset(
        self,
        *,
        dataset_dir: Path,
        sample_entries: Sequence[GuidedSVSFeatureExample],
    ) -> Dict[str, object]:
        entries = [entry.to_dict() for entry in sample_entries]
        dataset_dir.mkdir(parents=True, exist_ok=True)
        processed_metadata_path = dataset_dir / "processed_metadata.json"
        processed_metadata_path.write_text(json.dumps(entries, indent=2), encoding="utf-8")

        mel_sum = np.zeros(N_MELS, dtype=np.float64)
        mel_sq_sum = np.zeros(N_MELS, dtype=np.float64)
        mel_frames = 0
        log_f0_values: List[np.ndarray] = []
        energy_values: List[np.ndarray] = []
        nar_content_sum: Optional[np.ndarray] = None
        nar_content_sq_sum: Optional[np.ndarray] = None
        nar_content_frames = 0
        total_seconds = 0.0

        for entry in entries:
            feature_dir = dataset_dir / "features" / str(entry["feature_dir"])
            mel = np.load(feature_dir / "mel.npy").astype(np.float32)
            vuv = np.load(feature_dir / "vuv.npy").astype(np.float32)
            log_f0 = np.load(feature_dir / "log_f0.npy").astype(np.float32)
            energy = np.load(feature_dir / "energy.npy").astype(np.float32)
            mel_sum += mel.sum(axis=0)
            mel_sq_sum += np.square(mel).sum(axis=0)
            mel_frames += int(mel.shape[0])
            if np.any(vuv > 0.5):
                log_f0_values.append(log_f0[vuv > 0.5])
            energy_values.append(energy.reshape(-1))
            nar_content_path = feature_dir / "nar_content.npy"
            if nar_content_path.exists():
                nar_content = np.load(nar_content_path).astype(np.float32)
                if nar_content.ndim == 2 and nar_content.shape[0] > 0:
                    if nar_content_sum is None or nar_content_sq_sum is None:
                        nar_content_sum = np.zeros(nar_content.shape[1], dtype=np.float64)
                        nar_content_sq_sum = np.zeros(nar_content.shape[1], dtype=np.float64)
                    nar_content_sum += nar_content.sum(axis=0)
                    nar_content_sq_sum += np.square(nar_content).sum(axis=0)
                    nar_content_frames += int(nar_content.shape[0])
            total_seconds += float(entry.get("duration_seconds", 0.0))

        mel_mean = mel_sum / max(mel_frames, 1)
        mel_var = np.maximum((mel_sq_sum / max(mel_frames, 1)) - np.square(mel_mean), 1e-6)
        stacked_f0 = (
            np.concatenate(log_f0_values).astype(np.float32, copy=False)
            if log_f0_values
            else np.zeros(1, dtype=np.float32)
        )
        stacked_energy = (
            np.concatenate(energy_values).astype(np.float32, copy=False)
            if energy_values
            else np.zeros(1, dtype=np.float32)
        )
        log_f0_mean = float(np.mean(stacked_f0))
        log_f0_std = float(np.std(stacked_f0) + 1e-4)
        energy_mean = float(np.mean(stacked_energy))
        energy_std = float(np.std(stacked_energy) + 1e-4)
        nar_content_dim = 0
        nar_content_mean = np.zeros((0,), dtype=np.float32)
        nar_content_std = np.zeros((0,), dtype=np.float32)
        if nar_content_sum is not None and nar_content_sq_sum is not None and nar_content_frames > 0:
            nar_content_mean = (nar_content_sum / max(nar_content_frames, 1)).astype(np.float32, copy=False)
            nar_content_var = np.maximum(
                (nar_content_sq_sum / max(nar_content_frames, 1)) - np.square(nar_content_mean),
                1e-6,
            )
            nar_content_std = np.sqrt(nar_content_var).astype(np.float32, copy=False)
            nar_content_dim = int(nar_content_mean.shape[0])
        voice_signatures: List[np.ndarray] = []
        for entry in entries:
            feature_dir = dataset_dir / "features" / str(entry["feature_dir"])
            mel = np.load(feature_dir / "mel.npy").astype(np.float32)
            target_log_f0 = (
                np.load(feature_dir / "target_log_f0.npy").astype(np.float32)
                if (feature_dir / "target_log_f0.npy").exists()
                else np.load(feature_dir / "log_f0.npy").astype(np.float32)
            )
            target_vuv = (
                np.load(feature_dir / "target_vuv.npy").astype(np.float32)
                if (feature_dir / "target_vuv.npy").exists()
                else (target_log_f0 > 0.0).astype(np.float32)
            )
            norm_mel = (mel - mel_mean[np.newaxis, :]) / np.sqrt(mel_var)[np.newaxis, :]
            norm_target_log_f0 = np.zeros_like(target_log_f0, dtype=np.float32)
            target_voiced = target_vuv > 0.5
            if np.any(target_voiced):
                norm_target_log_f0[target_voiced] = (
                    target_log_f0[target_voiced] - log_f0_mean
                ) / log_f0_std
            voice_signatures.append(
                _compute_voice_signature_np(
                    log_mel=norm_mel,
                    log_f0=norm_target_log_f0,
                    vuv=target_vuv,
                )
            )
        global_voice_signature = (
            np.mean(np.stack(voice_signatures, axis=0), axis=0).astype(np.float32, copy=False)
            if voice_signatures
            else np.zeros(VOICE_SIGNATURE_DIM, dtype=np.float32)
        )
        stats = {
            "sample_rate": SAMPLE_RATE,
            "hop_length": HOP_LENGTH,
            "n_fft": N_FFT,
            "win_length": WIN_LENGTH,
            "n_mels": N_MELS,
            "voice_signature_bands": VOICE_SIGNATURE_BANDS,
            "voice_signature_dim": VOICE_SIGNATURE_DIM,
            "phone_tokens": list(PHONE_TOKENS),
            "mel_mean": mel_mean.astype(np.float32).tolist(),
            "mel_std": np.sqrt(mel_var).astype(np.float32).tolist(),
            "log_f0_mean": log_f0_mean,
            "log_f0_std": log_f0_std,
            "energy_mean": energy_mean,
            "energy_std": energy_std,
            "nar_content_dim": nar_content_dim,
            "nar_content_mean": nar_content_mean.tolist(),
            "nar_content_std": nar_content_std.tolist(),
            "global_voice_signature": global_voice_signature.tolist(),
            "sample_count": len(entries),
            "total_frames": mel_frames,
            "total_seconds": round(total_seconds, 3),
        }
        stats_path = dataset_dir / "stats.json"
        stats_path.write_text(json.dumps(stats, indent=2), encoding="utf-8")
        report_path = dataset_dir / "dataset_report.json"
        report_path.write_text(
            json.dumps(
                {
                    "sample_count": len(entries),
                    "total_frames": mel_frames,
                    "total_seconds": round(total_seconds, 3),
                    "avg_frames_per_sample": round(float(mel_frames / max(len(entries), 1)), 2),
                    "voice_signature_dim": VOICE_SIGNATURE_DIM,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        return {
            "dataset_dir": str(dataset_dir),
            "processed_metadata_path": str(processed_metadata_path),
            "stats_path": str(stats_path),
            "report_path": str(report_path),
            "sample_count": len(entries),
            "total_frames": int(mel_frames),
            "total_seconds": round(total_seconds, 3),
        }

    def _build_loaders(
        self,
        *,
        dataset_dir: Path,
        max_frames: int,
        batch_size: int,
        num_workers: int = 0,
        prefetch_factor: int = 2,
        persona_v11_recipe: bool = False,
    ) -> Tuple[
        DataLoader,
        DataLoader,
        Dict[str, object],
        List[Dict[str, object]],
        List[Dict[str, object]],
        Dict[str, object],
    ]:
        processed_metadata = json.loads((dataset_dir / "processed_metadata.json").read_text(encoding="utf-8"))
        stats = json.loads((dataset_dir / "stats.json").read_text(encoding="utf-8"))
        if not processed_metadata:
            raise RuntimeError("No voice-builder training feature entries were prepared.")
        filtered_entries, filter_metadata = self._filter_training_entries(
            [dict(entry) for entry in processed_metadata]
        )
        entries = filtered_entries
        random.Random(1337).shuffle(entries)
        val_count = 1 if len(entries) <= 6 else max(1, int(round(len(entries) * 0.15)))
        train_entries = entries[val_count:] or entries
        val_entries = entries[:val_count] or train_entries[:1]

        train_loader = self._create_loader(
            entries=train_entries,
            dataset_dir=dataset_dir,
            stats=stats,
            max_frames=max_frames,
            batch_size=batch_size,
            num_workers=num_workers,
            prefetch_factor=prefetch_factor,
            persona_v11_recipe=persona_v11_recipe,
            random_crop=True,
            shuffle=True,
        )
        val_loader = self._create_loader(
            entries=val_entries,
            dataset_dir=dataset_dir,
            stats=stats,
            max_frames=max_frames,
            batch_size=batch_size,
            num_workers=num_workers,
            prefetch_factor=prefetch_factor,
            persona_v11_recipe=persona_v11_recipe,
            random_crop=False,
            shuffle=False,
        )
        return train_loader, val_loader, stats, train_entries, val_entries, filter_metadata

    def _get_training_hardware_profile(self) -> Dict[str, object]:
        device_type = "cuda" if torch.cuda.is_available() else "cpu"
        gpu_memory_gb = 0.0
        if device_type == "cuda":
            try:
                gpu_memory_gb = float(torch.cuda.get_device_properties(0).total_memory) / float(1024**3)
            except Exception:
                gpu_memory_gb = 0.0
        cpu_count = max(1, os.cpu_count() or 1)
        bf16_supported = bool(
            device_type == "cuda"
            and hasattr(torch.cuda, "is_bf16_supported")
            and torch.cuda.is_bf16_supported()
        )
        # The trainer rebuilds loaders frequently across curriculum stages and
        # epochs. On Linux/Vast, large worker pools tend to pile up and stall
        # the run long before GPU memory is the bottleneck, so keep the loader
        # footprint intentionally moderate there.
        worker_cap = 8 if os.name == "nt" else 6
        if gpu_memory_gb >= 160.0:
            return {
                "gpu_memory_gb": round(gpu_memory_gb, 2),
                "max_frames": 1024,
                "num_workers": min(worker_cap, cpu_count),
                "prefetch_factor": 6,
                "model": {"d_model": 640, "n_heads": 8, "n_layers": 16, "dropout": 0.08},
                "compile_model": True,
                "precision_mode": "bf16" if bf16_supported else "fp16",
                "use_fused_adamw": True,
                "lr": 2.0e-4,
            }
        if gpu_memory_gb >= 80.0:
            return {
                "gpu_memory_gb": round(gpu_memory_gb, 2),
                "max_frames": 768,
                "num_workers": min(min(worker_cap, 16), cpu_count),
                "prefetch_factor": 4,
                "model": {"d_model": 512, "n_heads": 8, "n_layers": 14, "dropout": 0.08},
                "compile_model": True,
                "precision_mode": "bf16" if bf16_supported else "fp16",
                "use_fused_adamw": True,
                "lr": 2.0e-4,
            }
        if gpu_memory_gb >= 40.0:
            return {
                "gpu_memory_gb": round(gpu_memory_gb, 2),
                "max_frames": 704,
                "num_workers": min(worker_cap, cpu_count),
                "prefetch_factor": 3,
                "model": {"d_model": 384, "n_heads": 8, "n_layers": 10, "dropout": 0.09},
                "compile_model": True,
                "precision_mode": "bf16" if bf16_supported else "fp16",
                "use_fused_adamw": True,
                "lr": 2.0e-4,
            }
        if gpu_memory_gb >= 24.0:
            return {
                "gpu_memory_gb": round(gpu_memory_gb, 2),
                "max_frames": 512,
                "num_workers": min(worker_cap, cpu_count),
                "prefetch_factor": 3,
                "model": {"d_model": 320, "n_heads": 8, "n_layers": 9, "dropout": 0.09},
                "compile_model": True,
                "precision_mode": "bf16" if bf16_supported else "fp16",
                "use_fused_adamw": True,
                "lr": 2.0e-4,
            }
        return {
            "gpu_memory_gb": round(gpu_memory_gb, 2),
            "max_frames": 320,
            "num_workers": min(4, cpu_count),
            "prefetch_factor": 2,
            "model": {"d_model": 256, "n_heads": 4, "n_layers": 8, "dropout": 0.1},
            "compile_model": False,
            "precision_mode": "fp32",
            "use_fused_adamw": False,
            "lr": 2.0e-4,
        }

    def _get_vocoder_hardware_profile(
        self,
        *,
        guided_profile: Optional[Dict[str, object]] = None,
        requested_batch_size: int = 16,
        total_epochs: int = 0,
    ) -> Dict[str, object]:
        profile = dict(guided_profile or self._get_training_hardware_profile())
        gpu_memory_gb = float(profile.get("gpu_memory_gb", 0.0))
        cpu_count = max(1, os.cpu_count() or 1)
        worker_cap = 8 if os.name == "nt" else 6
        precision_mode = str(profile.get("precision_mode", "fp32"))
        if gpu_memory_gb >= 160.0:
            return {
                "gpu_memory_gb": gpu_memory_gb,
                "max_frames": 384,
                "batch_size": max(8, min(int(requested_batch_size), 48)),
                "num_workers": min(worker_cap, cpu_count),
                "prefetch_factor": 4,
                "base_channels": 512,
                "compile_model": True,
                "precision_mode": precision_mode,
                "use_fused_adamw": True,
                "lr": 1.5e-4,
                "total_epochs": max(80, min(260, int(max(total_epochs, 1) // 6))),
            }
        if gpu_memory_gb >= 80.0:
            return {
                "gpu_memory_gb": gpu_memory_gb,
                "max_frames": 320,
                "batch_size": max(6, min(int(requested_batch_size), 28)),
                "num_workers": min(min(worker_cap, 16), cpu_count),
                "prefetch_factor": 3,
                "base_channels": 448,
                "compile_model": True,
                "precision_mode": precision_mode,
                "use_fused_adamw": True,
                "lr": 1.6e-4,
                "total_epochs": max(72, min(220, int(max(total_epochs, 1) // 6))),
            }
        if gpu_memory_gb >= 40.0:
            return {
                "gpu_memory_gb": gpu_memory_gb,
                "max_frames": 288,
                "batch_size": max(4, min(int(requested_batch_size), 16)),
                "num_workers": min(worker_cap, cpu_count),
                "prefetch_factor": 3,
                "base_channels": 384,
                "compile_model": True,
                "precision_mode": precision_mode,
                "use_fused_adamw": True,
                "lr": 1.8e-4,
                "total_epochs": max(56, min(180, int(max(total_epochs, 1) // 7))),
            }
        if gpu_memory_gb >= 24.0:
            return {
                "gpu_memory_gb": gpu_memory_gb,
                "max_frames": 256,
                "batch_size": max(4, min(int(requested_batch_size), 12)),
                "num_workers": min(worker_cap, cpu_count),
                "prefetch_factor": 3,
                "base_channels": 320,
                "compile_model": True,
                "precision_mode": precision_mode,
                "use_fused_adamw": True,
                "lr": 1.9e-4,
                "total_epochs": max(48, min(160, int(max(total_epochs, 1) // 7))),
            }
        return {
            "gpu_memory_gb": gpu_memory_gb,
            "max_frames": 192,
            "batch_size": max(2, min(int(requested_batch_size), 8)),
            "num_workers": min(4, cpu_count),
            "prefetch_factor": 2,
            "base_channels": 256,
            "compile_model": False,
            "precision_mode": "fp32",
            "use_fused_adamw": False,
            "lr": 2.0e-4,
            "total_epochs": max(36, min(120, int(max(total_epochs, 1) // 8))),
        }

    def _create_loader(
        self,
        *,
        entries: Sequence[Dict[str, object]],
        dataset_dir: Path,
        stats: Dict[str, object],
        max_frames: int,
        batch_size: int,
        num_workers: int = 0,
        prefetch_factor: int = 2,
        persona_v11_recipe: bool = False,
        random_crop: bool,
        shuffle: bool,
    ) -> DataLoader:
        dataset: Dataset
        collate_fn = collate_guided_svs
        if persona_v11_recipe:
            dataset = PersonaMapperNARDataset(
                entries=entries,
                dataset_dir=dataset_dir,
                stats=stats,
                max_frames=max_frames,
                random_crop=random_crop,
            )
            collate_fn = collate_persona_mapper_nar
        else:
            dataset = GuidedSVSDataset(
                entries=entries,
                dataset_dir=dataset_dir,
                stats=stats,
                max_frames=max_frames,
                random_crop=random_crop,
            )
        worker_count = max(0, int(num_workers))
        loader_kwargs = {
            "batch_size": max(1, int(batch_size)),
            "num_workers": worker_count,
            # Vast/container CUDA environments can sporadically kill the
            # DataLoader pin-memory helper thread mid-epoch. Disabling pinned
            # memory here is slower in theory, but much more reliable for the
            # long Persona training runs we care about.
            "pin_memory": False,
            "collate_fn": collate_fn,
        }
        if worker_count > 0:
            # These loaders are rebuilt repeatedly as curriculum slices change.
            # Keeping workers persistent here can leak/accumulate loader
            # processes on Vast and make full-diversity look like it "freezes"
            # after a handful of epochs.
            loader_kwargs["persistent_workers"] = False
            loader_kwargs["prefetch_factor"] = max(2, int(prefetch_factor))
        return DataLoader(dataset, shuffle=shuffle, **loader_kwargs)

    def _create_vocoder_loader(
        self,
        *,
        entries: Sequence[Dict[str, object]],
        dataset_dir: Path,
        stats: Dict[str, object],
        max_frames: int,
        batch_size: int,
        num_workers: int = 0,
        prefetch_factor: int = 2,
        random_crop: bool,
        shuffle: bool,
    ) -> DataLoader:
        dataset = VocoderSliceDataset(
            entries=entries,
            dataset_dir=dataset_dir,
            stats=stats,
            max_frames=max_frames,
            random_crop=random_crop,
        )
        worker_count = max(0, int(num_workers))
        loader_kwargs = {
            "batch_size": max(1, int(batch_size)),
            "num_workers": worker_count,
            # Keep the waveform/vocoder phase on the same safer loader path.
            "pin_memory": False,
            "collate_fn": collate_vocoder_slices,
        }
        if worker_count > 0:
            loader_kwargs["persistent_workers"] = False
            loader_kwargs["prefetch_factor"] = max(2, int(prefetch_factor))
        return DataLoader(dataset, shuffle=shuffle, **loader_kwargs)

    def _is_loader_runtime_error(self, exc: BaseException) -> bool:
        message = str(exc or "").lower()
        if not message:
            return False
        return any(
            token in message
            for token in (
                "pin memory thread exited unexpectedly",
                "dataloader worker",
                "worker exited unexpectedly",
                "received 0 items of ancdata",
                "connection reset by peer",
            )
        )

    def _filter_training_entries(
        self,
        entries: Sequence[Dict[str, object]],
    ) -> Tuple[List[Dict[str, object]], Dict[str, object]]:
        prepared_entries = [dict(entry) for entry in entries]
        if not prepared_entries:
            return [], {
                "kept_total": 0,
                "dropped_total": 0,
                "kept_paired_count": 0,
                "dropped_paired_count": 0,
                "mode": "empty",
            }

        identity_entries = [
            entry for entry in prepared_entries if str(entry.get("slice_kind", "")) == "base-identity-window"
        ]
        paired_entries = [
            entry for entry in prepared_entries if str(entry.get("slice_kind", "")) != "base-identity-window"
        ]
        if len(paired_entries) < 12:
            return prepared_entries, {
                "kept_total": len(prepared_entries),
                "dropped_total": 0,
                "kept_paired_count": len(paired_entries),
                "dropped_paired_count": 0,
                "mode": "all-kept-small-dataset",
            }

        scored_pairs: List[Dict[str, object]] = []
        for entry in paired_entries:
            alignment_score = float(np.clip(float(entry.get("alignment_score", 0.0) or 0.0), 0.0, 1.0))
            conditioning_similarity = _normalize_similarity_score(entry.get("conditioning_similarity", 0.0))
            frame_phone_coverage = float(np.clip(float(entry.get("frame_phone_coverage", 0.0) or 0.0), 0.0, 1.0))
            quality_score = float(
                np.clip(
                    (0.5 * alignment_score)
                    + (0.3 * conditioning_similarity)
                    + (0.2 * frame_phone_coverage),
                    0.0,
                    1.0,
                )
            )
            if alignment_score < 0.1 or frame_phone_coverage < 0.12:
                continue
            scored_entry = dict(entry)
            scored_entry["quality_score"] = round(quality_score, 6)
            scored_pairs.append(scored_entry)

        if not scored_pairs:
            return prepared_entries, {
                "kept_total": len(prepared_entries),
                "dropped_total": 0,
                "kept_paired_count": len(paired_entries),
                "dropped_paired_count": 0,
                "mode": "all-kept-no-safe-filter-hit",
            }

        scored_pairs = sorted(
            scored_pairs,
            key=lambda entry: (
                float(entry.get("quality_score", 0.0)),
                float(entry.get("alignment_score", 0.0)),
                float(entry.get("conditioning_similarity", 0.0)),
            ),
            reverse=True,
        )
        keep_count = len(scored_pairs)
        if len(scored_pairs) > 24:
            keep_count = max(24, int(math.ceil(len(scored_pairs) * 0.9)))
        kept_pairs = scored_pairs[:keep_count]
        filtered_entries = identity_entries + kept_pairs
        return filtered_entries, {
            "kept_total": len(filtered_entries),
            "dropped_total": max(0, len(prepared_entries) - len(filtered_entries)),
            "kept_paired_count": len(kept_pairs),
            "dropped_paired_count": max(0, len(paired_entries) - len(kept_pairs)),
            "mode": "quality-filtered",
        }

    def _masked_l1_loss(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
        lengths: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        max_frames = int(target.shape[1])
        valid = (torch.arange(max_frames, device=lengths.device).unsqueeze(0) < lengths.unsqueeze(1)).unsqueeze(-1)
        if mask is not None:
            mask_tensor = mask
            if mask_tensor.dim() == 2:
                mask_tensor = mask_tensor.unsqueeze(-1)
            valid = valid & (mask_tensor > 0.5)
        valid = valid.to(dtype=prediction.dtype)
        diff = torch.abs(prediction - target) * valid
        return diff.sum() / valid.sum().clamp(min=1.0)

    def _masked_f0_loss(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
        lengths: torch.Tensor,
    ) -> torch.Tensor:
        max_frames = int(target.shape[1])
        time_mask = torch.arange(max_frames, device=lengths.device).unsqueeze(0) < lengths.unsqueeze(1)
        voiced_mask = torch.abs(target) > 1e-6
        mask = (time_mask & voiced_mask).to(dtype=prediction.dtype)
        diff = torch.abs(prediction - target) * mask
        return diff.sum() / mask.sum().clamp(min=1.0)

    def _masked_bce_loss(
        self,
        logits: torch.Tensor,
        target: torch.Tensor,
        lengths: torch.Tensor,
    ) -> torch.Tensor:
        max_frames = int(target.shape[1])
        mask = (torch.arange(max_frames, device=lengths.device).unsqueeze(0) < lengths.unsqueeze(1)).to(
            dtype=logits.dtype
        )
        loss = nn.functional.binary_cross_entropy_with_logits(logits, target, reduction="none") * mask
        return loss.sum() / mask.sum().clamp(min=1.0)

    def _masked_phone_cross_entropy_loss(
        self,
        logits: torch.Tensor,
        target: torch.Tensor,
        lengths: torch.Tensor,
        lyric_mask: torch.Tensor,
    ) -> torch.Tensor:
        max_frames = int(target.shape[1])
        valid = (
            (torch.arange(max_frames, device=lengths.device).unsqueeze(0) < lengths.unsqueeze(1))
            & (target != PHONE_TO_ID["PAD"])
            & (target != PHONE_TO_ID["SP"])
        )
        if not torch.any(valid):
            return torch.zeros((), device=logits.device, dtype=logits.dtype)
        per_frame_loss = nn.functional.cross_entropy(
            logits.transpose(1, 2),
            target,
            reduction="none",
        )
        frame_weights = (0.45 + (1.55 * torch.clamp(lyric_mask, 0.0, 1.0))).to(dtype=logits.dtype)
        frame_weights = frame_weights * valid.to(dtype=logits.dtype)
        return (per_frame_loss * frame_weights).sum() / frame_weights.sum().clamp(min=1.0)

    def _masked_delta_loss(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
        lengths: torch.Tensor,
        *,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if prediction.shape[1] <= 1:
            return torch.zeros((), device=prediction.device, dtype=prediction.dtype)
        pred_delta = prediction[:, 1:] - prediction[:, :-1]
        target_delta = target[:, 1:] - target[:, :-1]
        max_frames = int(prediction.shape[1] - 1)
        valid = (
            torch.arange(max_frames, device=lengths.device).unsqueeze(0)
            < torch.clamp(lengths - 1, min=0).unsqueeze(1)
        )
        if prediction.dim() == 3:
            valid = valid.unsqueeze(-1)
            if mask is not None:
                valid = valid * (mask[:, 1:] > 0.5).unsqueeze(-1)
        elif mask is not None:
            valid = valid * (mask[:, 1:] > 0.5)
        valid = valid.to(dtype=prediction.dtype)
        diff = torch.abs(pred_delta - target_delta) * valid
        return diff.sum() / valid.sum().clamp(min=1.0)

    def _masked_acceleration_loss(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
        lengths: torch.Tensor,
        *,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if prediction.shape[1] <= 2:
            return torch.zeros((), device=prediction.device, dtype=prediction.dtype)
        pred_accel = prediction[:, 2:] - (2.0 * prediction[:, 1:-1]) + prediction[:, :-2]
        target_accel = target[:, 2:] - (2.0 * target[:, 1:-1]) + target[:, :-2]
        max_frames = int(prediction.shape[1] - 2)
        valid = (
            torch.arange(max_frames, device=lengths.device).unsqueeze(0)
            < torch.clamp(lengths - 2, min=0).unsqueeze(1)
        )
        if prediction.dim() == 3:
            valid = valid.unsqueeze(-1)
            if mask is not None:
                valid = valid * (mask[:, 2:] > 0.5).unsqueeze(-1)
        elif mask is not None:
            valid = valid * (mask[:, 2:] > 0.5)
        valid = valid.to(dtype=prediction.dtype)
        diff = torch.abs(pred_accel - target_accel) * valid
        return diff.sum() / valid.sum().clamp(min=1.0)

    def _masked_smoothness_loss(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
        lengths: torch.Tensor,
        *,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if prediction.shape[1] <= 1:
            return torch.zeros((), device=prediction.device, dtype=prediction.dtype)
        pred_delta = prediction[:, 1:] - prediction[:, :-1]
        target_delta = target[:, 1:] - target[:, :-1]
        max_frames = int(prediction.shape[1] - 1)
        valid = (
            torch.arange(max_frames, device=lengths.device).unsqueeze(0)
            < torch.clamp(lengths - 1, min=0).unsqueeze(1)
        )
        if prediction.dim() == 3:
            pred_smoothness = torch.mean(torch.square(pred_delta), dim=-1)
            target_smoothness = torch.mean(torch.square(target_delta), dim=-1)
            if mask is not None:
                valid = valid * (mask[:, 1:] > 0.5)
        else:
            pred_smoothness = torch.square(pred_delta)
            target_smoothness = torch.square(target_delta)
            if mask is not None:
                valid = valid * (mask[:, 1:] > 0.5)
        valid = valid.to(dtype=prediction.dtype)
        diff = torch.abs(pred_smoothness - target_smoothness) * valid
        return diff.sum() / valid.sum().clamp(min=1.0)

    def _masked_high_band_loss(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
        lengths: torch.Tensor,
        *,
        mask: Optional[torch.Tensor] = None,
        start_ratio: float = 0.58,
    ) -> torch.Tensor:
        if prediction.dim() != 3 or prediction.shape[-1] <= 1:
            return self._masked_l1_loss(prediction, target, lengths, mask=mask)
        n_mels = int(prediction.shape[-1])
        start_bin = max(1, min(n_mels - 1, int(round(n_mels * float(start_ratio)))))
        return self._masked_l1_loss(
            prediction[..., start_bin:],
            target[..., start_bin:],
            lengths,
            mask=mask,
        )

    def _off_lyric_suppression_loss(
        self,
        prediction: torch.Tensor,
        lyric_mask: torch.Tensor,
        lengths: torch.Tensor,
    ) -> torch.Tensor:
        max_frames = int(prediction.shape[1])
        valid = (torch.arange(max_frames, device=lengths.device).unsqueeze(0) < lengths.unsqueeze(1)).unsqueeze(-1)
        off_lyric = (lyric_mask < 0.5).unsqueeze(-1)
        mask = (valid & off_lyric).to(dtype=prediction.dtype)
        if torch.sum(mask) <= 0:
            return torch.zeros((), device=prediction.device, dtype=prediction.dtype)
        suppressed_energy = torch.relu(prediction) * mask
        return suppressed_energy.sum() / mask.sum().clamp(min=1.0)

    def _multi_resolution_temporal_stft_loss(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
        lengths: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        masked_prediction = prediction
        masked_target = target
        max_frames = int(target.shape[1])
        time_mask = (torch.arange(max_frames, device=lengths.device).unsqueeze(0) < lengths.unsqueeze(1)).unsqueeze(-1)
        masked_prediction = masked_prediction * time_mask
        masked_target = masked_target * time_mask
        prediction_series = masked_prediction.transpose(1, 2).reshape(-1, max_frames).float()
        target_series = masked_target.transpose(1, 2).reshape(-1, max_frames).float()
        resolution_specs = [(8, 2), (16, 4), (32, 8), (64, 16)]
        stft_loss = torch.zeros((), device=prediction.device, dtype=torch.float32)
        phase_loss = torch.zeros((), device=prediction.device, dtype=torch.float32)
        used = 0
        for n_fft, hop_length in resolution_specs:
            if max_frames < n_fft:
                continue
            window = torch.hann_window(n_fft, device=prediction.device, dtype=torch.float32)
            pred_stft = torch.stft(
                prediction_series,
                n_fft=n_fft,
                hop_length=hop_length,
                win_length=n_fft,
                window=window,
                return_complex=True,
            )
            target_stft = torch.stft(
                target_series,
                n_fft=n_fft,
                hop_length=hop_length,
                win_length=n_fft,
                window=window,
                return_complex=True,
            )
            pred_mag = pred_stft.abs()
            target_mag = target_stft.abs()
            stft_loss = stft_loss + torch.mean(torch.abs(pred_mag - target_mag))
            log_pred_mag = torch.log(torch.clamp(pred_mag, min=1e-5))
            log_target_mag = torch.log(torch.clamp(target_mag, min=1e-5))
            stft_loss = stft_loss + (0.5 * torch.mean(torch.abs(log_pred_mag - log_target_mag)))
            phase_delta = torch.angle(pred_stft) - torch.angle(target_stft)
            phase_delta = torch.atan2(torch.sin(phase_delta), torch.cos(phase_delta)).abs()
            phase_loss = phase_loss + torch.mean(phase_delta)
            used += 1
        if used <= 0:
            zero = torch.zeros((), device=prediction.device, dtype=torch.float32)
            return zero, zero
        return stft_loss / float(used), phase_loss / float(used)

    def _masked_wave_l1_loss(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
        sample_lengths: torch.Tensor,
    ) -> torch.Tensor:
        masked_prediction = _mask_audio_by_lengths(prediction, sample_lengths)
        masked_target = _mask_audio_by_lengths(target, sample_lengths)
        valid = (
            torch.arange(prediction.shape[1], device=sample_lengths.device).unsqueeze(0)
            < sample_lengths.unsqueeze(1)
        ).to(dtype=prediction.dtype)
        return torch.sum(torch.abs(masked_prediction - masked_target) * valid) / valid.sum().clamp_min(1.0)

    def _multi_resolution_wave_stft_loss(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
        sample_lengths: torch.Tensor,
        fft_sizes: Sequence[int] = (1024, 2048, 512),
    ) -> torch.Tensor:
        masked_prediction = _mask_audio_by_lengths(prediction, sample_lengths)
        masked_target = _mask_audio_by_lengths(target, sample_lengths)
        total = torch.zeros((), device=prediction.device, dtype=prediction.dtype)
        used = 0
        for fft_size in fft_sizes:
            if int(masked_prediction.shape[1]) < fft_size:
                continue
            hop = max(64, fft_size // 4)
            window = torch.hann_window(fft_size, device=prediction.device, dtype=prediction.dtype)
            pred_stft = torch.stft(
                masked_prediction,
                n_fft=fft_size,
                hop_length=hop,
                win_length=fft_size,
                window=window,
                return_complex=True,
            )
            target_stft = torch.stft(
                masked_target,
                n_fft=fft_size,
                hop_length=hop,
                win_length=fft_size,
                window=window,
                return_complex=True,
            )
            pred_mag = pred_stft.abs()
            target_mag = target_stft.abs()
            total = total + torch.mean(torch.abs(pred_mag - target_mag))
            total = total + (0.5 * torch.mean(torch.abs(torch.log1p(pred_mag) - torch.log1p(target_mag))))
            used += 1
        if used <= 0:
            return torch.mean(torch.abs(masked_prediction - masked_target))
        return total / float(used)

    def _vocoder_discriminator_hinge_loss(
        self,
        discriminator: MultiScaleWaveDiscriminator,
        *,
        real_audio: torch.Tensor,
        fake_audio: torch.Tensor,
    ) -> torch.Tensor:
        real_outputs = discriminator(real_audio)
        fake_outputs = discriminator(fake_audio.detach())
        total = torch.zeros((), device=real_audio.device, dtype=real_audio.dtype)
        for (real_logits, _), (fake_logits, _) in zip(real_outputs, fake_outputs):
            total = total + torch.mean(F.relu(1.0 - real_logits))
            total = total + torch.mean(F.relu(1.0 + fake_logits))
        return total / float(max(len(real_outputs), 1))

    def _vocoder_generator_losses(
        self,
        discriminator: MultiScaleWaveDiscriminator,
        *,
        real_audio: torch.Tensor,
        fake_audio: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        real_outputs = discriminator(real_audio)
        fake_outputs = discriminator(fake_audio)
        adversarial = torch.zeros((), device=real_audio.device, dtype=real_audio.dtype)
        feature_matching = torch.zeros((), device=real_audio.device, dtype=real_audio.dtype)
        scale_count = max(len(fake_outputs), 1)
        for (real_logits, real_features), (fake_logits, fake_features) in zip(real_outputs, fake_outputs):
            adversarial = adversarial - fake_logits.mean()
            if real_features and fake_features:
                layer_count = max(len(fake_features), 1)
                for real_feature, fake_feature in zip(real_features, fake_features):
                    feature_matching = feature_matching + (
                        torch.mean(torch.abs(fake_feature - real_feature.detach())) / float(layer_count)
                    )
        return adversarial / float(scale_count), feature_matching / float(scale_count)

    def _save_vocoder_checkpoint(
        self,
        *,
        path: Path,
        generator: PersonaNeuralVocoder,
        discriminator: MultiScaleWaveDiscriminator,
        optimizer_g: torch.optim.Optimizer,
        optimizer_d: torch.optim.Optimizer,
        epoch: int,
        best_val_loss: float,
        config: Dict[str, object],
    ) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        checkpoint_generator = getattr(generator, "_orig_mod", generator)
        checkpoint_discriminator = getattr(discriminator, "_orig_mod", discriminator)
        torch.save(
            {
                "epoch": int(epoch),
                "best_val_loss": float(best_val_loss),
                "config": dict(config),
                "generator_state": checkpoint_generator.state_dict(),
                "discriminator_state": checkpoint_discriminator.state_dict(),
                "optimizer_g_state": optimizer_g.state_dict(),
                "optimizer_d_state": optimizer_d.state_dict(),
            },
            path,
        )

    def _load_vocoder_from_checkpoint(
        self,
        checkpoint_path: Path,
        *,
        device: torch.device,
    ) -> Tuple[PersonaNeuralVocoder, Dict[str, object]]:
        checkpoint = torch.load(checkpoint_path, map_location=device)
        config = dict(checkpoint.get("config", {}))
        generator = PersonaNeuralVocoder(
            base_channels=int(config.get("base_channels", 384)),
        ).to(device)
        generator.load_state_dict(checkpoint.get("generator_state", {}), strict=False)
        generator.eval()
        return generator, config

    def _phone_accuracy_counts(
        self,
        logits: torch.Tensor,
        target: torch.Tensor,
        lengths: torch.Tensor,
        lyric_mask: torch.Tensor,
    ) -> Dict[str, float]:
        max_frames = int(target.shape[1])
        valid = (
            (torch.arange(max_frames, device=lengths.device).unsqueeze(0) < lengths.unsqueeze(1))
            & (target != PHONE_TO_ID["PAD"])
        )
        predictions = torch.argmax(logits, dim=-1)
        correct = (predictions == target) & valid
        lyric_valid = valid & (lyric_mask > 0.5) & (target != PHONE_TO_ID["SP"])
        lyric_correct = (predictions == target) & lyric_valid
        return {
            "phone_correct": float(correct.sum().detach().cpu().item()),
            "phone_total": float(valid.sum().detach().cpu().item()),
            "lyric_phone_correct": float(lyric_correct.sum().detach().cpu().item()),
            "lyric_phone_total": float(lyric_valid.sum().detach().cpu().item()),
        }

    def _vuv_accuracy_counts(
        self,
        logits: torch.Tensor,
        target: torch.Tensor,
        lengths: torch.Tensor,
    ) -> Dict[str, float]:
        max_frames = int(target.shape[1])
        valid = (
            torch.arange(max_frames, device=lengths.device).unsqueeze(0) < lengths.unsqueeze(1)
        )
        predictions = torch.sigmoid(logits) >= 0.5
        target_voiced = target >= 0.5
        correct = (predictions == target_voiced) & valid
        return {
            "vuv_correct": float(correct.sum().detach().cpu().item()),
            "vuv_total": float(valid.sum().detach().cpu().item()),
        }

    def _phone_accuracy_per_sample(
        self,
        logits: torch.Tensor,
        target: torch.Tensor,
        lengths: torch.Tensor,
        lyric_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        max_frames = int(target.shape[1])
        valid = (
            (torch.arange(max_frames, device=lengths.device).unsqueeze(0) < lengths.unsqueeze(1))
            & (target != PHONE_TO_ID["PAD"])
        )
        predictions = torch.argmax(logits, dim=-1)
        correct = ((predictions == target) & valid).to(dtype=logits.dtype)
        phone_accuracy = correct.sum(dim=1) / valid.to(dtype=logits.dtype).sum(dim=1).clamp_min(1.0)
        lyric_valid = valid & (lyric_mask > 0.5) & (target != PHONE_TO_ID["SP"])
        lyric_correct = ((predictions == target) & lyric_valid).to(dtype=logits.dtype)
        lyric_accuracy = lyric_correct.sum(dim=1) / lyric_valid.to(dtype=logits.dtype).sum(dim=1).clamp_min(1.0)
        return phone_accuracy, lyric_accuracy

    def _vuv_accuracy_per_sample(
        self,
        logits: torch.Tensor,
        target: torch.Tensor,
        lengths: torch.Tensor,
    ) -> torch.Tensor:
        max_frames = int(target.shape[1])
        valid = torch.arange(max_frames, device=lengths.device).unsqueeze(0) < lengths.unsqueeze(1)
        predictions = torch.sigmoid(logits) >= 0.5
        target_voiced = target >= 0.5
        correct = ((predictions == target_voiced) & valid).to(dtype=logits.dtype)
        return correct.sum(dim=1) / valid.to(dtype=logits.dtype).sum(dim=1).clamp_min(1.0)

    def _voice_signature_losses(
        self,
        *,
        outputs: Dict[str, torch.Tensor],
        batch: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        predicted_voice_signature = _compute_voice_signature_torch(
            mel=outputs["mel"],
            log_f0=outputs["target_log_f0"],
            vuv=torch.sigmoid(outputs["target_vuv_logits"]),
            lengths=batch["lengths"],
        )
        target_voice_signature = batch["target_voice_signature"].to(
            device=predicted_voice_signature.device,
            dtype=predicted_voice_signature.dtype,
        )
        voice_prototype = batch["voice_prototype"].to(
            device=predicted_voice_signature.device,
            dtype=predicted_voice_signature.dtype,
        )
        voice_loss = 1.0 - F.cosine_similarity(predicted_voice_signature, target_voice_signature, dim=-1).mean()
        prototype_valid = voice_prototype.norm(dim=-1) > 1e-6
        if bool(torch.any(prototype_valid)):
            prototype_loss = 1.0 - F.cosine_similarity(
                predicted_voice_signature[prototype_valid],
                voice_prototype[prototype_valid],
                dim=-1,
            ).mean()
        else:
            prototype_loss = torch.zeros(
                (),
                device=predicted_voice_signature.device,
                dtype=predicted_voice_signature.dtype,
            )
        return voice_loss, prototype_loss

    def _describe_quality_state(
        self,
        *,
        improved: bool,
        epochs_since_best: int,
        lyric_phone_accuracy: float,
        vuv_accuracy: float,
        delta_mel_loss: float,
        phase_name: str = "",
        target_quality_loss: float = 0.0,
    ) -> str:
        normalized_phase = str(phase_name or "").strip().lower()
        if improved:
            return "new best"
        if normalized_phase == "general-refine":
            if epochs_since_best <= 4:
                return "locking temporal coherence"
            if target_quality_loss > 12.0:
                return "removing robotic jitter"
            if target_quality_loss > 7.0:
                return "tightening target texture"
            if delta_mel_loss > 14.0:
                return "steadying vocal continuity"
            if epochs_since_best >= 30:
                return "plateauing"
            return "coherence polish"
        if epochs_since_best <= 4:
            return "holding near best"
        if lyric_phone_accuracy < 0.72:
            return "still learning lyrics"
        if vuv_accuracy < 0.9:
            return "still stabilizing voicing"
        if delta_mel_loss > 0.06:
            return "smoothing transitions"
        if epochs_since_best >= 40:
            return "plateauing"
        return "steady refinement"

    def _compute_loss_terms(
        self,
        *,
        outputs: Dict[str, torch.Tensor],
        batch: Dict[str, torch.Tensor],
        loss_weights: Dict[str, float],
    ) -> Dict[str, torch.Tensor]:
        lyric_mask = batch["lyric_mask"]
        mel_loss = self._masked_l1_loss(outputs["mel"], batch["mel"], batch["lengths"])
        stft_loss, phase_loss = self._multi_resolution_temporal_stft_loss(
            outputs["mel"],
            batch["mel"],
            batch["lengths"],
        )
        f0_loss = self._masked_f0_loss(outputs["target_log_f0"], batch["target_log_f0"], batch["lengths"])
        delta_mel_loss = self._masked_delta_loss(
            outputs["mel"],
            batch["mel"],
            batch["lengths"],
            mask=lyric_mask,
        )
        accel_mel_loss = self._masked_acceleration_loss(
            outputs["mel"],
            batch["mel"],
            batch["lengths"],
            mask=lyric_mask,
        )
        smooth_mel_loss = self._masked_smoothness_loss(
            outputs["mel"],
            batch["mel"],
            batch["lengths"],
            mask=lyric_mask,
        )
        high_band_loss = self._masked_high_band_loss(
            outputs["mel"],
            batch["mel"],
            batch["lengths"],
            mask=lyric_mask,
        )
        delta_f0_loss = self._masked_delta_loss(
            outputs["target_log_f0"],
            batch["target_log_f0"],
            batch["lengths"],
            mask=batch["target_vuv"],
        )
        vuv_loss = self._masked_bce_loss(
            outputs["target_vuv_logits"],
            batch["target_vuv"],
            batch["lengths"],
        )
        phone_loss = self._masked_phone_cross_entropy_loss(
            outputs["phone_logits"],
            batch["phone_ids"],
            batch["lengths"],
            batch["lyric_mask"],
        )
        voice_loss, prototype_loss = self._voice_signature_losses(
            outputs=outputs,
            batch=batch,
        )
        target_quality_loss = (
            (0.26 * stft_loss)
            + (0.24 * delta_mel_loss)
            + (0.18 * accel_mel_loss)
            + (0.14 * smooth_mel_loss)
            + (0.14 * high_band_loss)
            + (0.10 * voice_loss)
            + (0.08 * prototype_loss)
        )
        silence_loss = self._off_lyric_suppression_loss(
            outputs["mel"],
            lyric_mask,
            batch["lengths"],
        )
        guide_delta_loss = torch.zeros((), device=outputs["mel"].device, dtype=outputs["mel"].dtype)
        guide_delta_transition_loss = torch.zeros((), device=outputs["mel"].device, dtype=outputs["mel"].dtype)
        guide_delta_smooth_loss = torch.zeros((), device=outputs["mel"].device, dtype=outputs["mel"].dtype)
        if "guide_delta" in outputs:
            guide_delta_target = batch["mel"] - batch["guide_mel"]
            guide_delta_loss = self._masked_l1_loss(
                outputs["guide_delta"],
                guide_delta_target,
                batch["lengths"],
                mask=lyric_mask,
            )
            guide_delta_transition_loss = self._masked_delta_loss(
                outputs["guide_delta"],
                guide_delta_target,
                batch["lengths"],
                mask=lyric_mask,
            )
            guide_delta_smooth_loss = self._masked_smoothness_loss(
                outputs["guide_delta"],
                guide_delta_target,
                batch["lengths"],
                mask=lyric_mask,
            )
            target_quality_loss = (
                target_quality_loss
                + (0.18 * guide_delta_loss)
                + (0.14 * guide_delta_transition_loss)
                + (0.10 * guide_delta_smooth_loss)
            )
        total = (
            (float(loss_weights.get("mel", 1.0)) * mel_loss)
            + (float(loss_weights.get("stft", 1.0)) * stft_loss)
            + (float(loss_weights.get("phase", 0.28)) * phase_loss)
            + (float(loss_weights.get("f0", 0.3)) * f0_loss)
            + (float(loss_weights.get("delta_mel", 1.0)) * delta_mel_loss)
            + (float(loss_weights.get("accel_mel", 0.5)) * accel_mel_loss)
            + (float(loss_weights.get("smooth_mel", 0.3)) * smooth_mel_loss)
            + (float(loss_weights.get("high_band", 0.36)) * high_band_loss)
            + (float(loss_weights.get("delta_f0", 0.24)) * delta_f0_loss)
            + (float(loss_weights.get("vuv", 0.2)) * vuv_loss)
            + (float(loss_weights.get("phones", 0.32)) * phone_loss)
            + (float(loss_weights.get("voice", 0.34)) * voice_loss)
            + (float(loss_weights.get("prototype", 0.22)) * prototype_loss)
            + (float(loss_weights.get("silence", 0.24)) * silence_loss)
            + (float(loss_weights.get("guide_delta", 0.34)) * guide_delta_loss)
            + (float(loss_weights.get("guide_delta_transition", 0.22)) * guide_delta_transition_loss)
            + (float(loss_weights.get("guide_delta_smooth", 0.14)) * guide_delta_smooth_loss)
            + (float(loss_weights.get("target_quality", 0.16)) * target_quality_loss)
        )
        return {
            "total": total,
            "mel": mel_loss,
            "stft": stft_loss,
            "phase": phase_loss,
            "f0": f0_loss,
            "delta_mel": delta_mel_loss,
            "accel_mel": accel_mel_loss,
            "smooth_mel": smooth_mel_loss,
            "high_band": high_band_loss,
            "delta_f0": delta_f0_loss,
            "vuv": vuv_loss,
            "phones": phone_loss,
            "voice": voice_loss,
            "prototype": prototype_loss,
            "silence": silence_loss,
            "target_quality": target_quality_loss,
            "guide_anchor": guide_delta_loss,
            "guide_transition": guide_delta_transition_loss,
        }

    def _compute_persona_v11_loss_terms(
        self,
        *,
        outputs: Dict[str, torch.Tensor],
        batch: Dict[str, torch.Tensor],
        loss_weights: Dict[str, float],
        teacher_outputs: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Dict[str, torch.Tensor]:
        terms = self._compute_loss_terms(
            outputs=outputs,
            batch=batch,
            loss_weights=loss_weights,
        )
        lyric_mask = batch.get("lyric_mask")
        predicted_voice_signature = _compute_voice_signature_torch(
            mel=outputs["mel"],
            log_f0=outputs["target_log_f0"],
            vuv=torch.sigmoid(outputs["target_vuv_logits"]),
            lengths=batch["lengths"],
        )
        target_voice_signature = batch["target_voice_signature"].to(
            device=predicted_voice_signature.device,
            dtype=predicted_voice_signature.dtype,
        )
        guide_voice_signature = _compute_voice_signature_torch(
            mel=batch["guide_mel"],
            log_f0=batch["log_f0"],
            vuv=batch["vuv"],
            lengths=batch["lengths"],
        ).to(
            device=predicted_voice_signature.device,
            dtype=predicted_voice_signature.dtype,
        )
        target_similarity = F.cosine_similarity(predicted_voice_signature, target_voice_signature, dim=-1)
        guide_similarity = F.cosine_similarity(predicted_voice_signature, guide_voice_signature, dim=-1)
        identity_margin_loss = F.relu(
            float(loss_weights.get("identity_margin_target", 0.10)) + guide_similarity - target_similarity
        ).mean()
        predicted_residual = outputs["mel"] - batch["guide_mel"]
        target_residual = batch["mel"] - batch["guide_mel"]
        guide_residual_loss = self._masked_l1_loss(
            predicted_residual,
            target_residual,
            batch["lengths"],
            mask=lyric_mask,
        )
        guide_residual_delta_loss = self._masked_delta_loss(
            predicted_residual,
            target_residual,
            batch["lengths"],
            mask=lyric_mask,
        )
        terms["target_quality"] = (
            terms["target_quality"]
            + (0.18 * identity_margin_loss)
            + (0.24 * guide_residual_loss)
            + (0.18 * guide_residual_delta_loss)
        )
        terms["total"] = terms["total"] + (
            (float(loss_weights.get("identity_margin", 0.44)) * identity_margin_loss)
            + (float(loss_weights.get("guide_residual", 0.76)) * guide_residual_loss)
            + (float(loss_weights.get("guide_residual_delta", 0.48)) * guide_residual_delta_loss)
        )
        if not teacher_outputs:
            return terms
        teacher_mel = teacher_outputs["mel"].detach()
        teacher_mel_loss = self._masked_l1_loss(
            outputs["mel"],
            teacher_mel,
            batch["lengths"],
            mask=lyric_mask,
        )
        teacher_delta_loss = self._masked_delta_loss(
            outputs["mel"],
            teacher_mel,
            batch["lengths"],
            mask=lyric_mask,
        )
        teacher_alignment = teacher_mel_loss + (0.35 * teacher_delta_loss)
        terms["target_quality"] = terms["target_quality"] + (0.14 * teacher_alignment)
        terms["total"] = terms["total"] + (
            float(loss_weights.get("teacher", 0.18)) * teacher_alignment
        )
        return terms

    def _compute_post_process_loss_terms(
        self,
        *,
        outputs: Dict[str, torch.Tensor],
        coarse_mel: torch.Tensor,
        batch: Dict[str, torch.Tensor],
        loss_weights: Dict[str, float],
    ) -> Dict[str, torch.Tensor]:
        base_terms = self._compute_loss_terms(
            outputs=outputs,
            batch=batch,
            loss_weights=loss_weights,
        )
        anchor_loss = self._masked_l1_loss(outputs["mel"], coarse_mel, batch["lengths"])
        total = base_terms["total"] + (float(loss_weights.get("anchor", 0.08)) * anchor_loss)
        return {
            **base_terms,
            "total": total,
            "anchor": anchor_loss,
        }

    def _compute_lyric_repair_loss_terms(
        self,
        *,
        outputs: Dict[str, torch.Tensor],
        batch: Dict[str, torch.Tensor],
        loss_weights: Dict[str, float],
    ) -> Dict[str, torch.Tensor]:
        base_terms = self._compute_loss_terms(
            outputs=outputs,
            batch=batch,
            loss_weights=loss_weights,
        )
        guide_anchor_loss = self._masked_l1_loss(
            outputs["mel"],
            batch["guide_mel"],
            batch["lengths"],
            mask=batch["lyric_mask"],
        )
        guide_transition_loss = self._masked_delta_loss(
            outputs["mel"],
            batch["guide_mel"],
            batch["lengths"],
            mask=batch["lyric_mask"],
        )
        total = (
            base_terms["total"]
            + (float(loss_weights.get("guide_anchor", 0.24)) * guide_anchor_loss)
            + (float(loss_weights.get("guide_transition", 0.16)) * guide_transition_loss)
        )
        target_quality = base_terms["target_quality"] + (0.14 * guide_anchor_loss) + (0.12 * guide_transition_loss)
        return {
            **base_terms,
            "total": total,
            "target_quality": target_quality,
            "guide_anchor": guide_anchor_loss,
            "guide_transition": guide_transition_loss,
        }

    def _compute_target_fit_loss_terms(
        self,
        *,
        outputs: Dict[str, torch.Tensor],
        batch: Dict[str, torch.Tensor],
        loss_weights: Dict[str, float],
    ) -> Dict[str, torch.Tensor]:
        base_terms = self._compute_loss_terms(
            outputs=outputs,
            batch=batch,
            loss_weights=loss_weights,
        )
        guide_anchor_loss = self._masked_l1_loss(
            outputs["mel"],
            batch["guide_mel"],
            batch["lengths"],
            mask=batch["lyric_mask"],
        )
        guide_transition_loss = self._masked_delta_loss(
            outputs["mel"],
            batch["guide_mel"],
            batch["lengths"],
            mask=batch["lyric_mask"],
        )
        total = (
            base_terms["total"]
            + (float(loss_weights.get("guide_anchor", 0.12)) * guide_anchor_loss)
            + (float(loss_weights.get("guide_transition", 0.1)) * guide_transition_loss)
        )
        target_quality = (
            base_terms["target_quality"]
            + (0.18 * guide_anchor_loss)
            + (0.14 * guide_transition_loss)
        )
        return {
            **base_terms,
            "total": total,
            "target_quality": target_quality,
            "guide_anchor": guide_anchor_loss,
            "guide_transition": guide_transition_loss,
        }

    def _compute_general_refine_loss_terms(
        self,
        *,
        outputs: Dict[str, torch.Tensor],
        batch: Dict[str, torch.Tensor],
        loss_weights: Dict[str, float],
    ) -> Dict[str, torch.Tensor]:
        base_terms = self._compute_loss_terms(
            outputs=outputs,
            batch=batch,
            loss_weights=loss_weights,
        )
        voiced_mask = torch.clamp(batch["lyric_mask"], 0.0, 1.0) * torch.clamp(batch["target_vuv"], 0.0, 1.0)
        flutter_transition_loss = self._masked_delta_loss(
            outputs["mel"],
            batch["mel"],
            batch["lengths"],
            mask=voiced_mask,
        )
        flutter_accel_loss = self._masked_acceleration_loss(
            outputs["mel"],
            batch["mel"],
            batch["lengths"],
            mask=voiced_mask,
        )
        flutter_high_band_loss = self._masked_high_band_loss(
            outputs["mel"],
            batch["mel"],
            batch["lengths"],
            mask=voiced_mask,
            start_ratio=0.52,
        )
        stability_reg = self._masked_smoothness_loss(
            outputs["mel"],
            batch["mel"],
            batch["lengths"],
            mask=voiced_mask,
        )
        total = (
            base_terms["total"]
            + (float(loss_weights.get("flutter_transition", 0.72)) * flutter_transition_loss)
            + (float(loss_weights.get("flutter_accel", 0.58)) * flutter_accel_loss)
            + (float(loss_weights.get("flutter_high_band", 0.54)) * flutter_high_band_loss)
            + (float(loss_weights.get("stability_reg", 0.20)) * stability_reg)
        )
        target_quality = (
            base_terms["target_quality"]
            + (0.22 * flutter_transition_loss)
            + (0.18 * flutter_accel_loss)
            + (0.16 * flutter_high_band_loss)
            + (0.14 * stability_reg)
        )
        return {
            **base_terms,
            "total": total,
            "target_quality": target_quality,
            "guide_anchor": torch.zeros((), device=outputs["mel"].device, dtype=outputs["mel"].dtype),
            "guide_transition": torch.zeros((), device=outputs["mel"].device, dtype=outputs["mel"].dtype),
            "flutter": flutter_transition_loss + (0.65 * flutter_accel_loss) + (0.55 * flutter_high_band_loss),
            "stability": stability_reg,
        }

    def _is_lyric_repair_recipe(self, config: Optional[Dict[str, object]] = None) -> bool:
        config = dict(config or {})
        recipe_mode = str(config.get("recipe_mode", "") or "").strip().lower()
        training_mode = str(config.get("training_mode", "") or "").strip().lower()
        if recipe_mode == "persona-lyric-repair":
            return True
        return "lyric-repair" in training_mode

    def _is_persona_v11_recipe(self, config: Optional[Dict[str, object]] = None) -> bool:
        config = dict(config or {})
        recipe_mode = str(config.get("recipe_mode", "") or "").strip().lower()
        training_mode = str(config.get("training_mode", "") or "").strip().lower()
        model_config = dict(config.get("model", {}) or {})
        persona_version = str(
            model_config.get("persona_version", config.get("persona_version", ""))
            or ""
        ).strip().lower()
        if recipe_mode in DIRECT_GUIDED_RECIPE_MODES:
            return True
        if training_mode in DIRECT_GUIDED_TRAINING_MODES:
            return True
        return persona_version == "v1.1"

    def _get_repair_strategy(self, config: Optional[Dict[str, object]] = None) -> Dict[str, object]:
        config = dict(config or {})
        strategy = dict(config.get("repair_strategy", {}) or {})
        return {
            "enabled": bool(strategy.get("enabled", self._is_lyric_repair_recipe(config))),
            "micro_window_frames": max(12, int(strategy.get("micro_window_frames", 28) or 28)),
            "micro_window_hop": max(2, int(strategy.get("micro_window_hop", 6) or 6)),
            "micro_window_pad": max(0, int(strategy.get("micro_window_pad", 6) or 6)),
            "micro_window_max": max(32, int(strategy.get("micro_window_max", 768) or 768)),
            "micro_candidate_count": max(1, int(strategy.get("micro_candidate_count", 4) or 4)),
            "dense_blend": float(np.clip(float(strategy.get("dense_blend", 0.84) or 0.84), 0.05, 1.25)),
            "guide_anchor_blend": float(np.clip(float(strategy.get("guide_anchor_blend", 0.06) or 0.06), 0.0, 0.4)),
            "off_lyric_floor": float(np.clip(float(strategy.get("off_lyric_floor", 0.01) or 0.01), 0.0, 0.2)),
            "wave_gate_smoothing_frames": max(0, int(strategy.get("wave_gate_smoothing_frames", 4) or 4)),
        }

    def _select_curriculum_entries(
        self,
        *,
        entries: Sequence[Dict[str, object]],
        epoch: int,
        total_epochs: int,
        warmup_end_epoch: int = 600,
        bridge_end_epoch: int = 1800,
        full_diversity_end_epoch: Optional[int] = None,
        general_refine_end_epoch: Optional[int] = None,
    ) -> Tuple[List[Dict[str, object]], Dict[str, object]]:
        all_entries = [dict(entry) for entry in entries]
        if not all_entries:
            return [], {"name": "empty", "fraction": 0.0}
        if total_epochs <= 1:
            return all_entries, {"name": "full-diversity", "fraction": 1.0}
        identity_entries = [entry for entry in all_entries if str(entry.get("slice_kind", "")) == "base-identity-window"]
        paired_entries = [entry for entry in all_entries if str(entry.get("slice_kind", "")) != "base-identity-window"]
        paired_entries = sorted(
            paired_entries,
            key=lambda entry: (
                float(entry.get("difficulty_score", 0.5)),
                -float(entry.get("alignment_score", 0.0)),
            ),
        )
        effective_warmup_end = max(0, int(warmup_end_epoch))
        effective_bridge_end = max(effective_warmup_end, int(bridge_end_epoch))
        effective_full_end = max(
            effective_bridge_end,
            int(full_diversity_end_epoch if full_diversity_end_epoch is not None else total_epochs),
        )
        effective_general_refine_end = max(
            effective_full_end,
            int(general_refine_end_epoch if general_refine_end_epoch is not None else effective_full_end),
        )
        if epoch <= effective_warmup_end and effective_warmup_end > 0:
            fraction = 0.35
            phase_name = "warm-up"
        elif epoch <= effective_bridge_end and effective_bridge_end > effective_warmup_end:
            fraction = 0.7
            phase_name = "curriculum-bridge"
        elif epoch <= effective_full_end and effective_full_end > effective_bridge_end:
            fraction = 1.0
            phase_name = "full-diversity"
        elif epoch <= effective_general_refine_end and effective_general_refine_end > effective_full_end:
            fraction = 1.0
            phase_name = "general-refine"
        else:
            fraction = 1.0
            phase_name = "general-refine"
        if phase_name == "general-refine":
            target_fit_entries = sorted(
                paired_entries,
                key=lambda entry: (
                    -float(entry.get("alignment_score", 0.0)),
                    float(entry.get("difficulty_score", 0.5)),
                ),
            )
            selected = identity_entries + target_fit_entries
            keep_count = len(target_fit_entries)
        else:
            keep_count = len(paired_entries) if fraction >= 0.999 else max(1, int(math.ceil(len(paired_entries) * fraction)))
            selected = identity_entries + paired_entries[:keep_count]
        if not selected:
            selected = all_entries
        return selected, {
            "name": phase_name,
            "fraction": round(float(fraction), 3),
            "selected_count": len(selected),
            "paired_count": len(paired_entries),
            "identity_count": len(identity_entries),
        }

    def _normalize_curriculum_phase(self, start_phase: str) -> str:
        normalized_phase = str(start_phase or "auto").strip().lower()
        return {
            "warmup": "warm-up",
            "bridge": "curriculum-bridge",
            "full": "full-diversity",
            "fit": "general-refine",
            "targetfit": "general-refine",
            "refine": "general-refine",
            "general": "general-refine",
            "generalrefine": "general-refine",
            "post": "general-refine",
            "postprocess": "general-refine",
        }.get(normalized_phase, normalized_phase)

    def _resolve_curriculum_schedule(
        self,
        *,
        start_phase: str,
        resume_epoch: int,
        warmup_stage_epochs: int,
        bridge_stage_epochs: int,
        full_diversity_stage_epochs: int,
        general_refine_stage_epochs: int,
        target_fit_stage_epochs: int = 0,
        post_process_stage_epochs: int = 0,
    ) -> Dict[str, int | str]:
        normalized_phase = self._normalize_curriculum_phase(start_phase)
        anchor_epoch = 0 if normalized_phase == "auto" else max(0, int(resume_epoch))
        effective_warmup = max(0, int(warmup_stage_epochs))
        effective_bridge = max(0, int(bridge_stage_epochs))
        effective_full = max(0, int(full_diversity_stage_epochs))
        effective_general_refine = (
            max(0, int(general_refine_stage_epochs))
            + max(0, int(target_fit_stage_epochs))
            + max(0, int(post_process_stage_epochs))
        )
        if normalized_phase == "warm-up":
            warmup_budget = effective_warmup
            bridge_budget = effective_bridge
            full_budget = effective_full
            general_refine_budget = effective_general_refine
        elif normalized_phase == "curriculum-bridge":
            warmup_budget = 0
            bridge_budget = effective_bridge
            full_budget = effective_full
            general_refine_budget = effective_general_refine
        elif normalized_phase == "full-diversity":
            warmup_budget = 0
            bridge_budget = 0
            full_budget = effective_full
            general_refine_budget = effective_general_refine
        elif normalized_phase == "general-refine":
            warmup_budget = 0
            bridge_budget = 0
            full_budget = 0
            general_refine_budget = effective_general_refine
        else:
            warmup_budget = effective_warmup
            bridge_budget = effective_bridge
            full_budget = effective_full
            general_refine_budget = effective_general_refine
        warmup_end_epoch = anchor_epoch + warmup_budget
        bridge_end_epoch = warmup_end_epoch + bridge_budget
        full_diversity_end_epoch = bridge_end_epoch + full_budget
        general_refine_end_epoch = full_diversity_end_epoch + general_refine_budget
        return {
            "normalized_start_phase": normalized_phase,
            "anchor_epoch": int(anchor_epoch),
            "warmup_stage_epochs": int(effective_warmup),
            "bridge_stage_epochs": int(effective_bridge),
            "full_diversity_stage_epochs": int(effective_full),
            "general_refine_stage_epochs": int(effective_general_refine),
            "target_fit_stage_epochs": int(effective_general_refine),
            "post_process_stage_epochs": 0,
            "warmup_end_epoch": int(warmup_end_epoch),
            "bridge_end_epoch": int(bridge_end_epoch),
            "full_diversity_end_epoch": int(full_diversity_end_epoch),
            "general_refine_end_epoch": int(general_refine_end_epoch),
            "target_fit_end_epoch": int(general_refine_end_epoch),
            "post_process_end_epoch": int(general_refine_end_epoch),
        }

    def _resolve_phase_loss_weights(
        self,
        config: Optional[Dict[str, object]],
        phase_name: str,
    ) -> Dict[str, float]:
        base_weights = dict((config or {}).get("loss_weights", {}) or {})
        normalized_phase = str(phase_name or "").strip().lower()
        if normalized_phase != "general-refine":
            return base_weights
        tuned = dict(base_weights)
        tuned["stft"] = max(float(tuned.get("stft", 1.0) or 1.0), 1.1)
        tuned["phase"] = max(float(tuned.get("phase", 0.28) or 0.28), 0.32)
        tuned["delta_mel"] = max(float(tuned.get("delta_mel", 1.0) or 1.0), 1.1)
        tuned["accel_mel"] = max(float(tuned.get("accel_mel", 0.5) or 0.5), 0.62)
        tuned["smooth_mel"] = max(float(tuned.get("smooth_mel", 0.3) or 0.3), 0.42)
        tuned["voice"] = max(float(tuned.get("voice", 0.34) or 0.34), 0.4)
        tuned["prototype"] = (
            max(float(tuned.get("prototype", 0.22) or 0.22), 0.28)
            if float(tuned.get("prototype", 0.0) or 0.0) > 0.0
            else 0.0
        )
        tuned["high_band"] = max(float(tuned.get("high_band", 0.36) or 0.36), 0.5)
        tuned["target_quality"] = max(float(tuned.get("target_quality", 0.16) or 0.16), 0.2)
        tuned["guide_anchor"] = 0.0
        tuned["guide_transition"] = 0.0
        tuned["phones"] = min(float(tuned.get("phones", 0.32) or 0.32), 0.24)
        tuned["vuv"] = min(float(tuned.get("vuv", 0.2) or 0.2), 0.16)
        tuned["identity_margin"] = max(float(tuned.get("identity_margin", 0.0) or 0.0), 0.50)
        tuned["guide_residual"] = max(float(tuned.get("guide_residual", 0.0) or 0.0), 0.88)
        tuned["guide_residual_delta"] = max(float(tuned.get("guide_residual_delta", 0.0) or 0.0), 0.58)
        tuned["flutter_transition"] = max(float(tuned.get("flutter_transition", 0.0) or 0.0), 0.72)
        tuned["flutter_accel"] = max(float(tuned.get("flutter_accel", 0.0) or 0.0), 0.58)
        tuned["flutter_high_band"] = max(float(tuned.get("flutter_high_band", 0.0) or 0.0), 0.54)
        tuned["stability_reg"] = max(float(tuned.get("stability_reg", 0.0) or 0.0), 0.2)
        return tuned

    def _set_general_refine_trainable(
        self,
        model: nn.Module,
        *,
        enabled: bool,
    ) -> None:
        base_model = getattr(model, "_orig_mod", model)
        if not isinstance(base_model, GuideConditionedMelRegenerator):
            return
        for name, parameter in base_model.named_parameters():
            parameter.requires_grad = True

    def _instantiate_model(self, config: Optional[Dict[str, object]] = None) -> nn.Module:
        config = dict(config or {})
        training_mode = str(config.get("training_mode", "") or "").strip().lower()
        voice_signature_dim = int(config.get("voice_signature_dim", VOICE_SIGNATURE_DIM) or VOICE_SIGNATURE_DIM)
        if bool(config.get("guide_conditioning", False)) or "paired" in training_mode or "persona" in training_mode:
            model_config = dict(config.get("model", {}))
            if self._is_persona_v11_recipe(config):
                if training_mode in DIRECT_GUIDED_TRAINING_MODES or model_config.get("content_dim") is not None:
                    return PersonaMapperNAR(
                        content_dim=int(model_config.get("content_dim", PERSONA_V11_CONTENT_DIM) or PERSONA_V11_CONTENT_DIM),
                        d_model=int(model_config.get("d_model", 256)),
                        n_heads=int(model_config.get("n_heads", 4)),
                        n_layers=int(model_config.get("n_layers", 8)),
                        dropout=float(model_config.get("dropout", 0.1)),
                        voice_signature_dim=voice_signature_dim,
                        use_guide_mel_conditioning=bool(model_config.get("use_guide_mel_conditioning", False)),
                        use_voice_prototype_conditioning=bool(
                            model_config.get("use_voice_prototype_conditioning", True)
                        ),
                        guide_condition_scale=float(model_config.get("guide_condition_scale", 0.55) or 0.55),
                    )
                return GuideConditionedMelRegeneratorV11(
                    d_model=int(model_config.get("d_model", 256)),
                    n_heads=int(model_config.get("n_heads", 4)),
                    n_layers=int(model_config.get("n_layers", 8)),
                    dropout=float(model_config.get("dropout", 0.1)),
                    voice_signature_dim=voice_signature_dim,
                    guide_residual_scale=float(model_config.get("guide_residual_scale", 0.72) or 0.72),
                    off_lyric_guide_floor=float(model_config.get("off_lyric_guide_floor", 0.02) or 0.02),
                    stability_refine_enabled=bool(model_config.get("stability_refine_enabled", False)),
                    stability_refine_scale=float(model_config.get("stability_refine_scale", 0.12) or 0.12),
                    contextual_voice_scale=float(model_config.get("contextual_voice_scale", 0.34) or 0.34),
                    guide_context_layers=int(model_config.get("guide_context_layers", 2) or 2),
                    guide_context_scale=float(model_config.get("guide_context_scale", 0.42) or 0.42),
                    guide_delta_scale=float(model_config.get("guide_delta_scale", 0.55) or 0.55),
                    coherence_refine_scale=float(model_config.get("coherence_refine_scale", 0.22) or 0.22),
                )
            return GuideConditionedMelRegenerator(
                d_model=int(model_config.get("d_model", 256)),
                n_heads=int(model_config.get("n_heads", 4)),
                n_layers=int(model_config.get("n_layers", 8)),
                dropout=float(model_config.get("dropout", 0.1)),
                voice_signature_dim=voice_signature_dim,
                guide_residual_refinement=bool(model_config.get("guide_residual_refinement", False)),
                guide_residual_scale=float(model_config.get("guide_residual_scale", 1.0) or 1.0),
                off_lyric_guide_floor=float(model_config.get("off_lyric_guide_floor", 0.04) or 0.04),
                stability_refine_enabled=bool(model_config.get("stability_refine_enabled", False)),
                stability_refine_scale=float(model_config.get("stability_refine_scale", 0.2) or 0.2),
                guide_mix_floor=float(model_config.get("guide_mix_floor", 0.04) or 0.04),
                guide_mix_ceiling=float(model_config.get("guide_mix_ceiling", 0.45) or 0.45),
                contextual_voice_scale=float(model_config.get("contextual_voice_scale", 0.28) or 0.28),
            )
        model_config = dict(config.get("model", {}))
        return FrameConditionedMelRegenerator(
            d_model=int(model_config.get("d_model", 192)),
            n_heads=int(model_config.get("n_heads", 4)),
            n_layers=int(model_config.get("n_layers", 6)),
            dropout=float(model_config.get("dropout", 0.1)),
            voice_signature_dim=voice_signature_dim,
        )

    def _reform_checkpoint_for_persona_v11(
        self,
        checkpoint: Dict[str, object],
        *,
        target_config: Dict[str, object],
    ) -> Dict[str, object]:
        source_config = dict(checkpoint.get("config", {}) or {})
        source_model_config = dict(source_config.get("model", {}) or {})
        reformed_config = dict(target_config)
        reformed_model_config = dict(reformed_config.get("model", {}) or {})
        for key in ("d_model", "n_heads", "n_layers", "dropout"):
            if key in source_model_config and source_model_config.get(key) is not None:
                reformed_model_config[key] = source_model_config.get(key)
        reformed_model_config["persona_version"] = "v1.1"
        reformed_model_config["content_dim"] = int(
            reformed_model_config.get("content_dim", PERSONA_V11_CONTENT_DIM) or PERSONA_V11_CONTENT_DIM
        )
        reformed_model_config["use_guide_mel_conditioning"] = True
        reformed_model_config["use_voice_prototype_conditioning"] = False
        reformed_model_config["guide_condition_scale"] = float(
            reformed_model_config.get("guide_condition_scale", 0.68) or 0.68
        )
        reformed_config["model"] = reformed_model_config
        reformed_config["recipe_mode"] = str(
            reformed_config.get("recipe_mode", "persona-v1.1") or "persona-v1.1"
        )
        reformed_config["training_mode"] = str(
            reformed_config.get("training_mode", "persona-paired-mapper-v1.1") or "persona-paired-mapper-v1.1"
        )
        reformed_config["guide_conditioning"] = False
        reform_model = self._instantiate_model(reformed_config).cpu()
        target_state = reform_model.state_dict()
        source_state = dict(checkpoint.get("model_state", {}) or {})

        for key, value in source_state.items():
            if key in target_state and target_state[key].shape == value.shape:
                target_state[key] = value

        reform_model.load_state_dict(target_state, strict=False)
        return {
            "epoch": 0,
            "best_val_loss": float("inf"),
            "best_target_quality": float("inf"),
            "config": reformed_config,
            "model_state": reform_model.state_dict(),
            "optimizer_state": None,
            "scheduler_state": None,
            "reformed_from_epoch": int(checkpoint.get("epoch", 0) or 0),
            "reformed_bootstrap": "legacy-teacher-distillation",
            "reformed_from_recipe_mode": str(
                source_config.get("recipe_mode", source_config.get("training_mode", "persona-v1"))
                or "persona-v1"
            ),
        }

    def _instantiate_post_process_model(self, config: Optional[Dict[str, object]] = None) -> PersonaPostProcessRefiner:
        config = dict(config or {})
        model_config = dict(config.get("model", {}))
        voice_signature_dim = int(config.get("voice_signature_dim", VOICE_SIGNATURE_DIM) or VOICE_SIGNATURE_DIM)
        return PersonaPostProcessRefiner(
            d_model=int(model_config.get("d_model", 192)),
            n_heads=int(model_config.get("n_heads", 4)),
            n_layers=int(model_config.get("n_layers", 6)),
            dropout=float(model_config.get("dropout", 0.1)),
            voice_signature_dim=voice_signature_dim,
        )

    def _predict_outputs(
        self,
        model: nn.Module,
        *,
        phone_ids: torch.Tensor,
        log_f0: torch.Tensor,
        vuv: torch.Tensor,
        energy: torch.Tensor,
        lengths: torch.Tensor,
        content: Optional[torch.Tensor] = None,
        beat_phase: Optional[torch.Tensor] = None,
        voice_prototype: Optional[torch.Tensor] = None,
        guide_mel: Optional[torch.Tensor] = None,
        lyric_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        base_model = getattr(model, "_orig_mod", model)
        if isinstance(base_model, PersonaMapperNAR):
            if content is None:
                raise RuntimeError("Persona v1.1 inference requires content features.")
            if beat_phase is None:
                raise RuntimeError("Persona v1.1 inference requires beat-phase features.")
            if log_f0.dim() == 2:
                log_f0 = log_f0.unsqueeze(-1)
            if beat_phase.dim() == 2:
                beat_phase = beat_phase.unsqueeze(-1)
            outputs = model(
                return_aux=True,
                content=content,
                log_f0=log_f0,
                beat_phase=beat_phase,
                lengths=lengths,
                voice_prototype=(
                    voice_prototype
                    if voice_prototype is not None
                    else torch.zeros(
                        content.shape[0],
                        VOICE_SIGNATURE_DIM,
                        device=content.device,
                        dtype=content.dtype,
                    )
                ),
                guide_mel=guide_mel,
            )
            if isinstance(outputs, dict):
                return outputs
            return {
                "mel": outputs,
                "target_log_f0": torch.zeros_like(log_f0.squeeze(-1) if log_f0.dim() == 3 else log_f0),
                "target_vuv_logits": torch.zeros_like(log_f0.squeeze(-1) if log_f0.dim() == 3 else log_f0),
                "phone_logits": torch.zeros(
                    phone_ids.shape[0],
                    phone_ids.shape[1],
                    len(PHONE_TOKENS),
                    device=phone_ids.device,
                    dtype=log_f0.dtype,
                ),
            }
        if log_f0.dim() == 3 and log_f0.shape[-1] == 1:
            log_f0 = log_f0.squeeze(-1)
        if vuv.dim() == 3 and vuv.shape[-1] == 1:
            vuv = vuv.squeeze(-1)
        if energy.dim() == 3 and energy.shape[-1] == 1:
            energy = energy.squeeze(-1)
        if lyric_mask is not None and lyric_mask.dim() == 3 and lyric_mask.shape[-1] == 1:
            lyric_mask = lyric_mask.squeeze(-1)
        kwargs = {
            "phone_ids": phone_ids,
            "log_f0": log_f0,
            "vuv": vuv,
            "energy": energy,
            "lengths": lengths,
            "voice_prototype": (
                voice_prototype
                if voice_prototype is not None
                else torch.zeros(
                    phone_ids.shape[0],
                    VOICE_SIGNATURE_DIM,
                    device=phone_ids.device,
                    dtype=log_f0.dtype,
                )
            ),
        }
        if isinstance(base_model, GuideConditionedMelRegenerator):
            kwargs["guide_mel"] = guide_mel if guide_mel is not None else torch.zeros(
                phone_ids.shape[0],
                phone_ids.shape[1],
                N_MELS,
                device=phone_ids.device,
                dtype=log_f0.dtype,
            )
            kwargs["lyric_mask"] = lyric_mask
        outputs = model(return_aux=True, **kwargs)
        if isinstance(outputs, dict):
            return outputs
        return {
            "mel": outputs,
            "target_log_f0": torch.zeros_like(log_f0),
            "target_vuv_logits": torch.zeros_like(log_f0),
            "phone_logits": torch.zeros(
                phone_ids.shape[0],
                phone_ids.shape[1],
                len(PHONE_TOKENS),
                device=phone_ids.device,
                dtype=log_f0.dtype,
            ),
        }

    def _predict_mel(
        self,
        model: nn.Module,
        *,
        phone_ids: torch.Tensor,
        log_f0: torch.Tensor,
        vuv: torch.Tensor,
        energy: torch.Tensor,
        lengths: torch.Tensor,
        content: Optional[torch.Tensor] = None,
        beat_phase: Optional[torch.Tensor] = None,
        voice_prototype: Optional[torch.Tensor] = None,
        guide_mel: Optional[torch.Tensor] = None,
        lyric_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return self._predict_outputs(
            model,
            phone_ids=phone_ids,
            log_f0=log_f0,
            vuv=vuv,
            energy=energy,
            lengths=lengths,
            content=content,
            beat_phase=beat_phase,
            voice_prototype=voice_prototype,
            guide_mel=guide_mel,
            lyric_mask=lyric_mask,
        )["mel"]

    def _save_checkpoint(
        self,
        *,
        path: Path,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        epoch: int,
        best_val_loss: float,
        config: Dict[str, object],
        best_target_quality: Optional[float] = None,
    ) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        checkpoint_model = getattr(model, "_orig_mod", model)
        torch.save(
            {
                "epoch": int(epoch),
                "best_val_loss": float(best_val_loss),
                "best_target_quality": (
                    float(best_target_quality)
                    if best_target_quality is not None and math.isfinite(float(best_target_quality))
                    else None
                ),
                "config": dict(config),
                "model_state": checkpoint_model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "scheduler_state": scheduler.state_dict() if scheduler is not None else None,
            },
            path,
        )

    def _save_post_process_checkpoint(
        self,
        *,
        path: Path,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        epoch: int,
        best_val_loss: float,
        config: Dict[str, object],
    ) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        checkpoint_model = getattr(model, "_orig_mod", model)
        torch.save(
            {
                "epoch": int(epoch),
                "best_val_loss": float(best_val_loss),
                "config": dict(config),
                "model_state": checkpoint_model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "scheduler_state": scheduler.state_dict() if scheduler is not None else None,
            },
            path,
        )

    def _load_model_from_checkpoint(
        self,
        checkpoint_path: Path,
        *,
        device: torch.device,
    ) -> Tuple[FrameConditionedMelRegenerator, Dict[str, object]]:
        checkpoint = torch.load(checkpoint_path, map_location=device)
        config = dict(checkpoint.get("config", {}))
        model = self._instantiate_model(config).to(device)
        model.load_state_dict(checkpoint["model_state"], strict=False)
        model.eval()
        return model, config

    def _load_post_process_from_checkpoint(
        self,
        checkpoint_path: Path,
        *,
        device: torch.device,
    ) -> Tuple[PersonaPostProcessRefiner, Dict[str, object]]:
        checkpoint = torch.load(checkpoint_path, map_location=device)
        config = dict(checkpoint.get("config", {}))
        model = self._instantiate_post_process_model(config).to(device)
        model.load_state_dict(checkpoint.get("model_state", {}), strict=False)
        model.eval()
        return model, config

    def _denormalize_log_mel(self, normalized_mel: np.ndarray, stats: Dict[str, object]) -> np.ndarray:
        mel_mean = np.asarray(stats.get("mel_mean", [0.0] * N_MELS), dtype=np.float32)
        mel_std = np.asarray(stats.get("mel_std", [1.0] * N_MELS), dtype=np.float32)
        return (normalized_mel * mel_std[np.newaxis, :]) + mel_mean[np.newaxis, :]

    def _get_inference_candidate_count(self, device: torch.device) -> int:
        if device.type != "cuda":
            return 1
        gpu_memory_gb = 0.0
        try:
            gpu_memory_gb = float(torch.cuda.get_device_properties(0).total_memory) / float(1024**3)
        except Exception:
            gpu_memory_gb = 0.0
        if gpu_memory_gb >= 160.0:
            return 8
        if gpu_memory_gb >= 80.0:
            return 6
        if gpu_memory_gb >= 40.0:
            return 4
        if gpu_memory_gb >= 20.0:
            return 2
        return 1

    def _run_inference_candidate_search(
        self,
        *,
        model: nn.Module,
        batch: Dict[str, torch.Tensor],
        candidate_count: int,
        post_process_model: Optional[nn.Module] = None,
    ) -> Tuple[np.ndarray, Dict[str, object]]:
        candidate_count = max(1, int(candidate_count))
        was_training = model.training
        post_process_was_training = bool(getattr(post_process_model, "training", False))
        search_batch = {key: value for key, value in batch.items()}
        if candidate_count > 1:
            search_batch = {
                key: value.repeat(candidate_count, *([1] * max(0, value.dim() - 1)))
                for key, value in batch.items()
            }
            model.train()
            if post_process_model is not None:
                post_process_model.train()
        else:
            model.eval()
            if post_process_model is not None:
                post_process_model.eval()

        autocast_context = (
            torch.autocast(
                device_type="cuda",
                dtype=(
                    torch.bfloat16
                    if hasattr(torch.cuda, "is_bf16_supported") and torch.cuda.is_bf16_supported()
                    else torch.float16
                ),
            )
            if search_batch["guide_mel"].device.type == "cuda"
            else nullcontext()
        )
        with torch.no_grad(), autocast_context:
            outputs = self._predict_outputs(
                model,
                guide_mel=search_batch["guide_mel"],
                phone_ids=search_batch["phone_ids"],
                log_f0=search_batch["log_f0"],
                vuv=search_batch["vuv"],
                energy=search_batch["energy"],
                lengths=search_batch["lengths"],
                content=search_batch.get("content"),
                beat_phase=search_batch.get("beat_phase"),
                voice_prototype=search_batch["voice_prototype"],
                lyric_mask=search_batch["lyric_mask"],
            )
            refined_mel = outputs["mel"]
            if post_process_model is not None:
                refined_mel = post_process_model(
                    coarse_mel=outputs["mel"],
                    guide_mel=search_batch["guide_mel"],
                    phone_ids=search_batch["phone_ids"],
                    log_f0=search_batch["log_f0"],
                    vuv=search_batch["vuv"],
                    energy=search_batch["energy"],
                    lengths=search_batch["lengths"],
                    voice_prototype=search_batch["voice_prototype"],
                    lyric_mask=search_batch["lyric_mask"],
                )
            outputs = {
                **outputs,
                "mel": refined_mel,
            }

        if was_training:
            model.train()
        else:
            model.eval()
        if post_process_model is not None:
            if post_process_was_training:
                post_process_model.train()
            else:
                post_process_model.eval()

        phone_accuracy, lyric_phone_accuracy = self._phone_accuracy_per_sample(
            outputs["phone_logits"],
            search_batch["phone_ids"],
            search_batch["lengths"],
            search_batch["lyric_mask"],
        )
        vuv_accuracy = self._vuv_accuracy_per_sample(
            outputs["target_vuv_logits"],
            search_batch["target_vuv"] if "target_vuv" in search_batch else search_batch["vuv"],
            search_batch["lengths"],
        )
        predicted_voice_signature = _compute_voice_signature_torch(
            mel=outputs["mel"],
            log_f0=outputs["target_log_f0"],
            vuv=torch.sigmoid(outputs["target_vuv_logits"]),
            lengths=search_batch["lengths"],
        )
        voice_similarity = F.cosine_similarity(
            predicted_voice_signature,
            search_batch["voice_prototype"].to(
                device=predicted_voice_signature.device,
                dtype=predicted_voice_signature.dtype,
            ),
            dim=-1,
        )
        candidate_score = (
            (0.52 * voice_similarity)
            + (0.24 * lyric_phone_accuracy)
            + (0.14 * phone_accuracy)
            + (0.10 * vuv_accuracy)
        )
        best_index = int(torch.argmax(candidate_score).detach().cpu().item())
        best_length = int(search_batch["lengths"][best_index].detach().cpu().item())
        return (
            outputs["mel"][best_index, :best_length].detach().float().cpu().numpy().astype(np.float32, copy=False),
            {
                "candidate_count": int(candidate_count),
                "best_index": int(best_index),
                "best_score": float(candidate_score[best_index].detach().cpu().item()),
                "best_voice_similarity": float(voice_similarity[best_index].detach().cpu().item()),
                "best_lyric_phone_accuracy": float(lyric_phone_accuracy[best_index].detach().cpu().item()),
                "best_phone_accuracy": float(phone_accuracy[best_index].detach().cpu().item()),
                "best_vuv_accuracy": float(vuv_accuracy[best_index].detach().cpu().item()),
                "post_process_applied": bool(post_process_model is not None),
                "search_mode": "mc-dropout-target-voice-rerank" if candidate_count > 1 else "deterministic-target-voice",
            },
        )

    def _normalized_silence_frame(self, stats: Dict[str, object]) -> np.ndarray:
        mel_mean = np.asarray(stats.get("mel_mean", [0.0] * N_MELS), dtype=np.float32)
        mel_std = np.asarray(stats.get("mel_std", [1.0] * N_MELS), dtype=np.float32)
        silence_log_mel = np.full((N_MELS,), -11.25, dtype=np.float32)
        return ((silence_log_mel - mel_mean) / np.maximum(mel_std, 1e-5)).astype(np.float32, copy=False)

    def _build_dense_repair_windows(
        self,
        lyric_mask: np.ndarray,
        *,
        micro_window_frames: int,
        micro_window_hop: int,
        micro_window_pad: int,
        micro_window_max: int,
    ) -> List[Dict[str, int]]:
        binary_mask = np.asarray(lyric_mask > 0.5, dtype=np.bool_)
        frame_count = int(binary_mask.shape[0])
        if frame_count <= 0 or not np.any(binary_mask):
            return []
        windows: List[Dict[str, int]] = []
        seen: set[tuple[int, int, int, int]] = set()
        cursor = 0
        while cursor < frame_count:
            if not binary_mask[cursor]:
                cursor += 1
                continue
            span_start = cursor
            while cursor < frame_count and binary_mask[cursor]:
                cursor += 1
            span_end = cursor
            if span_end <= span_start:
                continue
            local_core = min(micro_window_frames, max(4, span_end - span_start))
            first_core = max(0, span_start - micro_window_pad)
            last_core = max(first_core, span_end - local_core)
            core_starts = list(range(first_core, last_core + 1, max(1, micro_window_hop)))
            if not core_starts or core_starts[-1] != last_core:
                core_starts.append(last_core)
            for core_start in core_starts:
                core_end = min(frame_count, core_start + local_core)
                local_start = max(0, core_start - micro_window_pad)
                local_end = min(frame_count, core_end + micro_window_pad)
                key = (local_start, local_end, max(span_start, core_start), min(span_end, core_end))
                if local_end - local_start < 4 or key in seen:
                    continue
                seen.add(key)
                windows.append(
                    {
                        "start": int(local_start),
                        "end": int(local_end),
                        "core_start": int(key[2]),
                        "core_end": int(key[3]),
                    }
                )
                if len(windows) >= micro_window_max:
                    return windows
        return windows

    def _apply_dense_pronunciation_repairs(
        self,
        *,
        bundle: Dict[str, object],
        normalized_prediction: np.ndarray,
        guide_mel: np.ndarray,
        phone_ids: np.ndarray,
        log_f0: np.ndarray,
        vuv: np.ndarray,
        energy: np.ndarray,
        voice_prototype: torch.Tensor,
        candidate_count: int,
    ) -> Tuple[np.ndarray, Dict[str, object]]:
        config = dict(bundle.get("config", {}))
        strategy = self._get_repair_strategy(config)
        lyric_mask = (np.asarray(phone_ids, dtype=np.int64) != PHONE_TO_ID["SP"]).astype(np.float32, copy=False)
        if not bool(strategy.get("enabled", False)):
            return normalized_prediction.astype(np.float32, copy=False), {"repair_mode": "disabled", "repair_window_count": 0}
        windows = self._build_dense_repair_windows(
            lyric_mask,
            micro_window_frames=int(strategy["micro_window_frames"]),
            micro_window_hop=int(strategy["micro_window_hop"]),
            micro_window_pad=int(strategy["micro_window_pad"]),
            micro_window_max=int(strategy["micro_window_max"]),
        )
        repaired_prediction = normalized_prediction.astype(np.float32, copy=True)
        if not windows:
            silence_frame = self._normalized_silence_frame(dict(bundle.get("stats", {})))
            off_lyric_mask = lyric_mask < 0.5
            if np.any(off_lyric_mask):
                off_floor = float(strategy["off_lyric_floor"])
                repaired_prediction[off_lyric_mask] = (
                    (off_floor * repaired_prediction[off_lyric_mask])
                    + ((1.0 - off_floor) * silence_frame[np.newaxis, :])
                )
            return repaired_prediction, {"repair_mode": "dense-micro-windows", "repair_window_count": 0}

        device = bundle["device"]
        model = bundle["model"]
        post_process_model = bundle.get("post_process") if isinstance(bundle.get("post_process"), nn.Module) else None
        accum_mel = repaired_prediction.copy()
        accum_weight = np.ones((repaired_prediction.shape[0], 1), dtype=np.float32)
        repair_scores: List[float] = []
        repair_voice_similarity: List[float] = []
        repair_candidate_count = max(1, min(int(candidate_count), int(strategy["micro_candidate_count"])))
        dense_blend = float(strategy["dense_blend"])
        guide_anchor_blend = float(strategy["guide_anchor_blend"])

        for window in windows:
            start = int(window["start"])
            end = int(window["end"])
            core_start = int(window["core_start"])
            core_end = int(window["core_end"])
            if end - start < 4 or core_end <= core_start:
                continue
            local_phone_ids = phone_ids[start:end]
            local_mask = lyric_mask[start:end]
            batch = {
                "guide_mel": torch.from_numpy(guide_mel[start:end].astype(np.float32, copy=False)).unsqueeze(0).to(device),
                "phone_ids": torch.from_numpy(local_phone_ids.astype(np.int64, copy=False)).unsqueeze(0).to(device),
                "log_f0": torch.from_numpy(log_f0[start:end].astype(np.float32, copy=False)).unsqueeze(0).to(device),
                "vuv": torch.from_numpy(vuv[start:end].astype(np.float32, copy=False)).unsqueeze(0).to(device),
                "energy": torch.from_numpy(energy[start:end].astype(np.float32, copy=False)).unsqueeze(0).to(device),
                "voice_prototype": voice_prototype,
                "lyric_mask": torch.from_numpy(local_mask.astype(np.float32, copy=False)).unsqueeze(0).to(device),
                "lengths": torch.tensor([int(end - start)], dtype=torch.long, device=device),
            }
            repaired_chunk, candidate_metadata = self._run_inference_candidate_search(
                model=model,
                batch=batch,
                candidate_count=repair_candidate_count,
                post_process_model=post_process_model,
            )
            if guide_anchor_blend > 0.0:
                repaired_chunk = (
                    ((1.0 - guide_anchor_blend) * repaired_chunk)
                    + (guide_anchor_blend * guide_mel[start:end])
                ).astype(np.float32, copy=False)
            local_weights = np.zeros((end - start, 1), dtype=np.float32)
            core_local_start = max(0, core_start - start)
            core_local_end = min(end - start, core_end - start)
            local_weights[core_local_start:core_local_end] = 1.0
            fade = min(int(strategy["micro_window_pad"]), core_local_start)
            if fade > 0:
                fade_in = np.linspace(0.0, 1.0, num=fade + 1, dtype=np.float32)[1:].reshape(-1, 1)
                local_weights[core_local_start - fade:core_local_start] = np.maximum(
                    local_weights[core_local_start - fade:core_local_start],
                    fade_in,
                )
            fade = min(int(strategy["micro_window_pad"]), max(0, (end - start) - core_local_end))
            if fade > 0:
                fade_out = np.linspace(1.0, 0.0, num=fade + 1, dtype=np.float32)[:-1].reshape(-1, 1)
                local_weights[core_local_end:core_local_end + fade] = np.maximum(
                    local_weights[core_local_end:core_local_end + fade],
                    fade_out,
                )
            local_weights *= local_mask.reshape(-1, 1).astype(np.float32, copy=False)
            if float(local_weights.sum()) <= 1e-6:
                continue
            blended_weight = dense_blend * local_weights
            accum_mel[start:end] += repaired_chunk * blended_weight
            accum_weight[start:end] += blended_weight
            repair_scores.append(float(candidate_metadata.get("best_score", 0.0)))
            repair_voice_similarity.append(float(candidate_metadata.get("best_voice_similarity", 0.0)))

        repaired_prediction = accum_mel / np.maximum(accum_weight, 1e-6)
        silence_frame = self._normalized_silence_frame(dict(bundle.get("stats", {})))
        off_lyric_mask = lyric_mask < 0.5
        if np.any(off_lyric_mask):
            off_floor = float(strategy["off_lyric_floor"])
            repaired_prediction[off_lyric_mask] = (
                (off_floor * repaired_prediction[off_lyric_mask])
                + ((1.0 - off_floor) * silence_frame[np.newaxis, :])
            )
        return repaired_prediction.astype(np.float32, copy=False), {
            "repair_mode": "dense-micro-windows",
            "repair_window_count": int(len(windows)),
            "repair_candidate_count": int(repair_candidate_count),
            "repair_mean_score": float(np.mean(repair_scores)) if repair_scores else 0.0,
            "repair_mean_voice_similarity": float(np.mean(repair_voice_similarity)) if repair_voice_similarity else 0.0,
        }

    def _apply_lyric_wave_gate(
        self,
        audio: np.ndarray,
        lyric_mask: np.ndarray,
        *,
        off_floor: float,
        smoothing_frames: int,
    ) -> np.ndarray:
        waveform = np.asarray(audio, dtype=np.float32).reshape(-1)
        if waveform.size <= 0:
            return waveform
        frame_gate = np.asarray(lyric_mask, dtype=np.float32).reshape(-1)
        if frame_gate.size <= 0:
            return waveform
        if smoothing_frames > 0:
            kernel_size = max(1, (2 * int(smoothing_frames)) + 1)
            kernel = np.ones((kernel_size,), dtype=np.float32) / float(kernel_size)
            frame_gate = np.convolve(frame_gate, kernel, mode="same")
        frame_gate = np.clip(frame_gate, 0.0, 1.0)
        frame_positions = np.arange(frame_gate.shape[0], dtype=np.float32) * float(HOP_LENGTH)
        sample_positions = np.arange(waveform.shape[0], dtype=np.float32)
        if frame_positions.size == 1:
            sample_gate = np.full((waveform.shape[0],), frame_gate[0], dtype=np.float32)
        else:
            sample_gate = np.interp(sample_positions, frame_positions, frame_gate, left=frame_gate[0], right=frame_gate[-1]).astype(np.float32, copy=False)
        sample_gate = float(off_floor) + ((1.0 - float(off_floor)) * sample_gate)
        return (waveform * sample_gate).astype(np.float32, copy=False)

    def _render_preview(
        self,
        *,
        checkpoint_path: Path,
        dataset_dir: Path,
        stats: Dict[str, object],
        sample_entry: Dict[str, object],
        output_dir: Path,
    ) -> Dict[str, object]:
        bundle = self._load_inference_bundle(checkpoint_path=checkpoint_path)
        device = bundle["device"]
        model = bundle["model"]
        post_process_model = bundle.get("post_process")
        dataset = GuidedSVSDataset(
            entries=[sample_entry],
            dataset_dir=dataset_dir,
            stats=stats,
            max_frames=max(512, int(sample_entry.get("n_frames", 512))),
            random_crop=False,
        )
        batch = collate_guided_svs([dataset[0]])
        batch = {key: value.to(device) for key, value in batch.items()}
        prediction, candidate_metadata = self._run_inference_candidate_search(
            model=model,
            batch=batch,
            candidate_count=1,
            post_process_model=post_process_model if isinstance(post_process_model, nn.Module) else None,
        )
        predicted_audio, render_mode = self._render_audio_from_bundle(
            normalized_mel=prediction,
            bundle=bundle,
        )
        target_audio, _ = self._render_audio_from_bundle(
            normalized_mel=batch["mel"][0, : int(batch["lengths"][0].item())].detach().cpu().numpy(),
            bundle=bundle,
        )
        preview_path = output_dir / "guided_regeneration_preview.wav"
        target_preview_path = output_dir / "guided_regeneration_target_preview.wav"
        sf.write(preview_path, predicted_audio, SAMPLE_RATE, subtype="PCM_24")
        sf.write(target_preview_path, target_audio, SAMPLE_RATE, subtype="PCM_24")
        return {
            "preview_path": str(preview_path),
            "target_preview_path": str(target_preview_path),
            "preview_sample_id": str(sample_entry.get("id", "")),
            "preview_render_mode": render_mode,
            "preview_post_process_applied": bool(candidate_metadata.get("post_process_applied", False)),
        }

    def synthesize_phrase_from_blueprint(
        self,
        *,
        checkpoint_path: Path,
        guide_audio_path: Path,
        phrase_text: str,
        output_path: Path,
        config_path: Path | None = None,
        manifest_path: Path | None = None,
        training_report_path: Path | None = None,
        phrase_word_scores: Optional[Sequence[Dict[str, object]]] = None,
    ) -> Dict[str, object]:
        cleaned_phrase = normalize_lyrics(phrase_text)
        if not cleaned_phrase:
            raise RuntimeError("Blueprint synthesis needs non-empty phrase lyrics.")

        bundle = self._load_inference_bundle(
            checkpoint_path=checkpoint_path,
            config_path=config_path,
            manifest_path=manifest_path,
            training_report_path=training_report_path,
        )
        guide_audio, _ = librosa.load(str(guide_audio_path), sr=SAMPLE_RATE, mono=True)
        guide_audio = np.asarray(guide_audio, dtype=np.float32).reshape(-1)
        if guide_audio.size < max(HOP_LENGTH, 64):
            raise RuntimeError("Guide phrase is too short for pronunciation blueprint synthesis.")

        guide_log_mel = self._extract_log_mel(guide_audio, SAMPLE_RATE)
        frame_count = int(guide_log_mel.shape[0])
        if frame_count <= 3:
            raise RuntimeError("Guide phrase did not produce enough frames for regeneration.")

        duration_seconds = float(guide_audio.shape[0] / float(SAMPLE_RATE))
        normalized_word_scores = self._normalize_phrase_word_scores(
            phrase_text=cleaned_phrase,
            phrase_word_scores=phrase_word_scores,
            duration_seconds=duration_seconds,
        )
        phone_ids, coverage = self._build_phone_ids(
            lyrics=cleaned_phrase,
            word_scores=normalized_word_scores,
            frame_count=frame_count,
            sample_rate=SAMPLE_RATE,
        )
        f0 = self._extract_f0(guide_audio, SAMPLE_RATE, frame_count)
        energy = librosa.feature.rms(
            y=guide_audio,
            frame_length=N_FFT,
            hop_length=HOP_LENGTH,
            center=True,
        ).squeeze()
        energy = _align_1d(energy, frame_count)
        log_f0 = np.where(f0 > 0.0, np.log(np.maximum(f0, 1.0)), 0.0).astype(np.float32)
        vuv = (f0 > 0.0).astype(np.float32)

        stats = dict(bundle["stats"])
        mel_mean = np.asarray(stats.get("mel_mean", [0.0] * N_MELS), dtype=np.float32)
        mel_std = np.asarray(stats.get("mel_std", [1.0] * N_MELS), dtype=np.float32)
        norm_log_f0 = np.zeros_like(log_f0, dtype=np.float32)
        voiced = vuv > 0.5
        log_f0_std = float(_safe_std(float(stats.get("log_f0_std", 1.0))))
        if np.any(voiced):
            norm_log_f0[voiced] = (
                log_f0[voiced] - float(stats.get("log_f0_mean", 0.0))
            ) / log_f0_std
        norm_guide_mel = (guide_log_mel - mel_mean[np.newaxis, :]) / mel_std[np.newaxis, :]
        norm_energy = (
            energy - float(stats.get("energy_mean", 0.0))
        ) / float(_safe_std(float(stats.get("energy_std", 1.0))))

        device = bundle["device"]
        model = bundle["model"]
        candidate_count = self._get_inference_candidate_count(device)
        voice_prototype = torch.from_numpy(
            _ensure_voice_signature_dim(bundle.get("voice_prototype", []))
        ).unsqueeze(0).to(device=device, dtype=torch.float32)
        batch = {
            "guide_mel": torch.from_numpy(norm_guide_mel.astype(np.float32, copy=False)).unsqueeze(0).to(device),
            "phone_ids": torch.from_numpy(phone_ids.astype(np.int64, copy=False)).unsqueeze(0).to(device),
            "log_f0": torch.from_numpy(norm_log_f0.astype(np.float32, copy=False)).unsqueeze(0).to(device),
            "vuv": torch.from_numpy(vuv.astype(np.float32, copy=False)).unsqueeze(0).to(device),
            "energy": torch.from_numpy(norm_energy.astype(np.float32, copy=False)).unsqueeze(0).to(device),
            "voice_prototype": voice_prototype,
            "lyric_mask": (
                torch.from_numpy((phone_ids != PHONE_TO_ID["SP"]).astype(np.float32, copy=False))
                .unsqueeze(0)
                .to(device)
            ),
            "lengths": torch.tensor([frame_count], dtype=torch.long, device=device),
        }
        candidate_prediction, candidate_metadata = self._run_inference_candidate_search(
            model=model,
            batch=batch,
            candidate_count=candidate_count,
            post_process_model=bundle.get("post_process") if isinstance(bundle.get("post_process"), nn.Module) else None,
        )
        repair_metadata: Dict[str, object] = {}
        if self._is_lyric_repair_recipe(dict(bundle.get("config", {}))):
            candidate_prediction, repair_metadata = self._apply_dense_pronunciation_repairs(
                bundle=bundle,
                normalized_prediction=candidate_prediction,
                guide_mel=norm_guide_mel,
                phone_ids=phone_ids,
                log_f0=norm_log_f0,
                vuv=vuv,
                energy=norm_energy,
                voice_prototype=voice_prototype,
                candidate_count=candidate_count,
            )

        generated_audio, render_mode = self._render_audio_from_bundle(
            normalized_mel=candidate_prediction,
            bundle=bundle,
        )
        if repair_metadata:
            repair_strategy = self._get_repair_strategy(dict(bundle.get("config", {})))
            generated_audio = self._apply_lyric_wave_gate(
                generated_audio,
                (phone_ids != PHONE_TO_ID["SP"]).astype(np.float32, copy=False),
                off_floor=float(repair_strategy["off_lyric_floor"]),
                smoothing_frames=int(repair_strategy["wave_gate_smoothing_frames"]),
            )
        source_rms = float(np.sqrt(np.mean(np.square(guide_audio)) + 1e-9))
        generated_rms = float(np.sqrt(np.mean(np.square(generated_audio)) + 1e-9))
        if generated_rms > 1e-7:
            generated_audio = generated_audio * np.float32(np.clip(source_rms / generated_rms, 0.78, 1.32))
        stereo_audio = np.repeat(generated_audio[:, np.newaxis], 2, axis=1).astype(np.float32, copy=False)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        sf.write(output_path, stereo_audio, SAMPLE_RATE, subtype="PCM_24")
        return {
            "output_path": str(output_path),
            "sample_rate": SAMPLE_RATE,
            "frame_count": frame_count,
            "duration_seconds": round(float(generated_audio.shape[0] / float(SAMPLE_RATE)), 3),
            "phone_coverage": round(float(coverage), 4),
            "stats_path": str(bundle["stats_path"]),
            "checkpoint_path": str(bundle["checkpoint_path"]),
            "mode": "blueprint-regeneration-v1",
            "render_mode": render_mode,
            **candidate_metadata,
            **repair_metadata,
        }

    def synthesize_full_song_from_blueprint(
        self,
        *,
        checkpoint_path: Path,
        guide_audio_path: Path,
        lyrics: str,
        output_path: Path,
        config_path: Path | None = None,
        manifest_path: Path | None = None,
        training_report_path: Path | None = None,
        phrase_word_scores: Optional[Sequence[Dict[str, object]]] = None,
    ) -> Dict[str, object]:
        cleaned_lyrics = normalize_lyrics(lyrics)
        if not cleaned_lyrics:
            raise RuntimeError("Full blueprint synthesis needs pasted lyrics.")

        bundle = self._load_inference_bundle(
            checkpoint_path=checkpoint_path,
            config_path=config_path,
            manifest_path=manifest_path,
            training_report_path=training_report_path,
        )
        guide_audio, _ = librosa.load(str(guide_audio_path), sr=SAMPLE_RATE, mono=True)
        guide_audio = np.asarray(guide_audio, dtype=np.float32).reshape(-1)
        if guide_audio.size < max(HOP_LENGTH * 8, 512):
            raise RuntimeError("Guide vocal is too short for full blueprint conversion.")

        started = time.perf_counter()
        guide_log_mel = self._extract_log_mel(guide_audio, SAMPLE_RATE)
        frame_count = int(guide_log_mel.shape[0])
        if frame_count <= 8:
            raise RuntimeError("Guide vocal did not produce enough frames for full conversion.")

        duration_seconds = float(guide_audio.shape[0] / float(SAMPLE_RATE))
        normalized_word_scores = self._normalize_phrase_word_scores(
            phrase_text=cleaned_lyrics,
            phrase_word_scores=phrase_word_scores,
            duration_seconds=duration_seconds,
        )
        phone_ids, coverage = self._build_phone_ids(
            lyrics=cleaned_lyrics,
            word_scores=normalized_word_scores,
            frame_count=frame_count,
            sample_rate=SAMPLE_RATE,
        )
        f0 = self._extract_f0(guide_audio, SAMPLE_RATE, frame_count)
        energy = librosa.feature.rms(
            y=guide_audio,
            frame_length=N_FFT,
            hop_length=HOP_LENGTH,
            center=True,
        ).squeeze()
        energy = _align_1d(energy, frame_count)
        log_f0 = np.where(f0 > 0.0, np.log(np.maximum(f0, 1.0)), 0.0).astype(np.float32)
        vuv = (f0 > 0.0).astype(np.float32)

        stats = dict(bundle["stats"])
        mel_mean = np.asarray(stats.get("mel_mean", [0.0] * N_MELS), dtype=np.float32)
        mel_std = np.asarray(stats.get("mel_std", [1.0] * N_MELS), dtype=np.float32)
        norm_log_f0 = np.zeros_like(log_f0, dtype=np.float32)
        voiced = vuv > 0.5
        log_f0_std = float(_safe_std(float(stats.get("log_f0_std", 1.0))))
        if np.any(voiced):
            norm_log_f0[voiced] = (
                log_f0[voiced] - float(stats.get("log_f0_mean", 0.0))
            ) / log_f0_std
        norm_guide_mel = (guide_log_mel - mel_mean[np.newaxis, :]) / mel_std[np.newaxis, :]
        norm_energy = (
            energy - float(stats.get("energy_mean", 0.0))
        ) / float(_safe_std(float(stats.get("energy_std", 1.0))))

        device = bundle["device"]
        model = bundle["model"]
        config = dict(bundle["config"])
        candidate_count = self._get_inference_candidate_count(device)
        voice_prototype = torch.from_numpy(
            _ensure_voice_signature_dim(bundle.get("voice_prototype", []))
        ).unsqueeze(0).to(device=device, dtype=torch.float32)
        chunk_frames = max(120, int(config.get("max_frames", 240) or 240))
        overlap_frames = min(max(24, chunk_frames // 6), max(24, chunk_frames - 16))
        if frame_count <= chunk_frames:
            overlap_frames = 0
        step_frames = max(1, chunk_frames - overlap_frames)

        predicted_sum = np.zeros((frame_count, N_MELS), dtype=np.float32)
        weight_sum = np.zeros((frame_count, 1), dtype=np.float32)
        chunk_scores: List[float] = []
        chunk_voice_similarity: List[float] = []
        starts = list(range(0, max(frame_count - chunk_frames, 0) + 1, step_frames))
        if not starts or starts[-1] != max(0, frame_count - chunk_frames):
            starts.append(max(0, frame_count - chunk_frames))
        starts = sorted(set(starts))

        for start_frame in starts:
            end_frame = min(frame_count, start_frame + chunk_frames)
            local_phone_ids = phone_ids[start_frame:end_frame]
            local_log_f0 = norm_log_f0[start_frame:end_frame]
            local_vuv = vuv[start_frame:end_frame]
            local_energy = norm_energy[start_frame:end_frame]
            local_length = int(end_frame - start_frame)
            batch = {
                "guide_mel": torch.from_numpy(
                    norm_guide_mel[start_frame:end_frame].astype(np.float32, copy=False)
                ).unsqueeze(0).to(device),
                "phone_ids": torch.from_numpy(local_phone_ids.astype(np.int64, copy=False)).unsqueeze(0).to(device),
                "log_f0": torch.from_numpy(local_log_f0.astype(np.float32, copy=False)).unsqueeze(0).to(device),
                "vuv": torch.from_numpy(local_vuv.astype(np.float32, copy=False)).unsqueeze(0).to(device),
                "energy": torch.from_numpy(local_energy.astype(np.float32, copy=False)).unsqueeze(0).to(device),
                "voice_prototype": voice_prototype,
                "lyric_mask": torch.from_numpy(
                    (local_phone_ids != PHONE_TO_ID["SP"]).astype(np.float32, copy=False)
                ).unsqueeze(0).to(device),
                "lengths": torch.tensor([local_length], dtype=torch.long, device=device),
            }
            predicted_chunk, candidate_metadata = self._run_inference_candidate_search(
                model=model,
                batch=batch,
                candidate_count=candidate_count,
                post_process_model=bundle.get("post_process") if isinstance(bundle.get("post_process"), nn.Module) else None,
            )
            chunk_scores.append(float(candidate_metadata.get("best_score", 0.0)))
            chunk_voice_similarity.append(float(candidate_metadata.get("best_voice_similarity", 0.0)))

            weights = np.ones((local_length, 1), dtype=np.float32)
            if overlap_frames > 1 and local_length > 2:
                fade_length = min(overlap_frames, max(2, local_length // 3))
                fade_in = np.linspace(0.0, 1.0, num=fade_length, dtype=np.float32).reshape(-1, 1)
                fade_out = np.linspace(1.0, 0.0, num=fade_length, dtype=np.float32).reshape(-1, 1)
                if start_frame > 0:
                    weights[:fade_length] *= fade_in
                if end_frame < frame_count:
                    weights[-fade_length:] *= fade_out

            predicted_sum[start_frame:end_frame] += predicted_chunk * weights
            weight_sum[start_frame:end_frame] += weights

        normalized_prediction = predicted_sum / np.maximum(weight_sum, 1e-6)
        repair_metadata: Dict[str, object] = {}
        if self._is_lyric_repair_recipe(dict(bundle.get("config", {}))):
            normalized_prediction, repair_metadata = self._apply_dense_pronunciation_repairs(
                bundle=bundle,
                normalized_prediction=normalized_prediction.astype(np.float32, copy=False),
                guide_mel=norm_guide_mel,
                phone_ids=phone_ids,
                log_f0=norm_log_f0,
                vuv=vuv,
                energy=norm_energy,
                voice_prototype=voice_prototype,
                candidate_count=candidate_count,
            )
        generated_audio, render_mode = self._render_audio_from_bundle(
            normalized_mel=normalized_prediction.astype(np.float32, copy=False),
            bundle=bundle,
        )
        if repair_metadata:
            repair_strategy = self._get_repair_strategy(dict(bundle.get("config", {})))
            generated_audio = self._apply_lyric_wave_gate(
                generated_audio,
                (phone_ids != PHONE_TO_ID["SP"]).astype(np.float32, copy=False),
                off_floor=float(repair_strategy["off_lyric_floor"]),
                smoothing_frames=int(repair_strategy["wave_gate_smoothing_frames"]),
            )

        guide_rms = float(np.sqrt(np.mean(np.square(guide_audio)) + 1e-9))
        generated_rms = float(np.sqrt(np.mean(np.square(generated_audio)) + 1e-9))
        if generated_rms > 1e-7:
            generated_audio = generated_audio * np.float32(np.clip(guide_rms / generated_rms, 0.76, 1.28))

        stereo_audio = np.repeat(generated_audio[:, np.newaxis], 2, axis=1).astype(np.float32, copy=False)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        sf.write(output_path, stereo_audio, SAMPLE_RATE, subtype="PCM_24")
        elapsed = float(time.perf_counter() - started)
        return {
            "output_path": str(output_path),
            "sample_rate": SAMPLE_RATE,
            "frame_count": frame_count,
            "duration_seconds": round(float(generated_audio.shape[0] / float(SAMPLE_RATE)), 3),
            "phone_coverage": round(float(coverage), 4),
            "stats_path": str(bundle["stats_path"]),
            "checkpoint_path": str(bundle["checkpoint_path"]),
            "mode": "blueprint-full-conversion-v1",
            "chunk_frames": int(chunk_frames),
            "overlap_frames": int(overlap_frames),
            "synthesis_seconds": round(elapsed, 3),
            "candidate_count": int(candidate_count),
            "mean_chunk_score": round(float(np.mean(chunk_scores)) if chunk_scores else 0.0, 4),
            "mean_chunk_voice_similarity": round(float(np.mean(chunk_voice_similarity)) if chunk_voice_similarity else 0.0, 4),
            "search_mode": "mc-dropout-target-voice-rerank" if candidate_count > 1 else "deterministic-target-voice",
            "render_mode": render_mode,
            **repair_metadata,
        }

    def synthesize_direct_guide_v11(
        self,
        *,
        checkpoint_path: Path,
        guide_audio_path: Path,
        output_path: Path,
        config_path: Path | None = None,
        manifest_path: Path | None = None,
        training_report_path: Path | None = None,
    ) -> Dict[str, object]:
        bundle = self._load_inference_bundle(
            checkpoint_path=checkpoint_path,
            config_path=config_path,
            manifest_path=manifest_path,
            training_report_path=training_report_path,
        )
        config = dict(bundle.get("config", {}))
        model = bundle.get("model")
        base_model = getattr(model, "_orig_mod", model)
        training_mode = str(config.get("training_mode", "") or "").strip().lower()
        recipe_mode = str(config.get("recipe_mode", "") or "").strip().lower()
        legacy_direct_recipe_modes = {
            "persona-aligned-pth",
            "persona-aligned-pth-prefix-pairs",
        }
        is_direct_mapper_checkpoint = (
            training_mode in DIRECT_GUIDED_TRAINING_MODES
            or recipe_mode in DIRECT_GUIDED_RECIPE_MODES
            or isinstance(model, PersonaMapperNAR)
            or (
                recipe_mode in legacy_direct_recipe_modes
                and isinstance(base_model, GuideConditionedMelRegenerator)
            )
            or self._is_persona_v11_recipe(config)
        )
        if not is_direct_mapper_checkpoint:
            raise RuntimeError("Direct paired guide synthesis requires a direct-mapper checkpoint.")

        guide_audio, _ = librosa.load(str(guide_audio_path), sr=SAMPLE_RATE, mono=True)
        guide_audio = np.asarray(guide_audio, dtype=np.float32).reshape(-1)
        if guide_audio.size < max(HOP_LENGTH * 8, 512):
            raise RuntimeError("Guide vocal is too short for direct paired conversion.")

        started = time.perf_counter()
        guide_log_mel = self._extract_log_mel(guide_audio, SAMPLE_RATE)
        frame_count = int(guide_log_mel.shape[0])
        if frame_count <= 8:
            raise RuntimeError("Guide vocal did not produce enough frames for direct paired conversion.")

        f0 = self._extract_f0(guide_audio, SAMPLE_RATE, frame_count)
        energy = librosa.feature.rms(
            y=guide_audio,
            frame_length=N_FFT,
            hop_length=HOP_LENGTH,
            center=True,
        ).squeeze()
        energy = _align_1d(energy, frame_count)
        log_f0 = np.where(f0 > 0.0, np.log(np.maximum(f0, 1.0)), 0.0).astype(np.float32)
        vuv = (f0 > 0.0).astype(np.float32)
        content = self._extract_content_features(guide_audio, SAMPLE_RATE, frame_count)
        beat_phase = self._compute_beat_phase(guide_audio, SAMPLE_RATE, frame_count)

        stats = dict(bundle["stats"])
        mel_mean = np.asarray(stats.get("mel_mean", [0.0] * N_MELS), dtype=np.float32)
        mel_std = np.asarray(stats.get("mel_std", [1.0] * N_MELS), dtype=np.float32)
        norm_guide_mel = (guide_log_mel - mel_mean[np.newaxis, :]) / np.maximum(mel_std[np.newaxis, :], 1e-5)
        norm_log_f0 = np.zeros_like(log_f0, dtype=np.float32)
        voiced = vuv > 0.5
        log_f0_std = float(_safe_std(float(stats.get("log_f0_std", 1.0))))
        if np.any(voiced):
            norm_log_f0[voiced] = (
                log_f0[voiced] - float(stats.get("log_f0_mean", 0.0))
            ) / log_f0_std
        norm_energy = (
            energy - float(stats.get("energy_mean", 0.0))
        ) / float(_safe_std(float(stats.get("energy_std", 1.0))))
        content_dim = int(content.shape[1]) if content.ndim == 2 else PERSONA_V11_CONTENT_DIM
        content_mean = np.asarray(
            stats.get("nar_content_mean", [0.0] * content_dim),
            dtype=np.float32,
        )
        content_std = np.asarray(
            stats.get("nar_content_std", [1.0] * content_dim),
            dtype=np.float32,
        )
        if content_mean.shape[0] != content_dim:
            resized_mean = np.zeros(content_dim, dtype=np.float32)
            resized_std = np.ones(content_dim, dtype=np.float32)
            copy_dim = min(int(content_mean.shape[0]), int(content_dim))
            if copy_dim > 0:
                resized_mean[:copy_dim] = content_mean[:copy_dim]
                resized_std[:copy_dim] = content_std[:copy_dim]
            content_mean = resized_mean
            content_std = resized_std
        norm_content = (content - content_mean[np.newaxis, :]) / np.maximum(content_std[np.newaxis, :], 1e-4)

        device = bundle["device"]
        model = bundle["model"]
        model_config = dict(config.get("model", {}) or {})
        use_voice_prototype_conditioning = bool(model_config.get("use_voice_prototype_conditioning", True))
        aligned_search_recipe = recipe_mode in {
            "aligned-suno",
            "persona-aligned-pth",
            "persona-aligned-pth-prefix-pairs",
            "concert-remaster-paired",
        }
        voice_prototype_np = (
            _ensure_voice_signature_dim(bundle.get("voice_prototype", []))
            if use_voice_prototype_conditioning
            else np.zeros(VOICE_SIGNATURE_DIM, dtype=np.float32)
        )
        voice_prototype = torch.from_numpy(voice_prototype_np).unsqueeze(0).to(device=device, dtype=torch.float32)
        candidate_count = self._get_inference_candidate_count(device) if aligned_search_recipe else 1
        chunk_frames = max(160, int(config.get("max_frames", 320) or 320))
        overlap_frames = min(max(24, chunk_frames // 8), max(24, chunk_frames - 16))
        if frame_count <= chunk_frames:
            overlap_frames = 0
        step_frames = max(1, chunk_frames - overlap_frames)

        predicted_sum = np.zeros((frame_count, N_MELS), dtype=np.float32)
        weight_sum = np.zeros((frame_count, 1), dtype=np.float32)
        chunk_scores: List[float] = []
        chunk_voice_similarity: List[float] = []
        starts = list(range(0, max(frame_count - chunk_frames, 0) + 1, step_frames))
        if not starts or starts[-1] != max(0, frame_count - chunk_frames):
            starts.append(max(0, frame_count - chunk_frames))
        starts = sorted(set(starts))
        phone_ids, lyric_mask = self._build_proxy_activity_tokens(
            primary_f0=f0,
            primary_energy=energy,
        )

        autocast_context = (
            torch.autocast(
                device_type="cuda",
                dtype=(
                    torch.bfloat16
                    if hasattr(torch.cuda, "is_bf16_supported") and torch.cuda.is_bf16_supported()
                    else torch.float16
                ),
            )
            if device.type == "cuda"
            else nullcontext()
        )

        with torch.no_grad(), autocast_context:
            for start_frame in starts:
                end_frame = min(frame_count, start_frame + chunk_frames)
                local_length = int(end_frame - start_frame)
                batch = {
                    "guide_mel": torch.from_numpy(
                        norm_guide_mel[start_frame:end_frame].astype(np.float32, copy=False)
                    ).unsqueeze(0).to(device),
                    "phone_ids": torch.from_numpy(
                        phone_ids[start_frame:end_frame].astype(np.int64, copy=False)
                    ).unsqueeze(0).to(device),
                    "log_f0": torch.from_numpy(
                        norm_log_f0[start_frame:end_frame].astype(np.float32, copy=False)
                    ).unsqueeze(0).to(device),
                    "vuv": torch.from_numpy(
                        vuv[start_frame:end_frame].astype(np.float32, copy=False)
                    ).unsqueeze(0).to(device),
                    "energy": torch.from_numpy(
                        norm_energy[start_frame:end_frame].astype(np.float32, copy=False)
                    ).unsqueeze(0).to(device),
                    "lengths": torch.tensor([local_length], dtype=torch.long, device=device),
                    "content": torch.from_numpy(
                        norm_content[start_frame:end_frame].astype(np.float32, copy=False)
                    ).unsqueeze(0).to(device),
                    "beat_phase": torch.from_numpy(
                        beat_phase[start_frame:end_frame].astype(np.float32, copy=False)
                    ).unsqueeze(0).to(device),
                    "voice_prototype": voice_prototype,
                    "lyric_mask": torch.from_numpy(
                        lyric_mask[start_frame:end_frame].astype(np.float32, copy=False)
                    ).unsqueeze(0).to(device),
                }
                predicted_chunk, candidate_metadata = self._run_inference_candidate_search(
                    model=model,
                    batch=batch,
                    candidate_count=candidate_count,
                )
                chunk_scores.append(float(candidate_metadata.get("best_score", 0.0)))
                chunk_voice_similarity.append(float(candidate_metadata.get("best_voice_similarity", 0.0)))

                weights = np.ones((local_length, 1), dtype=np.float32)
                if overlap_frames > 1 and local_length > 2:
                    fade_length = min(overlap_frames, max(2, local_length // 3))
                    fade_in = np.linspace(0.0, 1.0, num=fade_length, dtype=np.float32).reshape(-1, 1)
                    fade_out = np.linspace(1.0, 0.0, num=fade_length, dtype=np.float32).reshape(-1, 1)
                    if start_frame > 0:
                        weights[:fade_length] *= fade_in
                    if end_frame < frame_count:
                        weights[-fade_length:] *= fade_out

                predicted_sum[start_frame:end_frame] += predicted_chunk * weights
                weight_sum[start_frame:end_frame] += weights

        normalized_prediction = predicted_sum / np.maximum(weight_sum, 1e-6)
        generated_audio, render_mode = self._render_audio_from_bundle(
            normalized_mel=normalized_prediction.astype(np.float32, copy=False),
            bundle=bundle,
        )

        guide_rms = float(np.sqrt(np.mean(np.square(guide_audio)) + 1e-9))
        generated_rms = float(np.sqrt(np.mean(np.square(generated_audio)) + 1e-9))
        if generated_rms > 1e-7:
            generated_audio = generated_audio * np.float32(np.clip(guide_rms / generated_rms, 0.76, 1.28))

        stereo_audio = np.repeat(generated_audio[:, np.newaxis], 2, axis=1).astype(np.float32, copy=False)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        sf.write(output_path, stereo_audio, SAMPLE_RATE, subtype="PCM_24")
        elapsed = float(time.perf_counter() - started)
        return {
            "output_path": str(output_path),
            "sample_rate": SAMPLE_RATE,
            "frame_count": frame_count,
            "duration_seconds": round(float(generated_audio.shape[0] / float(SAMPLE_RATE)), 3),
            "phone_coverage": round(float(np.mean(lyric_mask > 0.5)) if lyric_mask.size else 0.0, 4),
            "stats_path": str(bundle["stats_path"]),
            "checkpoint_path": str(bundle["checkpoint_path"]),
            "mode": "direct-frame-map-v1.1",
            "chunk_frames": int(chunk_frames),
            "overlap_frames": int(overlap_frames),
            "synthesis_seconds": round(elapsed, 3),
            "candidate_count": int(candidate_count),
            "mean_chunk_score": round(float(np.mean(chunk_scores)) if chunk_scores else 0.0, 4),
            "mean_chunk_voice_similarity": round(float(np.mean(chunk_voice_similarity)) if chunk_voice_similarity else 0.0, 4),
            "search_mode": "mc-dropout-target-voice-rerank" if candidate_count > 1 else "deterministic-target-voice",
            "render_mode": render_mode,
        }

    def train_post_process_refiner(
        self,
        *,
        dataset_dir: Path,
        output_dir: Path,
        stats: Dict[str, object],
        train_entries: Sequence[Dict[str, object]],
        val_entries: Sequence[Dict[str, object]],
        base_checkpoint_path: Path,
        requested_batch_size: int,
        start_epoch: int,
        total_end_epoch: int,
        guided_profile: Optional[Dict[str, object]],
        update_status: Callable[[str, str, str, int], None],
        cancel_event: Optional[object] = None,
        resume_checkpoint_path: Optional[Path] = None,
        resume_report_path: Optional[Path] = None,
    ) -> Dict[str, object]:
        output_dir.mkdir(parents=True, exist_ok=True)
        if total_end_epoch <= start_epoch:
            return {
                "checkpoint_path": "",
                "latest_checkpoint_path": "",
                "config_path": "",
                "report_path": "",
                "history_path": "",
                "best_val_total": 0.0,
                "best_epoch": 0,
                "quality_summary": "",
                "hardware_summary": "",
                "render_mode": "",
                "last_epoch": int(start_epoch),
                "stopped_early": False,
            }

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        hardware_profile = dict(guided_profile or self._get_training_hardware_profile())
        max_frames = int(hardware_profile.get("max_frames", 320))
        num_workers = int(hardware_profile.get("num_workers", 0))
        prefetch_factor = int(hardware_profile.get("prefetch_factor", 2))
        precision_mode = str(hardware_profile.get("precision_mode", "fp32"))
        autocast_dtype = torch.float32
        use_grad_scaler = False
        if device.type == "cuda":
            if precision_mode == "bf16":
                autocast_dtype = torch.bfloat16
            elif precision_mode == "fp16":
                autocast_dtype = torch.float16
                use_grad_scaler = True

        base_model, base_checkpoint_config = self._load_model_from_checkpoint(
            base_checkpoint_path,
            device=device,
        )
        base_model.eval()
        for parameter in base_model.parameters():
            parameter.requires_grad_(False)

        resolved_resume_checkpoint = Path(str(resume_checkpoint_path)) if resume_checkpoint_path else None
        resume_checkpoint: Dict[str, object] = {}
        resume_epoch = max(0, int(start_epoch))
        if resolved_resume_checkpoint is not None:
            if not resolved_resume_checkpoint.exists():
                raise FileNotFoundError(f"Post Process resume checkpoint was not found: {resolved_resume_checkpoint}")
            resume_checkpoint = torch.load(resolved_resume_checkpoint, map_location="cpu")
            resume_epoch = max(resume_epoch, int(resume_checkpoint.get("epoch", 0)))

        post_process_model_config = {
            "voice_signature_dim": VOICE_SIGNATURE_DIM,
            "model": {
                "d_model": 192,
                "n_heads": 4,
                "n_layers": 6,
                "dropout": 0.1,
            },
            "training_mode": "persona-post-process-refiner-v1",
            "base_checkpoint_path": str(base_checkpoint_path),
            "sample_rate": SAMPLE_RATE,
            "hop_length": HOP_LENGTH,
            "n_mels": N_MELS,
            "max_frames": int(max_frames),
            "batch_size": int(requested_batch_size),
            "num_workers": int(num_workers),
            "prefetch_factor": int(prefetch_factor),
            "precision_mode": precision_mode,
            "target_end_epoch": int(total_end_epoch),
            "loss_weights": {
                "mel": 0.92,
                "stft": 0.48,
                "phase": 0.16,
                "f0": 0.0,
                "delta_mel": 0.52,
                "delta_f0": 0.0,
                "vuv": 0.0,
                "phones": 0.0,
                "voice": 0.42,
                "prototype": 0.28,
                "silence": 0.12,
                "anchor": 0.1,
            },
        }
        if resume_checkpoint:
            merged_config = dict(resume_checkpoint.get("config", {}))
            merged_config.update(post_process_model_config)
            merged_config["loss_weights"] = dict(post_process_model_config["loss_weights"])
            post_process_model_config = merged_config

        model = self._instantiate_post_process_model(post_process_model_config).to(device)
        compiled_model = False
        compile_allowed = (
            device.type == "cuda"
            and bool(hardware_profile.get("compile_model", False))
            and hasattr(torch, "compile")
            and os.name == "nt"
        )
        if compile_allowed:
            try:
                model = torch.compile(model, mode="reduce-overhead")
                compiled_model = True
            except Exception:
                pass

        optimizer_kwargs = {
            "lr": min(float(hardware_profile.get("lr", 2e-4)), 1.5e-4),
            "betas": (0.9, 0.99),
            "weight_decay": 1e-4,
        }
        if device.type == "cuda" and bool(hardware_profile.get("use_fused_adamw", False)):
            optimizer_kwargs["fused"] = True
        try:
            optimizer = torch.optim.AdamW(model.parameters(), **optimizer_kwargs)
        except TypeError:
            optimizer_kwargs.pop("fused", None)
            optimizer = torch.optim.AdamW(model.parameters(), **optimizer_kwargs)

        scheduled_post_epochs = max(1, int(total_end_epoch - max(start_epoch, 0)))
        warmup_epochs = max(3, min(32, int(round(scheduled_post_epochs * 0.08))))
        min_lr_scale = 0.2

        def lr_lambda(epoch_index: int) -> float:
            epoch_number = epoch_index + 1
            if warmup_epochs > 0 and epoch_number <= warmup_epochs:
                return max(0.35, epoch_number / float(max(warmup_epochs, 1)))
            progress = (epoch_number - warmup_epochs) / float(max(scheduled_post_epochs - warmup_epochs, 1))
            cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
            return max(min_lr_scale, min_lr_scale + ((1.0 - min_lr_scale) * cosine))

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
        scaler = torch.cuda.amp.GradScaler(enabled=bool(device.type == "cuda" and use_grad_scaler))
        if resume_checkpoint:
            model.load_state_dict(dict(resume_checkpoint.get("model_state", {})), strict=False)
            optimizer_state = resume_checkpoint.get("optimizer_state")
            if isinstance(optimizer_state, dict):
                try:
                    optimizer.load_state_dict(optimizer_state)
                except Exception:
                    pass
            scheduler_state = resume_checkpoint.get("scheduler_state")
            if isinstance(scheduler_state, dict):
                try:
                    scheduler.load_state_dict(scheduler_state)
                except Exception:
                    pass

        resolved_resume_report = Path(str(resume_report_path)) if resume_report_path else None
        resume_report_payload: Dict[str, object] = {}
        if resolved_resume_report is not None and resolved_resume_report.exists():
            try:
                loaded_report = json.loads(resolved_resume_report.read_text(encoding="utf-8"))
                if isinstance(loaded_report, dict):
                    resume_report_payload = dict(loaded_report.get("guided_post_process", {}) or {})
            except Exception:
                resume_report_payload = {}

        config_path = output_dir / "guided_post_process_config.json"
        config_path.write_text(json.dumps(post_process_model_config, indent=2), encoding="utf-8")
        latest_checkpoint_path = output_dir / "guided_post_process_latest.pt"
        best_checkpoint_path = output_dir / "guided_post_process_best.pt"

        history: List[Dict[str, object]] = []
        if resolved_resume_checkpoint is not None:
            resume_history_path = resolved_resume_checkpoint.parent / "guided_post_process_history.json"
            if resume_history_path.exists():
                try:
                    loaded_history = json.loads(resume_history_path.read_text(encoding="utf-8"))
                    if isinstance(loaded_history, list):
                        history = [dict(entry) for entry in loaded_history if isinstance(entry, dict)]
                except Exception:
                    history = []

        best_val_total = float(resume_checkpoint.get("best_val_loss", resume_report_payload.get("best_val_total", float("inf"))))
        best_epoch = int(resume_report_payload.get("best_epoch", resume_epoch if resume_epoch > 0 else 0))
        last_epoch = max(resume_epoch, int(resume_report_payload.get("last_epoch", resume_epoch)))
        loader_safe_mode = False
        stopped_early = False
        start_time = time.time()
        hardware_summary = (
            f"{device.type} | {float(hardware_profile.get('gpu_memory_gb', 0.0)):.1f}GB | "
            f"{precision_mode} | {'fused AdamW' if optimizer_kwargs.get('fused') else 'AdamW'} | "
            f"{'compiled' if compiled_model else 'eager'} | frames {max_frames} | workers {num_workers}"
        )

        update_status(
            "guided-post-process-train",
            "Training Post Process target-vocal refiner...",
            (
                f"Windows {len(train_entries)} train / {len(val_entries)} val | "
                f"{hardware_summary} | base checkpoint {base_checkpoint_path.name} | "
                f"target polish without changing the learned content model"
            ),
            83,
        )

        for epoch in range(max(1, resume_epoch + 1), max(1, total_end_epoch) + 1):
            if cancel_event is not None and bool(getattr(cancel_event, "is_set", lambda: False)()):
                stopped_early = True
                break

            while True:
                active_worker_count = 0 if loader_safe_mode else num_workers
                active_prefetch_factor = 2 if loader_safe_mode else prefetch_factor
                train_loader = self._create_loader(
                    entries=train_entries,
                    dataset_dir=dataset_dir,
                    stats=stats,
                    max_frames=max_frames,
                    batch_size=requested_batch_size,
                    num_workers=active_worker_count,
                    prefetch_factor=active_prefetch_factor,
                    random_crop=True,
                    shuffle=True,
                )
                val_loader = self._create_loader(
                    entries=val_entries,
                    dataset_dir=dataset_dir,
                    stats=stats,
                    max_frames=max_frames,
                    batch_size=requested_batch_size,
                    num_workers=active_worker_count,
                    prefetch_factor=active_prefetch_factor,
                    random_crop=False,
                    shuffle=False,
                )
                model.train()
                train_loss_sums = {
                    "total": 0.0,
                    "mel": 0.0,
                    "stft": 0.0,
                    "phase": 0.0,
                    "f0": 0.0,
                    "delta_mel": 0.0,
                    "delta_f0": 0.0,
                    "vuv": 0.0,
                    "phones": 0.0,
                    "voice": 0.0,
                    "prototype": 0.0,
                    "silence": 0.0,
                    "anchor": 0.0,
                }
                train_metric_counts = {
                    "phone_correct": 0.0,
                    "phone_total": 0.0,
                    "lyric_phone_correct": 0.0,
                    "lyric_phone_total": 0.0,
                    "vuv_correct": 0.0,
                    "vuv_total": 0.0,
                }
                train_batches = 0
                train_frame_count = 0
                epoch_started = time.time()
                try:
                    for batch in train_loader:
                        if cancel_event is not None and bool(getattr(cancel_event, "is_set", lambda: False)()):
                            stopped_early = True
                            break
                        batch = {key: value.to(device) for key, value in batch.items()}
                        optimizer.zero_grad(set_to_none=True)
                        autocast_context = (
                            torch.autocast(device_type="cuda", dtype=autocast_dtype)
                            if device.type == "cuda" and precision_mode in {"bf16", "fp16"}
                            else nullcontext()
                        )
                        with torch.no_grad(), autocast_context:
                            coarse_outputs = self._predict_outputs(
                                base_model,
                                guide_mel=batch["guide_mel"],
                                phone_ids=batch["phone_ids"],
                                log_f0=batch["log_f0"],
                                vuv=batch["vuv"],
                                energy=batch["energy"],
                                lengths=batch["lengths"],
                                voice_prototype=batch["voice_prototype"],
                                lyric_mask=batch["lyric_mask"],
                            )
                        with autocast_context:
                            refined_mel = model(
                                coarse_mel=coarse_outputs["mel"],
                                guide_mel=batch["guide_mel"],
                                phone_ids=batch["phone_ids"],
                                log_f0=batch["log_f0"],
                                vuv=batch["vuv"],
                                energy=batch["energy"],
                                lengths=batch["lengths"],
                                voice_prototype=batch["voice_prototype"],
                                lyric_mask=batch["lyric_mask"],
                            )
                            refined_outputs = {
                                **coarse_outputs,
                                "mel": refined_mel,
                            }
                            loss_terms = self._compute_post_process_loss_terms(
                                outputs=refined_outputs,
                                coarse_mel=coarse_outputs["mel"],
                                batch=batch,
                                loss_weights=dict(post_process_model_config.get("loss_weights", {})),
                            )
                            loss = loss_terms["total"]
                        if use_grad_scaler:
                            scaler.scale(loss).backward()
                            scaler.unscale_(optimizer)
                        else:
                            loss.backward()
                        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                        if use_grad_scaler:
                            scaler.step(optimizer)
                            scaler.update()
                        else:
                            optimizer.step()
                        for loss_name in train_loss_sums:
                            loss_value = loss_terms.get(loss_name)
                            if isinstance(loss_value, torch.Tensor):
                                train_loss_sums[loss_name] += float(loss_value.detach().cpu().item())
                        phone_counts = self._phone_accuracy_counts(
                            refined_outputs["phone_logits"],
                            batch["phone_ids"],
                            batch["lengths"],
                            batch["lyric_mask"],
                        )
                        vuv_counts = self._vuv_accuracy_counts(
                            refined_outputs["target_vuv_logits"],
                            batch["target_vuv"],
                            batch["lengths"],
                        )
                        for metric_name, metric_value in {**phone_counts, **vuv_counts}.items():
                            train_metric_counts[metric_name] += float(metric_value)
                        train_batches += 1
                        train_frame_count += int(batch["lengths"].sum().detach().cpu().item())

                    model.eval()
                    val_loss_sums = {
                        "total": 0.0,
                        "mel": 0.0,
                        "stft": 0.0,
                        "phase": 0.0,
                        "f0": 0.0,
                        "delta_mel": 0.0,
                        "delta_f0": 0.0,
                        "vuv": 0.0,
                        "phones": 0.0,
                        "voice": 0.0,
                        "prototype": 0.0,
                        "silence": 0.0,
                        "anchor": 0.0,
                    }
                    val_metric_counts = {
                        "phone_correct": 0.0,
                        "phone_total": 0.0,
                        "lyric_phone_correct": 0.0,
                        "lyric_phone_total": 0.0,
                        "vuv_correct": 0.0,
                        "vuv_total": 0.0,
                    }
                    val_batches = 0
                    with torch.no_grad():
                        for batch in val_loader:
                            batch = {key: value.to(device) for key, value in batch.items()}
                            coarse_outputs = self._predict_outputs(
                                base_model,
                                guide_mel=batch["guide_mel"],
                                phone_ids=batch["phone_ids"],
                                log_f0=batch["log_f0"],
                                vuv=batch["vuv"],
                                energy=batch["energy"],
                                lengths=batch["lengths"],
                                voice_prototype=batch["voice_prototype"],
                                lyric_mask=batch["lyric_mask"],
                            )
                            refined_mel = model(
                                coarse_mel=coarse_outputs["mel"],
                                guide_mel=batch["guide_mel"],
                                phone_ids=batch["phone_ids"],
                                log_f0=batch["log_f0"],
                                vuv=batch["vuv"],
                                energy=batch["energy"],
                                lengths=batch["lengths"],
                                voice_prototype=batch["voice_prototype"],
                                lyric_mask=batch["lyric_mask"],
                            )
                            refined_outputs = {
                                **coarse_outputs,
                                "mel": refined_mel,
                            }
                            loss_terms = self._compute_post_process_loss_terms(
                                outputs=refined_outputs,
                                coarse_mel=coarse_outputs["mel"],
                                batch=batch,
                                loss_weights=dict(post_process_model_config.get("loss_weights", {})),
                            )
                            for loss_name in val_loss_sums:
                                loss_value = loss_terms.get(loss_name)
                                if isinstance(loss_value, torch.Tensor):
                                    val_loss_sums[loss_name] += float(loss_value.detach().cpu().item())
                            phone_counts = self._phone_accuracy_counts(
                                refined_outputs["phone_logits"],
                                batch["phone_ids"],
                                batch["lengths"],
                                batch["lyric_mask"],
                            )
                            vuv_counts = self._vuv_accuracy_counts(
                                refined_outputs["target_vuv_logits"],
                                batch["target_vuv"],
                                batch["lengths"],
                            )
                            for metric_name, metric_value in {**phone_counts, **vuv_counts}.items():
                                val_metric_counts[metric_name] += float(metric_value)
                            val_batches += 1
                    break
                except RuntimeError as exc:
                    if (not self._is_loader_runtime_error(exc)) or loader_safe_mode:
                        raise
                    loader_safe_mode = True
                    update_status(
                        "guided-post-process-train",
                        "Post Process loader became unstable, retrying in safe mode...",
                        "Switching the refiner to single-process loading so target polish can continue uninterrupted.",
                        83,
                    )
                    if device.type == "cuda":
                        try:
                            torch.cuda.empty_cache()
                        except Exception:
                            pass

            last_epoch = epoch
            train_total = train_loss_sums["total"] / max(train_batches, 1)
            val_total = val_loss_sums["total"] / max(val_batches, 1)
            train_phone_accuracy = train_metric_counts["phone_correct"] / max(train_metric_counts["phone_total"], 1.0)
            val_phone_accuracy = val_metric_counts["phone_correct"] / max(val_metric_counts["phone_total"], 1.0)
            train_lyric_phone_accuracy = train_metric_counts["lyric_phone_correct"] / max(train_metric_counts["lyric_phone_total"], 1.0)
            val_lyric_phone_accuracy = val_metric_counts["lyric_phone_correct"] / max(val_metric_counts["lyric_phone_total"], 1.0)
            train_vuv_accuracy = train_metric_counts["vuv_correct"] / max(train_metric_counts["vuv_total"], 1.0)
            val_vuv_accuracy = val_metric_counts["vuv_correct"] / max(val_metric_counts["vuv_total"], 1.0)
            improved = val_total <= best_val_total + 1e-8
            if improved:
                best_val_total = val_total
                best_epoch = epoch
            scheduler.step()
            epoch_duration = max(time.time() - epoch_started, 1e-6)
            train_frames_per_second = float(train_frame_count) / epoch_duration
            epochs_since_best = max(0, epoch - max(best_epoch, epoch if improved else 0))
            if improved:
                quality_state = "new best"
            elif epochs_since_best <= 4:
                quality_state = "holding near best"
            elif (val_loss_sums["voice"] / max(val_batches, 1)) > 0.05:
                quality_state = "pulling closer to target texture"
            elif (val_loss_sums["delta_mel"] / max(val_batches, 1)) > 0.05:
                quality_state = "smoothing texture"
            elif epochs_since_best >= 40:
                quality_state = "plateauing"
            else:
                quality_state = "steady polish"

            history.append(
                {
                    "epoch": epoch,
                    "phase": "post-process",
                    "active_train_slices": int(len(train_entries)),
                    "train_total": round(train_total, 6),
                    "train_mel": round(train_loss_sums["mel"] / max(train_batches, 1), 6),
                    "train_stft": round(train_loss_sums["stft"] / max(train_batches, 1), 6),
                    "train_phase": round(train_loss_sums["phase"] / max(train_batches, 1), 6),
                    "train_f0": round(train_loss_sums["f0"] / max(train_batches, 1), 6),
                    "train_delta_mel": round(train_loss_sums["delta_mel"] / max(train_batches, 1), 6),
                    "train_delta_f0": round(train_loss_sums["delta_f0"] / max(train_batches, 1), 6),
                    "train_vuv": round(train_loss_sums["vuv"] / max(train_batches, 1), 6),
                    "train_phones": round(train_loss_sums["phones"] / max(train_batches, 1), 6),
                    "train_voice": round(train_loss_sums["voice"] / max(train_batches, 1), 6),
                    "train_prototype": round(train_loss_sums["prototype"] / max(train_batches, 1), 6),
                    "train_silence": round(train_loss_sums["silence"] / max(train_batches, 1), 6),
                    "train_anchor": round(train_loss_sums["anchor"] / max(train_batches, 1), 6),
                    "val_total": round(val_total, 6),
                    "val_mel": round(val_loss_sums["mel"] / max(val_batches, 1), 6),
                    "val_stft": round(val_loss_sums["stft"] / max(val_batches, 1), 6),
                    "val_phase": round(val_loss_sums["phase"] / max(val_batches, 1), 6),
                    "val_f0": round(val_loss_sums["f0"] / max(val_batches, 1), 6),
                    "val_delta_mel": round(val_loss_sums["delta_mel"] / max(val_batches, 1), 6),
                    "val_delta_f0": round(val_loss_sums["delta_f0"] / max(val_batches, 1), 6),
                    "val_vuv": round(val_loss_sums["vuv"] / max(val_batches, 1), 6),
                    "val_phones": round(val_loss_sums["phones"] / max(val_batches, 1), 6),
                    "val_voice": round(val_loss_sums["voice"] / max(val_batches, 1), 6),
                    "val_prototype": round(val_loss_sums["prototype"] / max(val_batches, 1), 6),
                    "val_silence": round(val_loss_sums["silence"] / max(val_batches, 1), 6),
                    "val_anchor": round(val_loss_sums["anchor"] / max(val_batches, 1), 6),
                    "train_phone_accuracy": round(train_phone_accuracy, 6),
                    "val_phone_accuracy": round(val_phone_accuracy, 6),
                    "train_lyric_phone_accuracy": round(train_lyric_phone_accuracy, 6),
                    "val_lyric_phone_accuracy": round(val_lyric_phone_accuracy, 6),
                    "train_vuv_accuracy": round(train_vuv_accuracy, 6),
                    "val_vuv_accuracy": round(val_vuv_accuracy, 6),
                    "best_val_total": round(best_val_total, 6),
                    "best_epoch": int(best_epoch),
                    "epochs_since_best": int(epochs_since_best),
                    "quality_state": quality_state,
                    "epoch_seconds": round(epoch_duration, 3),
                    "train_frames_per_second": round(train_frames_per_second, 2),
                    "learning_rate": round(float(optimizer.param_groups[0]["lr"]), 10),
                    "elapsed_seconds": round(max(time.time() - start_time, 0.0), 2),
                }
            )

            self._save_post_process_checkpoint(
                path=latest_checkpoint_path,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
                best_val_loss=best_val_total,
                config=post_process_model_config,
            )
            if improved:
                self._save_post_process_checkpoint(
                    path=best_checkpoint_path,
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    epoch=epoch,
                    best_val_loss=best_val_total,
                    config=post_process_model_config,
                )

            completed_post_epochs = max(1, epoch - start_epoch)
            total_post_epochs = max(1, total_end_epoch - start_epoch)
            progress = min(90, max(83, 82 + int(round((completed_post_epochs / float(total_post_epochs)) * 8))))
            update_status(
                "guided-post-process-train",
                f"Post Process epoch {epoch}/{int(total_end_epoch)} | {quality_state}",
                (
                    f"best epoch {best_epoch or epoch} | best total {best_val_total:.4f} | "
                    f"train/val total {train_total:.4f}/{val_total:.4f} | "
                    f"voice match {train_loss_sums['voice'] / max(train_batches, 1):.4f}/{val_loss_sums['voice'] / max(val_batches, 1):.4f} | "
                    f"transition {train_loss_sums['delta_mel'] / max(train_batches, 1):.4f}/{val_loss_sums['delta_mel'] / max(val_batches, 1):.4f} | "
                    f"anchor {train_loss_sums['anchor'] / max(train_batches, 1):.4f}/{val_loss_sums['anchor'] / max(val_batches, 1):.4f} | "
                    f"lyric phones {train_lyric_phone_accuracy * 100.0:.1f}%/{val_lyric_phone_accuracy * 100.0:.1f}% | "
                    f"voicing {train_vuv_accuracy * 100.0:.1f}%/{val_vuv_accuracy * 100.0:.1f}% | "
                    f"frames/s {train_frames_per_second:.0f} | lr {optimizer.param_groups[0]['lr']:.2e} | "
                    f"plateau {epochs_since_best} epochs | active slices {len(train_entries)}"
                ),
                progress,
            )

        if not latest_checkpoint_path.exists():
            if stopped_early:
                raise InterruptedError("Post Process stopped before the first target-polish checkpoint was written.")
            raise RuntimeError("Post Process training never wrote a checkpoint.")

        history_path = output_dir / "guided_post_process_history.json"
        history_path.write_text(json.dumps(history, indent=2), encoding="utf-8")
        report_payload = {
            "checkpoint_path": str(best_checkpoint_path if best_checkpoint_path.exists() else latest_checkpoint_path),
            "latest_checkpoint_path": str(latest_checkpoint_path),
            "history_path": str(history_path),
            "config_path": str(config_path),
            "best_val_total": round(float(best_val_total), 6),
            "best_epoch": int(best_epoch),
            "quality_summary": str(history[-1]["quality_state"]) if history else "",
            "hardware_summary": hardware_summary,
            "last_epoch": int(last_epoch),
            "base_checkpoint_path": str(base_checkpoint_path),
            "training_mode": "persona-post-process-refiner-v1",
            "stopped_early": bool(stopped_early),
        }
        report_path = output_dir / "guided_post_process_report.json"
        report_path.write_text(json.dumps(report_payload, indent=2), encoding="utf-8")
        return {
            "checkpoint_path": str(best_checkpoint_path if best_checkpoint_path.exists() else latest_checkpoint_path),
            "latest_checkpoint_path": str(latest_checkpoint_path),
            "history_path": str(history_path),
            "config_path": str(config_path),
            "report_path": str(report_path),
            "best_val_total": float(best_val_total),
            "best_epoch": int(best_epoch),
            "quality_summary": str(history[-1]["quality_state"]) if history else "",
            "hardware_summary": hardware_summary,
            "render_mode": "persona-post-process-v1",
            "last_epoch": int(last_epoch),
            "stopped_early": bool(stopped_early),
        }

    def train_neural_vocoder(
        self,
        *,
        dataset_dir: Path,
        output_dir: Path,
        stats: Dict[str, object],
        train_entries: Sequence[Dict[str, object]],
        val_entries: Sequence[Dict[str, object]],
        requested_batch_size: int,
        total_epochs: int,
        guided_profile: Optional[Dict[str, object]],
        update_status: Callable[[str, str, str, int], None],
        requested_num_workers: int = -1,
        cancel_event: Optional[object] = None,
    ) -> Dict[str, object]:
        output_dir.mkdir(parents=True, exist_ok=True)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        profile = self._get_vocoder_hardware_profile(
            guided_profile=guided_profile,
            requested_batch_size=requested_batch_size,
            total_epochs=total_epochs,
        )
        if int(requested_num_workers) >= 0:
            profile["num_workers"] = int(requested_num_workers)
        precision_mode = str(profile.get("precision_mode", "fp32"))
        autocast_dtype = torch.float32
        use_grad_scaler = False
        if device.type == "cuda":
            if precision_mode == "bf16":
                autocast_dtype = torch.bfloat16
            elif precision_mode == "fp16":
                autocast_dtype = torch.float16
                use_grad_scaler = True
        config = {
            "sample_rate": SAMPLE_RATE,
            "hop_length": HOP_LENGTH,
            "n_mels": N_MELS,
            "base_channels": int(profile.get("base_channels", 384)),
            "max_frames": int(profile.get("max_frames", 256)),
            "batch_size": int(profile.get("batch_size", 8)),
            "total_epochs": int(profile.get("total_epochs", 80)),
            "precision_mode": precision_mode,
            "compile_model": bool(profile.get("compile_model", False)),
            "lr": float(profile.get("lr", 1.5e-4)),
            "gpu_memory_gb": float(profile.get("gpu_memory_gb", 0.0)),
        }
        config_path = output_dir / "guided_vocoder_config.json"
        config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")

        train_loader = self._create_vocoder_loader(
            entries=train_entries,
            dataset_dir=dataset_dir,
            stats=stats,
            max_frames=int(config["max_frames"]),
            batch_size=int(config["batch_size"]),
            num_workers=int(profile.get("num_workers", 0)),
            prefetch_factor=int(profile.get("prefetch_factor", 2)),
            random_crop=True,
            shuffle=True,
        )
        val_loader = self._create_vocoder_loader(
            entries=val_entries,
            dataset_dir=dataset_dir,
            stats=stats,
            max_frames=int(config["max_frames"]),
            batch_size=int(config["batch_size"]),
            num_workers=int(profile.get("num_workers", 0)),
            prefetch_factor=int(profile.get("prefetch_factor", 2)),
            random_crop=False,
            shuffle=False,
        )

        generator: nn.Module = PersonaNeuralVocoder(
            base_channels=int(config["base_channels"]),
        ).to(device)
        discriminator: nn.Module = MultiScaleWaveDiscriminator().to(device)
        compiled = False
        compile_allowed = (
            device.type == "cuda"
            and bool(config.get("compile_model", False))
            and hasattr(torch, "compile")
            and os.name == "nt"
        )
        if compile_allowed:
            try:
                generator = torch.compile(generator, mode="reduce-overhead")
                discriminator = torch.compile(discriminator, mode="reduce-overhead")
                compiled = True
            except Exception:
                compiled = False

        optimizer_kwargs = {
            "lr": float(config["lr"]),
            "betas": (0.8, 0.99),
            "weight_decay": 0.0,
        }
        if device.type == "cuda" and bool(profile.get("use_fused_adamw", False)):
            optimizer_kwargs["fused"] = True
        try:
            optimizer_g = torch.optim.AdamW(generator.parameters(), **optimizer_kwargs)
            optimizer_d = torch.optim.AdamW(discriminator.parameters(), **optimizer_kwargs)
        except TypeError:
            optimizer_kwargs.pop("fused", None)
            optimizer_g = torch.optim.AdamW(generator.parameters(), **optimizer_kwargs)
            optimizer_d = torch.optim.AdamW(discriminator.parameters(), **optimizer_kwargs)

        warmup_epochs = max(4, int(round(int(config["total_epochs"]) * 0.08)))

        def vocoder_lr_lambda(epoch_index: int) -> float:
            epoch_number = epoch_index + 1
            if epoch_number <= warmup_epochs:
                return max(0.25, epoch_number / float(max(warmup_epochs, 1)))
            progress = (epoch_number - warmup_epochs) / float(max(int(config["total_epochs"]) - warmup_epochs, 1))
            cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
            return max(0.08, 0.08 + (0.92 * cosine))

        scheduler_g = torch.optim.lr_scheduler.LambdaLR(optimizer_g, lr_lambda=vocoder_lr_lambda)
        scheduler_d = torch.optim.lr_scheduler.LambdaLR(optimizer_d, lr_lambda=vocoder_lr_lambda)
        scaler = torch.cuda.amp.GradScaler(enabled=bool(device.type == "cuda" and use_grad_scaler))

        best_checkpoint_path = output_dir / "guided_vocoder_best.pt"
        latest_checkpoint_path = output_dir / "guided_vocoder_latest.pt"
        history: List[Dict[str, object]] = []
        best_val_total = float("inf")
        best_epoch = 0
        stopped_early = False
        start_time = time.time()
        hardware_summary = (
            f"{device.type} | {float(config.get('gpu_memory_gb', 0.0)):.1f}GB | "
            f"{precision_mode} | {'compiled' if compiled else 'eager'} | "
            f"vocoder batch {int(config['batch_size'])} | frames {int(config['max_frames'])}"
        )

        update_status(
            "guided-vocoder-train",
            "Fine-tuning the neural waveform decoder for voice realism...",
            (
                f"Train {len(train_entries)} slices | val {len(val_entries)} | "
                f"{hardware_summary} | adversarial + STFT waveform refinement"
            ),
            84,
        )
        loader_safe_mode = False

        for epoch in range(1, int(config["total_epochs"]) + 1):
            if cancel_event is not None and bool(getattr(cancel_event, "is_set", lambda: False)()):
                stopped_early = True
                break
            while True:
                try:
                    generator.train()
                    discriminator.train()
                    train_sums = {
                        "generator_total": 0.0,
                        "wave": 0.0,
                        "stft": 0.0,
                        "adv": 0.0,
                        "feature_matching": 0.0,
                        "discriminator": 0.0,
                    }
                    train_batches = 0
                    train_samples = 0
                    epoch_started = time.time()
                    active_worker_count = 0 if loader_safe_mode else int(profile.get("num_workers", 0))
                    active_prefetch_factor = 2 if loader_safe_mode else int(profile.get("prefetch_factor", 2))
                    if loader_safe_mode:
                        train_loader = self._create_vocoder_loader(
                            entries=train_entries,
                            dataset_dir=dataset_dir,
                            stats=stats,
                            max_frames=int(config["max_frames"]),
                            batch_size=int(config["batch_size"]),
                            num_workers=active_worker_count,
                            prefetch_factor=active_prefetch_factor,
                            random_crop=True,
                            shuffle=True,
                        )
                        val_loader = self._create_vocoder_loader(
                            entries=val_entries,
                            dataset_dir=dataset_dir,
                            stats=stats,
                            max_frames=int(config["max_frames"]),
                            batch_size=int(config["batch_size"]),
                            num_workers=active_worker_count,
                            prefetch_factor=active_prefetch_factor,
                            random_crop=False,
                            shuffle=False,
                        )
                    for batch in train_loader:
                        if cancel_event is not None and bool(getattr(cancel_event, "is_set", lambda: False)()):
                            stopped_early = True
                            break
                        batch = {key: value.to(device) for key, value in batch.items()}
                        real_audio = _mask_audio_by_lengths(batch["audio"], batch["sample_lengths"])
                        autocast_context = (
                            torch.autocast(device_type="cuda", dtype=autocast_dtype)
                            if device.type == "cuda" and precision_mode in {"bf16", "fp16"}
                            else nullcontext()
                        )

                        for parameter in discriminator.parameters():
                            parameter.requires_grad_(True)
                        optimizer_d.zero_grad(set_to_none=True)
                        with autocast_context:
                            fake_audio = _mask_audio_by_lengths(generator(batch["mel"]), batch["sample_lengths"])
                            discriminator_loss = self._vocoder_discriminator_hinge_loss(
                                discriminator,
                                real_audio=real_audio,
                                fake_audio=fake_audio,
                            )
                        if use_grad_scaler:
                            scaler.scale(discriminator_loss).backward()
                            scaler.step(optimizer_d)
                            scaler.update()
                        else:
                            discriminator_loss.backward()
                            optimizer_d.step()

                        for parameter in discriminator.parameters():
                            parameter.requires_grad_(False)
                        optimizer_g.zero_grad(set_to_none=True)
                        with autocast_context:
                            fake_audio = _mask_audio_by_lengths(generator(batch["mel"]), batch["sample_lengths"])
                            wave_loss = self._masked_wave_l1_loss(fake_audio, real_audio, batch["sample_lengths"])
                            stft_loss = self._multi_resolution_wave_stft_loss(fake_audio, real_audio, batch["sample_lengths"])
                            adv_loss, feature_matching_loss = self._vocoder_generator_losses(
                                discriminator,
                                real_audio=real_audio,
                                fake_audio=fake_audio,
                            )
                            generator_total = (
                                wave_loss
                                + (0.8 * stft_loss)
                                + (0.18 * adv_loss)
                                + (0.32 * feature_matching_loss)
                            )
                        if use_grad_scaler:
                            scaler.scale(generator_total).backward()
                            scaler.step(optimizer_g)
                            scaler.update()
                        else:
                            generator_total.backward()
                            optimizer_g.step()
                        for parameter in discriminator.parameters():
                            parameter.requires_grad_(True)

                        train_sums["generator_total"] += float(generator_total.detach().cpu().item())
                        train_sums["wave"] += float(wave_loss.detach().cpu().item())
                        train_sums["stft"] += float(stft_loss.detach().cpu().item())
                        train_sums["adv"] += float(adv_loss.detach().cpu().item())
                        train_sums["feature_matching"] += float(feature_matching_loss.detach().cpu().item())
                        train_sums["discriminator"] += float(discriminator_loss.detach().cpu().item())
                        train_batches += 1
                        train_samples += int(batch["sample_lengths"].shape[0])

                    generator.eval()
                    discriminator.eval()
                    val_sums = {
                        "generator_total": 0.0,
                        "wave": 0.0,
                        "stft": 0.0,
                    }
                    val_batches = 0
                    with torch.no_grad():
                        for batch in val_loader:
                            batch = {key: value.to(device) for key, value in batch.items()}
                            real_audio = _mask_audio_by_lengths(batch["audio"], batch["sample_lengths"])
                            fake_audio = _mask_audio_by_lengths(generator(batch["mel"]), batch["sample_lengths"])
                            wave_loss = self._masked_wave_l1_loss(fake_audio, real_audio, batch["sample_lengths"])
                            stft_loss = self._multi_resolution_wave_stft_loss(fake_audio, real_audio, batch["sample_lengths"])
                            generator_total = wave_loss + (0.8 * stft_loss)
                            val_sums["generator_total"] += float(generator_total.detach().cpu().item())
                            val_sums["wave"] += float(wave_loss.detach().cpu().item())
                            val_sums["stft"] += float(stft_loss.detach().cpu().item())
                            val_batches += 1
                    break
                except RuntimeError as exc:
                    if (not self._is_loader_runtime_error(exc)) or loader_safe_mode:
                        raise
                    loader_safe_mode = True
                    update_status(
                        "guided-vocoder-train",
                        "Waveform loader became unstable, retrying in safe mode...",
                        "Switching to single-process data loading so training can continue uninterrupted.",
                        84,
                    )
                    try:
                        if device.type == "cuda":
                            torch.cuda.empty_cache()
                    except Exception:
                        pass

            train_total = train_sums["generator_total"] / max(train_batches, 1)
            val_total = val_sums["generator_total"] / max(val_batches, 1)
            val_wave = val_sums["wave"] / max(val_batches, 1)
            val_stft = val_sums["stft"] / max(val_batches, 1)
            improved = val_total <= best_val_total + 1e-8
            if improved:
                best_val_total = val_total
                best_epoch = epoch
            scheduler_g.step()
            scheduler_d.step()
            epoch_seconds = max(time.time() - epoch_started, 1e-6)
            samples_per_second = float(train_samples) / epoch_seconds
            epochs_since_best = max(0, epoch - max(best_epoch, epoch if improved else 0))
            quality_state = (
                "new best"
                if improved
                else ("still sharpening transients" if val_stft > 0.8 else ("locking realism" if epochs_since_best < 10 else "plateauing"))
            )
            history.append(
                {
                    "epoch": epoch,
                    "train_generator_total": round(train_total, 6),
                    "train_wave": round(train_sums["wave"] / max(train_batches, 1), 6),
                    "train_stft": round(train_sums["stft"] / max(train_batches, 1), 6),
                    "train_adv": round(train_sums["adv"] / max(train_batches, 1), 6),
                    "train_feature_matching": round(train_sums["feature_matching"] / max(train_batches, 1), 6),
                    "train_discriminator": round(train_sums["discriminator"] / max(train_batches, 1), 6),
                    "val_generator_total": round(val_total, 6),
                    "val_wave": round(val_wave, 6),
                    "val_stft": round(val_stft, 6),
                    "best_val_total": round(best_val_total, 6),
                    "best_epoch": int(best_epoch),
                    "epochs_since_best": int(epochs_since_best),
                    "quality_state": quality_state,
                    "samples_per_second": round(samples_per_second, 2),
                    "learning_rate": round(float(optimizer_g.param_groups[0]["lr"]), 10),
                    "elapsed_seconds": round(max(time.time() - start_time, 0.0), 2),
                }
            )

            self._save_vocoder_checkpoint(
                path=latest_checkpoint_path,
                generator=generator,
                discriminator=discriminator,
                optimizer_g=optimizer_g,
                optimizer_d=optimizer_d,
                epoch=epoch,
                best_val_loss=best_val_total,
                config=config,
            )
            if improved:
                self._save_vocoder_checkpoint(
                    path=best_checkpoint_path,
                    generator=generator,
                    discriminator=discriminator,
                    optimizer_g=optimizer_g,
                    optimizer_d=optimizer_d,
                    epoch=epoch,
                    best_val_loss=best_val_total,
                    config=config,
                )

            progress = min(96, 84 + int(round((epoch / float(max(int(config["total_epochs"]), 1))) * 12)))
            update_status(
                "guided-vocoder-train",
                f"Neural vocoder epoch {epoch}/{int(config['total_epochs'])} | {quality_state}",
                (
                    f"best epoch {best_epoch or epoch} | best total {best_val_total:.4f} | "
                    f"train/val total {train_total:.4f}/{val_total:.4f} | "
                    f"wave {train_sums['wave'] / max(train_batches, 1):.4f}/{val_wave:.4f} | "
                    f"stft {train_sums['stft'] / max(train_batches, 1):.4f}/{val_stft:.4f} | "
                    f"adv {train_sums['adv'] / max(train_batches, 1):.4f} | "
                    f"fm {train_sums['feature_matching'] / max(train_batches, 1):.4f} | "
                    f"disc {train_sums['discriminator'] / max(train_batches, 1):.4f} | "
                    f"samples/s {samples_per_second:.1f} | lr {optimizer_g.param_groups[0]['lr']:.2e}"
                ),
                progress,
            )

        if not latest_checkpoint_path.exists():
            if stopped_early:
                raise InterruptedError("Vocoder training stopped before the first neural decoder checkpoint was written.")
            raise RuntimeError("Neural vocoder training never wrote a checkpoint.")

        history_path = output_dir / "guided_vocoder_history.json"
        history_path.write_text(json.dumps(history, indent=2), encoding="utf-8")
        report_payload = {
            "checkpoint_path": str(best_checkpoint_path if best_checkpoint_path.exists() else latest_checkpoint_path),
            "latest_checkpoint_path": str(latest_checkpoint_path),
            "history_path": str(history_path),
            "config_path": str(config_path),
            "best_val_total": round(float(best_val_total), 6),
            "best_epoch": int(best_epoch),
            "quality_summary": str(history[-1]["quality_state"]) if history else "",
            "hardware_summary": hardware_summary,
            "render_mode": "persona-neural-vocoder-v1",
            "stopped_early": bool(stopped_early),
        }
        report_path = output_dir / "guided_vocoder_report.json"
        report_path.write_text(json.dumps(report_payload, indent=2), encoding="utf-8")
        return {
            "checkpoint_path": str(best_checkpoint_path if best_checkpoint_path.exists() else latest_checkpoint_path),
            "latest_checkpoint_path": str(latest_checkpoint_path),
            "history_path": str(history_path),
            "config_path": str(config_path),
            "report_path": str(report_path),
            "best_val_total": float(best_val_total),
            "best_epoch": int(best_epoch),
            "quality_summary": str(history[-1]["quality_state"]) if history else "",
            "hardware_summary": hardware_summary,
            "render_mode": "persona-neural-vocoder-v1",
            "stopped_early": bool(stopped_early),
        }

    def train_guided_regenerator(
        self,
        *,
        dataset_dir: Path,
        output_dir: Path,
        total_epochs: int,
        save_every_epoch: int,
        batch_size: int,
        requested_num_workers: int = -1,
        recipe_mode: str = "persona-v1",
        update_status: Callable[[str, str, str, int], None],
        cancel_event: Optional[object] = None,
        resume_checkpoint_path: Optional[Path] = None,
        resume_report_path: Optional[Path] = None,
        resume_post_process_checkpoint_path: Optional[Path] = None,
        start_phase: str = "auto",
        warmup_stage_epochs: int = 600,
        bridge_stage_epochs: int = 1200,
        full_diversity_stage_epochs: int = 600,
        general_refine_stage_epochs: int = 180,
        target_fit_stage_epochs: int = 0,
        post_process_stage_epochs: int = 0,
        checkpoint_callback: Optional[Callable[[Dict[str, object]], None]] = None,
    ) -> Dict[str, object]:
        output_dir.mkdir(parents=True, exist_ok=True)
        hardware_profile = self._get_training_hardware_profile()
        max_frames = int(hardware_profile.get("max_frames", 320))
        num_workers = int(hardware_profile.get("num_workers", 0))
        prefetch_factor = int(hardware_profile.get("prefetch_factor", 2))
        if int(requested_num_workers) >= 0:
            num_workers = int(requested_num_workers)
        normalized_recipe_mode = str(recipe_mode or "persona-v1").strip().lower() or "persona-v1"
        lyric_repair_recipe = normalized_recipe_mode == "persona-lyric-repair"
        aligned_suno_recipe = normalized_recipe_mode == "aligned-suno"
        concert_remaster_recipe = normalized_recipe_mode == "concert-remaster-paired"
        persona_v11_recipe = normalized_recipe_mode in DIRECT_GUIDED_RECIPE_MODES
        total_requested_run_epochs = max(1, int(total_epochs))
        normalized_start_phase = self._normalize_curriculum_phase(start_phase)
        train_loader, val_loader, stats, train_entries, val_entries, filter_metadata = self._build_loaders(
            dataset_dir=dataset_dir,
            max_frames=max_frames,
            batch_size=batch_size,
            num_workers=num_workers,
            prefetch_factor=prefetch_factor,
            persona_v11_recipe=persona_v11_recipe,
        )
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        precision_mode = str(hardware_profile.get("precision_mode", "fp32"))
        autocast_dtype = torch.float32
        use_grad_scaler = False
        resume_checkpoint: Dict[str, object] = {}
        resume_epoch = 0
        reformed_legacy_checkpoint = False
        legacy_teacher_model: Optional[nn.Module] = None
        resolved_resume_checkpoint = Path(str(resume_checkpoint_path)) if resume_checkpoint_path else None
        if resolved_resume_checkpoint is not None:
            if not resolved_resume_checkpoint.exists():
                raise FileNotFoundError(f"Resume checkpoint was not found: {resolved_resume_checkpoint}")
            resume_checkpoint = torch.load(resolved_resume_checkpoint, map_location="cpu")
            resume_epoch = max(0, int(resume_checkpoint.get("epoch", 0)))
            if persona_v11_recipe and not self._is_persona_v11_recipe(resume_checkpoint.get("config", {})):
                try:
                    legacy_teacher_model, _ = self._load_model_from_checkpoint(
                        resolved_resume_checkpoint,
                        device=device,
                    )
                    legacy_teacher_model.eval()
                    for parameter in legacy_teacher_model.parameters():
                        parameter.requires_grad_(False)
                except Exception:
                    legacy_teacher_model = None
                source_model_config = dict(dict(resume_checkpoint.get("config", {}) or {}).get("model", {}) or {})
                reform_model_config = dict(hardware_profile.get("model", {}) or {})
                for key in ("d_model", "n_heads", "n_layers", "dropout"):
                    if key in source_model_config and source_model_config.get(key) is not None:
                        reform_model_config[key] = source_model_config.get(key)
                reform_model_config.update(
                    {
                        "persona_version": "v1.1",
                        "content_dim": int(stats.get("nar_content_dim", PERSONA_V11_CONTENT_DIM) or PERSONA_V11_CONTENT_DIM),
                    }
                )
                reform_target_config = {
                    "recipe_mode": normalized_recipe_mode,
                    "training_mode": (
                        "concert-paired-mapper-v1.1"
                        if concert_remaster_recipe
                        else ("suno-aligned-mapper-v1.1" if aligned_suno_recipe else "persona-paired-mapper-v1.1")
                    ),
                    "guide_conditioning": False,
                    "voice_signature_dim": VOICE_SIGNATURE_DIM,
                    "model": reform_model_config,
                }
                resume_checkpoint = self._reform_checkpoint_for_persona_v11(
                    resume_checkpoint,
                    target_config=reform_target_config,
                )
                resume_epoch = 0
                reformed_legacy_checkpoint = True
        curriculum_schedule = self._resolve_curriculum_schedule(
            start_phase=normalized_start_phase,
            resume_epoch=resume_epoch,
            warmup_stage_epochs=warmup_stage_epochs,
            bridge_stage_epochs=bridge_stage_epochs,
            full_diversity_stage_epochs=full_diversity_stage_epochs,
            general_refine_stage_epochs=general_refine_stage_epochs,
            target_fit_stage_epochs=target_fit_stage_epochs,
            post_process_stage_epochs=post_process_stage_epochs,
        )
        warmup_end_epoch = int(curriculum_schedule["warmup_end_epoch"])
        bridge_end_epoch = int(curriculum_schedule["bridge_end_epoch"])
        full_diversity_end_epoch = int(curriculum_schedule["full_diversity_end_epoch"])
        general_refine_end_epoch = int(curriculum_schedule["general_refine_end_epoch"])
        post_process_enabled = False
        target_end_epoch = max(resume_epoch, 0) + int(total_requested_run_epochs)
        base_target_end_epoch = target_end_epoch
        base_requested_run_epochs = max(0, int(base_target_end_epoch - resume_epoch))
        if device.type == "cuda":
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            torch.backends.cudnn.benchmark = True
            if hasattr(torch.backends.cuda, "enable_flash_sdp"):
                try:
                    torch.backends.cuda.enable_flash_sdp(True)
                    torch.backends.cuda.enable_mem_efficient_sdp(True)
                    torch.backends.cuda.enable_math_sdp(True)
                except Exception:
                    pass
            try:
                torch.set_float32_matmul_precision("high")
            except Exception:
                pass
            if precision_mode == "bf16":
                autocast_dtype = torch.bfloat16
                use_grad_scaler = False
            elif precision_mode == "fp16":
                autocast_dtype = torch.float16
                use_grad_scaler = True
        model_config = dict(hardware_profile.get("model", {}))
        model_config.setdefault(
            "content_dim",
            int(stats.get("nar_content_dim", PERSONA_V11_CONTENT_DIM) or PERSONA_V11_CONTENT_DIM),
        )
        model_config.setdefault("guide_residual_refinement", False if persona_v11_recipe else True)
        model_config.setdefault("guide_residual_scale", 0.72 if (lyric_repair_recipe or persona_v11_recipe) else 0.8)
        model_config.setdefault("stability_refine_enabled", False)
        model_config.setdefault("stability_refine_scale", 0.0)
        model_config.setdefault(
            "off_lyric_guide_floor",
            0.02 if (lyric_repair_recipe or persona_v11_recipe) else 0.05,
        )
        model_config.setdefault("guide_mix_floor", 0.03 if lyric_repair_recipe else 0.04)
        model_config.setdefault("guide_mix_ceiling", 0.38 if lyric_repair_recipe else 0.45)
        model_config.setdefault("contextual_voice_scale", 0.22 if lyric_repair_recipe else (0.34 if persona_v11_recipe else 0.28))
        model_config.setdefault("use_guide_mel_conditioning", bool(persona_v11_recipe))
        model_config.setdefault("use_voice_prototype_conditioning", False if persona_v11_recipe else True)
        model_config.setdefault("guide_condition_scale", 0.68 if persona_v11_recipe else 0.55)
        model_config.setdefault("persona_version", "v1.1" if persona_v11_recipe else "v1.0")
        if persona_v11_recipe:
            model_config.setdefault("guide_context_layers", 2)
            model_config.setdefault("guide_context_scale", 0.42)
            model_config.setdefault("guide_delta_scale", 0.55)
            model_config.setdefault("coherence_refine_scale", 0.22)
        config: Dict[str, object] = {
            "sample_rate": SAMPLE_RATE,
            "hop_length": HOP_LENGTH,
            "n_fft": N_FFT,
            "win_length": WIN_LENGTH,
            "n_mels": N_MELS,
            "voice_signature_dim": VOICE_SIGNATURE_DIM,
            "voice_signature_bands": VOICE_SIGNATURE_BANDS,
            "max_frames": max_frames,
            "batch_size": int(batch_size),
            "total_epochs": int(max(base_requested_run_epochs, 1)),
            "phone_tokens": list(PHONE_TOKENS),
            "recipe_mode": normalized_recipe_mode,
            "num_workers": int(num_workers),
            "prefetch_factor": int(prefetch_factor),
            "gpu_memory_gb": float(hardware_profile.get("gpu_memory_gb", 0.0)),
            "model": model_config,
            "training_mode": (
                "persona-lyric-repair-v1"
                if lyric_repair_recipe
                else (
                    "concert-paired-mapper-v1.1"
                    if concert_remaster_recipe
                    else (
                        "suno-aligned-mapper-v1.1"
                        if aligned_suno_recipe
                        else ("persona-paired-mapper-v1.1" if persona_v11_recipe else "persona-paired-regeneration-v1")
                    )
                )
            ),
            "guide_conditioning": False if persona_v11_recipe else True,
            "window_strategy": (
                "base-identity-plus-paired-depersona-dense-lyric-repair-windows"
                if lyric_repair_recipe
                else (
                    "concert-source-plus-cd-target-paired-content-f0-beat-truth-windows-v1.1"
                    if concert_remaster_recipe
                    else (
                        "base-identity-plus-paired-suno-target-content-f0-beat-truth-windows-v1.1"
                        if aligned_suno_recipe
                        else "base-identity-plus-paired-content-f0-beat-truth-windows-v1.1"
                        if persona_v11_recipe
                        else "base-identity-plus-paired-depersona-contextual-word-windows"
                    )
                )
            ),
            "alignment_mode": (
                "lyric-guided-loose-repair-v1"
                if lyric_repair_recipe
                else (
                    "dtw-warped-concert-cd-conditioning-v1.1"
                    if concert_remaster_recipe
                    else (
                        "dtw-warped-suno-target-content-f0-beat-conditioning-v1.1"
                        if aligned_suno_recipe
                        else ("dtw-warped-content-f0-beat-conditioning-v1.1" if persona_v11_recipe else "dtw-warped-guide-conditioning-v1")
                    )
                )
            ),
            "precision_mode": precision_mode,
            "fused_optimizer": bool(hardware_profile.get("use_fused_adamw", False)),
            "pair_filtering": dict(filter_metadata),
            "repair_strategy": (
                {
                    "enabled": True,
                    "micro_window_frames": 28,
                    "micro_window_hop": 6,
                    "micro_window_pad": 6,
                    "micro_window_max": 768,
                    "micro_candidate_count": 4,
                    "dense_blend": 0.84,
                    "guide_anchor_blend": 0.06,
                    "off_lyric_floor": 0.01,
                    "wave_gate_smoothing_frames": 4,
                }
                if lyric_repair_recipe
                else {"enabled": False}
            ),
            "curriculum": {
                "enabled": True,
                "phases": [
                    {"name": "warm-up", "fraction": 0.35},
                    {"name": "curriculum-bridge", "fraction": 0.7},
                    {"name": "full-diversity", "fraction": 1.0},
                    {"name": "general-refine", "fraction": 1.0},
                ],
                "warmup_stage_epochs": int(curriculum_schedule["warmup_stage_epochs"]),
                "bridge_stage_epochs": int(curriculum_schedule["bridge_stage_epochs"]),
                "full_diversity_stage_epochs": int(curriculum_schedule["full_diversity_stage_epochs"]),
                "general_refine_stage_epochs": int(curriculum_schedule["general_refine_stage_epochs"]),
                "anchor_epoch": int(curriculum_schedule["anchor_epoch"]),
                "warmup_end_epoch": int(warmup_end_epoch),
                "bridge_end_epoch": int(bridge_end_epoch),
                "full_diversity_end_epoch": int(full_diversity_end_epoch),
                "general_refine_end_epoch": int(general_refine_end_epoch),
                "start_phase": normalized_start_phase,
            },
            "loss_weights": {
                "mel": 1.0,
                "stft": 1.0,
                "phase": 0.28,
                "f0": 0.36 if persona_v11_recipe else 0.3,
                "delta_mel": 1.0,
                "accel_mel": 0.5,
                "smooth_mel": 0.3,
                "high_band": 0.48 if persona_v11_recipe else 0.4,
                "delta_f0": 0.30 if persona_v11_recipe else 0.24,
                "vuv": 0.2,
                "phones": 0.22 if lyric_repair_recipe else (0.0 if aligned_suno_recipe else (0.18 if persona_v11_recipe else 0.32)),
                "voice": 0.54 if persona_v11_recipe else 0.34,
                "prototype": 0.0 if persona_v11_recipe else 0.22,
                "silence": 0.42 if lyric_repair_recipe else (0.30 if persona_v11_recipe else 0.24),
                "target_quality": 0.36 if persona_v11_recipe else 0.16,
                "guide_anchor": 0.24 if lyric_repair_recipe else 0.0,
                "guide_transition": 0.16 if lyric_repair_recipe else 0.0,
                "guide_delta": 0.0,
                "guide_delta_transition": 0.0,
                "guide_delta_smooth": 0.0,
                "teacher": 0.05 if persona_v11_recipe else 0.0,
                "identity_margin": 0.44 if persona_v11_recipe else 0.0,
                "identity_margin_target": 0.10 if persona_v11_recipe else 0.0,
                "guide_residual": 0.76 if persona_v11_recipe else 0.0,
                "guide_residual_delta": 0.48 if persona_v11_recipe else 0.0,
            },
        }
        if resume_checkpoint:
            checkpoint_config = dict(resume_checkpoint.get("config", {}))
            merged_config = dict(checkpoint_config)
            merged_config.update(
                {
                    "batch_size": int(batch_size),
                    "total_epochs": int(max(base_requested_run_epochs, 1)),
                    "num_workers": int(num_workers),
                    "prefetch_factor": int(prefetch_factor),
                    "gpu_memory_gb": float(hardware_profile.get("gpu_memory_gb", 0.0)),
                    "precision_mode": precision_mode,
                    "fused_optimizer": bool(hardware_profile.get("use_fused_adamw", False)),
                    "pair_filtering": dict(filter_metadata),
                }
            )
            merged_curriculum = dict(merged_config.get("curriculum", {}))
            merged_curriculum.update(
                {
                    "enabled": True,
                    "warmup_stage_epochs": int(curriculum_schedule["warmup_stage_epochs"]),
                    "bridge_stage_epochs": int(curriculum_schedule["bridge_stage_epochs"]),
                    "full_diversity_stage_epochs": int(curriculum_schedule["full_diversity_stage_epochs"]),
                    "general_refine_stage_epochs": int(curriculum_schedule["general_refine_stage_epochs"]),
                    "anchor_epoch": int(curriculum_schedule["anchor_epoch"]),
                    "warmup_end_epoch": int(warmup_end_epoch),
                    "bridge_end_epoch": int(bridge_end_epoch),
                    "full_diversity_end_epoch": int(full_diversity_end_epoch),
                    "general_refine_end_epoch": int(general_refine_end_epoch),
                    "start_phase": normalized_start_phase,
                }
            )
            merged_config["curriculum"] = merged_curriculum
            merged_model = dict(merged_config.get("model", {}) or {})
            merged_model.update(
                {
                    "content_dim": int(model_config.get("content_dim", merged_model.get("content_dim", PERSONA_V11_CONTENT_DIM)) or PERSONA_V11_CONTENT_DIM),
                    "guide_residual_refinement": bool(model_config.get("guide_residual_refinement", True)),
                    "guide_residual_scale": float(model_config.get("guide_residual_scale", 0.8) or 0.8),
                    "stability_refine_enabled": False,
                    "stability_refine_scale": 0.0,
                    "off_lyric_guide_floor": float(model_config.get("off_lyric_guide_floor", 0.05) or 0.05),
                    "guide_mix_floor": float(model_config.get("guide_mix_floor", 0.04) or 0.04),
                    "guide_mix_ceiling": float(model_config.get("guide_mix_ceiling", 0.45) or 0.45),
                    "contextual_voice_scale": float(model_config.get("contextual_voice_scale", 0.28) or 0.28),
                    "persona_version": str(model_config.get("persona_version", merged_model.get("persona_version", "v1.0")) or "v1.0"),
                    "guide_context_layers": int(model_config.get("guide_context_layers", merged_model.get("guide_context_layers", 2)) or 2),
                    "guide_context_scale": float(model_config.get("guide_context_scale", merged_model.get("guide_context_scale", 0.42)) or 0.42),
                    "guide_delta_scale": float(model_config.get("guide_delta_scale", merged_model.get("guide_delta_scale", 0.55)) or 0.55),
                    "coherence_refine_scale": float(model_config.get("coherence_refine_scale", merged_model.get("coherence_refine_scale", 0.22)) or 0.22),
                    "use_guide_mel_conditioning": bool(
                        model_config.get(
                            "use_guide_mel_conditioning",
                            merged_model.get("use_guide_mel_conditioning", persona_v11_recipe),
                        )
                    ),
                    "use_voice_prototype_conditioning": bool(
                        model_config.get(
                            "use_voice_prototype_conditioning",
                            merged_model.get("use_voice_prototype_conditioning", not persona_v11_recipe),
                        )
                    ),
                    "guide_condition_scale": float(
                        model_config.get(
                            "guide_condition_scale",
                            merged_model.get("guide_condition_scale", 0.68 if persona_v11_recipe else 0.55),
                        )
                        or (0.68 if persona_v11_recipe else 0.55)
                    ),
                }
            )
            merged_config["model"] = merged_model
            merged_config["recipe_mode"] = normalized_recipe_mode
            merged_config["training_mode"] = str(config.get("training_mode", merged_config.get("training_mode", "")) or "")
            merged_config["guide_conditioning"] = False if persona_v11_recipe else True
            merged_config["window_strategy"] = str(config.get("window_strategy", merged_config.get("window_strategy", "")) or "")
            merged_config["alignment_mode"] = str(config.get("alignment_mode", merged_config.get("alignment_mode", "")) or "")
            merged_config["repair_strategy"] = dict(config.get("repair_strategy", merged_config.get("repair_strategy", {})) or {})
            merged_config["loss_weights"] = dict(config.get("loss_weights", merged_config.get("loss_weights", {})) or {})
            config = merged_config
        model = self._instantiate_model(config).to(device)
        compiled_model = False
        compile_allowed = (
            device.type == "cuda"
            and bool(hardware_profile.get("compile_model", False))
            and hasattr(torch, "compile")
            and os.name == "nt"
        )
        if compile_allowed:
            try:
                model = torch.compile(model, mode="reduce-overhead")
                compiled_model = True
            except Exception:
                pass
        optimizer_kwargs = {
            "lr": float(hardware_profile.get("lr", 2e-4)),
            "betas": (0.9, 0.99),
            "weight_decay": 1e-4,
        }
        if device.type == "cuda" and bool(hardware_profile.get("use_fused_adamw", False)):
            optimizer_kwargs["fused"] = True
        try:
            optimizer = torch.optim.AdamW(model.parameters(), **optimizer_kwargs)
        except TypeError:
            optimizer_kwargs.pop("fused", None)
            optimizer = torch.optim.AdamW(model.parameters(), **optimizer_kwargs)
        lr_warmup_epochs = max(4, min(120, int(round(max(base_requested_run_epochs, 1) * 0.05))))
        if normalized_start_phase in {"curriculum-bridge", "full-diversity"} or resume_epoch >= warmup_end_epoch:
            lr_warmup_epochs = 0
        if normalized_start_phase == "general-refine" or base_requested_run_epochs <= 0:
            lr_warmup_epochs = 0
        min_lr_scale = 0.12

        def lr_lambda(epoch_index: int) -> float:
            epoch_number = epoch_index + 1
            if lr_warmup_epochs > 0 and epoch_number <= lr_warmup_epochs:
                return max(0.35, epoch_number / float(max(lr_warmup_epochs, 1)))
            progress = (epoch_number - lr_warmup_epochs) / float(max(base_requested_run_epochs - lr_warmup_epochs, 1))
            cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
            return max(min_lr_scale, min_lr_scale + ((1.0 - min_lr_scale) * cosine))

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
        scaler = torch.cuda.amp.GradScaler(enabled=bool(device.type == "cuda" and use_grad_scaler))
        resume_report_payload: Dict[str, object] = {}
        resolved_resume_report = Path(str(resume_report_path)) if resume_report_path else None
        if resolved_resume_report is not None and resolved_resume_report.exists():
            try:
                loaded_report = json.loads(resolved_resume_report.read_text(encoding="utf-8"))
                if isinstance(loaded_report, dict):
                    resume_report_payload = loaded_report
            except Exception:
                resume_report_payload = {}
        if reformed_legacy_checkpoint:
            resume_report_payload = {}
        config_path = output_dir / "guided_regeneration_config.json"
        config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")

        history: List[Dict[str, object]] = []
        if resolved_resume_checkpoint is not None:
            resume_history_path = resolved_resume_checkpoint.parent / "guided_regeneration_history.json"
            if resume_history_path.exists():
                try:
                    loaded_history = json.loads(resume_history_path.read_text(encoding="utf-8"))
                    if isinstance(loaded_history, list):
                        history = [dict(entry) for entry in loaded_history if isinstance(entry, dict)]
                except Exception:
                    history = []
        best_val_loss = float(resume_checkpoint.get("best_val_loss", resume_report_payload.get("best_val_total", float("inf"))))
        best_val_mel = float(resume_report_payload.get("best_val_l1", best_val_loss if math.isfinite(best_val_loss) else float("inf")))
        best_epoch = int(resume_report_payload.get("best_epoch", resume_epoch if resume_epoch > 0 else 0))
        best_phone_accuracy = float(resume_report_payload.get("best_phone_accuracy", 0.0))
        best_lyric_phone_accuracy = float(resume_report_payload.get("best_lyric_phone_accuracy", 0.0))
        best_vuv_accuracy = float(resume_report_payload.get("best_vuv_accuracy", 0.0))
        resume_checkpoint_epoch = int(resume_checkpoint.get("epoch", resume_epoch or 0)) if resume_checkpoint else 0
        resumed_from_best_checkpoint = bool(
            resume_checkpoint
            and resume_checkpoint_epoch > 0
            and best_epoch > 0
            and resume_checkpoint_epoch == best_epoch
            and normalized_start_phase in {"warm-up", "curriculum-bridge", "full-diversity"}
        )
        best_target_quality = float(
            resume_report_payload.get(
                "best_target_quality",
                resume_checkpoint.get("best_target_quality", float("inf")),
            )
        )
        if not math.isfinite(best_target_quality):
            best_target_quality = float("inf")
        if resume_checkpoint:
            model.load_state_dict(dict(resume_checkpoint.get("model_state", {})), strict=False)
            optimizer_state = resume_checkpoint.get("optimizer_state")
            if not resumed_from_best_checkpoint and isinstance(optimizer_state, dict):
                try:
                    optimizer.load_state_dict(optimizer_state)
                except Exception:
                    pass
            scheduler_state = resume_checkpoint.get("scheduler_state")
            if not resumed_from_best_checkpoint and isinstance(scheduler_state, dict):
                try:
                    scheduler.load_state_dict(scheduler_state)
                except Exception:
                    pass
            if resumed_from_best_checkpoint:
                # When we explicitly resume from the saved best builder checkpoint,
                # restarting with the old optimizer momentum and a fresh long manual-stop
                # schedule can immediately push the model far away from that good state.
                # Re-anchor with a smaller learning rate and clean optimizer state.
                resume_anchor_lr = min(
                    float(optimizer_kwargs.get("lr", 2e-4)),
                    max(2e-5, float(optimizer_kwargs.get("lr", 2e-4)) * 0.35),
                )
                for param_group in optimizer.param_groups:
                    param_group["lr"] = resume_anchor_lr
                    param_group["initial_lr"] = resume_anchor_lr
                if hasattr(scheduler, "base_lrs"):
                    scheduler.base_lrs = [resume_anchor_lr for _ in getattr(scheduler, "base_lrs", optimizer.param_groups)]
        best_checkpoint_path = output_dir / "guided_regeneration_best.pt"
        latest_checkpoint_path = output_dir / "guided_regeneration_latest.pt"
        stopped_early = False
        last_epoch = max(resume_epoch, int(resume_report_payload.get("last_epoch", resume_epoch)))
        if resume_checkpoint:
            self._save_checkpoint(
                path=latest_checkpoint_path,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=resume_checkpoint_epoch,
                best_val_loss=best_val_loss,
                best_target_quality=best_target_quality,
                config=config,
            )
            if resumed_from_best_checkpoint:
                self._save_checkpoint(
                    path=best_checkpoint_path,
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    epoch=resume_checkpoint_epoch,
                    best_val_loss=best_val_loss,
                    best_target_quality=best_target_quality,
                    config=config,
                )
        if resume_checkpoint and base_requested_run_epochs <= 0:
            if not best_checkpoint_path.exists():
                self._save_checkpoint(
                    path=best_checkpoint_path,
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    epoch=resume_checkpoint_epoch,
                    best_val_loss=best_val_loss,
                    best_target_quality=best_target_quality,
                    config=config,
                )
        start_time = time.time()
        hardware_summary = (
            f"{device.type} | {float(hardware_profile.get('gpu_memory_gb', 0.0)):.1f}GB | "
            f"{precision_mode} | {'fused AdamW' if optimizer_kwargs.get('fused') else 'AdamW'} | "
            f"{'compiled' if compiled_model else 'eager'} | "
            f"frames {max_frames} | workers {num_workers}"
        )
        loader_safe_mode = False
        general_refine_trainable = False
        self._set_general_refine_trainable(model, enabled=False)
        resume_note = (
            f" | resuming from epoch {resume_epoch} | start phase {normalized_start_phase}"
            if resume_checkpoint
            else ""
        )
        if reformed_legacy_checkpoint:
            resume_note = " | reformed legacy Persona checkpoint into v1.1 | start phase " + normalized_start_phase
        curriculum_summary = (
            f"W {int(curriculum_schedule['warmup_stage_epochs'])} | "
            f"B {int(curriculum_schedule['bridge_stage_epochs'])} | "
            f"F {int(curriculum_schedule['full_diversity_stage_epochs'])} | "
            f"G {int(curriculum_schedule['general_refine_stage_epochs'])}"
        )

        update_status(
            "guided-svs-train",
            (
                (
                    "Training the lyric-only Persona repair converter..."
                    if lyric_repair_recipe
                    else (
                        "Training the concert remaster direct frame mapper..."
                        if concert_remaster_recipe
                        else (
                            "Training the aligned SUNO -> target direct mapper..."
                            if aligned_suno_recipe
                            else "Training the Persona v1.1 direct frame mapper..."
                            if persona_v11_recipe
                            else "Training the paired voice-builder regenerator..."
                        )
                    )
                )
                if base_requested_run_epochs > 0
                else (
                    "Preparing general refinement from the saved concert remaster checkpoint..."
                    if concert_remaster_recipe
                    else "Preparing general refinement from the saved voice-builder checkpoint..."
                )
            ),
            (
                f"Windows {len(train_entries)} train / {len(val_entries)} val | "
                f"filtered out {int(filter_metadata.get('dropped_total', 0))} weak slices | "
                f"{hardware_summary} | "
                f"{'dense lyric repair + guide anchoring + aggressive off-lyric suppression' if lyric_repair_recipe else ('concert/CD truth-pair mapping with stable F0 + beat-phase conditioning' if concert_remaster_recipe else ('DTW-warped SUNO guide content + stable F0 + beat-phase frame mapping' if aligned_suno_recipe else ('DTW-warped guide content + stable F0 + beat-phase frame mapping' if persona_v11_recipe else 'DTW-warped guides + lyric supervision + curriculum')))} "
                f"({curriculum_summary}){resume_note}"
            ),
            6,
        )

        for epoch in range(max(1, resume_epoch + 1), max(1, base_target_end_epoch) + 1):
            if cancel_event is not None and bool(getattr(cancel_event, "is_set", lambda: False)()):
                stopped_early = True
                break

            active_train_entries, curriculum_state = self._select_curriculum_entries(
                entries=train_entries,
                epoch=epoch,
                total_epochs=max(base_target_end_epoch, 1),
                warmup_end_epoch=warmup_end_epoch,
                bridge_end_epoch=bridge_end_epoch,
                full_diversity_end_epoch=full_diversity_end_epoch,
                general_refine_end_epoch=general_refine_end_epoch,
            )
            phase_name = str(curriculum_state.get("name", "full-diversity"))
            if phase_name == "general-refine" and not general_refine_trainable:
                self._set_general_refine_trainable(model, enabled=True)
                general_refine_trainable = True
                general_refine_lr_cap = float(hardware_profile.get("lr", 2e-4)) * 0.45
                for group in optimizer.param_groups:
                    group["lr"] = min(float(group.get("lr", general_refine_lr_cap)), general_refine_lr_cap)
            elif phase_name != "general-refine" and general_refine_trainable:
                self._set_general_refine_trainable(model, enabled=False)
                general_refine_trainable = False
            phase_loss_weights = self._resolve_phase_loss_weights(config, phase_name)
            while True:
                active_worker_count = 0 if loader_safe_mode else num_workers
                active_prefetch_factor = 2 if loader_safe_mode else prefetch_factor
                if loader_safe_mode:
                    update_status(
                        "guided-svs-train",
                        (
                            "Training the concert remaster direct frame mapper..."
                            if concert_remaster_recipe
                            else (
                                "Training the aligned SUNO -> target direct mapper..."
                                if aligned_suno_recipe
                                else "Training the Persona v1.1 direct frame mapper..."
                                if persona_v11_recipe
                                else "Training the paired voice-builder regenerator..."
                            )
                        ),
                        "Data loading safe mode is active after a loader fault. Training is continuing with single-process loading for stability.",
                        6,
                    )
                train_loader = self._create_loader(
                    entries=active_train_entries,
                    dataset_dir=dataset_dir,
                    stats=stats,
                    max_frames=max_frames,
                    batch_size=batch_size,
                    num_workers=active_worker_count,
                    prefetch_factor=active_prefetch_factor,
                    persona_v11_recipe=persona_v11_recipe,
                    random_crop=True,
                    shuffle=True,
                )
                val_loader_epoch = (
                    self._create_loader(
                        entries=val_entries,
                        dataset_dir=dataset_dir,
                        stats=stats,
                        max_frames=max_frames,
                        batch_size=batch_size,
                        num_workers=active_worker_count,
                        prefetch_factor=active_prefetch_factor,
                        persona_v11_recipe=persona_v11_recipe,
                        random_crop=False,
                        shuffle=False,
                    )
                    if loader_safe_mode
                    else val_loader
                )
                model.train()
                train_loss_sums = {
                    "total": 0.0,
                    "mel": 0.0,
                    "stft": 0.0,
                    "phase": 0.0,
                    "f0": 0.0,
                    "delta_mel": 0.0,
                    "accel_mel": 0.0,
                    "smooth_mel": 0.0,
                    "high_band": 0.0,
                    "delta_f0": 0.0,
                    "vuv": 0.0,
                    "phones": 0.0,
                    "voice": 0.0,
                    "prototype": 0.0,
                    "silence": 0.0,
                    "target_quality": 0.0,
                    "guide_anchor": 0.0,
                    "guide_transition": 0.0,
                    "flutter": 0.0,
                    "stability": 0.0,
                }
                train_metric_counts = {
                    "phone_correct": 0.0,
                    "phone_total": 0.0,
                    "lyric_phone_correct": 0.0,
                    "lyric_phone_total": 0.0,
                    "vuv_correct": 0.0,
                    "vuv_total": 0.0,
                }
                train_batches = 0
                train_frame_count = 0
                train_sample_count = 0
                epoch_started = time.time()
                try:
                    for batch in train_loader:
                        if cancel_event is not None and bool(getattr(cancel_event, "is_set", lambda: False)()):
                            stopped_early = True
                            break
                        batch = {key: value.to(device) for key, value in batch.items()}
                        optimizer.zero_grad(set_to_none=True)
                        autocast_context = (
                            torch.autocast(device_type="cuda", dtype=autocast_dtype)
                            if device.type == "cuda" and precision_mode in {"bf16", "fp16"}
                            else nullcontext()
                        )
                        with autocast_context:
                            outputs = self._predict_outputs(
                                model,
                                guide_mel=batch["guide_mel"],
                                phone_ids=batch["phone_ids"],
                                log_f0=batch["log_f0"],
                                vuv=batch["vuv"],
                                energy=batch["energy"],
                                lengths=batch["lengths"],
                                content=batch.get("content"),
                                beat_phase=batch.get("beat_phase"),
                                voice_prototype=batch["voice_prototype"],
                                lyric_mask=batch["lyric_mask"],
                            )
                            teacher_outputs = None
                            if persona_v11_recipe and legacy_teacher_model is not None:
                                teacher_outputs = self._predict_outputs(
                                    legacy_teacher_model,
                                    guide_mel=batch["guide_mel"],
                                    phone_ids=batch["phone_ids"],
                                    log_f0=batch["log_f0"],
                                    vuv=batch["vuv"],
                                    energy=batch["energy"],
                                    lengths=batch["lengths"],
                                    voice_prototype=batch["voice_prototype"],
                                    lyric_mask=batch["lyric_mask"],
                                )
                            loss_terms = (
                                self._compute_general_refine_loss_terms(
                                    outputs=outputs,
                                    batch=batch,
                                    loss_weights=phase_loss_weights,
                                )
                                if phase_name == "general-refine"
                                else self._compute_lyric_repair_loss_terms(
                                    outputs=outputs,
                                    batch=batch,
                                    loss_weights=phase_loss_weights,
                                )
                                if lyric_repair_recipe
                                else self._compute_persona_v11_loss_terms(
                                    outputs=outputs,
                                    batch=batch,
                                    loss_weights=phase_loss_weights,
                                    teacher_outputs=teacher_outputs,
                                )
                                if persona_v11_recipe
                                else self._compute_loss_terms(
                                    outputs=outputs,
                                    batch=batch,
                                    loss_weights=phase_loss_weights,
                                )
                            )
                            loss = loss_terms["total"]
                        if use_grad_scaler:
                            scaler.scale(loss).backward()
                            scaler.unscale_(optimizer)
                        else:
                            loss.backward()
                        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                        if use_grad_scaler:
                            scaler.step(optimizer)
                            scaler.update()
                        else:
                            optimizer.step()
                        for loss_name in train_loss_sums:
                            loss_value = loss_terms.get(loss_name)
                            if isinstance(loss_value, torch.Tensor):
                                train_loss_sums[loss_name] += float(loss_value.detach().cpu().item())
                        phone_counts = self._phone_accuracy_counts(
                            outputs["phone_logits"],
                            batch["phone_ids"],
                            batch["lengths"],
                            batch["lyric_mask"],
                        )
                        vuv_counts = self._vuv_accuracy_counts(
                            outputs["target_vuv_logits"],
                            batch["target_vuv"],
                            batch["lengths"],
                        )
                        for metric_name, metric_value in {**phone_counts, **vuv_counts}.items():
                            train_metric_counts[metric_name] += float(metric_value)
                        train_batches += 1
                        train_frame_count += int(batch["lengths"].sum().detach().cpu().item())
                        train_sample_count += int(batch["lengths"].shape[0])

                    model.eval()
                    val_loss_sums = {
                        "total": 0.0,
                        "mel": 0.0,
                        "stft": 0.0,
                        "phase": 0.0,
                        "f0": 0.0,
                        "delta_mel": 0.0,
                        "accel_mel": 0.0,
                        "smooth_mel": 0.0,
                        "high_band": 0.0,
                        "delta_f0": 0.0,
                        "vuv": 0.0,
                        "phones": 0.0,
                        "voice": 0.0,
                        "prototype": 0.0,
                        "silence": 0.0,
                        "target_quality": 0.0,
                        "guide_anchor": 0.0,
                        "guide_transition": 0.0,
                        "flutter": 0.0,
                        "stability": 0.0,
                    }
                    val_metric_counts = {
                        "phone_correct": 0.0,
                        "phone_total": 0.0,
                        "lyric_phone_correct": 0.0,
                        "lyric_phone_total": 0.0,
                        "vuv_correct": 0.0,
                        "vuv_total": 0.0,
                    }
                    val_batches = 0
                    val_frame_count = 0
                    with torch.no_grad():
                        for batch in val_loader_epoch:
                            batch = {key: value.to(device) for key, value in batch.items()}
                            outputs = self._predict_outputs(
                                model,
                                guide_mel=batch["guide_mel"],
                                phone_ids=batch["phone_ids"],
                                log_f0=batch["log_f0"],
                                vuv=batch["vuv"],
                                energy=batch["energy"],
                                lengths=batch["lengths"],
                                content=batch.get("content"),
                                beat_phase=batch.get("beat_phase"),
                                voice_prototype=batch["voice_prototype"],
                                lyric_mask=batch["lyric_mask"],
                            )
                            teacher_outputs = None
                            if persona_v11_recipe and legacy_teacher_model is not None:
                                teacher_outputs = self._predict_outputs(
                                    legacy_teacher_model,
                                    guide_mel=batch["guide_mel"],
                                    phone_ids=batch["phone_ids"],
                                    log_f0=batch["log_f0"],
                                    vuv=batch["vuv"],
                                    energy=batch["energy"],
                                    lengths=batch["lengths"],
                                    voice_prototype=batch["voice_prototype"],
                                    lyric_mask=batch["lyric_mask"],
                                )
                            loss_terms = (
                                self._compute_general_refine_loss_terms(
                                    outputs=outputs,
                                    batch=batch,
                                    loss_weights=phase_loss_weights,
                                )
                                if phase_name == "general-refine"
                                else self._compute_lyric_repair_loss_terms(
                                    outputs=outputs,
                                    batch=batch,
                                    loss_weights=phase_loss_weights,
                                )
                                if lyric_repair_recipe
                                else self._compute_persona_v11_loss_terms(
                                    outputs=outputs,
                                    batch=batch,
                                    loss_weights=phase_loss_weights,
                                    teacher_outputs=teacher_outputs,
                                )
                                if persona_v11_recipe
                                else self._compute_loss_terms(
                                    outputs=outputs,
                                    batch=batch,
                                    loss_weights=phase_loss_weights,
                                )
                            )
                            for loss_name in val_loss_sums:
                                loss_value = loss_terms.get(loss_name)
                                if isinstance(loss_value, torch.Tensor):
                                    val_loss_sums[loss_name] += float(loss_value.detach().cpu().item())
                            phone_counts = self._phone_accuracy_counts(
                                outputs["phone_logits"],
                                batch["phone_ids"],
                                batch["lengths"],
                                batch["lyric_mask"],
                            )
                            vuv_counts = self._vuv_accuracy_counts(
                                outputs["target_vuv_logits"],
                                batch["target_vuv"],
                                batch["lengths"],
                            )
                            for metric_name, metric_value in {**phone_counts, **vuv_counts}.items():
                                val_metric_counts[metric_name] += float(metric_value)
                            val_batches += 1
                            val_frame_count += int(batch["lengths"].sum().detach().cpu().item())
                    train_loader = None
                    val_loader_epoch = None
                    gc.collect()
                    break
                except RuntimeError as exc:
                    if (not self._is_loader_runtime_error(exc)) or loader_safe_mode:
                        raise
                    loader_safe_mode = True
                    update_status(
                        "guided-svs-train",
                        "Training loader became unstable, retrying in safe mode...",
                        "Switching to single-process data loading so the current run can continue uninterrupted on Vast.",
                        6,
                    )
                    if device.type == "cuda":
                        try:
                            torch.cuda.empty_cache()
                        except Exception:
                            pass

            last_epoch = epoch
            train_loss = train_loss_sums["total"] / max(train_batches, 1)
            val_loss = val_loss_sums["total"] / max(val_batches, 1)
            val_mel_loss = val_loss_sums["mel"] / max(val_batches, 1)
            train_target_quality = train_loss_sums["target_quality"] / max(train_batches, 1)
            val_target_quality = val_loss_sums["target_quality"] / max(val_batches, 1)
            train_phone_accuracy = train_metric_counts["phone_correct"] / max(train_metric_counts["phone_total"], 1.0)
            val_phone_accuracy = val_metric_counts["phone_correct"] / max(val_metric_counts["phone_total"], 1.0)
            train_lyric_phone_accuracy = train_metric_counts["lyric_phone_correct"] / max(train_metric_counts["lyric_phone_total"], 1.0)
            val_lyric_phone_accuracy = val_metric_counts["lyric_phone_correct"] / max(val_metric_counts["lyric_phone_total"], 1.0)
            train_vuv_accuracy = train_metric_counts["vuv_correct"] / max(train_metric_counts["vuv_total"], 1.0)
            val_vuv_accuracy = val_metric_counts["vuv_correct"] / max(val_metric_counts["vuv_total"], 1.0)
            improved_total = val_loss <= best_val_loss + 1e-8
            improved_target_quality = val_target_quality <= best_target_quality + 1e-8
            improved = improved_target_quality if phase_name == "general-refine" else improved_total
            if improved_total:
                best_val_loss = val_loss
                best_val_mel = val_mel_loss
                best_phone_accuracy = val_phone_accuracy
                best_lyric_phone_accuracy = val_lyric_phone_accuracy
                best_vuv_accuracy = val_vuv_accuracy
            if improved_target_quality:
                best_target_quality = val_target_quality
            if improved:
                best_epoch = epoch
            scheduler.step()
            epoch_duration = max(time.time() - epoch_started, 1e-6)
            train_frames_per_second = float(train_frame_count) / epoch_duration
            train_samples_per_second = float(train_sample_count) / epoch_duration
            epochs_since_best = max(0, epoch - max(best_epoch, epoch if improved else 0))
            quality_state = self._describe_quality_state(
                improved=improved,
                epochs_since_best=epochs_since_best,
                lyric_phone_accuracy=val_lyric_phone_accuracy,
                vuv_accuracy=val_vuv_accuracy,
                delta_mel_loss=(val_loss_sums["delta_mel"] / max(val_batches, 1)),
                phase_name=phase_name,
                target_quality_loss=val_target_quality,
            )
            history.append(
                {
                    "epoch": epoch,
                    "phase": phase_name,
                    "active_train_slices": int(curriculum_state.get("selected_count", len(active_train_entries))),
                    "train_total": round(train_loss, 6),
                    "train_mel": round(train_loss_sums["mel"] / max(train_batches, 1), 6),
                    "train_stft": round(train_loss_sums["stft"] / max(train_batches, 1), 6),
                    "train_phase": round(train_loss_sums["phase"] / max(train_batches, 1), 6),
                    "train_f0": round(train_loss_sums["f0"] / max(train_batches, 1), 6),
                    "train_delta_mel": round(train_loss_sums["delta_mel"] / max(train_batches, 1), 6),
                    "train_accel_mel": round(train_loss_sums["accel_mel"] / max(train_batches, 1), 6),
                    "train_smooth_mel": round(train_loss_sums["smooth_mel"] / max(train_batches, 1), 6),
                    "train_high_band": round(train_loss_sums["high_band"] / max(train_batches, 1), 6),
                    "train_delta_f0": round(train_loss_sums["delta_f0"] / max(train_batches, 1), 6),
                    "train_vuv": round(train_loss_sums["vuv"] / max(train_batches, 1), 6),
                    "train_phones": round(train_loss_sums["phones"] / max(train_batches, 1), 6),
                    "train_voice": round(train_loss_sums["voice"] / max(train_batches, 1), 6),
                    "train_prototype": round(train_loss_sums["prototype"] / max(train_batches, 1), 6),
                    "train_silence": round(train_loss_sums["silence"] / max(train_batches, 1), 6),
                    "train_target_quality": round(train_target_quality, 6),
                    "train_guide_anchor": round(train_loss_sums.get("guide_anchor", 0.0) / max(train_batches, 1), 6),
                    "train_guide_transition": round(train_loss_sums.get("guide_transition", 0.0) / max(train_batches, 1), 6),
                    "train_flutter": round(train_loss_sums.get("flutter", 0.0) / max(train_batches, 1), 6),
                    "train_stability": round(train_loss_sums.get("stability", 0.0) / max(train_batches, 1), 6),
                    "val_total": round(val_loss, 6),
                    "val_mel": round(val_loss_sums["mel"] / max(val_batches, 1), 6),
                    "val_stft": round(val_loss_sums["stft"] / max(val_batches, 1), 6),
                    "val_phase": round(val_loss_sums["phase"] / max(val_batches, 1), 6),
                    "val_f0": round(val_loss_sums["f0"] / max(val_batches, 1), 6),
                    "val_delta_mel": round(val_loss_sums["delta_mel"] / max(val_batches, 1), 6),
                    "val_accel_mel": round(val_loss_sums["accel_mel"] / max(val_batches, 1), 6),
                    "val_smooth_mel": round(val_loss_sums["smooth_mel"] / max(val_batches, 1), 6),
                    "val_high_band": round(val_loss_sums["high_band"] / max(val_batches, 1), 6),
                    "val_delta_f0": round(val_loss_sums["delta_f0"] / max(val_batches, 1), 6),
                    "val_vuv": round(val_loss_sums["vuv"] / max(val_batches, 1), 6),
                    "val_phones": round(val_loss_sums["phones"] / max(val_batches, 1), 6),
                    "val_voice": round(val_loss_sums["voice"] / max(val_batches, 1), 6),
                    "val_prototype": round(val_loss_sums["prototype"] / max(val_batches, 1), 6),
                    "val_silence": round(val_loss_sums["silence"] / max(val_batches, 1), 6),
                    "val_target_quality": round(val_target_quality, 6),
                    "val_guide_anchor": round(val_loss_sums.get("guide_anchor", 0.0) / max(val_batches, 1), 6),
                    "val_guide_transition": round(val_loss_sums.get("guide_transition", 0.0) / max(val_batches, 1), 6),
                    "val_flutter": round(val_loss_sums.get("flutter", 0.0) / max(val_batches, 1), 6),
                    "val_stability": round(val_loss_sums.get("stability", 0.0) / max(val_batches, 1), 6),
                    "train_phone_accuracy": round(train_phone_accuracy, 6),
                    "val_phone_accuracy": round(val_phone_accuracy, 6),
                    "train_lyric_phone_accuracy": round(train_lyric_phone_accuracy, 6),
                    "val_lyric_phone_accuracy": round(val_lyric_phone_accuracy, 6),
                    "train_vuv_accuracy": round(train_vuv_accuracy, 6),
                    "val_vuv_accuracy": round(val_vuv_accuracy, 6),
                    "train_l1": round(train_loss_sums["mel"] / max(train_batches, 1), 6),
                    "val_l1": round(val_loss_sums["mel"] / max(val_batches, 1), 6),
                    "best_val_l1": round(best_val_mel, 6),
                    "best_val_total": round(best_val_loss, 6),
                    "best_epoch": int(best_epoch),
                    "best_phone_accuracy": round(best_phone_accuracy, 6),
                    "best_lyric_phone_accuracy": round(best_lyric_phone_accuracy, 6),
                    "best_vuv_accuracy": round(best_vuv_accuracy, 6),
                    "best_target_quality": round(best_target_quality, 6) if math.isfinite(best_target_quality) else 0.0,
                    "epochs_since_best": int(epochs_since_best),
                    "quality_state": quality_state,
                    "epoch_seconds": round(epoch_duration, 3),
                    "train_frames_per_second": round(train_frames_per_second, 2),
                    "train_samples_per_second": round(train_samples_per_second, 2),
                    "val_frame_count": int(val_frame_count),
                    "learning_rate": round(float(optimizer.param_groups[0]["lr"]), 10),
                    "elapsed_seconds": round(max(time.time() - start_time, 0.0), 2),
                }
            )

            self._save_checkpoint(
                path=latest_checkpoint_path,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
                best_val_loss=best_val_loss,
                best_target_quality=best_target_quality,
                config=config,
            )
            if improved:
                self._save_checkpoint(
                    path=best_checkpoint_path,
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    epoch=epoch,
                    best_val_loss=best_val_loss,
                    best_target_quality=best_target_quality,
                    config=config,
                )
            checkpoint_publish_path = latest_checkpoint_path
            if epoch % max(1, int(save_every_epoch)) == 0:
                checkpoint_publish_path = output_dir / f"guided_regeneration_epoch_{epoch:04d}.pt"
                self._save_checkpoint(
                    path=checkpoint_publish_path,
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    epoch=epoch,
                    best_val_loss=best_val_loss,
                    best_target_quality=best_target_quality,
                    config=config,
                )
                if checkpoint_callback is not None:
                    try:
                        checkpoint_callback(
                            {
                                "epoch": int(epoch),
                                "checkpoint_path": str(checkpoint_publish_path),
                                "latest_checkpoint_path": str(latest_checkpoint_path),
                                "config_path": str(config_path),
                                "best_val_l1": float(best_val_mel),
                                "best_val_total": float(best_val_loss),
                                "best_target_quality": float(best_target_quality)
                                if math.isfinite(best_target_quality)
                                else 0.0,
                                "best_quality": float(best_target_quality)
                                if math.isfinite(best_target_quality)
                                else 0.0,
                                "best_epoch": int(best_epoch),
                                "best_phone_accuracy": float(best_phone_accuracy),
                                "best_lyric_phone_accuracy": float(best_lyric_phone_accuracy),
                                "best_vuv_accuracy": float(best_vuv_accuracy),
                                "plateau_epochs": int(max(0, epoch - best_epoch)),
                                "quality_summary": str(history[-1]["quality_state"]) if history else "",
                                "hardware_summary": hardware_summary,
                                "last_epoch": int(epoch),
                                "sample_count": int(len(train_entries) + len(val_entries)),
                                "recipe_mode": normalized_recipe_mode,
                            }
                        )
                    except Exception:
                        pass

            completed_run_epochs = max(1, epoch - resume_epoch)
            progress = min(82, max(10, int(round((completed_run_epochs / max(base_requested_run_epochs, 1)) * 82))))
            phase_extra_summary = ""
            if lyric_repair_recipe:
                phase_extra_summary = (
                    " | guide anchor "
                    f"{train_loss_sums.get('guide_anchor', 0.0) / max(train_batches, 1):.4f}/"
                    f"{val_loss_sums.get('guide_anchor', 0.0) / max(val_batches, 1):.4f}"
                )
            elif phase_name == "general-refine":
                phase_extra_summary = (
                    " | flutter "
                    f"{train_loss_sums.get('flutter', 0.0) / max(train_batches, 1):.4f}/"
                    f"{val_loss_sums.get('flutter', 0.0) / max(val_batches, 1):.4f}"
                    " | stability "
                    f"{train_loss_sums.get('stability', 0.0) / max(train_batches, 1):.4f}/"
                    f"{val_loss_sums.get('stability', 0.0) / max(val_batches, 1):.4f}"
                )
            update_status(
                "guided-svs-train",
                (
                    f"Voice-builder epoch {epoch}/{max(base_target_end_epoch, 1)} | "
                    f"{curriculum_state.get('name', 'full-diversity')} | {quality_state}"
                ),
                (
                    f"best epoch {best_epoch or epoch} | best total {best_val_loss:.4f} | "
                    f"train/val total {train_loss:.4f}/{val_loss:.4f} | "
                    f"lyric phones {train_lyric_phone_accuracy * 100.0:.1f}%/{val_lyric_phone_accuracy * 100.0:.1f}% | "
                    f"all phones {train_phone_accuracy * 100.0:.1f}%/{val_phone_accuracy * 100.0:.1f}% | "
                    f"voicing {train_vuv_accuracy * 100.0:.1f}%/{val_vuv_accuracy * 100.0:.1f}% | "
                    f"voice match {train_loss_sums['voice'] / max(train_batches, 1):.4f}/{val_loss_sums['voice'] / max(val_batches, 1):.4f} | "
                    f"transition {train_loss_sums['delta_mel'] / max(train_batches, 1):.4f}/{val_loss_sums['delta_mel'] / max(val_batches, 1):.4f} | "
                    f"quality {train_target_quality:.4f}/{val_target_quality:.4f}"
                    f"{phase_extra_summary} | "
                    f"frames/s {train_frames_per_second:.0f} | lr {optimizer.param_groups[0]['lr']:.2e} | "
                    f"plateau {epochs_since_best} epochs | active slices {curriculum_state.get('selected_count', len(active_train_entries))}"
                ),
                progress,
            )

        post_process_metadata: Dict[str, object] = {}
        resolved_resume_post_process_checkpoint = (
            Path(str(resume_post_process_checkpoint_path))
            if resume_post_process_checkpoint_path
            else None
        )
        if (
            post_process_enabled
            and (cancel_event is None or not bool(getattr(cancel_event, "is_set", lambda: False)()))
            and target_end_epoch > max(base_target_end_epoch, resume_epoch)
        ):
            if not latest_checkpoint_path.exists():
                raise RuntimeError("Post Process needs a saved voice-builder checkpoint before it can begin.")
            try:
                post_process_metadata = self.train_post_process_refiner(
                    dataset_dir=dataset_dir,
                    output_dir=output_dir,
                    stats=stats,
                    train_entries=train_entries,
                    val_entries=val_entries,
                    base_checkpoint_path=latest_checkpoint_path,
                    requested_batch_size=batch_size,
                    start_epoch=max(base_target_end_epoch, resume_epoch),
                    total_end_epoch=target_end_epoch,
                    guided_profile=hardware_profile,
                    update_status=update_status,
                    cancel_event=cancel_event,
                    resume_checkpoint_path=resolved_resume_post_process_checkpoint,
                    resume_report_path=resolved_resume_report,
                )
                last_epoch = max(last_epoch, int(post_process_metadata.get("last_epoch", last_epoch)))
                stopped_early = bool(stopped_early or post_process_metadata.get("stopped_early", False))
            except InterruptedError:
                stopped_early = True
                post_process_metadata = {"stopped_early": True, "last_epoch": int(last_epoch)}
            except Exception as exc:
                post_process_metadata = {"error": str(exc), "last_epoch": int(last_epoch)}

        if not latest_checkpoint_path.exists():
            if stopped_early:
                raise InterruptedError(
                    "Training stopped before the first voice-builder checkpoint was written."
                )
            raise RuntimeError("Guided regeneration training never wrote a checkpoint.")

        history_path = output_dir / "guided_regeneration_history.json"
        history_path.write_text(json.dumps(history, indent=2), encoding="utf-8")
        vocoder_metadata: Dict[str, object] = {}
        if cancel_event is None or not bool(getattr(cancel_event, "is_set", lambda: False)()):
            try:
                vocoder_metadata = self.train_neural_vocoder(
                    dataset_dir=dataset_dir,
                    output_dir=output_dir,
                    stats=stats,
                    train_entries=train_entries,
                    val_entries=val_entries,
                    requested_batch_size=batch_size,
                    requested_num_workers=num_workers,
                    total_epochs=total_epochs,
                    guided_profile=hardware_profile,
                    update_status=update_status,
                    cancel_event=cancel_event,
                )
            except InterruptedError:
                stopped_early = True
                vocoder_metadata = {"stopped_early": True}
            except Exception as exc:
                vocoder_metadata = {"error": str(exc)}
        preview_metadata: Dict[str, object] = {}
        if cancel_event is not None and bool(getattr(cancel_event, "is_set", lambda: False)()):
            preview_metadata = {
                "preview_skipped": True,
                "preview_reason": "Training was stopped before preview rendering.",
            }
        else:
            try:
                preview_metadata = self._render_preview(
                    checkpoint_path=best_checkpoint_path if best_checkpoint_path.exists() else latest_checkpoint_path,
                    dataset_dir=dataset_dir,
                    stats=stats,
                    sample_entry=val_entries[0] if val_entries else train_entries[0],
                    output_dir=output_dir,
                )
            except Exception as exc:
                preview_metadata = {
                    "preview_error": str(exc),
                }
        report_payload = {
            "sample_count": int(stats.get("sample_count", 0)),
            "train_sample_count": len(train_entries),
            "val_sample_count": len(val_entries),
            "best_val_l1": round(float(best_val_mel), 6),
            "best_val_total": round(float(best_val_loss), 6),
            "best_target_quality": round(float(best_target_quality), 6) if math.isfinite(best_target_quality) else 0.0,
            "best_quality": round(float(best_target_quality), 6) if math.isfinite(best_target_quality) else 0.0,
            "best_epoch": int(best_epoch),
            "best_phone_accuracy": round(float(best_phone_accuracy), 6),
            "best_lyric_phone_accuracy": round(float(best_lyric_phone_accuracy), 6),
            "best_vuv_accuracy": round(float(best_vuv_accuracy), 6),
            "plateau_epochs": int(max(0, last_epoch - best_epoch)),
            "last_epoch": int(last_epoch),
            "stopped_early": bool(stopped_early),
            "device": device.type,
            "gpu_memory_gb": float(hardware_profile.get("gpu_memory_gb", 0.0)),
            "hardware_summary": hardware_summary,
            "quality_summary": history[-1]["quality_state"] if history else "",
            "precision_mode": precision_mode,
            "optimizer_mode": "fused-adamw" if optimizer_kwargs.get("fused") else "adamw",
            "pair_filtering": dict(filter_metadata),
            "recipe_mode": normalized_recipe_mode,
            "repair_strategy": dict(config.get("repair_strategy", {})),
            "search_mode": "mc-dropout-target-voice-rerank",
            "resume_epoch": int(resume_epoch),
            "target_end_epoch": int(target_end_epoch),
            "start_phase": normalized_start_phase,
            "curriculum": dict(config.get("curriculum", {})),
            "voice_signature_dim": VOICE_SIGNATURE_DIM,
            "checkpoint_path": str(best_checkpoint_path if best_checkpoint_path.exists() else latest_checkpoint_path),
            "latest_checkpoint_path": str(latest_checkpoint_path),
            "history_path": str(history_path),
            "guided_post_process": {
                "checkpoint_path": str(post_process_metadata.get("checkpoint_path", "")),
                "latest_checkpoint_path": str(post_process_metadata.get("latest_checkpoint_path", "")),
                "config_path": str(post_process_metadata.get("config_path", "")),
                "report_path": str(post_process_metadata.get("report_path", "")),
                "history_path": str(post_process_metadata.get("history_path", "")),
                "best_val_total": float(post_process_metadata.get("best_val_total", 0.0)),
                "best_epoch": int(post_process_metadata.get("best_epoch", 0)),
                "quality_summary": str(post_process_metadata.get("quality_summary", "")),
                "hardware_summary": str(post_process_metadata.get("hardware_summary", "")),
                "last_epoch": int(post_process_metadata.get("last_epoch", 0)),
                "render_mode": str(post_process_metadata.get("render_mode", "")),
                "error": str(post_process_metadata.get("error", "")),
            },
            "guided_vocoder": {
                "checkpoint_path": str(vocoder_metadata.get("checkpoint_path", "")),
                "latest_checkpoint_path": str(vocoder_metadata.get("latest_checkpoint_path", "")),
                "config_path": str(vocoder_metadata.get("config_path", "")),
                "report_path": str(vocoder_metadata.get("report_path", "")),
                "history_path": str(vocoder_metadata.get("history_path", "")),
                "best_val_total": float(vocoder_metadata.get("best_val_total", 0.0)),
                "best_epoch": int(vocoder_metadata.get("best_epoch", 0)),
                "quality_summary": str(vocoder_metadata.get("quality_summary", "")),
                "hardware_summary": str(vocoder_metadata.get("hardware_summary", "")),
                "render_mode": str(vocoder_metadata.get("render_mode", "")),
                "error": str(vocoder_metadata.get("error", "")),
            },
            **preview_metadata,
        }
        report_path = output_dir / "guided_regeneration_report.json"
        report_path.write_text(json.dumps(report_payload, indent=2), encoding="utf-8")
        return {
            "checkpoint_path": str(best_checkpoint_path if best_checkpoint_path.exists() else latest_checkpoint_path),
            "latest_checkpoint_path": str(latest_checkpoint_path),
            "config_path": str(config_path),
            "report_path": str(report_path),
            "history_path": str(history_path),
            "preview_path": str(preview_metadata.get("preview_path", "")),
            "target_preview_path": str(preview_metadata.get("target_preview_path", "")),
            "preview_sample_id": str(preview_metadata.get("preview_sample_id", "")),
            "preview_render_mode": str(preview_metadata.get("preview_render_mode", "")),
            "best_val_l1": float(best_val_mel),
            "best_val_total": float(best_val_loss),
            "best_target_quality": float(best_target_quality) if math.isfinite(best_target_quality) else 0.0,
            "best_quality": float(best_target_quality) if math.isfinite(best_target_quality) else 0.0,
            "best_epoch": int(best_epoch),
            "best_phone_accuracy": float(best_phone_accuracy),
            "best_lyric_phone_accuracy": float(best_lyric_phone_accuracy),
            "best_vuv_accuracy": float(best_vuv_accuracy),
            "plateau_epochs": int(max(0, last_epoch - best_epoch)),
            "hardware_summary": hardware_summary,
            "quality_summary": str(history[-1]["quality_state"]) if history else "",
            "recipe_mode": normalized_recipe_mode,
            "repair_strategy": dict(config.get("repair_strategy", {})),
            "search_mode": "mc-dropout-target-voice-rerank",
            "voice_signature_dim": VOICE_SIGNATURE_DIM,
            "post_process_checkpoint_path": str(post_process_metadata.get("checkpoint_path", "")),
            "post_process_latest_checkpoint_path": str(post_process_metadata.get("latest_checkpoint_path", "")),
            "post_process_config_path": str(post_process_metadata.get("config_path", "")),
            "post_process_report_path": str(post_process_metadata.get("report_path", "")),
            "post_process_history_path": str(post_process_metadata.get("history_path", "")),
            "post_process_best_val_total": float(post_process_metadata.get("best_val_total", 0.0)),
            "post_process_best_epoch": int(post_process_metadata.get("best_epoch", 0)),
            "post_process_quality_summary": str(post_process_metadata.get("quality_summary", "")),
            "post_process_hardware_summary": str(post_process_metadata.get("hardware_summary", "")),
            "vocoder_checkpoint_path": str(vocoder_metadata.get("checkpoint_path", "")),
            "vocoder_latest_checkpoint_path": str(vocoder_metadata.get("latest_checkpoint_path", "")),
            "vocoder_config_path": str(vocoder_metadata.get("config_path", "")),
            "vocoder_report_path": str(vocoder_metadata.get("report_path", "")),
            "vocoder_history_path": str(vocoder_metadata.get("history_path", "")),
            "vocoder_best_val_total": float(vocoder_metadata.get("best_val_total", 0.0)),
            "vocoder_best_epoch": int(vocoder_metadata.get("best_epoch", 0)),
            "vocoder_quality_summary": str(vocoder_metadata.get("quality_summary", "")),
            "vocoder_hardware_summary": str(vocoder_metadata.get("hardware_summary", "")),
            "render_mode": str(vocoder_metadata.get("render_mode", preview_metadata.get("preview_render_mode", "griffinlim_preview_only"))),
            "resume_epoch": int(resume_epoch),
            "target_end_epoch": int(target_end_epoch),
            "start_phase": normalized_start_phase,
            "curriculum": dict(config.get("curriculum", {})),
            "last_epoch": int(last_epoch),
            "sample_count": int(stats.get("sample_count", 0)),
            "stopped_early": bool(stopped_early),
        }
