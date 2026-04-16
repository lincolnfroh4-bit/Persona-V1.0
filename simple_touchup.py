from __future__ import annotations

import os
import random
import shutil
import subprocess
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import ffmpeg
import librosa
import numpy as np
import soundfile as sf
import torch
import torchaudio
from ctc_segmentation import (
    CtcSegmentationParameters,
    ctc_segmentation,
    determine_utterance_segments,
    prepare_text,
)
from scipy.ndimage import gaussian_filter1d
from scipy.signal import istft, stft


def normalize_lyrics(text: str) -> str:
    lowered = (text or "").lower()
    cleaned = "".join(char if char.isalnum() or char in {"'", " "} else " " for char in lowered)
    collapsed = " ".join(cleaned.split())
    return collapsed


def lyrics_to_words(text: str) -> List[str]:
    normalized = normalize_lyrics(text)
    return [word for word in normalized.split(" ") if word]


def word_to_letters(word: str) -> List[str]:
    cleaned = "".join(char for char in word.upper() if char.isalpha() or char == "'")
    return [char for char in cleaned if char]


class LetterAwarePronunciationScorer:
    def __init__(self):
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._model = None
        self._sample_rate = 16000
        self._labels: Tuple[str, ...] = ()
        self._config: Optional[CtcSegmentationParameters] = None
        self._lock = threading.Lock()

    def _ensure_model(self) -> None:
        if self._model is not None:
            return
        bundle = torchaudio.pipelines.WAV2VEC2_ASR_BASE_960H
        model = bundle.get_model().to(self._device)
        model.eval()
        self._model = model
        self._sample_rate = int(bundle.sample_rate)
        self._labels = tuple(bundle.get_labels())
        config = CtcSegmentationParameters()
        config.char_list = list(self._labels)
        config.blank = 0
        self._config = config

    def _prepare_waveform(self, audio: np.ndarray, sample_rate: int) -> torch.Tensor:
        working = np.asarray(audio, dtype=np.float32)
        if working.ndim == 1:
            mono = working.astype(np.float32, copy=False)
        elif working.size == 0 or working.shape[0] == 0:
            mono = np.zeros(1, dtype=np.float32)
        else:
            mono = working.mean(axis=1).astype(np.float32, copy=False)
        if mono.size == 0:
            mono = np.zeros(1, dtype=np.float32)
        waveform = torch.from_numpy(np.ascontiguousarray(mono))
        if sample_rate != self._sample_rate and int(waveform.numel()) > 1:
            waveform = torchaudio.functional.resample(
                waveform.unsqueeze(0),
                orig_freq=sample_rate,
                new_freq=self._sample_rate,
            ).squeeze(0)
        return waveform

    def _confidence_to_similarity(self, confidence: float) -> float:
        clamped = float(np.clip(confidence, -7.0, 0.0))
        return round(((clamped + 7.0) / 7.0) * 100.0, 2)

    def _run_ctc_alignment(
        self,
        waveform: torch.Tensor,
        entries: List[str],
    ) -> Tuple[np.ndarray, np.ndarray, List[str], List[Tuple[float, float, float]]]:
        assert self._model is not None
        assert self._config is not None
        with torch.no_grad():
            emissions, _ = self._model(waveform.unsqueeze(0).to(self._device))
        lpz = torch.log_softmax(emissions[0], dim=-1).detach().cpu().numpy()
        ground_truth_mat, utt_begin_indices = prepare_text(self._config, entries)
        timings, char_probs, state_list = ctc_segmentation(self._config, lpz, ground_truth_mat)
        segments = determine_utterance_segments(
            self._config,
            utt_begin_indices,
            char_probs,
            timings,
            entries,
        )
        return timings, char_probs, state_list, segments

    def _extract_letter_scores(
        self,
        waveform: torch.Tensor,
        word: str,
        *,
        global_start_seconds: float,
        word_index: int,
    ) -> List[Dict[str, float | str]]:
        letters = word_to_letters(word)
        if not letters:
            return []
        try:
            timings, char_probs, state_list, _ = self._run_ctc_alignment(waveform, ["".join(letters)])
        except Exception:
            return []

        matches: List[Dict[str, float | str]] = []
        cursor = 0
        for target in letters:
            best_index = None
            best_prob = -1e9
            for index in range(cursor, len(state_list)):
                token = str(state_list[index]).upper()
                if token == target and float(char_probs[index]) > best_prob:
                    best_index = index
                    best_prob = float(char_probs[index])
                if best_index is not None and token not in {"", "-", "\u03b5", target}:
                    break
            if best_index is None:
                matches.append(
                    {
                        "letter": target,
                        "word_index": int(word_index),
                        "start": float(global_start_seconds),
                        "end": float(global_start_seconds),
                        "confidence": -7.0,
                        "similarity": 0.0,
                    }
                )
                continue
            cursor = best_index + 1
            start_time = float(timings[min(best_index, len(timings) - 1)])
            end_time = float(timings[min(best_index + 1, len(timings) - 1)]) if len(timings) > 1 else start_time
            confidence = float(char_probs[best_index])
            matches.append(
                {
                    "letter": target,
                    "word_index": int(word_index),
                    "start": float(global_start_seconds + start_time),
                    "end": float(global_start_seconds + max(end_time, start_time)),
                    "confidence": confidence,
                    "similarity": self._confidence_to_similarity(confidence),
                }
            )
        return matches

    def _build_reports(
        self,
        word_scores: List[Dict[str, float | str]],
        letter_scores: List[Dict[str, float | str]],
    ) -> Tuple[str, str]:
        weak_words = sorted(word_scores, key=lambda entry: float(entry["similarity"]))[:8]
        weak_letters = sorted(letter_scores, key=lambda entry: float(entry["similarity"]))[:12]
        word_report = (
            "Weakest words: "
            + ", ".join(f"{entry['word']} ({float(entry['similarity']):.0f}%)" for entry in weak_words)
            if weak_words
            else "No weak words were found."
        )
        letter_report = (
            "Weakest letters: "
            + ", ".join(
                f"{entry['letter']} in {entry['word']} ({float(entry['similarity']):.0f}%)"
                for entry in weak_letters
            )
            if weak_letters
            else "No weak letters were found."
        )
        return word_report, letter_report

    def build_analysis_result(
        self,
        word_scores: List[Dict[str, float | str]],
        letter_scores: List[Dict[str, float | str]],
    ) -> Dict[str, object]:
        ordered_word_scores = sorted(
            (dict(entry) for entry in word_scores),
            key=lambda entry: int(entry.get("index", 0)),
        )
        ordered_letter_scores = sorted(
            (dict(entry) for entry in letter_scores),
            key=lambda entry: (int(entry.get("word_index", 0)), float(entry.get("start", 0.0))),
        )
        word_similarity = round(
            float(np.mean([float(entry["similarity"]) for entry in ordered_word_scores])) if ordered_word_scores else 0.0,
            2,
        )
        word_report, letter_report = self._build_reports(ordered_word_scores, ordered_letter_scores)
        return {
            "similarity_score": word_similarity,
            "word_scores": ordered_word_scores,
            "letter_scores": ordered_letter_scores,
            "word_report": word_report,
            "letter_report": letter_report,
        }

    def _build_failed_alignment_result(
        self,
        words: List[str],
        *,
        global_start_seconds: float = 0.0,
        absolute_word_indices: Optional[List[int]] = None,
    ) -> Dict[str, object]:
        fallback_scores: List[Dict[str, float | str]] = []
        cursor = float(global_start_seconds)
        for local_index, word in enumerate(words):
            absolute_index = (
                int(absolute_word_indices[local_index])
                if absolute_word_indices is not None and local_index < len(absolute_word_indices)
                else int(local_index)
            )
            fallback_scores.append(
                {
                    "index": absolute_index,
                    "word": word,
                    "start": cursor,
                    "end": cursor,
                    "confidence": -7.0,
                    "similarity": 0.0,
                }
            )
        return self.build_analysis_result(fallback_scores, [])

    def _analyze_words(
        self,
        audio: np.ndarray,
        sample_rate: int,
        words: List[str],
        *,
        global_start_seconds: float = 0.0,
        absolute_word_indices: Optional[List[int]] = None,
        letter_focus_limit: int = 12,
    ) -> Dict[str, object]:
        if not words:
            return {
                "similarity_score": 0.0,
                "word_scores": [],
                "letter_scores": [],
                "word_report": "No target lyrics were provided.",
                "letter_report": "No target lyrics were provided.",
            }

        try:
            with self._lock:
                self._ensure_model()
                waveform = self._prepare_waveform(audio, sample_rate)
                if int(waveform.numel()) <= 1:
                    return self._build_failed_alignment_result(
                        words,
                        global_start_seconds=global_start_seconds,
                        absolute_word_indices=absolute_word_indices,
                    )
                _, _, _, word_segments = self._run_ctc_alignment(
                    waveform,
                    [word.upper() for word in words],
                )
        except Exception:
            return self._build_failed_alignment_result(
                words,
                global_start_seconds=global_start_seconds,
                absolute_word_indices=absolute_word_indices,
            )

        total_samples = int(waveform.shape[0])
        scored_words: List[Tuple[Dict[str, float | str], torch.Tensor]] = []
        for local_index, (word, segment) in enumerate(zip(words, word_segments)):
            absolute_index = (
                int(absolute_word_indices[local_index])
                if absolute_word_indices is not None and local_index < len(absolute_word_indices)
                else int(local_index)
            )
            start, end, confidence = segment
            absolute_start = float(global_start_seconds + float(start))
            absolute_end = float(global_start_seconds + float(end))
            similarity = self._confidence_to_similarity(float(confidence))
            entry: Dict[str, float | str] = {
                "index": absolute_index,
                "word": word,
                "start": absolute_start,
                "end": absolute_end,
                "confidence": float(confidence),
                "similarity": similarity,
            }

            start_sample = max(0, int(float(start) * self._sample_rate))
            end_sample = min(total_samples, int(float(end) * self._sample_rate))
            if end_sample <= start_sample:
                word_waveform = waveform[max(0, start_sample) : max(0, start_sample + 1)]
            else:
                word_waveform = waveform[start_sample:end_sample]
            scored_words.append((entry, word_waveform))

        word_scores = [entry for entry, _ in scored_words]
        if not word_scores:
            return self.build_analysis_result([], [])

        weak_word_limit = max(1, min(int(letter_focus_limit), len(word_scores)))
        weak_words = sorted(word_scores, key=lambda entry: float(entry["similarity"]))[:weak_word_limit]
        weak_indices = {
            int(entry["index"])
            for entry in weak_words
            if float(entry["similarity"]) < 96.0
        }
        if not weak_indices:
            weak_indices = {int(weak_words[0]["index"])}

        letter_scores: List[Dict[str, float | str]] = []
        for entry, word_waveform in scored_words:
            if int(entry["index"]) not in weak_indices:
                continue
            letters = self._extract_letter_scores(
                word_waveform,
                str(entry["word"]),
                global_start_seconds=float(entry["start"]),
                word_index=int(entry["index"]),
            )
            for letter in letters:
                letter_entry = dict(letter)
                letter_entry["word"] = str(entry["word"])
                letter_scores.append(letter_entry)

        return self.build_analysis_result(word_scores, letter_scores)

    def analyze_audio(self, audio: np.ndarray, sample_rate: int, expected_lyrics: str) -> Dict[str, object]:
        return self._analyze_words(audio, sample_rate, lyrics_to_words(expected_lyrics))

    def analyze_segment(
        self,
        audio: np.ndarray,
        sample_rate: int,
        expected_words: List[str],
        *,
        global_start_seconds: float,
        absolute_word_indices: Optional[List[int]] = None,
        letter_focus_limit: int = 8,
    ) -> Dict[str, object]:
        cleaned_words = [normalize_lyrics(word) for word in expected_words]
        cleaned_words = [word for word in cleaned_words if word]
        return self._analyze_words(
            audio,
            sample_rate,
            cleaned_words,
            global_start_seconds=global_start_seconds,
            absolute_word_indices=absolute_word_indices,
            letter_focus_limit=letter_focus_limit,
        )


class NeuralClarityRepairEngine:
    def __init__(self, repo_root: Path):
        self.repo_root = repo_root
        self.scorer = LetterAwarePronunciationScorer()
        self._enhancer = None
        self._enhancer_sample_rate = 16000
        self._enhancer_lock = threading.Lock()
        self._tts_python = self.repo_root / ".venv-tts311" / "Scripts" / "python.exe"
        self._tts_runner = self.repo_root / "tools" / "xtts_word_regen.py"

    def _coqui_tos_agreed(self) -> bool:
        if os.environ.get("COQUI_TOS_AGREED") == "1":
            return True
        if os.name == "nt":
            try:
                import winreg

                with winreg.OpenKey(winreg.HKEY_CURRENT_USER, r"Environment") as key:
                    value, _ = winreg.QueryValueEx(key, "COQUI_TOS_AGREED")
                return str(value).strip() == "1"
            except OSError:
                return False
        return False

    def _ffmpeg_binary(self) -> str:
        local = self.repo_root / ("ffmpeg.exe" if os.name == "nt" else "ffmpeg")
        return str(local) if local.exists() else "ffmpeg"

    def _ensure_enhancer(self) -> None:
        if self._enhancer is not None:
            return
        with self._enhancer_lock:
            if self._enhancer is not None:
                return
            try:
                from speechbrain.inference.enhancement import SpectralMaskEnhancement
                from speechbrain.utils.fetching import LocalStrategy
            except ModuleNotFoundError as exc:
                raise RuntimeError(
                    "speechbrain is required only for the optional enhancement path and is not installed in the current Python environment."
                ) from exc
            savedir = self.repo_root / "pretrained_models" / "touchup_metricgan"
            self._enhancer = SpectralMaskEnhancement.from_hparams(
                source="speechbrain/metricgan-plus-voicebank",
                savedir=str(savedir),
                run_opts={"device": "cuda" if torch.cuda.is_available() else "cpu"},
                local_strategy=LocalStrategy.COPY,
            )

    def get_regenerator_status(self) -> Dict[str, object]:
        if not self._tts_python.exists():
            return {"available": False, "reason": "The separate XTTS Python environment was not found."}
        if not self._tts_runner.exists():
            return {"available": False, "reason": "The XTTS word regeneration runner is missing."}
        if not self._coqui_tos_agreed():
            return {
                "available": False,
                "reason": "XTTS regeneration is installed, but Coqui license acceptance is still required.",
            }
        return {"available": True, "reason": "XTTS word regeneration is ready."}

    def _load_audio(self, file_path: Path, sample_rate: int = 44100) -> Tuple[np.ndarray, int]:
        cleaned = str(file_path).strip().strip('"')
        out, _ = (
            ffmpeg.input(cleaned, threads=0)
            .output(
                "-",
                format="f32le",
                acodec="pcm_f32le",
                ac=2,
                ar=sample_rate,
            )
            .run(
                cmd=[self._ffmpeg_binary(), "-nostdin"],
                capture_stdout=True,
                capture_stderr=True,
            )
        )
        audio = np.frombuffer(out, np.float32)
        if audio.size == 0:
            raise RuntimeError(f"Could not decode audio from {file_path.name}.")
        return audio.reshape(-1, 2).copy(), sample_rate

    def _safe_rms_db(self, audio: np.ndarray) -> float:
        rms = float(np.sqrt(np.mean(np.square(audio)) + 1e-9))
        return float(20.0 * np.log10(max(rms, 1e-8)))

    def _cosine_fade(self, length: int) -> np.ndarray:
        if length <= 1:
            return np.ones(length, dtype=np.float32)
        return np.sin(np.linspace(0.0, np.pi, length, dtype=np.float32)) ** 2

    def _letter_window_to_samples(
        self,
        letter_score: Dict[str, float | str],
        sample_rate: int,
        total_samples: int,
        padding_ms: float,
    ) -> Tuple[int, int]:
        pad = int((padding_ms / 1000.0) * sample_rate)
        start = max(0, int(float(letter_score["start"]) * sample_rate) - pad)
        end = min(total_samples, int(float(letter_score["end"]) * sample_rate) + pad)
        if end <= start:
            end = min(total_samples, start + max(1, int(0.06 * sample_rate)))
        return start, end

    def _choose_fft_size(self, segment_length: int) -> int:
        if segment_length >= 2048:
            return 1024
        if segment_length >= 1024:
            return 512
        if segment_length >= 512:
            return 256
        return 128

    def _group_contiguous_indices(self, indices: List[int]) -> List[List[int]]:
        if not indices:
            return []
        ordered = sorted(set(int(index) for index in indices))
        groups: List[List[int]] = [[ordered[0]]]
        for index in ordered[1:]:
            if index == groups[-1][-1] + 1:
                groups[-1].append(index)
            else:
                groups.append([index])
        return groups

    def _word_window_to_samples(
        self,
        word_scores: List[Dict[str, float | str]],
        word_indices: List[int],
        sample_rate: int,
        total_samples: int,
        padding_ms: float,
    ) -> Tuple[int, int]:
        word_by_index = {int(entry["index"]): entry for entry in word_scores}
        starts = [float(word_by_index[index]["start"]) for index in word_indices if index in word_by_index]
        ends = [float(word_by_index[index]["end"]) for index in word_indices if index in word_by_index]
        if not starts or not ends:
            return 0, 0
        pad = int((padding_ms / 1000.0) * sample_rate)
        start = max(0, int(min(starts) * sample_rate) - pad)
        end = min(total_samples, int(max(ends) * sample_rate) + pad)
        return start, max(start + 1, end)

    def _fit_audio_length(self, audio: np.ndarray, target_length: int) -> np.ndarray:
        working = np.asarray(audio, dtype=np.float32)
        if working.ndim == 1:
            working = working[:, np.newaxis]
        channels = int(working.shape[1]) if working.ndim == 2 and working.shape[1] > 0 else 1
        if target_length <= 0:
            return np.zeros((0, channels), dtype=np.float32)
        if working.shape[0] == 0:
            return np.zeros((target_length, channels), dtype=np.float32)
        if working.shape[0] == target_length:
            return working.astype(np.float32, copy=False)
        if working.shape[0] == 1:
            return np.repeat(working.astype(np.float32, copy=False), target_length, axis=0)
        source_positions = np.linspace(0.0, 1.0, num=working.shape[0], dtype=np.float32)
        target_positions = np.linspace(0.0, 1.0, num=target_length, dtype=np.float32)
        resized_channels = []
        for channel in range(working.shape[1]):
            resized_channels.append(np.interp(target_positions, source_positions, working[:, channel]).astype(np.float32))
        return np.stack(resized_channels, axis=1)

    def _merge_sample_windows(
        self,
        windows: List[Tuple[int, int]],
        *,
        bridge_samples: int,
    ) -> List[Tuple[int, int]]:
        if not windows:
            return []
        ordered = sorted((max(0, int(start)), max(0, int(end))) for start, end in windows if int(end) > int(start))
        if not ordered:
            return []
        merged: List[Tuple[int, int]] = [ordered[0]]
        for start, end in ordered[1:]:
            prev_start, prev_end = merged[-1]
            if start <= (prev_end + bridge_samples):
                merged[-1] = (prev_start, max(prev_end, end))
            else:
                merged.append((start, end))
        return merged

    def smart_remove_non_lyrics(
        self,
        *,
        source_path: Path,
        intended_lyrics: str,
        output_dir: Path,
        strength: int,
        cancel_event: threading.Event,
        update_status: Optional[Callable[[Dict[str, object]], None]] = None,
    ) -> Dict[str, object]:
        output_dir.mkdir(parents=True, exist_ok=True)
        source_audio, sample_rate = self._load_audio(source_path, sample_rate=44100)
        baseline_scoring = self.scorer.analyze_audio(source_audio, sample_rate, intended_lyrics)
        word_scores = [dict(entry) for entry in baseline_scoring.get("word_scores", [])]

        if not word_scores:
            output_path = output_dir / f"{source_path.stem}_smart_removed.wav"
            removed_path = output_dir / f"{source_path.stem}_removed_layer.wav"
            sf.write(output_path, source_audio, sample_rate, subtype="PCM_24")
            sf.write(removed_path, np.zeros_like(source_audio, dtype=np.float32), sample_rate, subtype="PCM_24")
            return {
                "output_path": output_path,
                "removed_path": removed_path,
                "sample_rate": sample_rate,
                "source_rms_db": self._safe_rms_db(source_audio),
                "output_rms_db": self._safe_rms_db(source_audio),
                "best_similarity_score": float(baseline_scoring.get("similarity_score", 0.0)),
                "best_word_report": str(baseline_scoring.get("word_report", "")),
                "best_letter_report": str(baseline_scoring.get("letter_report", "")),
                "best_word_scores": word_scores,
                "best_letter_scores": list(baseline_scoring.get("letter_scores", [])),
                "detected_word_indices": [int(entry.get("index", 0)) for entry in word_scores],
                "kept_segment_count": 0,
                "kept_duration_seconds": 0.0,
                "removed_duration_seconds": float(source_audio.shape[0] / max(sample_rate, 1)),
                "repair_attempts": 0,
                "variants_tested": 0,
                "repaired_word_count": 0,
            }

        strength_value = max(1, min(int(strength), 100))
        strength_ratio = float((strength_value - 1) / 99.0)
        keep_padding_ms = float(np.interp(strength_ratio, [0.0, 1.0], [160.0, 42.0]))
        bridge_gap_ms = float(np.interp(strength_ratio, [0.0, 1.0], [260.0, 75.0]))
        fade_ms = float(np.interp(strength_ratio, [0.0, 1.0], [34.0, 16.0]))

        total_samples = int(source_audio.shape[0])
        keep_windows: List[Tuple[int, int]] = []
        for entry in word_scores:
            if cancel_event.is_set():
                break
            start = max(0, int(float(entry.get("start", 0.0)) * sample_rate) - int((keep_padding_ms / 1000.0) * sample_rate))
            end = min(
                total_samples,
                int(float(entry.get("end", 0.0)) * sample_rate) + int((keep_padding_ms / 1000.0) * sample_rate),
            )
            if end > start:
                keep_windows.append((start, end))

        merged_windows = self._merge_sample_windows(
            keep_windows,
            bridge_samples=int((bridge_gap_ms / 1000.0) * sample_rate),
        )

        keep_mask = np.zeros(total_samples, dtype=np.float32)
        fade_samples = max(8, int((fade_ms / 1000.0) * sample_rate))
        for start, end in merged_windows:
            if cancel_event.is_set():
                break
            segment_length = max(1, end - start)
            keep_mask[start:end] = 1.0
            edge = min(fade_samples, max(1, segment_length // 3))
            if edge > 1:
                fade_curve = self._cosine_fade(edge)
                keep_mask[start : start + edge] = np.maximum(keep_mask[start : start + edge], fade_curve[:edge])
                keep_mask[end - edge : end] = np.maximum(keep_mask[end - edge : end], fade_curve[-edge:])

        if update_status is not None:
            update_status(
                {
                    "progress": 72,
                    "message": (
                        f"Keeping {len(merged_windows)} lyric regions and muting everything else."
                    ),
                    "best_similarity_score": float(baseline_scoring.get("similarity_score", 0.0)),
                    "best_word_report": str(baseline_scoring.get("word_report", "")),
                    "best_letter_report": str(baseline_scoring.get("letter_report", "")),
                    "detected_word_indices": [int(entry.get("index", 0)) for entry in word_scores],
                    "repaired_word_count": 0,
                    "variants_tested": 0,
                    "repair_attempts": 0,
                }
            )

        kept_audio = source_audio * keep_mask[:, np.newaxis]
        removed_audio = source_audio - kept_audio

        kept_peak = float(np.max(np.abs(kept_audio)) + 1e-9)
        if kept_peak > 0.995:
            kept_audio = kept_audio * np.float32(0.995 / kept_peak)

        output_path = output_dir / f"{source_path.stem}_smart_removed.wav"
        removed_path = output_dir / f"{source_path.stem}_removed_layer.wav"
        sf.write(output_path, kept_audio.astype(np.float32, copy=False), sample_rate, subtype="PCM_24")
        sf.write(removed_path, removed_audio.astype(np.float32, copy=False), sample_rate, subtype="PCM_24")

        kept_duration_seconds = float(np.sum(keep_mask > 0.5) / max(sample_rate, 1))
        total_duration_seconds = float(total_samples / max(sample_rate, 1))
        return {
            "output_path": output_path,
            "removed_path": removed_path,
            "sample_rate": sample_rate,
            "source_rms_db": self._safe_rms_db(source_audio),
            "output_rms_db": self._safe_rms_db(kept_audio),
            "best_similarity_score": float(baseline_scoring.get("similarity_score", 0.0)),
            "best_word_report": str(baseline_scoring.get("word_report", "")),
            "best_letter_report": str(baseline_scoring.get("letter_report", "")),
            "best_word_scores": word_scores,
            "best_letter_scores": list(baseline_scoring.get("letter_scores", [])),
            "detected_word_indices": [int(entry.get("index", 0)) for entry in word_scores],
            "kept_segment_count": int(len(merged_windows)),
            "kept_duration_seconds": kept_duration_seconds,
            "removed_duration_seconds": max(0.0, total_duration_seconds - kept_duration_seconds),
            "repair_attempts": 0,
            "variants_tested": 0,
            "repaired_word_count": 0,
        }

    def _run_xtts_regenerator(
        self,
        *,
        text: str,
        speaker_wav: Path,
        output_path: Path,
    ) -> Path:
        status = self.get_regenerator_status()
        if not bool(status["available"]):
            raise RuntimeError(str(status["reason"]))
        output_path.parent.mkdir(parents=True, exist_ok=True)
        runner_env = os.environ.copy()
        if self._coqui_tos_agreed():
            runner_env["COQUI_TOS_AGREED"] = "1"
        completed = subprocess.run(
            [
                str(self._tts_python),
                str(self._tts_runner),
                "--text",
                text,
                "--speaker-wav",
                str(speaker_wav),
                "--output",
                str(output_path),
                "--language",
                "en",
            ],
            cwd=str(self.repo_root),
            capture_output=True,
            text=True,
            env=runner_env,
            check=False,
        )
        if completed.returncode != 0 or not output_path.exists():
            stderr = (completed.stderr or completed.stdout or "").strip()
            raise RuntimeError(stderr or "XTTS word generation failed.")
        return output_path

    def _resample_mono(self, audio: np.ndarray, source_rate: int, target_rate: int) -> np.ndarray:
        if source_rate == target_rate:
            return audio.astype(np.float32, copy=False)
        tensor = torch.from_numpy(audio.astype(np.float32, copy=False)).unsqueeze(0)
        resampled = torchaudio.functional.resample(tensor, orig_freq=source_rate, new_freq=target_rate)
        return resampled.squeeze(0).cpu().numpy().astype(np.float32, copy=False)

    def _enhance_mono_segment(self, mono_audio: np.ndarray, sample_rate: int) -> np.ndarray:
        self._ensure_enhancer()
        assert self._enhancer is not None
        target_rate = int(self._enhancer_sample_rate)
        working = self._resample_mono(mono_audio, sample_rate, target_rate)
        waveform = torch.from_numpy(working).unsqueeze(0).to(self._enhancer.device)
        lengths = torch.ones(1, device=self._enhancer.device)
        with torch.no_grad():
            enhanced = self._enhancer.enhance_batch(waveform, lengths).detach().cpu().squeeze(0).numpy()
        return self._resample_mono(enhanced, target_rate, sample_rate)

    def _build_pronunciation_embedding(
        self,
        audio: np.ndarray,
        sample_rate: int,
    ) -> Optional[np.ndarray]:
        working = np.asarray(audio, dtype=np.float32)
        if working.ndim == 2 and working.shape[1] > 0:
            mono = working.mean(axis=1).astype(np.float32, copy=False)
        else:
            mono = working.astype(np.float32, copy=False)
        if mono.size < max(96, int(0.03 * max(sample_rate, 1))):
            return None
        mono = mono - float(np.mean(mono))
        peak = float(np.max(np.abs(mono))) + 1e-8
        if not np.isfinite(peak) or peak <= 1e-8:
            return None
        mono = mono / peak
        n_fft = 1024 if int(sample_rate) >= 22050 else 512
        hop_length = max(64, n_fft // 4)
        fmax = float(min(8000.0, max(1200.0, sample_rate * 0.48)))
        try:
            mel = librosa.feature.melspectrogram(
                y=mono,
                sr=sample_rate,
                n_fft=n_fft,
                hop_length=hop_length,
                n_mels=64,
                fmin=60.0,
                fmax=fmax,
                power=2.0,
            )
            log_mel = librosa.power_to_db(np.maximum(mel, 1e-9))
            mfcc = librosa.feature.mfcc(S=log_mel, n_mfcc=20)
            delta = librosa.feature.delta(mfcc, mode="nearest")
            rms = librosa.feature.rms(y=mono, frame_length=n_fft, hop_length=hop_length)
            zcr = librosa.feature.zero_crossing_rate(y=mono, frame_length=n_fft, hop_length=hop_length)
            centroid = librosa.feature.spectral_centroid(
                y=mono,
                sr=sample_rate,
                n_fft=n_fft,
                hop_length=hop_length,
            )
            bandwidth = librosa.feature.spectral_bandwidth(
                y=mono,
                sr=sample_rate,
                n_fft=n_fft,
                hop_length=hop_length,
            )
            feature_stack = np.concatenate(
                [
                    mfcc,
                    delta,
                    rms,
                    zcr,
                    centroid / max(float(sample_rate), 1.0),
                    bandwidth / max(float(sample_rate), 1.0),
                ],
                axis=0,
            )
            vector = np.concatenate([feature_stack.mean(axis=1), feature_stack.std(axis=1)]).astype(np.float32)
        except Exception:
            return None
        if not np.all(np.isfinite(vector)):
            return None
        norm = float(np.linalg.norm(vector))
        if norm <= 1e-8:
            return None
        return (vector / norm).astype(np.float32, copy=False)

    def _embedding_similarity_score(self, left: np.ndarray, right: np.ndarray) -> float:
        left_vec = np.asarray(left, dtype=np.float32)
        right_vec = np.asarray(right, dtype=np.float32)
        if left_vec.size == 0 or right_vec.size == 0:
            return 0.0
        cosine = float(np.clip(np.dot(left_vec, right_vec), -1.0, 1.0))
        return float(np.clip((cosine * 50.0) + 50.0, 0.0, 100.0))

    def _build_bank_aware_word_priority_scores(
        self,
        *,
        source_audio: np.ndarray,
        sample_rate: int,
        intended_words: List[str],
        word_scores: List[Dict[str, object]],
        reference_bank_word_candidates_provider: Optional[
            Callable[[str, Dict[str, object]], List[Dict[str, object]]]
        ],
    ) -> Dict[int, float]:
        if reference_bank_word_candidates_provider is None:
            return {}
        total_samples = int(source_audio.shape[0])
        reference_cache: Dict[str, Optional[np.ndarray]] = {}
        bank_similarity: Dict[int, float] = {}

        for entry in word_scores:
            word_index = int(entry.get("index", -1))
            if word_index < 0 or word_index >= len(intended_words):
                continue
            target_word = normalize_lyrics(str(intended_words[word_index]))
            if not target_word:
                continue
            start, end = self._word_window_to_samples(
                [dict(item) for item in word_scores],
                [word_index],
                sample_rate,
                total_samples,
                padding_ms=58.0,
            )
            if end <= start:
                continue
            source_segment = np.asarray(source_audio[start:end], dtype=np.float32)
            source_embedding = self._build_pronunciation_embedding(source_segment, sample_rate)
            if source_embedding is None:
                continue
            try:
                candidates = reference_bank_word_candidates_provider(
                    target_word,
                    {
                        "word_index": word_index,
                        "word": target_word,
                        "start": float(entry.get("start", 0.0)),
                        "end": float(entry.get("end", 0.0)),
                    },
                )
            except Exception:
                candidates = []
            if not candidates:
                continue

            best_similarity: Optional[float] = None
            for bank_entry in list(candidates)[:6]:
                bank_file = Path(str(bank_entry.get("file_path", "")))
                if not bank_file.exists():
                    continue
                cache_key = f"{bank_file.resolve().as_posix()}::{int(sample_rate)}"
                cached_embedding = reference_cache.get(cache_key)
                if cached_embedding is None and cache_key not in reference_cache:
                    try:
                        reference_audio, _ = self._load_audio(bank_file, sample_rate=sample_rate)
                        cached_embedding = self._build_pronunciation_embedding(reference_audio, sample_rate)
                    except Exception:
                        cached_embedding = None
                    reference_cache[cache_key] = cached_embedding
                elif cache_key in reference_cache:
                    cached_embedding = reference_cache[cache_key]
                if cached_embedding is None:
                    continue
                similarity = self._embedding_similarity_score(source_embedding, cached_embedding)
                if best_similarity is None or similarity > best_similarity:
                    best_similarity = similarity

            if best_similarity is not None:
                bank_similarity[word_index] = float(np.clip(best_similarity, 0.0, 100.0))

        combined_priority: Dict[int, float] = {}
        for entry in word_scores:
            word_index = int(entry.get("index", -1))
            if word_index < 0:
                continue
            bank_score = bank_similarity.get(word_index)
            if bank_score is None:
                continue
            base_score = float(entry.get("similarity", 0.0))
            combined_priority[word_index] = float(
                np.clip((base_score * 0.55) + (bank_score * 0.45), 0.0, 100.0)
            )
        return combined_priority

    def _determine_target_word_indices(
        self,
        word_scores: List[Dict[str, float | str]],
        letter_scores: List[Dict[str, float | str]],
        *,
        max_words: int,
        word_priority_scores: Optional[Dict[int, float]] = None,
    ) -> List[int]:
        def score_for_word(entry: Dict[str, float | str]) -> float:
            index = int(entry.get("index", -1))
            if word_priority_scores is not None and index in word_priority_scores:
                return float(word_priority_scores[index])
            return float(entry.get("similarity", 0.0))

        weak_words = sorted(word_scores, key=score_for_word)
        chosen: List[int] = []
        for entry in weak_words:
            if score_for_word(entry) < 94.0:
                chosen.append(int(entry["index"]))
            if len(chosen) >= max_words:
                break
        if len(chosen) < max_words:
            weak_letter_words = [
                int(entry.get("word_index", -1))
                for entry in sorted(letter_scores, key=lambda entry: float(entry.get("similarity", 0.0)))
                if int(entry.get("word_index", -1)) >= 0
            ]
            for word_index in weak_letter_words:
                if word_index not in chosen:
                    chosen.append(word_index)
                if len(chosen) >= max_words:
                    break
        if not chosen and weak_words:
            chosen = [int(entry["index"]) for entry in weak_words[:max_words]]
        return sorted(set(chosen))

    def _apply_guided_ai_patch(
        self,
        source_segment: np.ndarray,
        *,
        sample_rate: int,
        weakness: float,
        ai_blend: float,
        detail_mix: float,
        transient_gain: float,
    ) -> np.ndarray:
        mono_source = source_segment.mean(axis=1).astype(np.float32, copy=False)
        enhanced_mono = self._enhance_mono_segment(mono_source, sample_rate)
        enhanced_mono = librosa.util.fix_length(enhanced_mono.astype(np.float32), size=mono_source.shape[0])

        source_low = gaussian_filter1d(mono_source, sigma=3.0, mode="nearest")
        enhanced_low = gaussian_filter1d(enhanced_mono, sigma=3.0, mode="nearest")
        source_detail = mono_source - source_low
        enhanced_detail = enhanced_mono - enhanced_low
        detail_layer = enhanced_detail - source_detail

        onset = np.abs(np.diff(np.concatenate([[enhanced_mono[0]], enhanced_mono])))
        onset = gaussian_filter1d(onset, sigma=1.0, mode="nearest")
        onset = onset / max(float(np.max(onset)), 1e-6)
        transient_layer = detail_layer * onset

        enhanced_stereo = np.repeat(enhanced_mono[:, np.newaxis], source_segment.shape[1], axis=1)
        repaired = ((1.0 - ai_blend) * source_segment) + (ai_blend * enhanced_stereo)
        repaired = repaired + (
            np.repeat(detail_layer[:, np.newaxis], source_segment.shape[1], axis=1)
            * detail_mix
            * (0.25 + weakness)
        )
        repaired = repaired + (
            np.repeat(transient_layer[:, np.newaxis], source_segment.shape[1], axis=1)
            * transient_gain
            * (0.15 + weakness)
        )
        return repaired.astype(np.float32, copy=False)

    def _render_guided_candidate(
        self,
        prepared: Dict[str, object],
        output_path: Path,
        params: Dict[str, float],
    ) -> Dict[str, object]:
        source_audio = np.asarray(prepared["source_audio"], dtype=np.float32)
        sample_rate = int(prepared["sample_rate"])
        word_scores = [dict(entry) for entry in prepared.get("word_scores", [])]
        letter_scores = [dict(entry) for entry in prepared.get("letter_scores", [])]
        output_audio = source_audio.copy()
        total_samples = output_audio.shape[0]

        target_word_indices = self._determine_target_word_indices(
            word_scores,
            letter_scores,
            max_words=max(2, min(8, int(params["max_target_words"]))),
        )
        groups = self._group_contiguous_indices(target_word_indices)
        fade_ms = float(params["fade_ms"])
        fade_samples = max(8, int((fade_ms / 1000.0) * sample_rate))

        for group in groups:
            start, end = self._word_window_to_samples(
                word_scores,
                group,
                sample_rate,
                total_samples,
                padding_ms=float(params["word_padding_ms"]),
            )
            if end <= start:
                continue
            segment = output_audio[start:end]
            if segment.shape[0] < int(0.05 * sample_rate):
                continue

            group_scores = [
                float(entry.get("similarity", 0.0))
                for entry in word_scores
                if int(entry.get("index", -1)) in group
            ]
            group_similarity = min(group_scores) if group_scores else 60.0
            weakness = 1.0 - (group_similarity / 100.0)
            patched = self._apply_guided_ai_patch(
                segment,
                sample_rate=sample_rate,
                weakness=weakness,
                ai_blend=float(params["ai_blend"]) * (0.45 + weakness),
                detail_mix=float(params["ai_detail_mix"]),
                transient_gain=float(params["ai_transient_gain"]),
            )

            fade_size = min(fade_samples, max(1, patched.shape[0] // 3))
            fade = self._cosine_fade(fade_size)
            mix_curve = np.ones(patched.shape[0], dtype=np.float32)
            mix_curve[:fade_size] = fade[:fade_size]
            mix_curve[-fade_size:] = fade[-fade_size:]
            output_audio[start:end] = (
                ((1.0 - mix_curve[:, np.newaxis]) * output_audio[start:end])
                + (mix_curve[:, np.newaxis] * patched)
            )

        peak = float(np.max(np.abs(output_audio))) if output_audio.size else 0.0
        if peak > 0.995:
            output_audio *= np.float32(0.995 / peak)

        output_path.parent.mkdir(parents=True, exist_ok=True)
        sf.write(output_path, output_audio, sample_rate, subtype="PCM_24")
        return {
            "output_path": output_path,
            "sample_rate": sample_rate,
            "output_audio": output_audio,
            "source_rms_db": self._safe_rms_db(source_audio),
            "output_rms_db": self._safe_rms_db(output_audio),
            "params": params,
            "target_word_indices": target_word_indices,
        }

    def _apply_micro_patch(
        self,
        source_segment: np.ndarray,
        params: Dict[str, float],
        weakness: float,
        sample_rate: int,
    ) -> np.ndarray:
        target_length = source_segment.shape[0]
        fft_size = self._choose_fft_size(target_length)
        hop_size = max(32, fft_size // 4)
        freqs = np.fft.rfftfreq(fft_size, d=1.0 / sample_rate)
        body_band = freqs < 700.0
        presence_band = (freqs >= 1500.0) & (freqs < 4500.0)
        sibilance_band = freqs >= 4500.0

        source_mono = source_segment.mean(axis=1).astype(np.float32, copy=False)
        smoothed = gaussian_filter1d(source_mono, sigma=float(params["time_smooth_sigma"]), mode="nearest")
        transient = source_mono - smoothed
        onset_env = np.abs(np.diff(np.concatenate([[source_mono[0]], source_mono])))
        onset_env = gaussian_filter1d(onset_env, sigma=float(params["onset_sigma"]), mode="nearest")
        onset_env = onset_env / max(float(np.max(onset_env)), 1e-6)
        letter_focus = np.linspace(1.0, float(params["tail_focus"]), target_length, dtype=np.float32)
        transient_drive = transient * onset_env * letter_focus

        repaired_channels = []
        for channel in range(source_segment.shape[1]):
            _, _, source_spec = stft(
                source_segment[:, channel].astype(np.float32),
                fs=sample_rate,
                window="hann",
                nperseg=fft_size,
                noverlap=fft_size - hop_size,
                boundary="zeros",
                padded=True,
            )
            source_mag = np.abs(source_spec).astype(np.float32, copy=False)
            source_phase = np.angle(source_spec).astype(np.float32, copy=False)
            smooth_mag = gaussian_filter1d(source_mag, sigma=float(params["freq_smooth_sigma"]), axis=0, mode="nearest")
            unsharp = np.maximum(source_mag - smooth_mag, 0.0)
            mixed_mag = source_mag.copy()

            presence_gain = 1.0 + (float(params["presence_gain"]) * weakness)
            sibilance_gain = 1.0 + (float(params["sibilance_gain"]) * weakness)
            body_anchor = 1.0 - (0.08 * weakness)

            if np.any(presence_band):
                mixed_mag[presence_band, :] *= presence_gain
                mixed_mag[presence_band, :] += unsharp[presence_band, :] * (
                    float(params["unsharp_amount"]) * weakness
                )

            if np.any(sibilance_band):
                mixed_mag[sibilance_band, :] *= sibilance_gain
                mixed_mag[sibilance_band, :] += unsharp[sibilance_band, :] * (
                    float(params["air_amount"]) * weakness
                )

            if np.any(body_band):
                mixed_mag[body_band, :] = (
                    body_anchor * source_mag[body_band, :]
                    + (1.0 - body_anchor) * mixed_mag[body_band, :]
                )

            harm_mag, perc_mag = librosa.decompose.hpss(
                mixed_mag,
                kernel_size=(13, 19),
                margin=(1.0, 2.5),
            )
            mixed_mag = harm_mag + (perc_mag * (1.0 + float(params["percussive_gain"]) * weakness))

            rebuilt_spec = mixed_mag * np.exp(1j * source_phase)
            _, repaired = istft(
                rebuilt_spec,
                fs=sample_rate,
                window="hann",
                nperseg=fft_size,
                noverlap=fft_size - hop_size,
                input_onesided=True,
                boundary=True,
            )
            repaired = librosa.util.fix_length(repaired.astype(np.float32), size=target_length)
            repaired_channels.append(repaired)

        repaired_audio = np.stack(repaired_channels, axis=1).astype(np.float32, copy=False)
        transient_boost = transient_drive[:, np.newaxis] * float(params["transient_gain"]) * (0.18 + 1.05 * weakness)
        repaired_audio = repaired_audio + transient_boost
        stereo_detail = source_segment - gaussian_filter1d(source_segment, sigma=2.0, axis=0, mode="nearest")
        repaired_audio = repaired_audio + (
            stereo_detail * float(params["detail_mix"]) * (0.08 + 0.55 * weakness)
        )
        blend = np.clip(float(params["local_blend"]) * (0.16 + 0.90 * weakness), 0.08, 0.92)
        return ((1.0 - blend) * source_segment) + (blend * repaired_audio)

    def _render_variant(
        self,
        prepared: Dict[str, object],
        output_path: Path,
        params: Dict[str, float],
    ) -> Dict[str, object]:
        source_audio = np.asarray(prepared["source_audio"], dtype=np.float32)
        sample_rate = int(prepared["sample_rate"])
        letter_scores = list(prepared.get("letter_scores", []))
        output_audio = source_audio.copy()
        total_samples = output_audio.shape[0]

        weak_threshold = float(params["weak_letter_threshold"])
        weak_letters = [
            entry for entry in letter_scores if float(entry.get("similarity", 0.0)) < weak_threshold
        ]
        if not weak_letters:
            weak_letters = sorted(letter_scores, key=lambda entry: float(entry.get("similarity", 0.0)))
        max_targets = max(1, min(int(params["max_target_letters"]), len(weak_letters) if weak_letters else 1))
        weak_letters = weak_letters[:max_targets]

        padding_ms = float(params["padding_ms"])
        fade_ms = float(params["fade_ms"])
        fade_samples = max(8, int((fade_ms / 1000.0) * sample_rate))
        target_word_indices: set[int] = set()

        for letter in weak_letters:
            word_index = int(letter.get("word_index", -1))
            if word_index >= 0:
                target_word_indices.add(word_index)
            start, end = self._letter_window_to_samples(letter, sample_rate, total_samples, padding_ms)
            segment = output_audio[start:end]
            if segment.shape[0] < int(0.04 * sample_rate):
                continue

            weakness = 1.0 - (float(letter.get("similarity", 0.0)) / 100.0)
            patched = self._apply_micro_patch(segment, params, weakness, sample_rate)

            fade_size = min(fade_samples, max(1, patched.shape[0] // 3))
            fade = self._cosine_fade(fade_size)
            mix_curve = np.ones(patched.shape[0], dtype=np.float32)
            mix_curve[:fade_size] = fade[:fade_size]
            mix_curve[-fade_size:] = fade[-fade_size:]
            output_audio[start:end] = (
                ((1.0 - mix_curve[:, np.newaxis]) * output_audio[start:end])
                + (mix_curve[:, np.newaxis] * patched)
            )

        peak = float(np.max(np.abs(output_audio))) if output_audio.size else 0.0
        if peak > 0.995:
            output_audio *= np.float32(0.995 / peak)

        output_path.parent.mkdir(parents=True, exist_ok=True)
        sf.write(output_path, output_audio, sample_rate, subtype="PCM_24")
        return {
            "output_path": output_path,
            "sample_rate": sample_rate,
            "output_audio": output_audio,
            "source_rms_db": self._safe_rms_db(source_audio),
            "output_rms_db": self._safe_rms_db(output_audio),
            "params": params,
            "target_word_indices": sorted(target_word_indices),
        }

    def _score_candidate(
        self,
        prepared: Dict[str, object],
        output_path: Path,
        params: Dict[str, float],
        intended_lyrics: str,
        candidate_mode: str = "micro",
    ) -> Dict[str, object]:
        if candidate_mode == "guided":
            metadata = self._render_guided_candidate(prepared, output_path, params)
        else:
            metadata = self._render_variant(prepared, output_path, params)
        full_output = np.asarray(metadata["output_audio"], dtype=np.float32)
        sample_rate = int(metadata["sample_rate"])
        intended_words = list(prepared.get("intended_words", lyrics_to_words(intended_lyrics)))
        previous_word_scores = [dict(entry) for entry in prepared.get("word_scores", [])]
        previous_letter_scores = [dict(entry) for entry in prepared.get("letter_scores", [])]
        target_word_indices = [int(index) for index in metadata.get("target_word_indices", [])]

        if not previous_word_scores or not target_word_indices:
            scoring = self.scorer.analyze_audio(full_output, sample_rate, intended_lyrics)
        else:
            expanded_indices: List[int] = []
            for index in target_word_indices:
                expanded_indices.extend(range(max(0, index - 1), min(len(intended_words), index + 2)))
            groups = self._group_contiguous_indices(expanded_indices)
            word_by_index = {int(entry["index"]): dict(entry) for entry in previous_word_scores}
            rescored_indices = {index for group in groups for index in group}
            merged_letter_scores = [
                dict(entry)
                for entry in previous_letter_scores
                if int(entry.get("word_index", -1)) not in rescored_indices
            ]

            full_duration_seconds = full_output.shape[0] / max(sample_rate, 1)
            for group in groups:
                if not group:
                    continue
                first_index = group[0]
                last_index = group[-1]
                if first_index not in word_by_index or last_index not in word_by_index:
                    continue
                segment_start = max(0.0, float(word_by_index[first_index]["start"]) - 0.08)
                segment_end = min(full_duration_seconds, float(word_by_index[last_index]["end"]) + 0.08)
                start_sample = max(0, int(segment_start * sample_rate))
                end_sample = min(full_output.shape[0], int(segment_end * sample_rate))
                if end_sample <= start_sample:
                    continue
                local_words = intended_words[first_index : last_index + 1]
                local_result = self.scorer.analyze_segment(
                    full_output[start_sample:end_sample],
                    sample_rate,
                    local_words,
                    global_start_seconds=(start_sample / sample_rate),
                    absolute_word_indices=list(range(first_index, last_index + 1)),
                    letter_focus_limit=max(4, min(12, len(local_words) * 2)),
                )
                for entry in local_result["word_scores"]:
                    word_by_index[int(entry["index"])] = dict(entry)
                merged_letter_scores.extend(dict(entry) for entry in local_result["letter_scores"])

            merged_word_scores = [word_by_index[index] for index in sorted(word_by_index)]
            scoring = self.scorer.build_analysis_result(merged_word_scores, merged_letter_scores)

        metadata["word_report"] = str(scoring["word_report"])
        metadata["letter_report"] = str(scoring["letter_report"])
        metadata["word_scores"] = list(scoring["word_scores"])
        metadata["letter_scores"] = list(scoring["letter_scores"])
        metadata["similarity_score"] = float(scoring["similarity_score"])
        return metadata

    def _score_audio_for_indices(
        self,
        *,
        full_output: np.ndarray,
        sample_rate: int,
        intended_words: List[str],
        previous_word_scores: List[Dict[str, object]],
        previous_letter_scores: List[Dict[str, object]],
        target_word_indices: List[int],
        intended_lyrics: str,
    ) -> Dict[str, object]:
        if not previous_word_scores or not target_word_indices:
            return self.scorer.analyze_audio(full_output, sample_rate, intended_lyrics)

        expanded_indices: List[int] = []
        for index in target_word_indices:
            expanded_indices.extend(range(max(0, index - 1), min(len(intended_words), index + 2)))
        groups = self._group_contiguous_indices(expanded_indices)
        word_by_index = {int(entry["index"]): dict(entry) for entry in previous_word_scores}
        rescored_indices = {index for group in groups for index in group}
        merged_letter_scores = [
            dict(entry)
            for entry in previous_letter_scores
            if int(entry.get("word_index", -1)) not in rescored_indices
        ]

        full_duration_seconds = full_output.shape[0] / max(sample_rate, 1)
        for group in groups:
            if not group:
                continue
            first_index = group[0]
            last_index = group[-1]
            if first_index not in word_by_index or last_index not in word_by_index:
                continue
            segment_start = max(0.0, float(word_by_index[first_index]["start"]) - 0.08)
            segment_end = min(full_duration_seconds, float(word_by_index[last_index]["end"]) + 0.08)
            start_sample = max(0, int(segment_start * sample_rate))
            end_sample = min(full_output.shape[0], int(segment_end * sample_rate))
            if end_sample <= start_sample:
                continue
            local_words = intended_words[first_index : last_index + 1]
            local_result = self.scorer.analyze_segment(
                full_output[start_sample:end_sample],
                sample_rate,
                local_words,
                global_start_seconds=(start_sample / sample_rate),
                absolute_word_indices=list(range(first_index, last_index + 1)),
                letter_focus_limit=max(4, min(12, len(local_words) * 2)),
            )
            for entry in local_result["word_scores"]:
                word_by_index[int(entry["index"])] = dict(entry)
            merged_letter_scores.extend(dict(entry) for entry in local_result["letter_scores"])

        merged_word_scores = [word_by_index[index] for index in sorted(word_by_index)]
        return self.scorer.build_analysis_result(merged_word_scores, merged_letter_scores)

    def _render_regenerated_candidate(
        self,
        prepared: Dict[str, object],
        output_path: Path,
        *,
        word_indices: List[int],
        phrase_text: str,
        speaker_wav: Path,
        blend: float,
        padding_ms: float,
    ) -> Dict[str, object]:
        source_audio = np.asarray(prepared["source_audio"], dtype=np.float32)
        sample_rate = int(prepared["sample_rate"])
        word_scores = [dict(entry) for entry in prepared.get("word_scores", [])]
        output_audio = source_audio.copy()
        total_samples = output_audio.shape[0]

        start, end = self._word_window_to_samples(
            word_scores,
            word_indices,
            sample_rate,
            total_samples,
            padding_ms=padding_ms,
        )
        if end <= start:
            raise RuntimeError("Could not determine the target word timing window.")

        generated_path = output_path.parent / f"{output_path.stem}_xtts.wav"
        self._run_xtts_regenerator(
            text=phrase_text,
            speaker_wav=speaker_wav,
            output_path=generated_path,
        )
        generated_audio, _ = self._load_audio(generated_path, sample_rate=sample_rate)
        generated_audio = self._fit_audio_length(generated_audio, end - start)

        source_segment = output_audio[start:end]
        mixed_segment = ((1.0 - blend) * source_segment) + (blend * generated_audio)
        fade_size = min(max(8, int(0.02 * sample_rate)), max(1, mixed_segment.shape[0] // 3))
        fade = self._cosine_fade(fade_size)
        mix_curve = np.ones(mixed_segment.shape[0], dtype=np.float32)
        mix_curve[:fade_size] = fade[:fade_size]
        mix_curve[-fade_size:] = fade[-fade_size:]
        output_audio[start:end] = (
            ((1.0 - mix_curve[:, np.newaxis]) * source_segment)
            + (mix_curve[:, np.newaxis] * mixed_segment)
        )

        peak = float(np.max(np.abs(output_audio))) if output_audio.size else 0.0
        if peak > 0.995:
            output_audio *= np.float32(0.995 / peak)

        output_path.parent.mkdir(parents=True, exist_ok=True)
        sf.write(output_path, output_audio, sample_rate, subtype="PCM_24")
        return {
            "output_path": output_path,
            "sample_rate": sample_rate,
            "output_audio": output_audio,
            "source_rms_db": self._safe_rms_db(source_audio),
            "output_rms_db": self._safe_rms_db(output_audio),
            "target_word_indices": sorted(word_indices),
        }

    def _build_guided_candidate_params(self, strength: int) -> List[Dict[str, float]]:
        base_blend = float(np.clip(0.22 + (0.40 * (strength / 100.0)), 0.18, 0.72))
        return [
            {
                "strength": float(strength),
                "ai_blend": base_blend,
                "ai_detail_mix": 0.32,
                "ai_transient_gain": 0.48,
                "word_padding_ms": 85.0,
                "fade_ms": 24.0,
                "max_target_words": 4.0,
            },
            {
                "strength": float(strength),
                "ai_blend": min(0.82, base_blend + 0.14),
                "ai_detail_mix": 0.46,
                "ai_transient_gain": 0.68,
                "word_padding_ms": 105.0,
                "fade_ms": 28.0,
                "max_target_words": 6.0,
            },
            {
                "strength": float(strength),
                "ai_blend": min(0.9, base_blend + 0.22),
                "ai_detail_mix": 0.62,
                "ai_transient_gain": 0.84,
                "word_padding_ms": 120.0,
                "fade_ms": 32.0,
                "max_target_words": 8.0,
            },
        ]

    def _build_micro_candidate_base_params(
        self,
        *,
        strength: int,
        max_target_words: int,
    ) -> Dict[str, float]:
        strength_ratio = float(np.clip(strength / 100.0, 0.0, 1.0))
        return {
            "strength": float(strength),
            "presence_gain": float(0.65 + (1.05 * strength_ratio)),
            "sibilance_gain": float(0.55 + (1.15 * strength_ratio)),
            "transient_gain": float(0.45 + (1.05 * strength_ratio)),
            "percussive_gain": float(0.12 + (0.78 * strength_ratio)),
            "unsharp_amount": float(0.28 + (0.90 * strength_ratio)),
            "air_amount": float(0.06 + (0.54 * strength_ratio)),
            "detail_mix": float(0.14 + (0.52 * strength_ratio)),
            "local_blend": float(0.24 + (0.38 * strength_ratio)),
            "freq_smooth_sigma": float(np.clip(1.45 - (0.55 * strength_ratio), 0.45, 2.1)),
            "time_smooth_sigma": float(np.clip(1.95 - (0.65 * strength_ratio), 0.55, 2.8)),
            "onset_sigma": float(np.clip(1.2 - (0.25 * strength_ratio), 0.25, 1.4)),
            "tail_focus": float(1.12 + (0.92 * strength_ratio)),
            "weak_letter_threshold": float(np.clip(82.0 - (26.0 * strength_ratio), 40.0, 88.0)),
            "max_target_letters": float(max(4, min(26, int(max_target_words) * 4))),
            "padding_ms": float(44.0 + (52.0 * strength_ratio)),
            "fade_ms": float(14.0 + (10.0 * strength_ratio)),
            "max_target_words": float(max_target_words),
        }

    def _build_variant_params(
        self,
        *,
        base_params: Dict[str, float],
        batch_index: int,
        variants_per_batch: int,
    ) -> List[Dict[str, float]]:
        rng = random.Random((batch_index * 92821) + int(base_params["strength"] * 137))
        variants: List[Dict[str, float]] = [dict(base_params)]
        while len(variants) < variants_per_batch:
            variants.append(
                {
                    "strength": float(np.clip(base_params["strength"] + rng.randint(-20, 20), 1, 100)),
                    "presence_gain": float(np.clip(base_params["presence_gain"] + rng.uniform(-0.25, 0.35), 0.10, 2.50)),
                    "sibilance_gain": float(np.clip(base_params["sibilance_gain"] + rng.uniform(-0.25, 0.35), 0.05, 2.80)),
                    "transient_gain": float(np.clip(base_params["transient_gain"] + rng.uniform(-0.25, 0.35), 0.05, 2.50)),
                    "percussive_gain": float(np.clip(base_params["percussive_gain"] + rng.uniform(-0.18, 0.26), 0.00, 2.20)),
                    "unsharp_amount": float(np.clip(base_params["unsharp_amount"] + rng.uniform(-0.18, 0.24), 0.00, 2.40)),
                    "air_amount": float(np.clip(base_params["air_amount"] + rng.uniform(-0.10, 0.18), 0.00, 1.80)),
                    "detail_mix": float(np.clip(base_params["detail_mix"] + rng.uniform(-0.10, 0.18), 0.05, 1.2)),
                    "local_blend": float(np.clip(base_params["local_blend"] + rng.uniform(-0.12, 0.18), 0.08, 0.95)),
                    "freq_smooth_sigma": float(np.clip(base_params["freq_smooth_sigma"] + rng.uniform(-0.4, 0.8), 0.2, 4.0)),
                    "time_smooth_sigma": float(np.clip(base_params["time_smooth_sigma"] + rng.uniform(-0.6, 0.8), 0.4, 6.0)),
                    "onset_sigma": float(np.clip(base_params["onset_sigma"] + rng.uniform(-0.4, 0.6), 0.2, 4.0)),
                    "tail_focus": float(np.clip(base_params["tail_focus"] + rng.uniform(-0.3, 0.4), 0.8, 2.6)),
                    "weak_letter_threshold": float(np.clip(base_params["weak_letter_threshold"] + rng.uniform(-10.0, 10.0), 25.0, 90.0)),
                    "max_target_letters": float(np.clip(base_params["max_target_letters"] + rng.randint(-2, 3), 1, 30)),
                    "padding_ms": float(np.clip(base_params["padding_ms"] + rng.uniform(-15.0, 25.0), 20.0, 180.0)),
                    "fade_ms": float(np.clip(base_params["fade_ms"] + rng.uniform(-6.0, 8.0), 6.0, 70.0)),
                }
            )
        return variants[:variants_per_batch]

    def optimize_pronunciation(
        self,
        *,
        source_path: Path,
        intended_lyrics: str,
        output_dir: Path,
        strength: int,
        variants_per_batch: int,
        parallel_variants: int,
        max_batches: int,
        cancel_event: threading.Event,
        update_status: Optional[Callable[[Dict[str, object]], None]] = None,
        mode: str = "detect-regenerate",
        max_target_words: int = 5,
    ) -> Dict[str, object]:
        output_dir.mkdir(parents=True, exist_ok=True)
        candidates_dir = output_dir / "candidates"
        candidates_dir.mkdir(parents=True, exist_ok=True)

        source_audio, sample_rate = self._load_audio(source_path, sample_rate=44100)
        intended_words = lyrics_to_words(intended_lyrics)
        baseline_scoring = self.scorer.analyze_audio(source_audio, sample_rate, intended_lyrics)
        baseline_score = float(baseline_scoring["similarity_score"])
        best_output_path = output_dir / f"{source_path.stem}_best_touchup.wav"
        sf.write(best_output_path, source_audio, sample_rate, subtype="PCM_24")

        best_state: Dict[str, object] = {
            "output_path": best_output_path,
            "sample_rate": sample_rate,
            "source_rms_db": self._safe_rms_db(source_audio),
            "output_rms_db": self._safe_rms_db(source_audio),
            "output_audio": source_audio.copy(),
            "word_report": str(baseline_scoring["word_report"]),
            "letter_report": str(baseline_scoring["letter_report"]),
            "word_scores": list(baseline_scoring["word_scores"]),
            "letter_scores": list(baseline_scoring["letter_scores"]),
            "similarity_score": baseline_score,
            "params": {"mode": "baseline"},
        }

        supported_modes = {"detect-only", "detect-regenerate", "repair", "auto-repair"}
        normalized_mode = str(mode or "detect-regenerate").strip().lower()
        if normalized_mode not in supported_modes:
            normalized_mode = "detect-regenerate"

        attempts_tested = 0
        target_word_indices = self._determine_target_word_indices(
            best_state["word_scores"],
            best_state["letter_scores"],
            max_words=max(1, min(int(max_target_words), 12)),
        )
        regeneration_status = self.get_regenerator_status()
        regeneration_available = bool(regeneration_status["available"])
        allow_local_repairs = normalized_mode != "detect-only"
        allow_xtts_regeneration = (
            normalized_mode in {"detect-regenerate", "repair", "auto-repair"}
            and regeneration_available
        )
        detected_only = not allow_local_repairs and not allow_xtts_regeneration
        repaired_word_indices: set[int] = set()

        def working_state_payload(current_state: Dict[str, object]) -> Dict[str, object]:
            return {
                "source_audio": np.asarray(current_state["output_audio"], dtype=np.float32).copy(),
                "sample_rate": sample_rate,
                "word_scores": list(current_state["word_scores"]),
                "letter_scores": list(current_state["letter_scores"]),
                "intended_words": list(intended_words),
            }

        def accept_candidate(candidate: Dict[str, object]) -> bool:
            nonlocal best_state
            candidate_score = float(candidate.get("similarity_score", 0.0))
            best_score = float(best_state.get("similarity_score", 0.0))
            if candidate_score <= (best_score + 0.05):
                return False
            best_state = candidate
            shutil.copy2(str(candidate["output_path"]), str(best_output_path))
            best_state["output_path"] = best_output_path
            repaired_word_indices.update(
                int(index) for index in candidate.get("target_word_indices", []) if int(index) >= 0
            )
            return True

        def emit_status(message: str, *, progress: int) -> None:
            if update_status is None:
                return
            current_targets = self._determine_target_word_indices(
                list(best_state["word_scores"]),
                list(best_state["letter_scores"]),
                max_words=max(1, min(int(max_target_words), 12)),
            )
            update_status(
                {
                    "progress": int(progress),
                    "message": message,
                    "best_similarity_score": float(best_state["similarity_score"]),
                    "best_word_report": str(best_state["word_report"]),
                    "best_letter_report": str(best_state["letter_report"]),
                    "variants_tested": attempts_tested,
                    "repair_attempts": attempts_tested,
                    "repaired_word_count": len(repaired_word_indices),
                    "batch_index": attempts_tested,
                    "detected_word_indices": list(current_targets),
                    "regeneration_available": regeneration_available,
                    "regeneration_reason": str(regeneration_status["reason"]),
                    "detected_only": detected_only,
                }
            )

        weak_summary = ", ".join(intended_words[index] for index in target_word_indices[:8]) if target_word_indices else "none"
        emit_status(
            (
                f"Detected {len(target_word_indices)} weak word regions. "
                f"Priority words: {weak_summary}."
            ),
            progress=max(12, int(best_state["similarity_score"])),
        )

        if allow_local_repairs:
            guided_candidates = self._build_guided_candidate_params(strength)
            for candidate_index, params in enumerate(guided_candidates, start=1):
                if cancel_event.is_set():
                    break
                emit_status(
                    f"Testing guided repair pass {candidate_index}/{len(guided_candidates)}.",
                    progress=min(52, max(18, int(best_state["similarity_score"]))),
                )
                candidate_path = candidates_dir / f"guided_{candidate_index:02d}.wav"
                attempts_tested += 1
                try:
                    candidate = self._score_candidate(
                        working_state_payload(best_state),
                        candidate_path,
                        params,
                        intended_lyrics,
                        candidate_mode="guided",
                    )
                    candidate["params"] = {"mode": "guided", **params}
                    accept_candidate(candidate)
                except Exception:
                    pass
                emit_status(
                    (
                        f"Guided repair pass {candidate_index}/{len(guided_candidates)} finished. "
                        f"Best ordered-word similarity is {float(best_state['similarity_score']):.2f}%."
                    ),
                    progress=min(60, max(24, int(best_state["similarity_score"]))),
                )

            micro_base_params = self._build_micro_candidate_base_params(
                strength=strength,
                max_target_words=max_target_words,
            )
            total_batches = max(1, int(max_batches))
            variants_per_batch = max(1, int(variants_per_batch))
            for batch_index in range(total_batches):
                if cancel_event.is_set():
                    break
                params_list = self._build_variant_params(
                    base_params=micro_base_params,
                    batch_index=batch_index,
                    variants_per_batch=variants_per_batch,
                )
                for variant_index, params in enumerate(params_list, start=1):
                    if cancel_event.is_set():
                        break
                    emit_status(
                        (
                            f"Testing consonant repair variant {variant_index}/{len(params_list)} "
                            f"in batch {batch_index + 1}/{total_batches}."
                        ),
                        progress=min(78, max(28, int(best_state["similarity_score"]))),
                    )
                    candidate_path = candidates_dir / (
                        f"micro_b{batch_index + 1:02d}_v{variant_index:02d}.wav"
                    )
                    attempts_tested += 1
                    try:
                        candidate = self._score_candidate(
                            working_state_payload(best_state),
                            candidate_path,
                            params,
                            intended_lyrics,
                            candidate_mode="micro",
                        )
                        candidate["params"] = {"mode": "micro", **params}
                        if accept_candidate(candidate):
                            micro_base_params = dict(params)
                    except Exception:
                        pass

                emit_status(
                    (
                        f"Finished consonant repair batch {batch_index + 1}/{total_batches}. "
                        f"Best ordered-word similarity is {float(best_state['similarity_score']):.2f}%."
                    ),
                    progress=min(84, max(34, int(best_state["similarity_score"]))),
                )

        if allow_xtts_regeneration and not cancel_event.is_set():
            speaker_reference = best_output_path
            target_word_indices = self._determine_target_word_indices(
                list(best_state["word_scores"]),
                list(best_state["letter_scores"]),
                max_words=max(1, min(int(max_target_words), 12)),
            )
            contiguous_groups = self._group_contiguous_indices(target_word_indices)
            for group_index, word_group in enumerate(contiguous_groups, start=1):
                if cancel_event.is_set():
                    break
                phrase_text = " ".join(intended_words[index] for index in word_group).strip()
                if not phrase_text:
                    continue
                emit_status(
                    f"Regenerating weak region {group_index}/{len(contiguous_groups)}: \"{phrase_text}\".",
                    progress=min(92, max(40, int(best_state["similarity_score"]))),
                )
                candidate_path = candidates_dir / f"regen_{group_index:02d}.wav"
                attempts_tested += 1
                try:
                    candidate = self._render_regenerated_candidate(
                        working_state_payload(best_state),
                        candidate_path,
                        word_indices=word_group,
                        phrase_text=phrase_text,
                        speaker_wav=speaker_reference,
                        blend=float(np.clip(0.58 + (0.30 * (strength / 100.0)), 0.45, 0.92)),
                        padding_ms=95.0,
                    )
                    scoring = self._score_audio_for_indices(
                        full_output=np.asarray(candidate["output_audio"], dtype=np.float32),
                        sample_rate=int(candidate["sample_rate"]),
                        intended_words=intended_words,
                        previous_word_scores=list(best_state["word_scores"]),
                        previous_letter_scores=list(best_state["letter_scores"]),
                        target_word_indices=list(word_group),
                        intended_lyrics=intended_lyrics,
                    )
                    candidate["word_report"] = str(scoring["word_report"])
                    candidate["letter_report"] = str(scoring["letter_report"])
                    candidate["word_scores"] = list(scoring["word_scores"])
                    candidate["letter_scores"] = list(scoring["letter_scores"])
                    candidate["similarity_score"] = float(scoring["similarity_score"])
                    candidate["params"] = {"mode": "xtts-regenerated"}
                    candidate["target_word_indices"] = list(word_group)
                    accept_candidate(candidate)
                except Exception:
                    pass
                emit_status(
                    (
                        f"Finished XTTS region {group_index}/{len(contiguous_groups)}. "
                        f"Best ordered-word similarity is {float(best_state['similarity_score']):.2f}%."
                    ),
                    progress=min(96, max(44, int(best_state["similarity_score"]))),
                )

        return {
            "output_path": best_output_path,
            "sample_rate": sample_rate,
            "strength": int(strength),
            "source_word": intended_lyrics.strip(),
            "source_rms_db": float(best_state["source_rms_db"]),
            "output_rms_db": float(best_state["output_rms_db"]),
            "best_similarity_score": float(best_state["similarity_score"]),
            "best_word_report": str(best_state["word_report"]),
            "best_letter_report": str(best_state["letter_report"]),
            "best_word_scores": list(best_state["word_scores"]),
            "best_letter_scores": list(best_state["letter_scores"]),
            "variants_tested": attempts_tested,
            "batch_index": attempts_tested,
            "stopped_early": bool(cancel_event.is_set()),
            "detected_word_indices": list(
                self._determine_target_word_indices(
                    list(best_state["word_scores"]),
                    list(best_state["letter_scores"]),
                    max_words=max(1, min(int(max_target_words), 12)),
                )
            ),
            "regeneration_available": regeneration_available,
            "regeneration_reason": str(regeneration_status["reason"]),
            "detected_only": bool(detected_only),
            "repair_attempts": int(attempts_tested),
            "repaired_word_count": int(len(repaired_word_indices)),
        }

    def repair_with_reference_phrase_patches(
        self,
        *,
        source_path: Path,
        reference_path: Path,
        intended_lyrics: str,
        output_dir: Path,
        cancel_event: threading.Event,
        max_target_words: int,
        patch_renderer: Callable[[Path, Path, str, Dict[str, object]], Path],
        reference_bank_candidates_provider: Optional[
            Callable[[str, List[str], Dict[str, object]], List[Dict[str, object]]]
        ] = None,
        reference_bank_word_candidates_provider: Optional[
            Callable[[str, Dict[str, object]], List[Dict[str, object]]]
        ] = None,
        update_status: Optional[Callable[[Dict[str, object]], None]] = None,
        blend_strength: float = 0.82,
        padding_ms: float = 90.0,
        replacement_strategy: str = "blend",
    ) -> Dict[str, object]:
        output_dir.mkdir(parents=True, exist_ok=True)
        references_dir = output_dir / "reference-segments"
        rendered_dir = output_dir / "rendered-patches"
        references_dir.mkdir(parents=True, exist_ok=True)
        rendered_dir.mkdir(parents=True, exist_ok=True)

        source_audio, sample_rate = self._load_audio(source_path, sample_rate=44100)
        reference_audio, reference_rate = self._load_audio(reference_path, sample_rate=44100)
        if int(reference_rate) != int(sample_rate):
            raise RuntimeError("Reference patch repair expects matching sample rates.")

        intended_words = lyrics_to_words(intended_lyrics)
        baseline_scoring = self.scorer.analyze_audio(source_audio, sample_rate, intended_lyrics)
        reference_scoring = self.scorer.analyze_audio(reference_audio, sample_rate, intended_lyrics)
        best_output_path = output_dir / f"{source_path.stem}_reference_phrase_repaired.wav"
        sf.write(best_output_path, source_audio, sample_rate, subtype="PCM_24")

        best_state: Dict[str, object] = {
            "output_path": best_output_path,
            "sample_rate": sample_rate,
            "source_rms_db": self._safe_rms_db(source_audio),
            "output_rms_db": self._safe_rms_db(source_audio),
            "output_audio": source_audio.copy(),
            "word_report": str(baseline_scoring["word_report"]),
            "letter_report": str(baseline_scoring["letter_report"]),
            "word_scores": list(baseline_scoring["word_scores"]),
            "letter_scores": list(baseline_scoring["letter_scores"]),
            "similarity_score": float(baseline_scoring["similarity_score"]),
            "params": {"mode": "baseline"},
        }

        word_priority_scores = self._build_bank_aware_word_priority_scores(
            source_audio=np.asarray(source_audio, dtype=np.float32),
            sample_rate=sample_rate,
            intended_words=intended_words,
            word_scores=[dict(entry) for entry in best_state["word_scores"]],
            reference_bank_word_candidates_provider=reference_bank_word_candidates_provider,
        )

        target_word_indices = self._determine_target_word_indices(
            best_state["word_scores"],
            best_state["letter_scores"],
            max_words=max(1, min(int(max_target_words), 12)),
            word_priority_scores=word_priority_scores,
        )
        contiguous_groups = self._group_contiguous_indices(target_word_indices)
        repaired_word_indices: set[int] = set()
        attempts_tested = 0
        normalized_replacement_strategy = str(replacement_strategy or "blend").strip().lower()
        if normalized_replacement_strategy not in {"blend", "replace"}:
            normalized_replacement_strategy = "blend"

        def try_candidate_patch(
            *,
            patch_audio: np.ndarray,
            source_segment: np.ndarray,
            target_start: int,
            target_end: int,
            word_group: List[int],
            mode: str,
            extra_params: Optional[Dict[str, object]] = None,
            blend_floor: float,
            blend_ceil: float,
        ) -> bool:
            nonlocal best_state

            fitted_patch = self._fit_audio_length(
                np.asarray(patch_audio, dtype=np.float32),
                source_segment.shape[0],
            )
            source_rms = float(np.sqrt(np.mean(np.square(source_segment)) + 1e-9))
            patch_rms = float(np.sqrt(np.mean(np.square(fitted_patch)) + 1e-9))
            if patch_rms > 1e-7:
                patch_gain = float(np.clip(source_rms / patch_rms, 0.72, 1.35))
                fitted_patch = fitted_patch * patch_gain

            group_scores = [
                float(entry.get("similarity", 0.0))
                for entry in best_state["word_scores"]
                if int(entry.get("index", -1)) in word_group
            ]
            group_similarity = min(group_scores) if group_scores else 60.0
            weakness = 1.0 - (group_similarity / 100.0)
            blend = float(np.clip(blend_floor + ((blend_ceil - blend_floor) * weakness), blend_floor, blend_ceil))
            mixed_segment = ((1.0 - blend) * source_segment) + (blend * fitted_patch)

            fade_size = min(
                max(8, int(0.025 * sample_rate)),
                max(1, mixed_segment.shape[0] // 3),
            )
            fade = self._cosine_fade(fade_size)
            mix_curve = np.ones(mixed_segment.shape[0], dtype=np.float32)
            mix_curve[:fade_size] = fade[:fade_size]
            mix_curve[-fade_size:] = fade[-fade_size:]

            candidate_audio = np.asarray(best_state["output_audio"], dtype=np.float32).copy()
            if normalized_replacement_strategy == "replace":
                candidate_audio[target_start:target_end] = fitted_patch
                if fade_size > 1:
                    candidate_audio[target_start:target_end] = (
                        ((1.0 - mix_curve[:, np.newaxis]) * source_segment)
                        + (mix_curve[:, np.newaxis] * fitted_patch)
                    )
            else:
                candidate_audio[target_start:target_end] = (
                    ((1.0 - mix_curve[:, np.newaxis]) * source_segment)
                    + (mix_curve[:, np.newaxis] * mixed_segment)
                )
            peak = float(np.max(np.abs(candidate_audio))) if candidate_audio.size else 0.0
            if peak > 0.995:
                candidate_audio *= np.float32(0.995 / peak)

            scoring = self._score_audio_for_indices(
                full_output=candidate_audio,
                sample_rate=sample_rate,
                intended_words=intended_words,
                previous_word_scores=list(best_state["word_scores"]),
                previous_letter_scores=list(best_state["letter_scores"]),
                target_word_indices=list(word_group),
                intended_lyrics=intended_lyrics,
            )
            candidate_score = float(scoring.get("similarity_score", 0.0))
            if candidate_score > (float(best_state["similarity_score"]) + 0.05):
                best_state = {
                    "output_path": best_output_path,
                    "sample_rate": sample_rate,
                    "source_rms_db": self._safe_rms_db(source_audio),
                    "output_rms_db": self._safe_rms_db(candidate_audio),
                    "output_audio": candidate_audio,
                    "word_report": str(scoring["word_report"]),
                    "letter_report": str(scoring["letter_report"]),
                    "word_scores": list(scoring["word_scores"]),
                    "letter_scores": list(scoring["letter_scores"]),
                    "similarity_score": candidate_score,
                    "params": {
                        "mode": mode,
                        "blend": blend,
                        "replacement_strategy": normalized_replacement_strategy,
                        **(extra_params or {}),
                    },
                }
                repaired_word_indices.update(int(index) for index in word_group if int(index) >= 0)
                sf.write(best_output_path, candidate_audio, sample_rate, subtype="PCM_24")
                return True
            return False

        def emit_status(message: str, *, progress: int) -> None:
            if update_status is None:
                return
            update_status(
                {
                    "progress": int(progress),
                    "message": message,
                    "best_similarity_score": float(best_state["similarity_score"]),
                    "best_word_report": str(best_state["word_report"]),
                    "best_letter_report": str(best_state["letter_report"]),
                    "variants_tested": attempts_tested,
                    "repair_attempts": attempts_tested,
                    "repaired_word_count": len(repaired_word_indices),
                    "batch_index": attempts_tested,
                    "detected_word_indices": list(target_word_indices),
                }
            )

        emit_status(
            (
                f"Reference phrase repair found {len(contiguous_groups)} weak regions. "
                f"Trying targeted phrase rerenders"
                f"{' with full gap replacement' if normalized_replacement_strategy == 'replace' else ''}."
            ),
            progress=max(12, int(best_state["similarity_score"])),
        )

        total_source_samples = int(reference_audio.shape[0])
        total_target_samples = int(source_audio.shape[0])
        reference_word_scores = list(reference_scoring["word_scores"])

        for group_index, word_group in enumerate(contiguous_groups, start=1):
            if cancel_event.is_set():
                break
            phrase_words = [
                intended_words[index]
                for index in word_group
                if 0 <= int(index) < len(intended_words)
            ]
            phrase_text = " ".join(phrase_words).strip()
            if not phrase_text:
                continue

            emit_status(
                (
                    f"Rendering phrase patch {group_index}/{len(contiguous_groups)}: "
                    f"\"{phrase_text}\"."
                ),
                progress=min(90, max(18, int(best_state["similarity_score"]))),
            )

            target_start, target_end = self._word_window_to_samples(
                list(best_state["word_scores"]),
                word_group,
                sample_rate,
                total_target_samples,
                padding_ms=padding_ms,
            )
            reference_start, reference_end = self._word_window_to_samples(
                reference_word_scores,
                word_group,
                sample_rate,
                total_source_samples,
                padding_ms=(padding_ms + 15.0),
            )
            if target_end <= target_start or reference_end <= reference_start:
                continue

            reference_segment_path = references_dir / f"reference_group_{group_index:02d}.wav"
            sf.write(
                reference_segment_path,
                reference_audio[reference_start:reference_end],
                sample_rate,
                subtype="PCM_24",
            )

            rendered_patch_path = rendered_dir / f"rendered_group_{group_index:02d}.wav"
            attempts_tested += 1
            try:
                target_phrase_scores = [
                    {
                        "index": local_index,
                        "word": intended_words[int(word_index)]
                        if 0 <= int(word_index) < len(intended_words)
                        else str(entry.get("word", "")),
                        "start": max(0.0, float(entry.get("start", 0.0)) - (target_start / float(sample_rate))),
                        "end": max(0.0, float(entry.get("end", 0.0)) - (target_start / float(sample_rate))),
                        "similarity": float(entry.get("similarity", 0.0)),
                    }
                    for local_index, (word_index, entry) in enumerate(
                        (
                            (int(entry.get("index", -1)), entry)
                            for entry in best_state["word_scores"]
                            if int(entry.get("index", -1)) in word_group
                        ),
                        start=0,
                    )
                ]
                reference_phrase_scores = [
                    {
                        "index": local_index,
                        "word": intended_words[int(word_index)]
                        if 0 <= int(word_index) < len(intended_words)
                        else str(entry.get("word", "")),
                        "start": max(0.0, float(entry.get("start", 0.0)) - (reference_start / float(sample_rate))),
                        "end": max(0.0, float(entry.get("end", 0.0)) - (reference_start / float(sample_rate))),
                        "similarity": float(entry.get("similarity", 0.0)),
                    }
                    for local_index, (word_index, entry) in enumerate(
                        (
                            (int(entry.get("index", -1)), entry)
                            for entry in reference_word_scores
                            if int(entry.get("index", -1)) in word_group
                        ),
                        start=0,
                    )
                ]
                rendered_output_path = patch_renderer(
                    reference_segment_path,
                    rendered_patch_path,
                    phrase_text,
                    {
                        "group_index": group_index,
                        "word_indices": list(word_group),
                        "target_start": target_start,
                        "target_end": target_end,
                        "reference_start": reference_start,
                        "reference_end": reference_end,
                        "target_phrase_word_scores": target_phrase_scores,
                        "reference_phrase_word_scores": reference_phrase_scores,
                    },
                )
                patch_audio, _ = self._load_audio(
                    Path(str(rendered_output_path)),
                    sample_rate=sample_rate,
                )
            except Exception:
                continue

            source_segment = np.asarray(
                best_state["output_audio"][target_start:target_end],
                dtype=np.float32,
            )
            if source_segment.size == 0:
                continue
            try_candidate_patch(
                patch_audio=np.asarray(patch_audio, dtype=np.float32),
                source_segment=source_segment,
                target_start=target_start,
                target_end=target_end,
                word_group=list(word_group),
                mode="reference-phrase-patch",
                blend_floor=float(np.clip(float(blend_strength), 0.72, 0.90)),
                blend_ceil=0.97,
            )

            if reference_bank_candidates_provider is not None:
                reference_candidates = reference_bank_candidates_provider(
                    phrase_text,
                    list(phrase_words),
                    {
                        "group_index": group_index,
                        "word_indices": list(word_group),
                        "target_start": target_start,
                        "target_end": target_end,
                    },
                )
                for bank_entry in reference_candidates or []:
                    bank_file = Path(str(bank_entry.get("file_path", "")))
                    if not bank_file.exists():
                        continue
                    attempts_tested += 1
                    try:
                        bank_audio, _ = self._load_audio(bank_file, sample_rate=sample_rate)
                    except Exception:
                        continue
                    try_candidate_patch(
                        patch_audio=np.asarray(bank_audio, dtype=np.float32),
                        source_segment=source_segment,
                        target_start=target_start,
                        target_end=target_end,
                        word_group=list(word_group),
                        mode=f"pipa-{str(bank_entry.get('kind', 'reference'))}",
                        extra_params={
                            "reference_text": str(
                                bank_entry.get("phrase")
                                or bank_entry.get("word")
                                or phrase_text
                            ),
                            "reference_path": str(bank_file),
                        },
                        blend_floor=0.18 if str(bank_entry.get("kind", "")) == "word" else 0.24,
                        blend_ceil=0.42 if str(bank_entry.get("kind", "")) == "word" else 0.58,
                    )

            emit_status(
                (
                    f"Finished phrase patch {group_index}/{len(contiguous_groups)}. "
                    f"Best ordered-word similarity is {float(best_state['similarity_score']):.2f}%."
                ),
                progress=min(96, max(22, int(best_state["similarity_score"]))),
            )

        return {
            "output_path": best_output_path,
            "sample_rate": sample_rate,
            "source_rms_db": float(best_state["source_rms_db"]),
            "output_rms_db": float(best_state["output_rms_db"]),
            "best_similarity_score": float(best_state["similarity_score"]),
            "best_word_report": str(best_state["word_report"]),
            "best_letter_report": str(best_state["letter_report"]),
            "best_word_scores": list(best_state["word_scores"]),
            "best_letter_scores": list(best_state["letter_scores"]),
            "variants_tested": attempts_tested,
            "repair_attempts": attempts_tested,
            "repaired_word_count": len(repaired_word_indices),
            "detected_word_indices": list(target_word_indices),
            "bank_scored_words": int(len(word_priority_scores)),
            "replacement_strategy": normalized_replacement_strategy,
            "stopped_early": bool(cancel_event.is_set()),
        }
