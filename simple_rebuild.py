from __future__ import annotations

from datetime import datetime, timezone
import math
from pathlib import Path
from typing import Dict, List, Tuple

import ffmpeg
import librosa
import numpy as np
from scipy.signal import resample_poly

from simple_touchup import lyrics_to_words, normalize_lyrics


def approximate_pronunciation_units(word: str) -> List[str]:
    normalized = normalize_lyrics(word).replace(" ", "")
    if not normalized:
        return []

    digraphs = {
        "ch": "CH",
        "sh": "SH",
        "th": "TH",
        "ph": "F",
        "ng": "NG",
        "ck": "K",
        "qu": "KW",
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
        "or": "AOR",
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
        duo = normalized[index : index + 2]
        if duo in digraphs:
            units.append(digraphs[duo])
            index += 2
            continue
        mapped = singles.get(normalized[index], normalized[index].upper())
        if mapped:
            units.extend(unit for unit in mapped.split(" ") if unit)
        index += 1
    return units


class RebuildFeatureBuilder:
    def __init__(self, repo_root: Path):
        self.repo_root = Path(repo_root)
        self.analysis_sample_rate = 22050
        self.hop_length = 256
        self.frame_length = 1024

    def _ffmpeg_binary(self) -> str:
        local = self.repo_root / ("ffmpeg.exe" if Path().anchor else "ffmpeg")
        return str(local) if local.exists() else "ffmpeg"

    def load_audio(
        self, file_path: Path, sample_rate: int = 44100
    ) -> Tuple[np.ndarray, int]:
        cleaned = str(file_path).strip().strip('"')
        out, _ = (
            ffmpeg.input(cleaned, threads=0)
            .output("-", format="f32le", acodec="pcm_f32le", ac=2, ar=sample_rate)
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

    def _to_analysis_mono(
        self, audio: np.ndarray, sample_rate: int
    ) -> Tuple[np.ndarray, int]:
        working = np.asarray(audio, dtype=np.float32)
        mono = working if working.ndim == 1 else working.mean(axis=1)
        if int(sample_rate) != int(self.analysis_sample_rate):
            orig_sr = int(sample_rate)
            target_sr = int(self.analysis_sample_rate)
            common_divisor = math.gcd(orig_sr, target_sr)
            up = max(1, target_sr // common_divisor)
            down = max(1, orig_sr // common_divisor)
            mono = resample_poly(
                mono.astype(np.float32, copy=False),
                up,
                down,
            ).astype(np.float32, copy=False)
        return mono.astype(np.float32, copy=False), int(self.analysis_sample_rate)

    def _estimate_pitch_track(
        self, mono: np.ndarray, sample_rate: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        try:
            f0, _, _ = librosa.pyin(
                mono,
                fmin=librosa.note_to_hz("C2"),
                fmax=librosa.note_to_hz("C7"),
                sr=sample_rate,
                hop_length=self.hop_length,
                frame_length=self.frame_length,
            )
            pitch = np.asarray(f0, dtype=np.float32)
            pitch[~np.isfinite(pitch)] = 0.0
        except Exception:
            fallback = librosa.yin(
                mono,
                fmin=librosa.note_to_hz("C2"),
                fmax=librosa.note_to_hz("C7"),
                sr=sample_rate,
                hop_length=self.hop_length,
                frame_length=self.frame_length,
            )
            pitch = np.asarray(fallback, dtype=np.float32)
            pitch[~np.isfinite(pitch)] = 0.0
        times = librosa.frames_to_time(
            np.arange(pitch.shape[0]),
            sr=sample_rate,
            hop_length=self.hop_length,
        ).astype(np.float32)
        return times, pitch

    def _estimate_energy_track(
        self, mono: np.ndarray, sample_rate: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        rms = librosa.feature.rms(
            y=mono,
            frame_length=self.frame_length,
            hop_length=self.hop_length,
            center=True,
        )[0]
        times = librosa.frames_to_time(
            np.arange(rms.shape[0]),
            sr=sample_rate,
            hop_length=self.hop_length,
        ).astype(np.float32)
        return times, np.asarray(rms, dtype=np.float32)

    def _estimate_onset_track(
        self, mono: np.ndarray, sample_rate: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        onset = librosa.onset.onset_strength(
            y=mono,
            sr=sample_rate,
            hop_length=self.hop_length,
        )
        times = librosa.frames_to_time(
            np.arange(onset.shape[0]),
            sr=sample_rate,
            hop_length=self.hop_length,
        ).astype(np.float32)
        return times, np.asarray(onset, dtype=np.float32)

    def _window_values(
        self,
        track_times: np.ndarray,
        values: np.ndarray,
        start_seconds: float,
        end_seconds: float,
    ) -> np.ndarray:
        if values.size == 0:
            return np.zeros(0, dtype=np.float32)
        mask = (track_times >= float(start_seconds)) & (
            track_times <= float(end_seconds)
        )
        if np.any(mask):
            return np.asarray(values[mask], dtype=np.float32)
        nearest_start = int(
            np.searchsorted(track_times, float(start_seconds), side="left")
        )
        nearest_end = int(
            np.searchsorted(track_times, float(end_seconds), side="right")
        )
        nearest_start = max(0, min(nearest_start, len(values) - 1))
        nearest_end = max(nearest_start + 1, min(nearest_end, len(values)))
        return np.asarray(values[nearest_start:nearest_end], dtype=np.float32)

    def summarize_segment(
        self,
        audio: np.ndarray,
        sample_rate: int,
        *,
        start_sample: int = 0,
        end_sample: int | None = None,
    ) -> Dict[str, float]:
        working = np.asarray(audio, dtype=np.float32)
        if working.ndim == 2 and working.shape[1] > 0:
            mono = working.mean(axis=1)
        else:
            mono = working.reshape(-1)
        total_samples = int(mono.shape[0])
        segment_start = max(0, int(start_sample))
        segment_end = (
            total_samples if end_sample is None else min(total_samples, int(end_sample))
        )
        if segment_end <= segment_start:
            segment_end = min(total_samples, segment_start + 1)
        segment = mono[segment_start:segment_end].astype(np.float32, copy=False)
        if segment.size == 0:
            segment = np.zeros(1, dtype=np.float32)

        analysis_mono, analysis_sr = self._to_analysis_mono(segment, sample_rate)
        pitch_times, pitch_values = self._estimate_pitch_track(
            analysis_mono, analysis_sr
        )
        energy_times, energy_values = self._estimate_energy_track(
            analysis_mono, analysis_sr
        )
        onset_times, onset_values = self._estimate_onset_track(
            analysis_mono, analysis_sr
        )

        voiced_pitch = pitch_values[pitch_values > 1.0]
        duration_seconds = float(segment.shape[0]) / float(max(sample_rate, 1))
        return {
            "duration_seconds": round(duration_seconds, 4),
            "pitch_median_hz": round(
                float(np.median(voiced_pitch)) if voiced_pitch.size else 0.0, 3
            ),
            "pitch_p10_hz": round(
                float(np.percentile(voiced_pitch, 10)) if voiced_pitch.size else 0.0, 3
            ),
            "pitch_p90_hz": round(
                float(np.percentile(voiced_pitch, 90)) if voiced_pitch.size else 0.0, 3
            ),
            "pitch_mean_hz": round(
                float(np.mean(voiced_pitch)) if voiced_pitch.size else 0.0, 3
            ),
            "voiced_ratio": round(
                float(voiced_pitch.size / max(1, pitch_values.size)), 4
            ),
            "energy_mean": round(
                float(np.mean(energy_values)) if energy_values.size else 0.0, 6
            ),
            "energy_peak": round(
                float(np.max(energy_values)) if energy_values.size else 0.0, 6
            ),
            "onset_mean": round(
                float(np.mean(onset_values)) if onset_values.size else 0.0, 6
            ),
            "onset_peak": round(
                float(np.max(onset_values)) if onset_values.size else 0.0, 6
            ),
        }

    def analyze_aligned_audio(
        self,
        *,
        audio: np.ndarray,
        sample_rate: int,
        lyrics: str,
        word_scores: List[Dict[str, object]],
        source_name: str,
    ) -> Dict[str, object]:
        duration_seconds = float(audio.shape[0]) / float(max(sample_rate, 1))
        if duration_seconds >= 120.0 or len(word_scores) >= 320:
            return self._analyze_aligned_audio_fast(
                audio=audio,
                sample_rate=sample_rate,
                lyrics=lyrics,
                word_scores=word_scores,
                source_name=source_name,
            )

        analysis_mono, analysis_sr = self._to_analysis_mono(audio, sample_rate)
        pitch_times, pitch_values = self._estimate_pitch_track(
            analysis_mono, analysis_sr
        )
        energy_times, energy_values = self._estimate_energy_track(
            analysis_mono, analysis_sr
        )
        onset_times, onset_values = self._estimate_onset_track(
            analysis_mono, analysis_sr
        )

        enriched_words: List[Dict[str, object]] = []
        for entry in sorted(
            (
                dict(item)
                for item in word_scores
                if normalize_lyrics(str(item.get("word", "")))
            ),
            key=lambda item: int(item.get("index", 0)),
        ):
            start_seconds = float(entry.get("start", 0.0))
            end_seconds = max(start_seconds, float(entry.get("end", start_seconds)))
            pitch_window = self._window_values(
                pitch_times, pitch_values, start_seconds, end_seconds
            )
            voiced_pitch = pitch_window[pitch_window > 1.0]
            energy_window = self._window_values(
                energy_times, energy_values, start_seconds, end_seconds
            )
            onset_window = self._window_values(
                onset_times, onset_values, start_seconds, end_seconds
            )
            word = normalize_lyrics(str(entry.get("word", "")))
            enriched_words.append(
                {
                    "index": int(entry.get("index", len(enriched_words))),
                    "word": word,
                    "units": approximate_pronunciation_units(word),
                    "start": round(start_seconds, 4),
                    "end": round(end_seconds, 4),
                    "duration_seconds": round(max(0.0, end_seconds - start_seconds), 4),
                    "similarity": round(float(entry.get("similarity", 0.0)), 2),
                    "pitch_median_hz": round(
                        float(np.median(voiced_pitch)) if voiced_pitch.size else 0.0, 3
                    ),
                    "pitch_mean_hz": round(
                        float(np.mean(voiced_pitch)) if voiced_pitch.size else 0.0, 3
                    ),
                    "pitch_min_hz": round(
                        float(np.min(voiced_pitch)) if voiced_pitch.size else 0.0, 3
                    ),
                    "pitch_max_hz": round(
                        float(np.max(voiced_pitch)) if voiced_pitch.size else 0.0, 3
                    ),
                    "voiced_ratio": round(
                        float(voiced_pitch.size / max(1, pitch_window.size)), 4
                    ),
                    "energy_mean": round(
                        float(np.mean(energy_window)) if energy_window.size else 0.0, 6
                    ),
                    "energy_peak": round(
                        float(np.max(energy_window)) if energy_window.size else 0.0, 6
                    ),
                    "onset_mean": round(
                        float(np.mean(onset_window)) if onset_window.size else 0.0, 6
                    ),
                    "onset_peak": round(
                        float(np.max(onset_window)) if onset_window.size else 0.0, 6
                    ),
                }
            )

        phrase_groups: List[Dict[str, object]] = []
        current_group: List[Dict[str, object]] = []
        for word_entry in enriched_words:
            if not current_group:
                current_group = [word_entry]
                continue
            previous = current_group[-1]
            gap = float(word_entry["start"]) - float(previous["end"])
            if gap > 0.45 or len(current_group) >= 6:
                phrase_groups.append(self._summarize_phrase_group(current_group))
                current_group = [word_entry]
            else:
                current_group.append(word_entry)
        if current_group:
            phrase_groups.append(self._summarize_phrase_group(current_group))

        voiced_all = pitch_values[pitch_values > 1.0]
        style_summary = {
            "source_name": source_name,
            "duration_seconds": round(
                float(audio.shape[0]) / float(max(sample_rate, 1)), 3
            ),
            "lyric_word_count": len(lyrics_to_words(lyrics)),
            "aligned_word_count": len(enriched_words),
            "phrase_count": len(phrase_groups),
            "pitch_median_hz": round(
                float(np.median(voiced_all)) if voiced_all.size else 0.0, 3
            ),
            "pitch_p10_hz": round(
                float(np.percentile(voiced_all, 10)) if voiced_all.size else 0.0, 3
            ),
            "pitch_p90_hz": round(
                float(np.percentile(voiced_all, 90)) if voiced_all.size else 0.0, 3
            ),
            "energy_mean": round(
                float(np.mean(energy_values)) if energy_values.size else 0.0, 6
            ),
            "energy_p90": round(
                float(np.percentile(energy_values, 90)) if energy_values.size else 0.0,
                6,
            ),
            "onset_mean": round(
                float(np.mean(onset_values)) if onset_values.size else 0.0, 6
            ),
            "voiced_ratio": round(
                float(voiced_all.size / max(1, pitch_values.size)), 4
            ),
            "mean_word_duration": round(
                (
                    float(
                        np.mean(
                            [float(item["duration_seconds"]) for item in enriched_words]
                        )
                    )
                    if enriched_words
                    else 0.0
                ),
                4,
            ),
        }

        return {
            "source_name": source_name,
            "lyrics": normalize_lyrics(lyrics),
            "style_summary": style_summary,
            "word_performance": enriched_words,
            "phrase_performance": phrase_groups,
        }

    def _analyze_aligned_audio_fast(
        self,
        *,
        audio: np.ndarray,
        sample_rate: int,
        lyrics: str,
        word_scores: List[Dict[str, object]],
        source_name: str,
    ) -> Dict[str, object]:
        working = np.asarray(audio, dtype=np.float32)
        mono = working.mean(axis=1) if working.ndim == 2 else working.reshape(-1)
        total_samples = int(mono.shape[0])
        duration_seconds = float(total_samples) / float(max(sample_rate, 1))

        def summarize_raw_window(
            start_seconds: float, end_seconds: float
        ) -> Dict[str, float]:
            start_sample = max(
                0, min(total_samples, int(round(float(start_seconds) * sample_rate)))
            )
            end_sample = max(
                start_sample + 1,
                min(total_samples, int(round(float(end_seconds) * sample_rate))),
            )
            segment = mono[start_sample:end_sample].astype(np.float32, copy=False)
            if segment.size == 0:
                segment = np.zeros(1, dtype=np.float32)
            rms = float(np.sqrt(np.mean(np.square(segment)))) if segment.size else 0.0
            diffs = (
                np.abs(np.diff(segment))
                if segment.size > 1
                else np.zeros(1, dtype=np.float32)
            )
            return {
                "energy_mean": round(rms, 6),
                "energy_peak": round(
                    float(np.max(np.abs(segment))) if segment.size else 0.0, 6
                ),
                "onset_mean": round(float(np.mean(diffs)) if diffs.size else 0.0, 6),
                "onset_peak": round(float(np.max(diffs)) if diffs.size else 0.0, 6),
            }

        normalized_words = [
            dict(item)
            for item in sorted(
                (
                    dict(entry)
                    for entry in word_scores
                    if normalize_lyrics(str(entry.get("word", "")))
                ),
                key=lambda item: int(item.get("index", 0)),
            )
        ]
        enriched_words: List[Dict[str, object]] = []
        for entry in normalized_words:
            start_seconds = float(entry.get("start", 0.0))
            end_seconds = max(start_seconds, float(entry.get("end", start_seconds)))
            raw_summary = summarize_raw_window(start_seconds, end_seconds)
            word = normalize_lyrics(str(entry.get("word", "")))
            enriched_words.append(
                {
                    "index": int(entry.get("index", len(enriched_words))),
                    "word": word,
                    "units": approximate_pronunciation_units(word),
                    "start": round(start_seconds, 4),
                    "end": round(end_seconds, 4),
                    "duration_seconds": round(max(0.0, end_seconds - start_seconds), 4),
                    "similarity": round(float(entry.get("similarity", 0.0)), 2),
                    "pitch_median_hz": 0.0,
                    "pitch_mean_hz": 0.0,
                    "pitch_min_hz": 0.0,
                    "pitch_max_hz": 0.0,
                    "voiced_ratio": 0.0,
                    **raw_summary,
                }
            )

        # Estimate pitch only on a representative subset so long-form package builds stay tractable.
        pitch_candidates = [
            word
            for word in enriched_words
            if float(word.get("similarity", 0.0)) >= 40.0
            and 0.06 <= float(word.get("duration_seconds", 0.0)) <= 1.2
        ]
        if pitch_candidates:
            step = max(1, len(pitch_candidates) // 64)
            sampled = pitch_candidates[::step][:64]
            sampled_pitch_values: List[float] = []
            sampled_pitch_by_index: Dict[int, float] = {}
            for word in sampled:
                start_sample = max(
                    0,
                    min(total_samples, int(round(float(word["start"]) * sample_rate))),
                )
                end_sample = max(
                    start_sample + 1,
                    min(total_samples, int(round(float(word["end"]) * sample_rate))),
                )
                segment = mono[start_sample:end_sample]
                summary = self.summarize_segment(segment, sample_rate)
                pitch_value = float(summary.get("pitch_median_hz", 0.0))
                if pitch_value > 0.0:
                    sampled_pitch_values.append(pitch_value)
                    sampled_pitch_by_index[int(word["index"])] = pitch_value
            for word in enriched_words:
                sampled_pitch = float(
                    sampled_pitch_by_index.get(int(word["index"]), 0.0)
                )
                if sampled_pitch > 0.0:
                    word["pitch_median_hz"] = round(sampled_pitch, 3)
                    word["pitch_mean_hz"] = round(sampled_pitch, 3)
                    word["pitch_min_hz"] = round(sampled_pitch, 3)
                    word["pitch_max_hz"] = round(sampled_pitch, 3)
                    word["voiced_ratio"] = 1.0
        else:
            sampled_pitch_values = []

        phrase_groups: List[Dict[str, object]] = []
        current_group: List[Dict[str, object]] = []
        for word_entry in enriched_words:
            if not current_group:
                current_group = [word_entry]
                continue
            previous = current_group[-1]
            gap = float(word_entry["start"]) - float(previous["end"])
            if gap > 0.45 or len(current_group) >= 6:
                phrase_groups.append(self._summarize_phrase_group(current_group))
                current_group = [word_entry]
            else:
                current_group.append(word_entry)
        if current_group:
            phrase_groups.append(self._summarize_phrase_group(current_group))

        clip_energy = [
            float(item.get("energy_mean", 0.0))
            for item in enriched_words
            if float(item.get("energy_mean", 0.0)) > 0.0
        ]
        clip_onset = [
            float(item.get("onset_mean", 0.0))
            for item in enriched_words
            if float(item.get("onset_mean", 0.0)) > 0.0
        ]
        style_summary = {
            "source_name": source_name,
            "duration_seconds": round(duration_seconds, 3),
            "lyric_word_count": len(lyrics_to_words(lyrics)),
            "aligned_word_count": len(enriched_words),
            "phrase_count": len(phrase_groups),
            "pitch_median_hz": round(
                float(np.median(sampled_pitch_values)) if sampled_pitch_values else 0.0,
                3,
            ),
            "pitch_p10_hz": round(
                (
                    float(np.percentile(sampled_pitch_values, 10))
                    if sampled_pitch_values
                    else 0.0
                ),
                3,
            ),
            "pitch_p90_hz": round(
                (
                    float(np.percentile(sampled_pitch_values, 90))
                    if sampled_pitch_values
                    else 0.0
                ),
                3,
            ),
            "energy_mean": round(
                float(np.mean(clip_energy)) if clip_energy else 0.0, 6
            ),
            "energy_p90": round(
                float(np.percentile(clip_energy, 90)) if clip_energy else 0.0, 6
            ),
            "onset_mean": round(float(np.mean(clip_onset)) if clip_onset else 0.0, 6),
            "voiced_ratio": round(
                float(len(sampled_pitch_values) / max(1, len(enriched_words))), 4
            ),
            "mean_word_duration": round(
                (
                    float(
                        np.mean(
                            [float(item["duration_seconds"]) for item in enriched_words]
                        )
                    )
                    if enriched_words
                    else 0.0
                ),
                4,
            ),
            "analysis_mode": "fast-long-form",
        }

        return {
            "source_name": source_name,
            "lyrics": normalize_lyrics(lyrics),
            "style_summary": style_summary,
            "word_performance": enriched_words,
            "phrase_performance": phrase_groups,
        }

    def _summarize_phrase_group(
        self, words: List[Dict[str, object]]
    ) -> Dict[str, object]:
        if not words:
            return {"words": [], "phrase": ""}
        pitch_values = [
            float(item.get("pitch_median_hz", 0.0))
            for item in words
            if float(item.get("pitch_median_hz", 0.0)) > 0.0
        ]
        energy_values = [float(item.get("energy_mean", 0.0)) for item in words]
        onset_values = [float(item.get("onset_mean", 0.0)) for item in words]
        return {
            "phrase": " ".join(str(item.get("word", "")) for item in words).strip(),
            "word_indices": [int(item.get("index", 0)) for item in words],
            "start": round(float(words[0].get("start", 0.0)), 4),
            "end": round(float(words[-1].get("end", 0.0)), 4),
            "duration_seconds": round(
                float(words[-1].get("end", 0.0)) - float(words[0].get("start", 0.0)),
                4,
            ),
            "pitch_median_hz": round(
                float(np.median(pitch_values)) if pitch_values else 0.0, 3
            ),
            "energy_mean": round(
                float(np.mean(energy_values)) if energy_values else 0.0, 6
            ),
            "onset_mean": round(
                float(np.mean(onset_values)) if onset_values else 0.0, 6
            ),
        }

    def build_package_profile(
        self,
        *,
        package_id: str,
        clip_analyses: List[Dict[str, object]],
        word_entries: List[Dict[str, object]],
        phrase_entries: List[Dict[str, object]],
        alignment_tolerance: str,
    ) -> Dict[str, object]:
        clip_summaries = [
            dict(item.get("style_summary", {}))
            for item in clip_analyses
            if isinstance(item, dict)
            and isinstance(item.get("style_summary", {}), dict)
        ]
        clip_pitch = [
            float(item.get("pitch_median_hz", 0.0))
            for item in clip_summaries
            if float(item.get("pitch_median_hz", 0.0)) > 0.0
        ]
        clip_energy = [
            float(item.get("energy_mean", 0.0))
            for item in clip_summaries
            if float(item.get("energy_mean", 0.0)) > 0.0
        ]
        word_durations = [
            float(item.get("duration_seconds", 0.0))
            for clip in clip_analyses
            for item in clip.get("word_performance", [])
            if float(item.get("duration_seconds", 0.0)) > 0.0
        ]
        phrase_durations = [
            float(item.get("duration_seconds", 0.0))
            for clip in clip_analyses
            for item in clip.get("phrase_performance", [])
            if float(item.get("duration_seconds", 0.0)) > 0.0
        ]
        return {
            "package_id": package_id,
            "created_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
            "alignment_tolerance": alignment_tolerance,
            "strategy": "guide-conditioned-rebuild-foundation-v1",
            "clip_count": len(clip_analyses),
            "reference_word_count": len(word_entries),
            "reference_phrase_count": len(phrase_entries),
            "voice_style": {
                "pitch_median_hz": round(
                    float(np.median(clip_pitch)) if clip_pitch else 0.0, 3
                ),
                "pitch_p10_hz": round(
                    float(np.percentile(clip_pitch, 10)) if clip_pitch else 0.0, 3
                ),
                "pitch_p90_hz": round(
                    float(np.percentile(clip_pitch, 90)) if clip_pitch else 0.0, 3
                ),
                "energy_mean": round(
                    float(np.mean(clip_energy)) if clip_energy else 0.0, 6
                ),
                "mean_word_duration": round(
                    float(np.mean(word_durations)) if word_durations else 0.0, 4
                ),
                "mean_phrase_duration": round(
                    float(np.mean(phrase_durations)) if phrase_durations else 0.0, 4
                ),
            },
            "training_usage": {
                "uses_lyrics": True,
                "uses_word_alignment": True,
                "uses_word_pitch_energy": True,
                "uses_phrase_style_templates": True,
                "supports_guide_conditioning": True,
            },
        }

    def build_guide_plan(
        self,
        *,
        guide_analysis: Dict[str, object],
        reference_bank: Dict[str, object],
        package_label: str,
        top_k: int = 3,
    ) -> Dict[str, object]:
        word_entries = [
            dict(entry)
            for entry in reference_bank.get("words", [])
            if isinstance(entry, dict)
        ]
        phrase_entries = [
            dict(entry)
            for entry in reference_bank.get("phrases", [])
            if isinstance(entry, dict)
        ]
        plan_phrases: List[Dict[str, object]] = []
        for phrase in guide_analysis.get("phrase_performance", []):
            phrase_text = normalize_lyrics(str(phrase.get("phrase", "")))
            phrase_words = [
                normalize_lyrics(word)
                for word in phrase_text.split(" ")
                if normalize_lyrics(word)
            ]
            phrase_candidates: List[Dict[str, object]] = []
            for entry in phrase_entries:
                score = self._reference_match_score(
                    guide_text=phrase_text,
                    guide_words=phrase_words,
                    guide_pitch=float(phrase.get("pitch_median_hz", 0.0)),
                    guide_duration=float(phrase.get("duration_seconds", 0.0)),
                    reference_entry=entry,
                )
                if score <= 0.0:
                    continue
                phrase_candidates.append(
                    {
                        "relative_path": str(entry.get("relative_path", "")),
                        "phrase": str(entry.get("phrase", "")),
                        "score": round(score, 3),
                        "target_pitch_shift_semitones": round(
                            self._pitch_shift_semitones(
                                float(phrase.get("pitch_median_hz", 0.0)),
                                float(
                                    entry.get("performance", {}).get(
                                        "pitch_median_hz", 0.0
                                    )
                                ),
                            ),
                            3,
                        ),
                        "target_duration_ratio": round(
                            self._duration_ratio(
                                float(phrase.get("duration_seconds", 0.0)),
                                float(entry.get("duration_seconds", 0.0)),
                            ),
                            4,
                        ),
                    }
                )
            phrase_candidates.sort(
                key=lambda item: float(item.get("score", 0.0)), reverse=True
            )
            plan_phrases.append(
                {
                    "phrase": phrase_text,
                    "word_indices": list(phrase.get("word_indices", [])),
                    "start": float(phrase.get("start", 0.0)),
                    "end": float(phrase.get("end", 0.0)),
                    "duration_seconds": float(phrase.get("duration_seconds", 0.0)),
                    "pitch_median_hz": float(phrase.get("pitch_median_hz", 0.0)),
                    "energy_mean": float(phrase.get("energy_mean", 0.0)),
                    "reference_candidates": phrase_candidates[: max(1, int(top_k))],
                }
            )

        plan_words: List[Dict[str, object]] = []
        for word in guide_analysis.get("word_performance", []):
            word_text = normalize_lyrics(str(word.get("word", "")))
            if not word_text:
                continue
            word_candidates: List[Dict[str, object]] = []
            for entry in word_entries:
                score = self._reference_match_score(
                    guide_text=word_text,
                    guide_words=[word_text],
                    guide_pitch=float(word.get("pitch_median_hz", 0.0)),
                    guide_duration=float(word.get("duration_seconds", 0.0)),
                    reference_entry=entry,
                )
                if score <= 0.0:
                    continue
                word_candidates.append(
                    {
                        "relative_path": str(entry.get("relative_path", "")),
                        "word": str(entry.get("word", "")),
                        "score": round(score, 3),
                        "target_pitch_shift_semitones": round(
                            self._pitch_shift_semitones(
                                float(word.get("pitch_median_hz", 0.0)),
                                float(
                                    entry.get("performance", {}).get(
                                        "pitch_median_hz", 0.0
                                    )
                                ),
                            ),
                            3,
                        ),
                        "target_duration_ratio": round(
                            self._duration_ratio(
                                float(word.get("duration_seconds", 0.0)),
                                float(entry.get("duration_seconds", 0.0)),
                            ),
                            4,
                        ),
                    }
                )
            word_candidates.sort(
                key=lambda item: float(item.get("score", 0.0)), reverse=True
            )
            plan_words.append(
                {
                    "index": int(word.get("index", 0)),
                    "word": word_text,
                    "units": list(word.get("units", [])),
                    "start": float(word.get("start", 0.0)),
                    "end": float(word.get("end", 0.0)),
                    "duration_seconds": float(word.get("duration_seconds", 0.0)),
                    "pitch_median_hz": float(word.get("pitch_median_hz", 0.0)),
                    "energy_mean": float(word.get("energy_mean", 0.0)),
                    "reference_candidates": word_candidates[: max(1, int(top_k))],
                }
            )

        return {
            "strategy": "guide-conditioned-resynthesis-plan-v1",
            "package_label": package_label,
            "guide_summary": dict(guide_analysis.get("style_summary", {})),
            "planned_phrases": plan_phrases,
            "planned_words": plan_words,
            "summary": {
                "phrase_plan_count": len(plan_phrases),
                "word_plan_count": len(plan_words),
                "phrases_with_matches": int(
                    sum(1 for item in plan_phrases if item["reference_candidates"])
                ),
                "words_with_matches": int(
                    sum(1 for item in plan_words if item["reference_candidates"])
                ),
            },
        }

    def analyze_file(
        self,
        *,
        guide_path: Path,
        lyrics: str,
        scorer,
    ) -> Dict[str, object]:
        audio, sample_rate = self.load_audio(guide_path, sample_rate=44100)
        scoring = scorer.analyze_audio(audio, sample_rate, lyrics)
        return self.analyze_aligned_audio(
            audio=audio,
            sample_rate=sample_rate,
            lyrics=lyrics,
            word_scores=list(scoring.get("word_scores", [])),
            source_name=guide_path.name,
        )

    def _reference_match_score(
        self,
        *,
        guide_text: str,
        guide_words: List[str],
        guide_pitch: float,
        guide_duration: float,
        reference_entry: Dict[str, object],
    ) -> float:
        entry_phrase = normalize_lyrics(str(reference_entry.get("phrase", "")))
        entry_word = normalize_lyrics(str(reference_entry.get("word", "")))
        entry_text = entry_phrase or entry_word
        if not entry_text:
            return 0.0

        score = 0.0
        if guide_text == entry_text:
            score += 120.0
        elif guide_words and len(guide_words) == 1 and guide_words[0] == entry_word:
            score += 110.0
        else:
            guide_units = {
                unit
                for word in guide_words
                for unit in approximate_pronunciation_units(word)
            }
            entry_units = {
                str(unit) for unit in reference_entry.get("units", []) if str(unit)
            }
            union = len(guide_units | entry_units)
            if union:
                score += 40.0 * (len(guide_units & entry_units) / float(union))

        score += float(reference_entry.get("similarity", 0.0))
        entry_performance = dict(reference_entry.get("performance", {}))
        entry_pitch = float(entry_performance.get("pitch_median_hz", 0.0))
        entry_duration = float(reference_entry.get("duration_seconds", 0.0))
        if guide_pitch > 0.0 and entry_pitch > 0.0:
            delta = abs(self._pitch_shift_semitones(guide_pitch, entry_pitch))
            score += max(0.0, 28.0 - (delta * 3.5))
        if guide_duration > 0.0 and entry_duration > 0.0:
            ratio = self._duration_ratio(guide_duration, entry_duration)
            score += max(0.0, 18.0 - (abs(1.0 - ratio) * 24.0))
        return float(score)

    def _pitch_shift_semitones(self, target_pitch: float, source_pitch: float) -> float:
        if target_pitch <= 0.0 or source_pitch <= 0.0:
            return 0.0
        return float(12.0 * np.log2(target_pitch / source_pitch))

    def _duration_ratio(self, target_duration: float, source_duration: float) -> float:
        if target_duration <= 0.0 or source_duration <= 0.0:
            return 1.0
        return float(target_duration / source_duration)
