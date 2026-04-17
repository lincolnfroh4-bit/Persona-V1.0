from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import librosa
import numpy as np
import soundfile as sf

from simple_touchup import LetterAwarePronunciationScorer, lyrics_to_words


class VoiceSuitabilityOptimizer:
    """
    Lyric-guided multi-take stitcher.

    The previous "optimize rank" tab is replaced by a comping workflow:
    1) score word intelligibility for each uploaded vocal against intended lyrics,
    2) keep one anchor take for flow/timing,
    3) replace weak anchor words with stronger word regions from other takes.
    """

    def __init__(self, sample_rate: int = 44100):
        self.sample_rate = int(sample_rate)
        self.scorer = LetterAwarePronunciationScorer()

    def _load_mono(self, path: Path) -> np.ndarray:
        audio, _ = librosa.load(str(path), sr=self.sample_rate, mono=True)
        if audio.size == 0:
            return np.zeros(1, dtype=np.float32)
        return np.asarray(audio, dtype=np.float32)

    def _to_2d(self, audio: np.ndarray) -> np.ndarray:
        if audio.ndim == 1:
            return audio[:, None]
        return np.asarray(audio, dtype=np.float32)

    def _time_to_sample(self, seconds: float, length: int) -> int:
        return int(
            np.clip(round(float(seconds) * self.sample_rate), 0, max(length - 1, 0))
        )

    def _segment_with_margin(
        self,
        audio: np.ndarray,
        start_sec: float,
        end_sec: float,
        margin_samples: int,
    ) -> tuple[np.ndarray, int, int]:
        length = int(audio.shape[0])
        start = self._time_to_sample(start_sec, length)
        end = self._time_to_sample(end_sec, length)
        if end <= start:
            end = min(length, start + max(32, int(0.02 * self.sample_rate)))

        start = max(0, start - int(margin_samples))
        end = min(length, end + int(margin_samples))
        if end <= start:
            end = min(length, start + 32)
        return np.asarray(audio[start:end], dtype=np.float32), start, end

    def _fit_to_length_no_stretch(
        self, segment: np.ndarray, target_length: int
    ) -> np.ndarray:
        """
        Match length without time-stretching (no tempo/pitch warp).
        """
        target_length = int(max(target_length, 1))
        segment = np.asarray(segment, dtype=np.float32)
        seg_len = int(segment.shape[0])
        if seg_len == target_length:
            return segment
        if seg_len <= 1:
            return np.zeros(target_length, dtype=np.float32)
        if seg_len > target_length:
            trim = seg_len - target_length
            left = trim // 2
            right = trim - left
            return segment[left : seg_len - right]

        # seg_len < target_length: center-pad to preserve local timing.
        pad = target_length - seg_len
        left = pad // 2
        right = pad - left
        return np.pad(segment, (left, right), mode="constant")

    def _window_dbfs(self, audio: np.ndarray, start: int, end: int) -> float:
        start_i = int(max(0, start))
        end_i = int(max(start_i, end))
        if end_i <= start_i:
            return -120.0
        window = np.asarray(audio[start_i:end_i], dtype=np.float32)
        if window.size == 0:
            return -120.0
        rms = float(np.sqrt(np.mean(np.square(window), dtype=np.float64) + 1e-12))
        if rms <= 1e-12:
            return -120.0
        return float(20.0 * np.log10(rms))

    def _seam_db_gate(
        self,
        merged: np.ndarray,
        target_start: int,
        target_end: int,
        replacement: np.ndarray,
        max_cut_db: float,
        seam_window: int,
    ) -> tuple[bool, Dict[str, float]]:
        merged_len = int(merged.shape[0])
        repl_len = int(replacement.shape[0])
        seam_window = int(max(8, min(seam_window, max(repl_len, 8))))

        prev_anchor_db = self._window_dbfs(
            merged,
            max(0, target_start - seam_window),
            min(merged_len, target_start),
        )
        next_anchor_db = self._window_dbfs(
            merged,
            max(0, target_end),
            min(merged_len, target_end + seam_window),
        )
        prev_replacement_db = self._window_dbfs(
            replacement, 0, min(repl_len, seam_window)
        )
        next_replacement_db = self._window_dbfs(
            replacement,
            max(0, repl_len - seam_window),
            repl_len,
        )

        previous_side_db = max(prev_anchor_db, prev_replacement_db)
        following_side_db = max(next_anchor_db, next_replacement_db)
        allow_cut = previous_side_db <= max_cut_db and following_side_db <= max_cut_db

        return allow_cut, {
            "previous_side_db": round(previous_side_db, 2),
            "following_side_db": round(following_side_db, 2),
            "prev_anchor_db": round(prev_anchor_db, 2),
            "next_anchor_db": round(next_anchor_db, 2),
            "prev_replacement_db": round(prev_replacement_db, 2),
            "next_replacement_db": round(next_replacement_db, 2),
        }

    def analyze_candidate(self, vocal_path: Path, lyrics: str) -> Dict[str, object]:
        audio = self._load_mono(vocal_path)
        words = lyrics_to_words(lyrics)
        try:
            scoring = self.scorer.analyze_audio(
                self._to_2d(audio),
                self.sample_rate,
                lyrics,
            )
        except Exception as exc:
            scoring = {
                "similarity_score": 0.0,
                "word_scores": [],
                "word_report": f"Lyric scoring failed: {exc}",
            }
        word_scores = [
            dict(entry)
            for entry in (scoring.get("word_scores") or [])
            if isinstance(entry, dict)
        ]
        word_by_index = {
            int(entry.get("index", idx)): dict(entry)
            for idx, entry in enumerate(word_scores)
        }
        weak_words = sorted(
            (
                {
                    "index": int(entry.get("index", 0)),
                    "word": str(entry.get("word", "")),
                    "similarity": float(entry.get("similarity", 0.0)),
                }
                for entry in word_scores
            ),
            key=lambda entry: float(entry["similarity"]),
        )[:10]
        weak_summary = ", ".join(
            f"{w['word']} ({float(w['similarity']):.0f}%)" for w in weak_words[:5]
        )

        return {
            "path": str(vocal_path),
            "audio": audio,
            "duration_seconds": round(
                float(audio.shape[0]) / float(self.sample_rate), 2
            ),
            "score": float(scoring.get("similarity_score", 0.0)),
            "summary": str(scoring.get("word_report", "")),
            "issues": [
                (
                    "No usable lyric scoring was produced."
                    if not word_scores
                    else f"Weakest words: {weak_summary}"
                )
            ],
            "word_scores": word_scores,
            "word_by_index": word_by_index,
            "word_count": len(words),
            "weak_words": weak_words,
        }

    def stitch_best_parts(
        self,
        analyses: List[Dict[str, object]],
        lyrics: str,
        output_path: Path,
        max_cut_db: float = -24.0,
    ) -> Dict[str, object]:
        if not analyses:
            raise RuntimeError("No analyses were provided for stitching.")

        words = lyrics_to_words(lyrics)
        max_cut_db = float(max(-60.0, min(-3.0, float(max_cut_db))))
        min_gain = 2.0
        min_candidate_similarity = 55.0
        margin_samples = int(0.0095 * self.sample_rate)
        seam_duck_db = 4.0
        seam_duck_factor = float(10.0 ** (-seam_duck_db / 20.0))
        pre_duck_len = int(0.012 * self.sample_rate)
        seam_gate_window = int(0.012 * self.sample_rate)
        max_word_span_sec = 1.2

        anchor_index = int(
            max(
                range(len(analyses)),
                key=lambda idx: float(analyses[idx].get("score", 0.0)),
            )
        )
        anchor = analyses[anchor_index]
        anchor_audio = np.asarray(
            anchor.get("audio", np.zeros(1, dtype=np.float32)), dtype=np.float32
        )
        merged = np.copy(anchor_audio)
        anchor_words = anchor.get("word_by_index", {})
        if not isinstance(anchor_words, dict):
            anchor_words = {}

        edits: List[Dict[str, object]] = []
        replaced_word_count = 0
        skipped_by_db_gate = 0

        for word_index, word in enumerate(words):
            anchor_entry = anchor_words.get(word_index)
            if not isinstance(anchor_entry, dict):
                continue

            anchor_similarity = float(anchor_entry.get("similarity", 0.0))
            anchor_start_sec = float(anchor_entry.get("start", 0.0))
            anchor_end_sec = float(anchor_entry.get("end", anchor_start_sec + 0.02))
            anchor_span = max(0.01, anchor_end_sec - anchor_start_sec)
            if anchor_span > max_word_span_sec:
                continue

            replacement_candidate: Optional[Dict[str, object]] = None
            candidate_pool: List[tuple[float, int, Dict[str, object]]] = []
            for candidate_index, candidate in enumerate(analyses):
                if candidate_index == anchor_index:
                    continue
                word_by_index = candidate.get("word_by_index", {})
                if not isinstance(word_by_index, dict):
                    continue
                candidate_entry = word_by_index.get(word_index)
                if not isinstance(candidate_entry, dict):
                    continue
                candidate_similarity = float(candidate_entry.get("similarity", 0.0))
                if candidate_similarity < min_candidate_similarity:
                    continue
                if (candidate_similarity - anchor_similarity) < min_gain:
                    continue
                candidate_pool.append(
                    (candidate_similarity, candidate_index, candidate_entry)
                )

            candidate_pool.sort(key=lambda item: item[0], reverse=True)
            if not candidate_pool:
                continue

            target_raw, target_start, target_end = self._segment_with_margin(
                merged,
                anchor_start_sec,
                anchor_end_sec,
                margin_samples=margin_samples,
            )
            target_length = int(target_raw.shape[0])
            if target_length <= 32:
                continue

            original = np.copy(target_raw)
            chosen_take_index = anchor_index
            chosen_similarity = anchor_similarity
            chosen_replacement: Optional[np.ndarray] = None
            chosen_gate_metrics: Dict[str, float] = {}

            for (
                candidate_similarity,
                candidate_index,
                _candidate_entry,
            ) in candidate_pool:
                candidate_audio = np.asarray(
                    analyses[candidate_index].get(
                        "audio", np.zeros(1, dtype=np.float32)
                    ),
                    dtype=np.float32,
                )
                replacement_raw, _, _ = self._segment_with_margin(
                    candidate_audio,
                    anchor_start_sec,
                    anchor_end_sec,
                    margin_samples=margin_samples,
                )
                replacement = self._fit_to_length_no_stretch(
                    replacement_raw, target_length
                )

                original_rms = float(
                    np.sqrt(np.mean(np.square(original), dtype=np.float64) + 1e-9)
                )
                replacement_rms = float(
                    np.sqrt(np.mean(np.square(replacement), dtype=np.float64) + 1e-9)
                )
                if replacement_rms > 1e-7 and original_rms > 1e-7:
                    gain = float(np.clip(original_rms / replacement_rms, 0.8, 1.25))
                    replacement = replacement * gain

                seam_ok, seam_metrics = self._seam_db_gate(
                    merged=merged,
                    target_start=target_start,
                    target_end=target_end,
                    replacement=replacement,
                    max_cut_db=max_cut_db,
                    seam_window=seam_gate_window,
                )
                if not seam_ok:
                    skipped_by_db_gate += 1
                    continue

                chosen_take_index = int(candidate_index)
                chosen_similarity = float(candidate_similarity)
                chosen_replacement = replacement
                chosen_gate_metrics = seam_metrics
                break

            if chosen_replacement is None:
                continue
            replacement = chosen_replacement

            # Gently duck the outgoing anchor region right before the stitch start
            # so consonant boundaries don't "double-hit" across takes.
            if pre_duck_len > 4 and target_start > 0:
                duck_start = max(0, target_start - pre_duck_len)
                duck_slice = merged[duck_start:target_start]
                if duck_slice.size > 0:
                    duck_ramp = np.linspace(
                        1.0,
                        seam_duck_factor,
                        num=int(duck_slice.shape[0]),
                        dtype=np.float32,
                    )
                    merged[duck_start:target_start] = duck_slice * duck_ramp

            fade = max(0, min(int(0.012 * self.sample_rate), target_length // 4))
            if fade >= 4:
                in_ramp = np.linspace(0.0, 1.0, num=fade, dtype=np.float32)
                out_ramp = np.linspace(1.0, 0.0, num=fade, dtype=np.float32)
                # Keep less of the old edge during crossfades so stitches read cleaner.
                start_old_weight = (1.0 - in_ramp) * seam_duck_factor
                end_old_weight = out_ramp * seam_duck_factor
                replacement[:fade] = (original[:fade] * start_old_weight) + (
                    replacement[:fade] * (1.0 - start_old_weight)
                )
                replacement[-fade:] = (original[-fade:] * end_old_weight) + (
                    replacement[-fade:] * (1.0 - end_old_weight)
                )

            merged[target_start:target_end] = replacement
            replaced_word_count += 1
            edits.append(
                {
                    "index": int(word_index),
                    "word": word,
                    "from_source": Path(str(anchor.get("path", ""))).name,
                    "to_source": Path(
                        str(analyses[chosen_take_index].get("path", ""))
                    ).name,
                    "before": round(anchor_similarity, 2),
                    "after": round(chosen_similarity, 2),
                    "gain": round(chosen_similarity - anchor_similarity, 2),
                    "previous_side_db": float(
                        chosen_gate_metrics.get("previous_side_db", -120.0)
                    ),
                    "following_side_db": float(
                        chosen_gate_metrics.get("following_side_db", -120.0)
                    ),
                }
            )

        peak = float(np.max(np.abs(merged)) + 1e-9)
        if peak > 0.99:
            merged = (merged / peak) * 0.99
        merged = np.asarray(merged, dtype=np.float32)

        output_path.parent.mkdir(parents=True, exist_ok=True)
        sf.write(str(output_path), merged, self.sample_rate, subtype="PCM_24")

        edits_sorted = sorted(
            edits, key=lambda entry: float(entry.get("gain", 0.0)), reverse=True
        )
        return {
            "anchor_index": int(anchor_index),
            "anchor_source_name": Path(str(anchor.get("path", ""))).name,
            "replaced_word_count": int(replaced_word_count),
            "total_word_count": int(len(words)),
            "edits": edits_sorted,
            "edits_preview": edits_sorted[:24],
            "max_cut_db": round(max_cut_db, 2),
            "min_gain": round(min_gain, 2),
            "min_candidate_similarity": round(min_candidate_similarity, 2),
            "skipped_by_db_gate": int(skipped_by_db_gate),
            "sample_rate": int(self.sample_rate),
            "output_path": str(output_path),
        }
