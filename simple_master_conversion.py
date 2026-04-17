from __future__ import annotations

import json
import shutil
import threading
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import ffmpeg
import numpy as np
import soundfile as sf


class MasterConversionEngine:
    DEFAULT_PROFILE = "studio"
    PROFILES: Dict[str, Dict[str, object]] = {
        "studio": {
            "label": "Studio balanced",
            "description": "Best all-around direct conversion flow with pronunciation repair and final lyric cleanup.",
            "outside_gain": 0.025,
        },
        "clarity": {
            "label": "Pronunciation first",
            "description": "Pushes harder toward lyric clarity and more assertive cleanup between spoken words.",
            "outside_gain": 0.018,
        },
        "body": {
            "label": "Body preserving",
            "description": "Keeps more of the source body and natural tone before pronunciation repair.",
            "outside_gain": 0.04,
        },
        "adlib-heavy": {
            "label": "Adlib heavy",
            "description": "More aggressive direct cleanup for tracks with lots of doubles, stacks, and overlap.",
            "outside_gain": 0.015,
        },
    }
    REPAIR_PRESETS: Dict[str, Dict[str, int]] = {
        "fast": {"max_target_words": 4},
        "balanced": {"max_target_words": 6},
        "clean": {"max_target_words": 8},
    }

    def __init__(self, repo_root: Path, backend, repair_engine):
        self.repo_root = Path(repo_root)
        self.backend = backend
        self.repair_engine = repair_engine

    def get_options(self) -> Dict[str, object]:
        return {
            "profiles": [
                {
                    "id": profile_id,
                    "label": str(profile["label"]),
                    "description": str(profile["description"]),
                }
                for profile_id, profile in self.PROFILES.items()
            ],
            "defaults": {
                "profile": self.DEFAULT_PROFILE,
                "quality_preset": "balanced",
                "candidate_strength": 1,
            },
            "description": (
                "Master Conversion uses one selected voice model plus one optional pronunciation package, "
                "and when a PIPA pronunciation package is available it synthesizes the full target vocal from the uploaded guide or de-personafied vocal blueprint. "
                "Without a PIPA package it falls back to direct voice conversion and weak-region rebuilding."
            ),
        }

    def _ffmpeg_binary(self) -> str:
        return self.backend._ffmpeg_binary()

    def _load_audio(
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

    def _write_audio(
        self, output_path: Path, audio: np.ndarray, sample_rate: int
    ) -> Path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        peak = float(np.max(np.abs(audio))) if audio.size else 0.0
        normalized = np.asarray(audio, dtype=np.float32)
        if peak > 0.995:
            normalized = normalized * np.float32(0.995 / peak)
        sf.write(output_path, normalized, int(sample_rate), subtype="PCM_24")
        return output_path

    def _safe_rms_db(self, audio: np.ndarray) -> float:
        working = np.asarray(audio, dtype=np.float32)
        if working.size == 0:
            return -120.0
        rms = float(np.sqrt(np.mean(np.square(working)) + 1e-9))
        return float(20.0 * np.log10(max(rms, 1e-8)))

    def _fit_audio_length(self, audio: np.ndarray, target_length: int) -> np.ndarray:
        working = np.asarray(audio, dtype=np.float32)
        if working.ndim == 1:
            working = working[:, np.newaxis]
        if working.shape[1] == 1:
            working = np.repeat(working, 2, axis=1)
        elif working.shape[1] > 2:
            working = working[:, :2]
        if target_length <= 0:
            return np.zeros((0, 2), dtype=np.float32)
        if working.shape[0] == 0:
            return np.zeros((target_length, 2), dtype=np.float32)
        if working.shape[0] == target_length:
            return working.astype(np.float32, copy=False)
        if working.shape[0] == 1:
            return np.repeat(
                working.astype(np.float32, copy=False), target_length, axis=0
            )
        source_positions = np.linspace(0.0, 1.0, num=working.shape[0], dtype=np.float32)
        target_positions = np.linspace(0.0, 1.0, num=target_length, dtype=np.float32)
        channels = [
            np.interp(target_positions, source_positions, working[:, channel]).astype(
                np.float32
            )
            for channel in range(working.shape[1])
        ]
        return np.stack(channels, axis=1)

    def _cosine_fade(self, length: int) -> np.ndarray:
        if length <= 1:
            return np.ones(max(1, length), dtype=np.float32)
        return np.sin(np.linspace(0.0, np.pi / 2.0, length, dtype=np.float32)) ** 2

    def _merge_windows(
        self,
        windows: List[Tuple[int, int]],
        *,
        bridge_samples: int,
    ) -> List[Tuple[int, int]]:
        if not windows:
            return []
        ordered = sorted(
            (max(0, int(start)), max(0, int(end)))
            for start, end in windows
            if int(end) > int(start)
        )
        if not ordered:
            return []
        merged: List[Tuple[int, int]] = [ordered[0]]
        for start, end in ordered[1:]:
            prev_start, prev_end = merged[-1]
            if start <= (prev_end + int(bridge_samples)):
                merged[-1] = (prev_start, max(prev_end, end))
            else:
                merged.append((start, end))
        return merged

    def _word_window_to_samples(
        self,
        word_scores: List[Dict[str, object]],
        word_indices: List[int],
        sample_rate: int,
        total_samples: int,
        *,
        padding_ms: float,
    ) -> Tuple[int, int]:
        word_by_index = {
            int(entry.get("index", -1)): entry
            for entry in word_scores
            if int(entry.get("index", -1)) >= 0
        }
        starts = [
            float(word_by_index[index]["start"])
            for index in word_indices
            if index in word_by_index
        ]
        ends = [
            float(word_by_index[index]["end"])
            for index in word_indices
            if index in word_by_index
        ]
        if not starts or not ends:
            return 0, 0
        pad = int((float(padding_ms) / 1000.0) * sample_rate)
        start = max(0, int(min(starts) * sample_rate) - pad)
        end = min(total_samples, int(max(ends) * sample_rate) + pad)
        return start, max(start + 1, end)

    def _build_keep_mask(
        self,
        word_scores: List[Dict[str, object]],
        sample_rate: int,
        total_samples: int,
        *,
        pre_padding_ms: float,
        post_padding_ms: float,
        bridge_gap_ms: float,
        fade_ms: float,
    ) -> Tuple[np.ndarray, List[Tuple[int, int]]]:
        windows: List[Tuple[int, int]] = []
        pre_pad = int((float(pre_padding_ms) / 1000.0) * sample_rate)
        post_pad = int((float(post_padding_ms) / 1000.0) * sample_rate)
        for entry in word_scores:
            start = max(0, int(float(entry.get("start", 0.0)) * sample_rate) - pre_pad)
            end = min(
                total_samples,
                int(float(entry.get("end", 0.0)) * sample_rate) + post_pad,
            )
            if end > start:
                windows.append((start, end))
        merged = self._merge_windows(
            windows,
            bridge_samples=int((float(bridge_gap_ms) / 1000.0) * sample_rate),
        )
        mask = np.zeros(total_samples, dtype=np.float32)
        fade_samples = max(8, int((float(fade_ms) / 1000.0) * sample_rate))
        for start, end in merged:
            segment_length = max(1, end - start)
            mask[start:end] = 1.0
            edge = min(fade_samples, max(1, segment_length // 3))
            if edge > 1:
                fade_curve = self._cosine_fade(edge)
                mask[start : start + edge] = np.maximum(
                    mask[start : start + edge], fade_curve
                )
                mask[end - edge : end] = np.maximum(
                    mask[end - edge : end], fade_curve[::-1]
                )
        return mask, merged

    def _lead_focus_score(self, audio: np.ndarray) -> float:
        working = np.asarray(audio, dtype=np.float32)
        if working.ndim == 1:
            working = working[:, np.newaxis]
        if working.shape[1] == 1:
            working = np.repeat(working, 2, axis=1)
        if working.size == 0:
            return 0.0
        mid = (working[:, 0] + working[:, 1]) * 0.5
        side = (working[:, 0] - working[:, 1]) * 0.5
        mid_rms = float(np.sqrt(np.mean(np.square(mid)) + 1e-9))
        side_rms = float(np.sqrt(np.mean(np.square(side)) + 1e-9))
        ratio = side_rms / max(mid_rms, 1e-6)
        return float(np.clip(100.0 * ((1.35 - ratio) / 1.35), 0.0, 100.0))

    def _silence_clean_score(self, gap_rms_db: float) -> float:
        return float(
            np.clip(((-float(gap_rms_db)) - 18.0) * (100.0 / 42.0), 0.0, 100.0)
        )

    def _build_phrase_groups(
        self,
        word_scores: List[Dict[str, object]],
        *,
        gap_seconds: float = 0.48,
        max_words: int = 7,
    ) -> List[List[int]]:
        ordered = sorted(
            (dict(entry) for entry in word_scores if int(entry.get("index", -1)) >= 0),
            key=lambda entry: int(entry.get("index", 0)),
        )
        if not ordered:
            return []
        groups: List[List[int]] = []
        current: List[int] = [int(ordered[0]["index"])]
        previous_end = float(ordered[0].get("end", 0.0))
        for entry in ordered[1:]:
            current_index = int(entry["index"])
            start = float(entry.get("start", previous_end))
            should_split = (start - previous_end) > float(gap_seconds) or len(
                current
            ) >= int(max_words)
            if should_split:
                groups.append(current)
                current = [current_index]
            else:
                current.append(current_index)
            previous_end = float(entry.get("end", start))
        if current:
            groups.append(current)
        return groups

    def _group_similarity_from_word_scores(
        self,
        word_scores: List[Dict[str, object]],
        word_group: List[int],
    ) -> float:
        target_indices = {int(index) for index in word_group}
        values = [
            float(entry.get("similarity", 0.0))
            for entry in word_scores
            if int(entry.get("index", -1)) in target_indices
        ]
        if not values:
            return 0.0
        return float((0.65 * min(values)) + (0.35 * float(np.mean(values))))

    def _score_full_candidate(
        self,
        *,
        audio: np.ndarray,
        sample_rate: int,
        word_scores: List[Dict[str, object]],
        similarity_score: float,
        bridge_gap_ms: float,
    ) -> Dict[str, float]:
        keep_mask, _ = self._build_keep_mask(
            word_scores,
            sample_rate,
            int(audio.shape[0]),
            pre_padding_ms=95.0,
            post_padding_ms=165.0,
            bridge_gap_ms=bridge_gap_ms,
            fade_ms=24.0,
        )
        kept_audio = np.asarray(audio, dtype=np.float32) * keep_mask[:, np.newaxis]
        removed_audio = np.asarray(audio, dtype=np.float32) - kept_audio
        gap_rms_db = self._safe_rms_db(removed_audio)
        silence_clean_score = self._silence_clean_score(gap_rms_db)
        focus_score = self._lead_focus_score(kept_audio)
        combined_score = (
            (0.64 * float(similarity_score))
            + (0.22 * float(silence_clean_score))
            + (0.14 * float(focus_score))
        )
        return {
            "gap_rms_db": float(gap_rms_db),
            "silence_clean_score": float(silence_clean_score),
            "lead_focus_score": float(focus_score),
            "combined_score": round(float(combined_score), 2),
        }

    def _candidate_modes(self, profile_id: str, preferred_pipeline: str) -> List[str]:
        profile = self.PROFILES.get(profile_id, self.PROFILES[self.DEFAULT_PROFILE])
        ordered = list(profile.get("pipelines", []))
        preferred = str(preferred_pipeline or "").strip().lower()
        if preferred and preferred != "off":
            ordered = [preferred] + [
                pipeline for pipeline in ordered if pipeline != preferred
            ]
        deduped: List[str] = []
        for pipeline in ordered:
            if pipeline and pipeline not in deduped:
                deduped.append(str(pipeline))
        return deduped

    def _render_candidate(
        self,
        *,
        input_path: Path,
        output_dir: Path,
        lyrics: str,
        pipeline_id: str,
        strength: int,
        bridge_gap_ms: float,
    ) -> Dict[str, object]:
        candidate_root = output_dir / pipeline_id
        candidate_root.mkdir(parents=True, exist_ok=True)
        lead_path = self.backend.preprocess_for_conversion_pipeline(
            input_path,
            candidate_root / "prep",
            preprocess_mode=pipeline_id,
            strength=int(strength),
        )
        normalized_path = candidate_root / f"{pipeline_id}_lead.wav"
        if Path(str(lead_path)).resolve() != normalized_path.resolve():
            shutil.copy2(str(lead_path), str(normalized_path))
        audio, sample_rate = self._load_audio(normalized_path, sample_rate=44100)
        scoring = self.repair_engine.scorer.analyze_audio(audio, sample_rate, lyrics)
        score_bits = self._score_full_candidate(
            audio=audio,
            sample_rate=sample_rate,
            word_scores=list(scoring.get("word_scores", [])),
            similarity_score=float(scoring.get("similarity_score", 0.0)),
            bridge_gap_ms=bridge_gap_ms,
        )
        return {
            "pipeline_id": pipeline_id,
            "path": normalized_path,
            "audio": audio,
            "sample_rate": sample_rate,
            "word_scores": list(scoring.get("word_scores", [])),
            "letter_scores": list(scoring.get("letter_scores", [])),
            "similarity_score": float(scoring.get("similarity_score", 0.0)),
            "word_report": str(scoring.get("word_report", "")),
            "letter_report": str(scoring.get("letter_report", "")),
            **score_bits,
        }

    def _replace_segment(
        self,
        target_audio: np.ndarray,
        replacement: np.ndarray,
        *,
        start: int,
        end: int,
        sample_rate: int,
    ) -> None:
        target_segment = np.asarray(target_audio[start:end], dtype=np.float32)
        replacement_segment = self._fit_audio_length(replacement, end - start)
        if target_segment.shape[0] == 0 or replacement_segment.shape[0] == 0:
            return
        fade_size = min(
            max(12, int(0.03 * sample_rate)),
            max(1, replacement_segment.shape[0] // 4),
        )
        mix_curve = np.ones(replacement_segment.shape[0], dtype=np.float32)
        if fade_size > 1:
            fade = self._cosine_fade(fade_size)
            mix_curve[:fade_size] = fade
            mix_curve[-fade_size:] = fade[::-1]
        target_audio[start:end] = (
            (1.0 - mix_curve[:, np.newaxis]) * target_segment
        ) + (mix_curve[:, np.newaxis] * replacement_segment)

    def _reconstruct_lead(
        self,
        *,
        candidates: List[Dict[str, object]],
        intended_words: List[str],
        output_dir: Path,
        profile_id: str,
        update_status: Optional[Callable[[Dict[str, object]], None]] = None,
    ) -> Dict[str, object]:
        if not candidates:
            raise RuntimeError("No lead candidates were rendered.")

        anchor = max(
            candidates, key=lambda entry: float(entry.get("combined_score", 0.0))
        )
        output_audio = np.asarray(anchor["audio"], dtype=np.float32).copy()
        sample_rate = int(anchor["sample_rate"])
        total_samples = int(output_audio.shape[0])
        phrase_groups = self._build_phrase_groups(list(anchor.get("word_scores", [])))
        phrase_choices: List[Dict[str, object]] = []

        if update_status is not None:
            update_status(
                {
                    "progress": 40,
                    "message": f"Reconstructing the lead from {len(candidates)} candidate pipelines.",
                }
            )

        for group_index, word_group in enumerate(phrase_groups, start=1):
            target_start, target_end = self._word_window_to_samples(
                list(anchor["word_scores"]),
                word_group,
                sample_rate,
                total_samples,
                padding_ms=150.0,
            )
            if target_end <= target_start:
                continue

            expected_words = [
                intended_words[index]
                for index in word_group
                if 0 <= int(index) < len(intended_words)
            ]
            if not expected_words:
                continue

            anchor_segment = np.asarray(
                output_audio[target_start:target_end], dtype=np.float32
            )
            anchor_rms_db = self._safe_rms_db(anchor_segment)
            best_pipeline = str(anchor["pipeline_id"])
            best_segment = anchor_segment
            best_local_score = -1.0
            per_pipeline_scores: List[Dict[str, object]] = []

            for candidate in candidates:
                candidate_audio = np.asarray(candidate["audio"], dtype=np.float32)
                candidate_start, candidate_end = self._word_window_to_samples(
                    list(candidate["word_scores"]),
                    word_group,
                    sample_rate,
                    int(candidate_audio.shape[0]),
                    padding_ms=150.0,
                )
                if candidate_end <= candidate_start:
                    continue
                segment = self._fit_audio_length(
                    candidate_audio[candidate_start:candidate_end],
                    target_end - target_start,
                )
                scoring_mode = "local-align"
                try:
                    local_scoring = self.repair_engine.scorer.analyze_segment(
                        segment,
                        sample_rate,
                        expected_words,
                        global_start_seconds=0.0,
                        absolute_word_indices=list(word_group),
                        letter_focus_limit=max(4, min(10, len(expected_words) * 2)),
                    )
                    similarity = float(local_scoring.get("similarity_score", 0.0))
                except Exception:
                    # Short or malformed phrase slices should never kill a full
                    # run. Fall back to the candidate's already computed
                    # full-song word scores instead of crashing the pipeline.
                    similarity = self._group_similarity_from_word_scores(
                        list(candidate.get("word_scores", [])),
                        word_group,
                    )
                    scoring_mode = "global-fallback"
                focus_score = self._lead_focus_score(segment)
                body_score = float(
                    np.clip(
                        100.0 - (abs(self._safe_rms_db(segment) - anchor_rms_db) * 8.0),
                        0.0,
                        100.0,
                    )
                )
                local_score = (
                    (0.72 * similarity) + (0.18 * focus_score) + (0.10 * body_score)
                )
                per_pipeline_scores.append(
                    {
                        "pipeline": str(candidate["pipeline_id"]),
                        "score": round(float(local_score), 2),
                        "similarity": round(similarity, 2),
                        "focus": round(float(focus_score), 2),
                        "body": round(float(body_score), 2),
                        "mode": scoring_mode,
                    }
                )
                if local_score > (best_local_score + 0.2):
                    best_local_score = float(local_score)
                    best_pipeline = str(candidate["pipeline_id"])
                    best_segment = segment

            self._replace_segment(
                output_audio,
                best_segment,
                start=target_start,
                end=target_end,
                sample_rate=sample_rate,
            )
            phrase_choices.append(
                {
                    "group_index": int(group_index),
                    "word_indices": list(word_group),
                    "words": " ".join(expected_words),
                    "pipeline": best_pipeline,
                    "score": round(float(best_local_score), 2),
                    "pipelines": per_pipeline_scores,
                }
            )
            if update_status is not None:
                update_status(
                    {
                        "progress": min(
                            56,
                            40
                            + int(
                                round((group_index / max(len(phrase_groups), 1)) * 16)
                            ),
                        ),
                        "message": (
                            f"Reconstructing lead phrase {group_index}/{len(phrase_groups)} "
                            f"with {best_pipeline}."
                        ),
                    }
                )

        reconstruction_scoring = self.repair_engine.scorer.analyze_audio(
            output_audio,
            sample_rate,
            " ".join(intended_words),
        )
        profile = self.PROFILES.get(profile_id, self.PROFILES[self.DEFAULT_PROFILE])
        keep_mask, merged_windows = self._build_keep_mask(
            list(reconstruction_scoring.get("word_scores", [])),
            sample_rate,
            total_samples,
            pre_padding_ms=92.0,
            post_padding_ms=165.0,
            bridge_gap_ms=float(profile.get("bridge_gap_ms", 240.0)),
            fade_ms=26.0,
        )
        outside_gain = float(profile.get("outside_gain", 0.025))
        reconstructed_lead = (keep_mask[:, np.newaxis] * output_audio) + (
            (1.0 - keep_mask[:, np.newaxis]) * (output_audio * outside_gain)
        )
        removed_layer = output_audio - reconstructed_lead

        lead_path = self._write_audio(
            output_dir / "reconstructed_lead.wav", reconstructed_lead, sample_rate
        )
        removed_path = self._write_audio(
            output_dir / "reconstructed_removed.wav", removed_layer, sample_rate
        )
        return {
            "lead_path": lead_path,
            "removed_path": removed_path,
            "sample_rate": sample_rate,
            "lead_audio": reconstructed_lead,
            "removed_audio": removed_layer,
            "word_scores": list(reconstruction_scoring.get("word_scores", [])),
            "letter_scores": list(reconstruction_scoring.get("letter_scores", [])),
            "similarity_score": float(
                reconstruction_scoring.get("similarity_score", 0.0)
            ),
            "word_report": str(reconstruction_scoring.get("word_report", "")),
            "letter_report": str(reconstruction_scoring.get("letter_report", "")),
            "phrase_choices": phrase_choices,
            "kept_segment_count": len(merged_windows),
        }

    def _blend_audio_paths(
        self,
        *,
        primary_path: Path,
        secondary_path: Path,
        output_path: Path,
        primary_percentage: int,
    ) -> int:
        primary_audio, primary_sr = sf.read(str(primary_path), always_2d=True)
        secondary_audio, secondary_sr = sf.read(str(secondary_path), always_2d=True)
        if int(primary_sr) != int(secondary_sr):
            raise RuntimeError("Blended conversions must have matching sample rates.")
        target_length = max(int(primary_audio.shape[0]), int(secondary_audio.shape[0]))
        primary = self._fit_audio_length(
            np.asarray(primary_audio, dtype=np.float32), target_length
        )
        secondary = self._fit_audio_length(
            np.asarray(secondary_audio, dtype=np.float32), target_length
        )
        primary_weight = float(np.clip(float(primary_percentage) / 100.0, 0.0, 1.0))
        blended = (primary * primary_weight) + (secondary * (1.0 - primary_weight))
        self._write_audio(output_path, blended, int(primary_sr))
        return int(primary_sr)

    def _render_voice_output(
        self,
        *,
        source_path: Path,
        output_path: Path,
        output_format: str,
        model_name: str,
        settings: Dict[str, object],
        work_dir: Path,
        secondary_model_name: str = "",
        blend_percentage: int = 50,
    ) -> Dict[str, object]:
        normalized_settings = dict(settings)
        normalized_settings["preprocess_mode"] = "off"
        work_dir.mkdir(parents=True, exist_ok=True)
        if secondary_model_name:
            primary_path = work_dir / "primary.wav"
            secondary_path = work_dir / "secondary.wav"
            primary_meta = self.backend.convert_file(
                model_name,
                source_path,
                primary_path,
                preprocess_mode="off",
                preprocess_strength=int(
                    normalized_settings.get("preprocess_strength", 9)
                ),
                work_dir=work_dir / "prep-primary",
                speaker_id=int(normalized_settings["speaker_id"]),
                transpose=int(normalized_settings["transpose"]),
                f0_method=str(normalized_settings["f0_method"]),
                index_path=str(normalized_settings["index_path"]),
                index_rate=float(normalized_settings["index_rate"]),
                filter_radius=int(normalized_settings["filter_radius"]),
                resample_sr=int(normalized_settings["resample_sr"]),
                rms_mix_rate=float(normalized_settings["rms_mix_rate"]),
                protect=float(normalized_settings["protect"]),
                crepe_hop_length=int(normalized_settings["crepe_hop_length"]),
            )
            secondary_meta = self.backend.convert_file(
                secondary_model_name,
                source_path,
                secondary_path,
                preprocess_mode="off",
                preprocess_strength=int(
                    normalized_settings.get("preprocess_strength", 9)
                ),
                work_dir=work_dir / "prep-secondary",
                speaker_id=int(normalized_settings["speaker_id"]),
                transpose=int(normalized_settings["transpose"]),
                f0_method=str(normalized_settings["f0_method"]),
                index_path="",
                index_rate=float(normalized_settings["index_rate"]),
                filter_radius=int(normalized_settings["filter_radius"]),
                resample_sr=int(normalized_settings["resample_sr"]),
                rms_mix_rate=float(normalized_settings["rms_mix_rate"]),
                protect=float(normalized_settings["protect"]),
                crepe_hop_length=int(normalized_settings["crepe_hop_length"]),
            )
            temp_output = (
                output_path
                if output_format == "wav"
                else output_path.with_suffix(".wav")
            )
            sample_rate = self._blend_audio_paths(
                primary_path=primary_path,
                secondary_path=secondary_path,
                output_path=temp_output,
                primary_percentage=int(blend_percentage),
            )
            if output_format != "wav":
                self.backend._transcode_audio(temp_output, output_path)
                temp_output.unlink(missing_ok=True)
            return {
                "sample_rate": sample_rate,
                "timings": {
                    "npy": round(
                        float(primary_meta["timings"]["npy"])
                        + float(secondary_meta["timings"]["npy"]),
                        2,
                    ),
                    "f0": round(
                        float(primary_meta["timings"]["f0"])
                        + float(secondary_meta["timings"]["f0"]),
                        2,
                    ),
                    "infer": round(
                        float(primary_meta["timings"]["infer"])
                        + float(secondary_meta["timings"]["infer"]),
                        2,
                    ),
                },
            }

        metadata = self.backend.convert_file(
            model_name,
            source_path,
            output_path,
            preprocess_mode="off",
            preprocess_strength=int(normalized_settings.get("preprocess_strength", 9)),
            work_dir=work_dir / "prep",
            speaker_id=int(normalized_settings["speaker_id"]),
            transpose=int(normalized_settings["transpose"]),
            f0_method=str(normalized_settings["f0_method"]),
            index_path=str(normalized_settings["index_path"]),
            index_rate=float(normalized_settings["index_rate"]),
            filter_radius=int(normalized_settings["filter_radius"]),
            resample_sr=int(normalized_settings["resample_sr"]),
            rms_mix_rate=float(normalized_settings["rms_mix_rate"]),
            protect=float(normalized_settings["protect"]),
            crepe_hop_length=int(normalized_settings["crepe_hop_length"]),
        )
        return metadata

    def _resolve_guided_regeneration_bundle(
        self, settings: Dict[str, object]
    ) -> Dict[str, str]:
        manifest_value = str(settings.get("pipa_manifest_path", "") or "").strip()
        if not manifest_value:
            return {}
        manifest_path = Path(manifest_value)
        if not manifest_path.exists():
            return {}
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except Exception:
            return {}
        if not isinstance(manifest, dict):
            return {}
        checkpoint_value = str(
            manifest.get("guided_regeneration_path", "") or ""
        ).strip()
        if not checkpoint_value:
            return {}
        checkpoint_path = Path(checkpoint_value)
        if not checkpoint_path.is_absolute():
            checkpoint_path = manifest_path.parent / checkpoint_path
        if not checkpoint_path.exists():
            return {}
        config_value = str(
            manifest.get("guided_regeneration_config_path", "") or ""
        ).strip()
        config_path = Path(config_value) if config_value else Path()
        if config_value and not config_path.is_absolute():
            config_path = manifest_path.parent / config_path
        report_value = str(
            manifest.get("guided_regeneration_report_path", "") or ""
        ).strip()
        report_path = Path(report_value) if report_value else Path()
        if report_value and not report_path.is_absolute():
            report_path = manifest_path.parent / report_path
        training_report_value = str(
            manifest.get("training_report_path", "") or ""
        ).strip()
        training_report_path = (
            Path(training_report_value) if training_report_value else Path()
        )
        if training_report_value and not training_report_path.is_absolute():
            training_report_path = manifest_path.parent / training_report_path
        stats_value = str(
            manifest.get("guided_regeneration_stats_path", "") or ""
        ).strip()
        stats_path = Path(stats_value) if stats_value else Path()
        if stats_value and not stats_path.is_absolute():
            stats_path = manifest_path.parent / stats_path
        return {
            "manifest_path": str(manifest_path),
            "checkpoint_path": str(checkpoint_path),
            "config_path": str(config_path) if config_value else "",
            "training_report_path": (
                str(training_report_path) if training_report_value else ""
            ),
            "report_path": str(report_path) if report_value else "",
            "stats_path": str(stats_path) if stats_value else "",
        }

    def _build_patch_variant_settings(
        self, settings: Dict[str, object]
    ) -> List[Dict[str, object]]:
        base = dict(settings)
        base["preprocess_mode"] = "off"
        protect_value = float(base.get("protect", 0.33))
        index_rate = float(base.get("index_rate", 0.10))
        rms_mix = float(base.get("rms_mix_rate", 0.25))
        variants: List[Dict[str, object]] = [
            dict(base),
            {
                **base,
                "index_rate": min(index_rate, 0.08),
                "protect": max(protect_value, 0.40),
                "rms_mix_rate": min(rms_mix, 0.18),
            },
            {
                **base,
                "index_path": "",
                "index_rate": 0.0,
                "protect": max(protect_value, 0.44),
                "rms_mix_rate": min(rms_mix, 0.16),
                "filter_radius": max(5, int(base.get("filter_radius", 3))),
            },
        ]
        deduped: List[Dict[str, object]] = []
        seen: set[str] = set()
        for variant in variants:
            signature = json.dumps(
                {
                    "index_path": str(variant.get("index_path", "")),
                    "index_rate": round(float(variant.get("index_rate", 0.0)), 4),
                    "protect": round(float(variant.get("protect", 0.0)), 4),
                    "rms_mix_rate": round(float(variant.get("rms_mix_rate", 0.0)), 4),
                    "filter_radius": int(variant.get("filter_radius", 0)),
                },
                sort_keys=True,
            )
            if signature in seen:
                continue
            seen.add(signature)
            deduped.append(variant)
        return deduped

    def _patch_renderer(
        self,
        *,
        model_name: str,
        secondary_model_name: str,
        blend_percentage: int,
        settings: Dict[str, object],
        phrase_text: str,
        reference_segment_path: Path,
        output_path: Path,
        work_dir: Path,
    ) -> Path:
        variants = self._build_patch_variant_settings(settings)
        best_score = -1.0
        best_path: Optional[Path] = None
        for variant_index, variant_settings in enumerate(variants, start=1):
            variant_dir = work_dir / f"variant_{variant_index:02d}"
            variant_output = variant_dir / "patch.wav"
            variant_dir.mkdir(parents=True, exist_ok=True)
            self._render_voice_output(
                source_path=reference_segment_path,
                output_path=variant_output,
                output_format="wav",
                model_name=model_name,
                settings=variant_settings,
                work_dir=variant_dir / "render",
                secondary_model_name=secondary_model_name,
                blend_percentage=blend_percentage,
            )
            patch_audio, sample_rate = self._load_audio(
                variant_output, sample_rate=44100
            )
            scoring = self.repair_engine.scorer.analyze_audio(
                patch_audio, sample_rate, phrase_text
            )
            local_score = (0.84 * float(scoring.get("similarity_score", 0.0))) + (
                0.16 * self._lead_focus_score(patch_audio)
            )
            if local_score > best_score:
                best_score = float(local_score)
                best_path = variant_output
        if best_path is None:
            raise RuntimeError("Could not render any usable phrase patch.")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(str(best_path), str(output_path))
        return output_path

    def _guided_patch_renderer(
        self,
        *,
        model_name: str,
        secondary_model_name: str,
        blend_percentage: int,
        settings: Dict[str, object],
        phrase_text: str,
        reference_segment_path: Path,
        output_path: Path,
        work_dir: Path,
        guided_bundle: Dict[str, str],
        phrase_word_scores: Optional[List[Dict[str, object]]] = None,
    ) -> Path:
        checkpoint_path = Path(
            str(guided_bundle.get("checkpoint_path", "") or "").strip()
        )
        if checkpoint_path.exists():
            config_value = str(guided_bundle.get("config_path", "") or "").strip()
            training_report_value = str(
                guided_bundle.get("training_report_path", "") or ""
            ).strip()
            manifest_value = str(guided_bundle.get("manifest_path", "") or "").strip()
            config_path = Path(config_value) if config_value else None
            training_report_path = (
                Path(training_report_value) if training_report_value else None
            )
            manifest_path = Path(manifest_value) if manifest_value else None
            try:
                synthesized = (
                    self.backend.pipa_store.guided_svs.synthesize_phrase_from_blueprint(
                        checkpoint_path=checkpoint_path,
                        config_path=(
                            config_path
                            if config_path is not None and config_path.exists()
                            else None
                        ),
                        manifest_path=(
                            manifest_path
                            if manifest_path is not None and manifest_path.exists()
                            else None
                        ),
                        training_report_path=(
                            training_report_path
                            if training_report_path is not None
                            and training_report_path.exists()
                            else None
                        ),
                        guide_audio_path=reference_segment_path,
                        phrase_text=phrase_text,
                        output_path=output_path,
                        phrase_word_scores=list(phrase_word_scores or []),
                    )
                )
                rendered_path = Path(
                    str(synthesized.get("output_path", "") or "").strip()
                )
                if rendered_path.exists():
                    return rendered_path
            except Exception:
                pass
        return self._patch_renderer(
            model_name=model_name,
            secondary_model_name=secondary_model_name,
            blend_percentage=blend_percentage,
            settings=settings,
            phrase_text=phrase_text,
            reference_segment_path=reference_segment_path,
            output_path=output_path,
            work_dir=work_dir,
        )

    def _final_lyric_gate(
        self,
        *,
        input_path: Path,
        lyrics: str,
        output_dir: Path,
        outside_gain: float,
    ) -> Dict[str, object]:
        audio, sample_rate = self._load_audio(input_path, sample_rate=44100)
        scoring = self.repair_engine.scorer.analyze_audio(audio, sample_rate, lyrics)
        keep_mask, merged = self._build_keep_mask(
            list(scoring.get("word_scores", [])),
            sample_rate,
            int(audio.shape[0]),
            pre_padding_ms=76.0,
            post_padding_ms=148.0,
            bridge_gap_ms=210.0,
            fade_ms=24.0,
        )
        cleaned = (keep_mask[:, np.newaxis] * audio) + (
            (1.0 - keep_mask[:, np.newaxis]) * (audio * float(outside_gain))
        )
        removed = audio - cleaned
        final_path = self._write_audio(
            output_dir / "master_conversion_final.wav", cleaned, sample_rate
        )
        removed_path = self._write_audio(
            output_dir / "master_conversion_removed_noise.wav", removed, sample_rate
        )
        return {
            "output_path": final_path,
            "removed_path": removed_path,
            "sample_rate": sample_rate,
            "best_similarity_score": float(scoring.get("similarity_score", 0.0)),
            "best_word_report": str(scoring.get("word_report", "")),
            "best_letter_report": str(scoring.get("letter_report", "")),
            "best_word_scores": list(scoring.get("word_scores", [])),
            "best_letter_scores": list(scoring.get("letter_scores", [])),
            "kept_segment_count": len(merged),
            "source_rms_db": self._safe_rms_db(audio),
            "output_rms_db": self._safe_rms_db(cleaned),
        }

    def _prepare_source_reference(
        self,
        *,
        input_path: Path,
        output_dir: Path,
        preprocess_mode: str,
        preprocess_strength: int,
    ) -> Dict[str, object]:
        output_dir.mkdir(parents=True, exist_ok=True)
        original_audio, sample_rate = self._load_audio(input_path, sample_rate=44100)
        _ = preprocess_mode
        _ = preprocess_strength
        removed_audio = np.zeros_like(original_audio, dtype=np.float32)
        prepared_path = self._write_audio(
            output_dir / "prepared_source.wav",
            original_audio,
            sample_rate,
        )
        removed_path = self._write_audio(
            output_dir / "prepared_removed.wav",
            removed_audio,
            sample_rate,
        )
        return {
            "prepared_path": prepared_path,
            "removed_path": removed_path,
            "prepared_audio": np.asarray(original_audio, dtype=np.float32),
            "removed_audio": removed_audio,
            "sample_rate": sample_rate,
            "cleanup_mode": "off",
        }

    def run(
        self,
        *,
        model_name: str,
        input_path: Path,
        lyrics: str,
        output_dir: Path,
        settings: Dict[str, object],
        quality_preset: str,
        master_profile: str,
        preferred_pipeline: str,
        candidate_strength: int,
        output_format: str,
        secondary_model_name: str = "",
        blend_percentage: int = 50,
        cancel_event: Optional[threading.Event] = None,
        update_status: Optional[Callable[[Dict[str, object]], None]] = None,
    ) -> Dict[str, object]:
        cancel_event = cancel_event or threading.Event()
        output_dir.mkdir(parents=True, exist_ok=True)
        normalized_profile = str(master_profile or self.DEFAULT_PROFILE).strip().lower()
        if normalized_profile not in self.PROFILES:
            normalized_profile = self.DEFAULT_PROFILE
        _ = preferred_pipeline
        normalized_preprocess_mode = "off"

        if update_status is not None:
            update_status(
                {
                    "progress": 8,
                    "message": "Preparing the source audio for direct pronunciation-guided conversion.",
                }
            )

        profile = self.PROFILES[normalized_profile]
        source_reference = self._prepare_source_reference(
            input_path=input_path,
            output_dir=output_dir / "source",
            preprocess_mode=normalized_preprocess_mode,
            preprocess_strength=int(candidate_strength),
        )

        if cancel_event.is_set():
            raise RuntimeError("Master Conversion was cancelled.")

        guided_bundle = self._resolve_guided_regeneration_bundle(settings)
        use_full_blueprint_conversion = bool(
            str(guided_bundle.get("checkpoint_path", "") or "").strip()
        )

        if update_status is not None:
            update_status(
                {
                    "progress": 34,
                    "message": (
                        "Extracting the guide blueprint and rebuilding the full target voice from the de-personafied vocal."
                        if use_full_blueprint_conversion
                        else "Converting the prepared source into the selected voice."
                    ),
                }
            )

        raw_render_dir = output_dir / "conversion"
        raw_render_dir.mkdir(parents=True, exist_ok=True)
        raw_converted_path = raw_render_dir / "converted_raw.wav"
        if use_full_blueprint_conversion:
            guide_scoring = self.repair_engine.scorer.analyze_audio(
                source_reference["prepared_audio"],
                int(source_reference["sample_rate"]),
                lyrics,
            )
            checkpoint_path = Path(
                str(guided_bundle.get("checkpoint_path", "") or "").strip()
            )
            config_value = str(guided_bundle.get("config_path", "") or "").strip()
            manifest_value = str(guided_bundle.get("manifest_path", "") or "").strip()
            training_report_value = str(
                guided_bundle.get("training_report_path", "") or ""
            ).strip()
            synthesis_metadata = (
                self.backend.pipa_store.guided_svs.synthesize_full_song_from_blueprint(
                    checkpoint_path=checkpoint_path,
                    config_path=(Path(config_value) if config_value else None),
                    manifest_path=(Path(manifest_value) if manifest_value else None),
                    training_report_path=(
                        Path(training_report_value) if training_report_value else None
                    ),
                    guide_audio_path=Path(str(source_reference["prepared_path"])),
                    lyrics=lyrics,
                    output_path=raw_converted_path,
                    phrase_word_scores=list(guide_scoring.get("word_scores", [])),
                )
            )
            raw_metadata = {
                "sample_rate": int(
                    synthesis_metadata.get(
                        "sample_rate", source_reference["sample_rate"]
                    )
                ),
                "timings": {
                    "npy": 0.0,
                    "f0": 0.0,
                    "infer": round(
                        float(synthesis_metadata.get("synthesis_seconds", 0.0)), 2
                    ),
                },
            }
        else:
            raw_metadata = self._render_voice_output(
                source_path=Path(str(source_reference["prepared_path"])),
                output_path=raw_converted_path,
                output_format="wav",
                model_name=model_name,
                settings=settings,
                work_dir=raw_render_dir / "work",
                secondary_model_name=secondary_model_name,
                blend_percentage=int(blend_percentage),
            )

        repair_limits = self.REPAIR_PRESETS.get(
            quality_preset, self.REPAIR_PRESETS["balanced"]
        )
        if use_full_blueprint_conversion:
            if update_status is not None:
                update_status(
                    {
                        "progress": 78,
                        "message": "Scoring the full synthesized lead against the intended lyrics.",
                    }
                )
            synthesized_audio, synthesized_sample_rate = self._load_audio(
                raw_converted_path, sample_rate=44100
            )
            synthesis_scoring = self.repair_engine.scorer.analyze_audio(
                synthesized_audio,
                synthesized_sample_rate,
                lyrics,
            )
            repair_metadata = {
                "output_path": raw_converted_path,
                "sample_rate": int(synthesized_sample_rate),
                "source_rms_db": self._safe_rms_db(synthesized_audio),
                "output_rms_db": self._safe_rms_db(synthesized_audio),
                "best_similarity_score": float(
                    synthesis_scoring.get("similarity_score", 0.0)
                ),
                "best_word_report": str(synthesis_scoring.get("word_report", "")),
                "best_letter_report": str(synthesis_scoring.get("letter_report", "")),
                "best_word_scores": list(synthesis_scoring.get("word_scores", [])),
                "best_letter_scores": list(synthesis_scoring.get("letter_scores", [])),
                "variants_tested": 0,
                "repair_attempts": 0,
                "repaired_word_count": 0,
                "detected_word_indices": [],
                "replacement_strategy": "full-song",
            }
        else:
            if update_status is not None:
                update_status(
                    {
                        "progress": 64,
                        "message": "Scoring weak lyric regions and rebuilding those gaps with the pronunciation model.",
                    }
                )

            reference_bank_path = str(
                settings.get("pipa_reference_bank_path", "") or ""
            ).strip()

            def resolve_reference_candidates(
                phrase_text: str,
                phrase_words: List[str],
                *,
                limit: int,
            ) -> List[Dict[str, object]]:
                if not reference_bank_path:
                    return []
                candidates: List[Dict[str, object]] = []
                base_dir = Path(reference_bank_path).parent / "reference_bank"
                for entry in self.backend.pipa_store.find_reference_candidates(
                    reference_bank_path=reference_bank_path,
                    phrase_text=phrase_text,
                    words=list(phrase_words),
                    limit=max(1, int(limit)),
                ):
                    candidate = dict(entry)
                    candidate["file_path"] = str(
                        base_dir / str(entry.get("relative_path", ""))
                    )
                    candidates.append(candidate)
                return candidates

            repair_metadata = self.repair_engine.repair_with_reference_phrase_patches(
                source_path=raw_converted_path,
                reference_path=Path(str(source_reference["prepared_path"])),
                intended_lyrics=lyrics,
                output_dir=output_dir / "repair",
                cancel_event=cancel_event,
                max_target_words=int(repair_limits["max_target_words"]),
                reference_bank_candidates_provider=(
                    None
                    if not reference_bank_path
                    else (
                        lambda phrase_text, phrase_words, metadata: resolve_reference_candidates(
                            phrase_text,
                            phrase_words,
                            limit=3,
                        )
                    )
                ),
                reference_bank_word_candidates_provider=(
                    None
                    if not reference_bank_path
                    else (
                        lambda target_word, metadata: resolve_reference_candidates(
                            target_word,
                            [target_word],
                            limit=6,
                        )
                    )
                ),
                patch_renderer=(
                    lambda reference_segment_path, rendered_patch_path, phrase_text, metadata: self._guided_patch_renderer(
                        model_name=model_name,
                        secondary_model_name=secondary_model_name,
                        blend_percentage=int(blend_percentage),
                        settings=settings,
                        phrase_text=phrase_text,
                        reference_segment_path=reference_segment_path,
                        output_path=rendered_patch_path,
                        work_dir=(
                            output_dir
                            / "repair"
                            / "patch-work"
                            / f"group_{int(metadata.get('group_index', 0)):02d}"
                        ),
                        guided_bundle=guided_bundle,
                        phrase_word_scores=list(
                            metadata.get("target_phrase_word_scores", [])
                            or metadata.get("reference_phrase_word_scores", [])
                            or []
                        ),
                    )
                ),
                update_status=(
                    None
                    if update_status is None
                    else lambda payload: update_status(
                        {
                            "progress": min(
                                92,
                                64
                                + int(
                                    round(
                                        max(
                                            0.0,
                                            min(
                                                float(payload.get("progress", 0.0)),
                                                100.0,
                                            ),
                                        )
                                        * 0.24
                                    )
                                ),
                            ),
                            "message": str(payload.get("message", ""))
                            or "Repairing weak phrase regions.",
                            "best_similarity_score": float(
                                payload.get("best_similarity_score", 0.0)
                            ),
                            "best_word_report": str(
                                payload.get("best_word_report", "")
                            ),
                            "best_letter_report": str(
                                payload.get("best_letter_report", "")
                            ),
                            "repair_attempts": int(payload.get("repair_attempts", 0)),
                            "repaired_word_count": int(
                                payload.get("repaired_word_count", 0)
                            ),
                        }
                    )
                ),
                blend_strength=0.98,
                padding_ms=95.0,
                replacement_strategy="replace",
            )

        if update_status is not None:
            update_status(
                {
                    "progress": 94,
                    "message": "Removing non-lyric tails from the final converted lead.",
                }
            )

        final_cleanup = self._final_lyric_gate(
            input_path=Path(str(repair_metadata["output_path"])),
            lyrics=lyrics,
            output_dir=output_dir / "final",
            outside_gain=float(profile.get("outside_gain", 0.025)),
        )

        final_wav_path = Path(str(final_cleanup["output_path"]))
        final_output_path = (
            output_dir / f"{input_path.stem}_master_conversion.{output_format}"
        )
        if output_format == "wav":
            shutil.copy2(str(final_wav_path), str(final_output_path))
        else:
            self.backend._transcode_audio(final_wav_path, final_output_path)

        metadata_path = output_dir / "master_conversion_report.json"
        metadata_payload = {
            "profile": normalized_profile,
            "model_name": model_name,
            "secondary_model_name": secondary_model_name,
            "blend_percentage": int(blend_percentage),
            "cleanup_mode": str(
                source_reference.get("cleanup_mode", normalized_preprocess_mode)
            ),
            "cleanup_strength": int(candidate_strength),
            "quality_preset": quality_preset,
            "output_format": output_format,
            "pipa_manifest_path": str(settings.get("pipa_manifest_path", "")),
            "pipa_reference_bank_path": str(
                settings.get("pipa_reference_bank_path", "")
            ),
            "prepared_source_path": str(source_reference["prepared_path"]),
            "prepared_removed_path": str(source_reference["removed_path"]),
            "repair_similarity": round(
                float(repair_metadata.get("best_similarity_score", 0.0)), 2
            ),
            "final_similarity": round(
                float(final_cleanup.get("best_similarity_score", 0.0)), 2
            ),
            "repair_attempts": int(repair_metadata.get("repair_attempts", 0)),
            "repaired_word_count": int(repair_metadata.get("repaired_word_count", 0)),
            "repair_patch_mode": (
                "blueprint-full-conversion-v1"
                if use_full_blueprint_conversion
                else "blueprint-gap-fill-v1"
            ),
            "guided_regeneration_checkpoint": str(
                guided_bundle.get("checkpoint_path", "") or ""
            ),
            "timings": {
                "npy": round(float(raw_metadata["timings"]["npy"]), 2),
                "f0": round(float(raw_metadata["timings"]["f0"]), 2),
                "infer": round(float(raw_metadata["timings"]["infer"]), 2),
            },
        }
        metadata_path.write_text(
            json.dumps(metadata_payload, indent=2), encoding="utf-8"
        )

        return {
            "output_path": final_output_path,
            "sample_rate": int(final_cleanup["sample_rate"]),
            "raw_conversion_path": raw_converted_path,
            "reconstructed_lead_path": Path(str(source_reference["prepared_path"])),
            "reconstructed_removed_path": Path(str(source_reference["removed_path"])),
            "repaired_path": Path(str(repair_metadata["output_path"])),
            "final_removed_path": Path(str(final_cleanup["removed_path"])),
            "metadata_path": metadata_path,
            "candidate_reports": [],
            "phrase_choices": [],
            "best_similarity_score": float(final_cleanup["best_similarity_score"]),
            "best_word_report": str(final_cleanup["best_word_report"]),
            "best_letter_report": str(final_cleanup["best_letter_report"]),
            "repair_attempts": int(repair_metadata.get("repair_attempts", 0)),
            "repaired_word_count": int(repair_metadata.get("repaired_word_count", 0)),
            "timings": {
                "npy": round(float(raw_metadata["timings"]["npy"]), 2),
                "f0": round(float(raw_metadata["timings"]["f0"]), 2),
                "infer": round(float(raw_metadata["timings"]["infer"]), 2),
            },
        }
