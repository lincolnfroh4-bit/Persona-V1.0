from __future__ import annotations

import csv
import json
import shutil
import threading
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import ffmpeg
import numpy as np
import soundfile as sf

from simple_rebuild import RebuildFeatureBuilder
from simple_svs import GuidedSVSManager
from simple_touchup import LetterAwarePronunciationScorer, lyrics_to_words, normalize_lyrics


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def slugify_name(value: str) -> str:
    cleaned = []
    for char in str(value or "").strip():
        if char.isalnum() or char in {"-", "_"}:
            cleaned.append(char)
        elif char in {" ", "."}:
            cleaned.append("_")
    collapsed = "".join(cleaned).strip("_")
    return collapsed or "item"


def normalize_match_key(value: str) -> str:
    text = Path(str(value or "")).stem.lower()
    return "".join(char for char in text if char.isalnum())


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
        single = singles.get(normalized[index], normalized[index].upper())
        if single:
            units.extend(unit for unit in single.split(" ") if unit)
        index += 1

    if units and units[-1] in {"AH", "EH", "IH", "UH"} and normalized.endswith("e"):
        units[-1] = "IY"
    return units


class PIPAModelStore:
    ALIGNMENT_THRESHOLDS = {
        "forgiving": {"word": 52.0, "phrase": 58.0},
        "balanced": {"word": 66.0, "phrase": 72.0},
        "strict": {"word": 78.0, "phrase": 84.0},
    }

    def __init__(self, repo_root: Path):
        self.repo_root = Path(repo_root)
        self.root = self.repo_root / "pipa_models"
        self.root.mkdir(parents=True, exist_ok=True)
        self._reference_index_cache: Dict[str, Dict[str, object]] = {}
        self.rebuild_builder = RebuildFeatureBuilder(self.repo_root)
        self.guided_svs = GuidedSVSManager(self.repo_root)

    def _ffmpeg_binary(self) -> str:
        local = self.repo_root / "ffmpeg.exe"
        return str(local) if local.exists() else "ffmpeg"

    def _load_audio(self, file_path: Path, sample_rate: int = 44100) -> Tuple[np.ndarray, int]:
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

    def list_bundles(self) -> List[Dict[str, object]]:
        manifests = sorted(self.root.glob("*/manifest.json"))
        bundles: List[Dict[str, object]] = []
        for manifest_path in manifests:
            try:
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            except Exception:
                continue
            if not isinstance(manifest, dict):
                continue
            bundle_id = str(manifest.get("id", manifest_path.parent.name)).strip()
            selection_name = str(
                manifest.get("selection_name", f"pipa:{bundle_id}")
            ).strip()
            label = str(manifest.get("label", bundle_id.replace("_", " "))).strip()
            rvc_model_name = str(manifest.get("rvc_model_name", "")).strip()
            package_mode = str(
                manifest.get("package_mode", manifest.get("backbone", {}).get("output_mode", "pipa-full"))
                or "pipa-full"
            ).strip()
            if not bundle_id or not selection_name:
                continue
            bundles.append(
                {
                    "id": bundle_id,
                    "name": selection_name,
                    "label": label,
                    "kind": "pipa",
                    "rvc_model_name": rvc_model_name,
                    "has_backing_model": bool(rvc_model_name),
                    "package_mode": package_mode,
                    "default_index": str(manifest.get("index_path", "") or ""),
                    "has_index": bool(str(manifest.get("index_path", "") or "")),
                    "manifest_path": manifest_path.as_posix(),
                    "phoneme_profile_path": str(
                        manifest.get("phoneme_profile_path", "") or ""
                    ),
                    "rebuild_profile_path": str(
                        manifest.get("rebuild_profile_path", "") or ""
                    ),
                    "reference_bank_path": str(
                        manifest.get("reference_bank_index_path", "") or ""
                    ),
                    "training_report_path": str(
                        manifest.get("training_report_path", "") or ""
                    ),
                    "guided_regeneration_path": str(
                        manifest.get("guided_regeneration_path", "") or ""
                    ),
                    "guided_regeneration_config_path": str(
                        manifest.get("guided_regeneration_config_path", "") or ""
                    ),
                    "guided_regeneration_stats_path": str(
                        manifest.get("guided_regeneration_stats_path", "") or ""
                    ),
                    "guided_regeneration_report_path": str(
                        manifest.get("guided_regeneration_report_path", "") or ""
                    ),
                    "guided_vocoder_path": str(
                        manifest.get("guided_vocoder_path", "") or ""
                    ),
                    "guided_vocoder_config_path": str(
                        manifest.get("guided_vocoder_config_path", "") or ""
                    ),
                    "guided_vocoder_report_path": str(
                        manifest.get("guided_vocoder_report_path", "") or ""
                    ),
                    "guided_regeneration_preview_path": str(
                        manifest.get("guided_regeneration_preview_path", "") or ""
                    ),
                    "guided_regeneration_target_preview_path": str(
                        manifest.get("guided_regeneration_target_preview_path", "") or ""
                    ),
                    "has_guided_regeneration": bool(
                        str(manifest.get("guided_regeneration_path", "") or "")
                    ),
                    "alignment_tolerance": str(
                        manifest.get("pronunciation_strategy", {})
                        .get("alignment_tolerance", "balanced")
                    ),
                    "phoneme_mode": str(
                        manifest.get("pronunciation_strategy", {})
                        .get("unit_mode", "approx-pronunciation")
                    ),
                    "transcripted_clips": int(
                        manifest.get("dataset", {}).get("matched_audio_files", 0)
                    ),
                    "reference_word_count": int(
                        manifest.get("dataset", {}).get("reference_word_count", 0)
                    ),
                    "reference_phrase_count": int(
                        manifest.get("dataset", {}).get("reference_phrase_count", 0)
                    ),
                }
            )
        return bundles

    def resolve_bundle(self, selection_name: str) -> Optional[Dict[str, object]]:
        target = str(selection_name or "").strip()
        if not target:
            return None
        for bundle in self.list_bundles():
            if bundle["name"] == target or bundle["id"] == target:
                return dict(bundle)
        return None

    def load_reference_index(self, reference_bank_path: str) -> Dict[str, object]:
        cache_key = str(reference_bank_path or "").strip()
        if not cache_key:
            return {"words": [], "phrases": []}
        cached = self._reference_index_cache.get(cache_key)
        if cached is not None:
            return {
                "words": [dict(entry) for entry in cached.get("words", [])],
                "phrases": [dict(entry) for entry in cached.get("phrases", [])],
            }
        path = Path(cache_key)
        if not path.exists():
            return {"words": [], "phrases": []}
        try:
            loaded = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return {"words": [], "phrases": []}
        if not isinstance(loaded, dict):
            return {"words": [], "phrases": []}
        words = [
            dict(entry)
            for entry in loaded.get("words", [])
            if isinstance(entry, dict)
        ]
        phrases = [
            dict(entry)
            for entry in loaded.get("phrases", [])
            if isinstance(entry, dict)
        ]
        self._reference_index_cache[cache_key] = {"words": words, "phrases": phrases}
        return {
            "words": [dict(entry) for entry in words],
            "phrases": [dict(entry) for entry in phrases],
        }

    def find_reference_candidates(
        self,
        *,
        reference_bank_path: str,
        phrase_text: str,
        words: List[str],
        limit: int = 3,
    ) -> List[Dict[str, object]]:
        index_payload = self.load_reference_index(reference_bank_path)
        normalized_phrase = normalize_lyrics(phrase_text)
        normalized_words = [normalize_lyrics(word) for word in words if normalize_lyrics(word)]

        ranked: List[Tuple[float, Dict[str, object]]] = []
        for entry in index_payload.get("phrases", []):
            entry_phrase = normalize_lyrics(str(entry.get("phrase", "")))
            if entry_phrase and entry_phrase == normalized_phrase:
                ranked.append((220.0 + float(entry.get("similarity", 0.0)), dict(entry)))

        if len(normalized_words) == 1:
            target_word = normalized_words[0]
            target_units = pronunciation_units(target_word)
            for entry in index_payload.get("words", []):
                entry_word = normalize_lyrics(str(entry.get("word", "")))
                if not entry_word:
                    continue
                score = 0.0
                if entry_word == target_word:
                    score = 180.0 + float(entry.get("similarity", 0.0))
                else:
                    entry_units = [str(unit) for unit in entry.get("units", []) if str(unit)]
                    overlap = len(set(target_units) & set(entry_units))
                    union = len(set(target_units) | set(entry_units))
                    if union:
                        ratio = overlap / float(union)
                        if ratio >= 0.65:
                            score = 110.0 + (ratio * 40.0) + float(entry.get("similarity", 0.0))
                if score > 0.0:
                    ranked.append((score, dict(entry)))

        ranked.sort(key=lambda item: item[0], reverse=True)
        selected: List[Dict[str, object]] = []
        seen_paths: set[str] = set()
        for _, entry in ranked:
            rel_path = str(entry.get("relative_path", "")).strip()
            if not rel_path or rel_path in seen_paths:
                continue
            seen_paths.add(rel_path)
            selected.append(entry)
            if len(selected) >= max(1, int(limit)):
                break
        return selected

    def _load_training_plan(
        self,
        *,
        plan_paths: Optional[List[Path]],
        transcript_paths: Optional[List[Path]] = None,
    ) -> Tuple[Dict[str, object], Optional[Path]]:
        candidates: List[Path] = [Path(path) for path in (plan_paths or []) if Path(path).exists()]
        if not candidates:
            for transcript_path in transcript_paths or []:
                path = Path(transcript_path)
                if path.suffix.lower() == ".json" and path.exists():
                    candidates.append(path)

        for candidate in candidates:
            try:
                payload = json.loads(candidate.read_text(encoding="utf-8"))
            except Exception:
                continue
            normalized = self._normalize_training_plan_payload(payload)
            if normalized.get("base_vocals") or normalized.get("paired_songs"):
                return normalized, candidate
        return {}, None

    def _normalize_training_plan_payload(self, payload: object) -> Dict[str, object]:
        if not isinstance(payload, dict):
            return {}

        def normalize_file_entry(item: object) -> Dict[str, str]:
            if isinstance(item, str):
                return {"file": item.strip(), "lyrics": ""}
            if not isinstance(item, dict):
                return {}
            file_value = (
                item.get("file")
                or item.get("filename")
                or item.get("audio_file")
                or item.get("path")
                or ""
            )
            return {
                "file": str(file_value).strip(),
                "lyrics": normalize_lyrics(
                    item.get("lyrics")
                    or item.get("transcript")
                    or item.get("text")
                    or ""
                ),
            }

        base_vocals: List[Dict[str, str]] = []
        for item in (
            payload.get("base_vocals")
            or payload.get("pure_vocals")
            or payload.get("voice_dataset")
            or payload.get("identity_clips")
            or []
        ):
            normalized = normalize_file_entry(item)
            if normalized.get("file"):
                base_vocals.append(normalized)

        paired_songs: List[Dict[str, object]] = []
        raw_songs = payload.get("paired_songs") or payload.get("songs") or payload.get("pairs") or []
        for song_index, raw_song in enumerate(raw_songs, start=1):
            if not isinstance(raw_song, dict):
                continue
            target_block = raw_song.get("target") if isinstance(raw_song.get("target"), dict) else {}
            target_file = (
                raw_song.get("target_file")
                or target_block.get("file")
                or target_block.get("filename")
                or target_block.get("audio_file")
                or ""
            )
            lyrics = normalize_lyrics(
                raw_song.get("lyrics")
                or raw_song.get("transcript")
                or target_block.get("lyrics")
                or target_block.get("transcript")
                or ""
            )
            guides_raw = (
                raw_song.get("depersonafied_files")
                or raw_song.get("de_personafied_files")
                or raw_song.get("depersona_files")
                or raw_song.get("guide_files")
                or raw_song.get("guides")
                or raw_song.get("sources")
                or []
            )
            guides: List[Dict[str, str]] = []
            for guide in guides_raw:
                normalized_guide = normalize_file_entry(guide)
                if normalized_guide.get("file"):
                    guides.append(normalized_guide)
            if str(target_file).strip() and guides:
                paired_songs.append(
                    {
                        "song_id": str(raw_song.get("song_id") or raw_song.get("id") or f"song_{song_index:02d}"),
                        "lyrics": lyrics,
                        "target_file": str(target_file).strip(),
                        "guides": guides,
                    }
                )

        return {
            "plan_version": str(payload.get("plan_version") or "persona-builder-v1"),
            "base_vocals": base_vocals,
            "paired_songs": paired_songs,
        }

    def _build_audio_lookup(self, audio_paths: List[Path]) -> Dict[str, Path]:
        lookup: Dict[str, Path] = {}
        for audio_path in audio_paths:
            for key in {normalize_match_key(audio_path.name), normalize_match_key(audio_path.stem)}:
                if key:
                    lookup[key] = audio_path
        return lookup

    def _resolve_audio_path(self, audio_lookup: Dict[str, Path], file_name: str) -> Optional[Path]:
        cleaned = str(file_name or "").strip()
        if not cleaned:
            return None
        return audio_lookup.get(normalize_match_key(cleaned))

    def _prepare_persona_training_assets(
        self,
        *,
        package_id: str,
        training_plan: Dict[str, object],
        training_plan_path: Optional[Path],
        audio_paths: List[Path],
        transcript_paths: List[Path],
        output_dir: Path,
        alignment_tolerance: str,
        update_status: Callable[[str, str, str, int], None],
        cancel_event: Optional[threading.Event] = None,
    ) -> Dict[str, object]:
        reference_bank_dir = output_dir / "reference_bank"
        guided_svs_dataset_dir = output_dir / "_guided_svs_dataset"
        if reference_bank_dir.exists():
            shutil.rmtree(reference_bank_dir, ignore_errors=True)
        reference_bank_dir.mkdir(parents=True, exist_ok=True)
        if guided_svs_dataset_dir.exists():
            shutil.rmtree(guided_svs_dataset_dir, ignore_errors=True)
        guided_svs_dataset_dir.mkdir(parents=True, exist_ok=True)

        matched_transcripts, transcript_report = self._match_transcripts(
            audio_paths=audio_paths,
            transcript_paths=transcript_paths,
        )
        transcript_manifest_path = output_dir / "transcript_manifest.json"
        transcript_manifest_path.write_text(
            json.dumps(
                {
                    "package_id": package_id,
                    "created_at": _utc_now_iso(),
                    "training_plan_path": str(training_plan_path or ""),
                    **transcript_report,
                },
                indent=2,
            ),
            encoding="utf-8",
        )

        audio_lookup = self._build_audio_lookup(audio_paths)
        base_specs = list(training_plan.get("base_vocals") or [])
        paired_specs = list(training_plan.get("paired_songs") or [])
        if not base_specs and not paired_specs:
            raise RuntimeError("Training plan did not include any base vocals or paired songs.")

        threshold_config = self.ALIGNMENT_THRESHOLDS.get(
            str(alignment_tolerance or "balanced").strip().lower(),
            self.ALIGNMENT_THRESHOLDS["balanced"],
        )
        scorer = LetterAwarePronunciationScorer()
        word_candidates: List[Dict[str, object]] = []
        phrase_candidates: List[Dict[str, object]] = []
        clip_reports: List[Dict[str, object]] = []
        rebuild_clip_analyses: List[Dict[str, object]] = []
        guided_svs_entries: List[object] = []
        base_reports: List[Dict[str, object]] = []
        paired_song_reports: List[Dict[str, object]] = []
        total_units = max(1, len(base_specs) + sum(len(song.get("guides", [])) for song in paired_specs))
        processed_units = 0

        def ensure_running() -> None:
            if cancel_event is not None and cancel_event.is_set():
                raise InterruptedError("Training stopped by user.")

        def progress_at(start: int, width: int, fraction: float) -> int:
            return int(start + round(float(np.clip(fraction, 0.0, 1.0)) * width))

        for base_index, base_spec in enumerate(base_specs, start=1):
            ensure_running()
            base_path = self._resolve_audio_path(audio_lookup, str(base_spec.get("file", "")))
            if base_path is None:
                raise RuntimeError(f"Base vocal file '{base_spec.get('file', '')}' was not uploaded.")
            update_status(
                "persona-base",
                f"Preparing base voice clip {base_index}/{len(base_specs)}: {base_path.name}",
                "Building identity windows from pure target vocals.",
                progress_at(6, 8, processed_units / float(total_units)),
            )
            audio, sample_rate = self._load_audio(base_path, sample_rate=44100)
            clip_lyrics = normalize_lyrics(
                str(base_spec.get("lyrics", "") or matched_transcripts.get(str(base_path.resolve()), "") or "")
            )
            if clip_lyrics:
                analysis, analysis_metadata = self._analyze_clip_pronunciation(
                    scorer=scorer,
                    audio=audio,
                    sample_rate=sample_rate,
                    clip_text=clip_lyrics,
                    clip_name=base_path.name,
                    progress_callback=lambda fraction, message, detail: update_status(
                        "persona-base",
                        message,
                        detail,
                        progress_at(6, 8, (processed_units + float(fraction)) / float(total_units)),
                    ),
                    cancel_event=cancel_event,
                )
                scored_words = [dict(entry) for entry in analysis.get("word_scores", [])]
                rebuild_clip_analysis = self.rebuild_builder.analyze_aligned_audio(
                    audio=audio,
                    sample_rate=sample_rate,
                    lyrics=clip_lyrics,
                    word_scores=scored_words,
                    source_name=base_path.name,
                )
                rebuild_clip_analyses.append(rebuild_clip_analysis)
                word_performance_lookup = {
                    int(entry.get("index", -1)): dict(entry)
                    for entry in rebuild_clip_analysis.get("word_performance", [])
                    if int(entry.get("index", -1)) >= 0
                }
                word_candidates.extend(
                    self._extract_word_candidates(
                        audio_path=base_path,
                        audio=audio,
                        sample_rate=sample_rate,
                        word_scores=scored_words,
                        threshold=float(threshold_config["word"]),
                        performance_lookup=word_performance_lookup,
                    )
                )
                phrase_candidates.extend(
                    self._extract_phrase_candidates(
                        audio_path=base_path,
                        audio=audio,
                        sample_rate=sample_rate,
                        word_scores=scored_words,
                        threshold=float(threshold_config["phrase"]),
                        performance_lookup=word_performance_lookup,
                    )
                )
                clip_reports.append(
                    {
                        "audio_file": base_path.name,
                        "transcript": clip_lyrics,
                        "used_for_profile": True,
                        "duration_seconds": round(float(audio.shape[0]) / float(sample_rate), 3),
                        "word_similarity": float(analysis.get("similarity_score", 0.0)),
                        "word_count": len(scored_words),
                        "analysis_strategy": str(analysis_metadata.get("strategy", "full-audio")),
                        "dataset_role": "base-vocal",
                    }
                )
            guided_svs_entries.extend(
                self.guided_svs.build_identity_training_examples(
                    sample_id_prefix=f"base_{base_index:03d}_{slugify_name(base_path.stem)}",
                    source_name=base_path.name,
                    audio=audio,
                    sample_rate=sample_rate,
                    output_dir=guided_svs_dataset_dir,
                    progress_callback=lambda fraction, message, detail: update_status(
                        "persona-base",
                        message,
                        detail,
                        progress_at(6, 8, (processed_units + float(fraction)) / float(total_units)),
                    ),
                    cancel_event=cancel_event,
                )
            )
            base_reports.append(
                {
                    "audio_file": base_path.name,
                    "lyrics_supplied": bool(clip_lyrics),
                    "dataset_role": "base-vocal",
                }
            )
            processed_units += 1

        for song_index, song_spec in enumerate(paired_specs, start=1):
            ensure_running()
            target_path = self._resolve_audio_path(audio_lookup, str(song_spec.get("target_file", "")))
            if target_path is None:
                raise RuntimeError(f"Target song file '{song_spec.get('target_file', '')}' was not uploaded.")
            song_lyrics = normalize_lyrics(
                str(song_spec.get("lyrics", "") or matched_transcripts.get(str(target_path.resolve()), "") or "")
            )
            if not song_lyrics:
                raise RuntimeError(f"Paired song '{song_spec.get('song_id', song_index)}' is missing lyrics.")

            update_status(
                "persona-paired",
                f"Aligning target song {song_index}/{len(paired_specs)}: {target_path.name}",
                "Extracting target timing and pronunciation references for paired regeneration.",
                progress_at(14, 12, processed_units / float(total_units)),
            )
            target_audio, sample_rate = self._load_audio(target_path, sample_rate=44100)
            target_analysis, target_analysis_metadata = self._analyze_clip_pronunciation(
                scorer=scorer,
                audio=target_audio,
                sample_rate=sample_rate,
                clip_text=song_lyrics,
                clip_name=target_path.name,
                progress_callback=lambda fraction, message, detail: update_status(
                    "persona-paired",
                    message,
                    detail,
                    progress_at(14, 12, (processed_units + (float(fraction) * 0.35)) / float(total_units)),
                ),
                cancel_event=cancel_event,
            )
            target_scored_words = [dict(entry) for entry in target_analysis.get("word_scores", [])]
            rebuild_clip_analysis = self.rebuild_builder.analyze_aligned_audio(
                audio=target_audio,
                sample_rate=sample_rate,
                lyrics=song_lyrics,
                word_scores=target_scored_words,
                source_name=target_path.name,
            )
            rebuild_clip_analyses.append(rebuild_clip_analysis)
            word_performance_lookup = {
                int(entry.get("index", -1)): dict(entry)
                for entry in rebuild_clip_analysis.get("word_performance", [])
                if int(entry.get("index", -1)) >= 0
            }
            word_candidates.extend(
                self._extract_word_candidates(
                    audio_path=target_path,
                    audio=target_audio,
                    sample_rate=sample_rate,
                    word_scores=target_scored_words,
                    threshold=float(threshold_config["word"]),
                    performance_lookup=word_performance_lookup,
                )
            )
            phrase_candidates.extend(
                self._extract_phrase_candidates(
                    audio_path=target_path,
                    audio=target_audio,
                    sample_rate=sample_rate,
                    word_scores=target_scored_words,
                    threshold=float(threshold_config["phrase"]),
                    performance_lookup=word_performance_lookup,
                )
            )
            clip_reports.append(
                {
                    "audio_file": target_path.name,
                    "transcript": song_lyrics,
                    "used_for_profile": True,
                    "duration_seconds": round(float(target_audio.shape[0]) / float(sample_rate), 3),
                    "word_similarity": float(target_analysis.get("similarity_score", 0.0)),
                    "word_count": len(target_scored_words),
                    "analysis_strategy": str(target_analysis_metadata.get("strategy", "full-audio")),
                    "dataset_role": "paired-target",
                    "song_id": str(song_spec.get("song_id", "")),
                }
            )

            song_report = {
                "song_id": str(song_spec.get("song_id", f"song_{song_index:02d}")),
                "target_file": target_path.name,
                "lyrics_word_count": len(lyrics_to_words(song_lyrics)),
                "guide_reports": [],
            }
            for guide_index, guide_spec in enumerate(song_spec.get("guides", []), start=1):
                ensure_running()
                guide_path = self._resolve_audio_path(audio_lookup, str(guide_spec.get("file", "")))
                if guide_path is None:
                    raise RuntimeError(
                        f"De-personafied guide '{guide_spec.get('file', '')}' was not uploaded for {song_report['song_id']}."
                    )
                guide_audio, _ = self._load_audio(guide_path, sample_rate=sample_rate)
                guide_analysis, guide_analysis_metadata = self._analyze_clip_pronunciation(
                    scorer=scorer,
                    audio=guide_audio,
                    sample_rate=sample_rate,
                    clip_text=song_lyrics,
                    clip_name=guide_path.name,
                    progress_callback=lambda fraction, message, detail: update_status(
                        "persona-paired",
                        message,
                        detail,
                        progress_at(14, 12, (processed_units + float(fraction)) / float(total_units)),
                    ),
                    cancel_event=cancel_event,
                )
                guide_scored_words = [dict(entry) for entry in guide_analysis.get("word_scores", [])]
                guided_svs_entries.extend(
                    self.guided_svs.build_paired_training_examples(
                        sample_id_prefix=(
                            f"song_{song_index:02d}_{slugify_name(target_path.stem)}_"
                            f"guide_{guide_index:02d}_{slugify_name(guide_path.stem)}"
                        ),
                        source_name=target_path.name,
                        conditioning_name=guide_path.name,
                        guide_audio=guide_audio,
                        target_audio=target_audio,
                        sample_rate=sample_rate,
                        lyrics=song_lyrics,
                        target_word_scores=target_scored_words,
                        guide_word_scores=guide_scored_words,
                        guide_similarity_score=float(guide_analysis.get("similarity_score", 0.0)),
                        output_dir=guided_svs_dataset_dir,
                        progress_callback=lambda fraction, message, detail: update_status(
                            "persona-paired",
                            message,
                            f"{detail} | song {song_report['song_id']} | guide {guide_index}/{max(1, len(song_spec.get('guides', [])))}",
                            progress_at(14, 12, (processed_units + float(fraction)) / float(total_units)),
                        ),
                        cancel_event=cancel_event,
                    )
                )
                song_report["guide_reports"].append(
                    {
                        "guide_file": guide_path.name,
                        "similarity_to_lyrics": float(guide_analysis.get("similarity_score", 0.0)),
                        "analysis_strategy": str(guide_analysis_metadata.get("strategy", "full-audio")),
                    }
                )
                processed_units += 1
            paired_song_reports.append(song_report)

        ensure_running()
        update_status(
            "pipa-refs",
            "Selecting the strongest target-word and target-phrase references...",
            f"Raw target candidates: {len(word_candidates)} words | {len(phrase_candidates)} phrases",
            26,
        )
        selected_word_candidates = self._select_best_candidates(
            word_candidates,
            key_name="word",
            per_key_limit=3,
            overall_limit=480,
        )
        selected_phrase_candidates = self._select_best_candidates(
            phrase_candidates,
            key_name="phrase",
            per_key_limit=2,
            overall_limit=280,
        )
        word_entries = self._materialize_reference_candidates(
            selected_word_candidates,
            target_dir=reference_bank_dir / "words",
            text_key="word",
            kind="word",
            progress_callback=lambda index, total, kind_name: update_status(
                "pipa-refs",
                f"Writing target reference snippets ({kind_name})...",
                f"{kind_name.title()} refs {index}/{total}",
                27 if kind_name == "word" else 28,
            ),
            cancel_event=cancel_event,
        )
        phrase_entries = self._materialize_reference_candidates(
            selected_phrase_candidates,
            target_dir=reference_bank_dir / "phrases",
            text_key="phrase",
            kind="phrase",
            progress_callback=lambda index, total, kind_name: update_status(
                "pipa-refs",
                f"Writing target reference snippets ({kind_name})...",
                f"{kind_name.title()} refs {index}/{total}",
                28,
            ),
            cancel_event=cancel_event,
        )

        ensure_running()
        update_status(
            "persona-dataset",
            "Finalizing the base + paired voice-builder dataset...",
            (
                f"Base clips {len(base_specs)} | paired songs {len(paired_specs)} | "
                f"slices {len(guided_svs_entries)}"
            ),
            29,
        )
        guided_svs_dataset = self.guided_svs.finalize_training_dataset(
            dataset_dir=guided_svs_dataset_dir,
            sample_entries=guided_svs_entries,
        )

        phoneme_profile = self._build_phoneme_profile(
            package_id=package_id,
            clip_reports=clip_reports,
            word_entries=word_entries,
            phrase_entries=phrase_entries,
            alignment_tolerance=str(alignment_tolerance or "balanced"),
        )
        depersonafied_variant_count = sum(len(song.get("guides", [])) for song in paired_specs)
        phoneme_profile["training_recipe"] = {
            "mode": "persona-builder-v1",
            "plan_version": str(training_plan.get("plan_version", "persona-builder-v1")),
            "base_voice_clip_count": len(base_specs),
            "paired_song_count": len(paired_specs),
            "depersonafied_variant_count": depersonafied_variant_count,
        }
        phoneme_profile_path = output_dir / "phoneme_profile.json"
        phoneme_profile_path.write_text(json.dumps(phoneme_profile, indent=2), encoding="utf-8")

        rebuild_clip_reports_path = output_dir / "rebuild_clip_reports.json"
        rebuild_clip_reports_path.write_text(
            json.dumps(
                {
                    "package_id": package_id,
                    "created_at": _utc_now_iso(),
                    "clip_analyses": rebuild_clip_analyses,
                    "base_reports": base_reports,
                    "paired_song_reports": paired_song_reports,
                },
                indent=2,
            ),
            encoding="utf-8",
        )

        rebuild_profile = self.rebuild_builder.build_package_profile(
            package_id=package_id,
            clip_analyses=rebuild_clip_analyses,
            word_entries=word_entries,
            phrase_entries=phrase_entries,
            alignment_tolerance=str(alignment_tolerance or "balanced"),
        )
        rebuild_profile["created_at"] = _utc_now_iso()
        rebuild_profile["training_recipe"] = dict(phoneme_profile["training_recipe"])
        rebuild_profile_path = output_dir / "rebuild_profile.json"
        rebuild_profile_path.write_text(json.dumps(rebuild_profile, indent=2), encoding="utf-8")

        reference_bank_index_path = output_dir / "reference_bank_index.json"
        reference_bank_index_path.write_text(
            json.dumps(
                {
                    "created_at": _utc_now_iso(),
                    "package_id": package_id,
                    "words": word_entries,
                    "phrases": phrase_entries,
                },
                indent=2,
            ),
            encoding="utf-8",
        )

        training_report_path = output_dir / "training_report.json"
        training_report_path.write_text(
            json.dumps(
                {
                    "package_id": package_id,
                    "created_at": _utc_now_iso(),
                    "alignment_tolerance": str(alignment_tolerance or "balanced"),
                    "training_plan_path": str(training_plan_path or ""),
                    "training_plan": training_plan,
                    "transcript_report": transcript_report,
                    "base_reports": base_reports,
                    "paired_song_reports": paired_song_reports,
                    "clip_reports": clip_reports,
                    "reference_word_count": len(word_entries),
                    "reference_phrase_count": len(phrase_entries),
                    "guided_regeneration_dataset": {
                        "sample_count": int(guided_svs_dataset.get("sample_count", 0)),
                        "total_frames": int(guided_svs_dataset.get("total_frames", 0)),
                        "total_seconds": float(guided_svs_dataset.get("total_seconds", 0.0)),
                        "stats_path": str(guided_svs_dataset.get("stats_path", "")),
                        "report_path": str(guided_svs_dataset.get("report_path", "")),
                    },
                    "training_recipe": dict(phoneme_profile["training_recipe"]),
                },
                indent=2,
            ),
            encoding="utf-8",
        )

        update_status(
            "persona-ready",
            "Finished the base-voice + paired-song prep pass.",
            (
                f"Base clips {len(base_specs)} | paired songs {len(paired_specs)} | "
                f"depersonafied guides {depersonafied_variant_count} | "
                f"training slices {int(guided_svs_dataset.get('sample_count', 0))}"
            ),
            30,
        )
        return {
            "phoneme_profile_path": str(phoneme_profile_path),
            "rebuild_profile_path": str(rebuild_profile_path),
            "rebuild_clip_reports_path": str(rebuild_clip_reports_path),
            "training_report_path": str(training_report_path),
            "reference_bank_index_path": str(reference_bank_index_path),
            "transcript_manifest_path": str(transcript_manifest_path),
            "reference_word_count": len(word_entries),
            "reference_phrase_count": len(phrase_entries),
            "matched_audio_files": int(transcript_report["matched_audio_files"]),
            "total_audio_files": int(transcript_report["total_audio_files"]),
            "skipped_audio_files": int(transcript_report["skipped_audio_files"]),
            "alignment_tolerance": str(alignment_tolerance or "balanced"),
            "phoneme_mode": "persona-builder-conditioning",
            "guided_svs_dataset_dir": str(guided_svs_dataset.get("dataset_dir", "")),
            "guided_svs_stats_path": str(guided_svs_dataset.get("stats_path", "")),
            "guided_svs_report_path": str(guided_svs_dataset.get("report_path", "")),
            "guided_svs_sample_count": int(guided_svs_dataset.get("sample_count", 0)),
            "guided_svs_total_frames": int(guided_svs_dataset.get("total_frames", 0)),
            "guided_svs_total_seconds": float(guided_svs_dataset.get("total_seconds", 0.0)),
            "training_plan_path": str(training_plan_path or ""),
            "base_voice_clip_count": len(base_specs),
            "paired_song_count": len(paired_specs),
            "depersonafied_variant_count": depersonafied_variant_count,
            "build_dir": str(output_dir),
        }

    def prepare_training_assets(
        self,
        *,
        package_id: str,
        audio_paths: List[Path],
        transcript_paths: List[Path],
        plan_paths: Optional[List[Path]] = None,
        output_dir: Path,
        alignment_tolerance: str,
        update_status: Callable[[str, str, str, int], None],
        cancel_event: Optional[threading.Event] = None,
    ) -> Dict[str, object]:
        output_dir.mkdir(parents=True, exist_ok=True)
        training_plan, training_plan_path = self._load_training_plan(
            plan_paths=plan_paths,
            transcript_paths=transcript_paths,
        )
        if training_plan.get("base_vocals") or training_plan.get("paired_songs"):
            return self._prepare_persona_training_assets(
                package_id=package_id,
                training_plan=training_plan,
                training_plan_path=training_plan_path,
                audio_paths=audio_paths,
                transcript_paths=transcript_paths,
                output_dir=output_dir,
                alignment_tolerance=alignment_tolerance,
                update_status=update_status,
                cancel_event=cancel_event,
            )
        reference_bank_dir = output_dir / "reference_bank"
        guided_svs_dataset_dir = output_dir / "_guided_svs_dataset"
        if reference_bank_dir.exists():
            shutil.rmtree(reference_bank_dir, ignore_errors=True)
        reference_bank_dir.mkdir(parents=True, exist_ok=True)
        if guided_svs_dataset_dir.exists():
            shutil.rmtree(guided_svs_dataset_dir, ignore_errors=True)
        guided_svs_dataset_dir.mkdir(parents=True, exist_ok=True)

        matched_transcripts, transcript_report = self._match_transcripts(
            audio_paths=audio_paths,
            transcript_paths=transcript_paths,
        )
        transcript_manifest_path = output_dir / "transcript_manifest.json"
        transcript_manifest_path.write_text(
            json.dumps(
                {
                    "package_id": package_id,
                    "created_at": _utc_now_iso(),
                    **transcript_report,
                    "matched_clips": [
                        {
                            "audio_file": audio_path.name,
                            "transcript": matched_transcripts[str(audio_path.resolve())],
                        }
                        for audio_path in audio_paths
                        if str(audio_path.resolve()) in matched_transcripts
                    ],
                },
                indent=2,
            ),
            encoding="utf-8",
        )

        if not matched_transcripts:
            raise RuntimeError(
                "No transcripts matched your audio clips. Upload matching .txt files or a manifest with filename + transcript."
            )

        threshold_config = self.ALIGNMENT_THRESHOLDS.get(
            str(alignment_tolerance or "balanced").strip().lower(),
            self.ALIGNMENT_THRESHOLDS["balanced"],
        )
        scorer = LetterAwarePronunciationScorer()
        word_candidates: List[Dict[str, object]] = []
        phrase_candidates: List[Dict[str, object]] = []
        clip_reports: List[Dict[str, object]] = []
        rebuild_clip_analyses: List[Dict[str, object]] = []
        guided_svs_entries: List[object] = []
        total_files = max(1, len(audio_paths))

        for index, audio_path in enumerate(audio_paths, start=1):
            if cancel_event is not None and cancel_event.is_set():
                raise InterruptedError("Training stopped by user.")
            clip_text = matched_transcripts.get(str(audio_path.resolve()), "")
            progress = 6 + int(round(((index - 1) / total_files) * 18))
            update_status(
                "pipa-profile",
                f"Analyzing transcripts and pronunciation coverage for {audio_path.name}...",
                "",
                progress,
            )
            if not clip_text:
                clip_reports.append(
                    {
                        "audio_file": audio_path.name,
                        "transcript": "",
                        "used_for_profile": False,
                        "reason": "No transcript matched this clip.",
                    }
                )
                continue

            audio, sample_rate = self._load_audio(audio_path, sample_rate=44100)
            duration_seconds = round(float(audio.shape[0]) / float(sample_rate), 3)

            def report_analysis_progress(
                clip_fraction: float,
                detail_message: str,
                detail_tail: str,
            ) -> None:
                bounded_fraction = float(np.clip(float(clip_fraction), 0.0, 1.0))
                overall_fraction = ((index - 1) + bounded_fraction) / float(total_files)
                overall_progress = 6 + int(round(overall_fraction * 18))
                update_status(
                    "pipa-profile",
                    detail_message,
                    detail_tail,
                    min(23, overall_progress),
                )

            analysis, analysis_metadata = self._analyze_clip_pronunciation(
                scorer=scorer,
                audio=audio,
                sample_rate=sample_rate,
                clip_text=clip_text,
                clip_name=audio_path.name,
                progress_callback=report_analysis_progress,
                cancel_event=cancel_event,
            )
            scored_words = [dict(entry) for entry in analysis.get("word_scores", [])]
            similarity = float(analysis.get("similarity_score", 0.0))
            update_status(
                "pipa-rebuild",
                f"Building rebuild timing map for {audio_path.name}...",
                (
                    f"Alignment ready | strategy {analysis_metadata.get('strategy', 'unknown')} | "
                    f"scored words {len(scored_words)} | similarity {similarity:.1f}%"
                ),
                min(24, 7 + int(round((index / total_files) * 18))),
            )
            rebuild_clip_analysis = self.rebuild_builder.analyze_aligned_audio(
                audio=audio,
                sample_rate=sample_rate,
                lyrics=clip_text,
                word_scores=scored_words,
                source_name=audio_path.name,
            )
            rebuild_clip_analyses.append(rebuild_clip_analysis)
            update_status(
                "pipa-rebuild",
                f"Rebuild timing map ready for {audio_path.name}.",
                (
                    f"Aligned words {len(rebuild_clip_analysis.get('word_performance', []))} | "
                    f"phrases {len(rebuild_clip_analysis.get('phrase_performance', []))} | "
                    f"mode {rebuild_clip_analysis.get('style_summary', {}).get('analysis_mode', 'dense')}"
                ),
                min(25, 8 + int(round((index / total_files) * 18))),
            )
            clip_reports.append(
                {
                    "audio_file": audio_path.name,
                    "transcript": clip_text,
                    "used_for_profile": True,
                    "duration_seconds": duration_seconds,
                    "word_similarity": similarity,
                    "word_count": len(scored_words),
                    "analysis_strategy": str(analysis_metadata.get("strategy", "full-audio")),
                    "chunk_count": int(analysis_metadata.get("chunk_count", 0)),
                    "chunk_reports": list(analysis_metadata.get("chunk_reports", [])),
                    "word_report": str(analysis.get("word_report", "")),
                    "letter_report": str(analysis.get("letter_report", "")),
                }
            )
            update_status(
                "pipa-svs-data",
                f"Cutting pronunciation windows from {audio_path.name}...",
                (
                    f"Sub-word windows from aligned lyrics + pitch contour | "
                    f"similarity {similarity:.1f}%"
                ),
                min(24, 8 + int(round((index / total_files) * 18))),
            )
            def report_guided_feature_progress(
                clip_fraction: float,
                detail_message: str,
                detail_tail: str,
            ) -> None:
                bounded_fraction = float(np.clip(float(clip_fraction), 0.0, 1.0))
                overall_fraction = ((index - 1) + bounded_fraction) / float(total_files)
                overall_progress = 24 + int(round(overall_fraction * 5))
                update_status(
                    "pipa-svs-data",
                    detail_message,
                    (
                        f"{detail_tail} | "
                        f"clip {index}/{total_files} | "
                        f"similarity {similarity:.1f}%"
                    ),
                    min(29, max(24, overall_progress)),
                )
            guided_svs_entries.extend(
                self.guided_svs.build_pronunciation_training_examples(
                    sample_id_prefix=f"{index:04d}_{slugify_name(audio_path.stem)}",
                    source_name=audio_path.name,
                    audio=audio,
                    sample_rate=sample_rate,
                    lyrics=clip_text,
                    word_scores=scored_words,
                    output_dir=guided_svs_dataset_dir,
                    progress_callback=report_guided_feature_progress,
                    cancel_event=cancel_event,
                )
            )

            word_threshold = float(threshold_config["word"])
            phrase_threshold = float(threshold_config["phrase"])
            strategy_name = str(analysis_metadata.get("strategy", "full-audio"))
            if strategy_name.startswith("chunked"):
                word_threshold = min(word_threshold, 36.0)
                phrase_threshold = min(phrase_threshold, 42.0)
                if similarity < 24.0:
                    word_threshold = min(word_threshold, 24.0)
                    phrase_threshold = min(phrase_threshold, 30.0)

            word_performance_lookup = {
                int(entry.get("index", -1)): dict(entry)
                for entry in rebuild_clip_analysis.get("word_performance", [])
                if int(entry.get("index", -1)) >= 0
            }
            word_candidates.extend(
                self._extract_word_candidates(
                    audio_path=audio_path,
                    audio=audio,
                    sample_rate=sample_rate,
                    word_scores=scored_words,
                    threshold=word_threshold,
                    performance_lookup=word_performance_lookup,
                )
            )
            phrase_candidates.extend(
                self._extract_phrase_candidates(
                    audio_path=audio_path,
                    audio=audio,
                    sample_rate=sample_rate,
                    word_scores=scored_words,
                    threshold=phrase_threshold,
                    performance_lookup=word_performance_lookup,
                )
            )
            update_status(
                "pipa-profile",
                f"Profiled {index}/{total_files} clip{'s' if total_files != 1 else ''}: {audio_path.name}",
                (
                    f"Similarity {similarity:.1f}% | "
                    f"running refs {len(word_candidates)} words / {len(phrase_candidates)} phrases | "
                    f"strategy {strategy_name}"
                ),
                min(23, 6 + int(round((index / total_files) * 18))),
            )

        if cancel_event is not None and cancel_event.is_set():
            raise InterruptedError("Training stopped by user.")
        update_status(
            "pipa-refs",
            "Selecting the strongest word and phrase references from the aligned run...",
            (
                f"Raw candidates so far: {len(word_candidates)} words | "
                f"{len(phrase_candidates)} phrases"
            ),
            24,
        )
        selected_word_candidates = self._select_best_candidates(
            word_candidates,
            key_name="word",
            per_key_limit=3,
            overall_limit=480,
        )
        selected_phrase_candidates = self._select_best_candidates(
            phrase_candidates,
            key_name="phrase",
            per_key_limit=2,
            overall_limit=280,
        )

        if cancel_event is not None and cancel_event.is_set():
            raise InterruptedError("Training stopped by user.")
        update_status(
            "pipa-refs",
            "Writing the reference bank to disk...",
            (
                f"Selected references: {len(selected_word_candidates)} words | "
                f"{len(selected_phrase_candidates)} phrases"
            ),
            26,
        )
        word_entries = self._materialize_reference_candidates(
            selected_word_candidates,
            target_dir=reference_bank_dir / "words",
            text_key="word",
            kind="word",
            progress_callback=lambda index, total, kind_name: update_status(
                "pipa-refs",
                f"Writing reference snippets ({kind_name})...",
                f"{kind_name.title()} refs {index}/{total}",
                27 if kind_name == "word" else 28,
            ),
            cancel_event=cancel_event,
        )
        phrase_entries = self._materialize_reference_candidates(
            selected_phrase_candidates,
            target_dir=reference_bank_dir / "phrases",
            text_key="phrase",
            kind="phrase",
            progress_callback=lambda index, total, kind_name: update_status(
                "pipa-refs",
                f"Writing reference snippets ({kind_name})...",
                f"{kind_name.title()} refs {index}/{total}",
                28,
            ),
            cancel_event=cancel_event,
        )

        if cancel_event is not None and cancel_event.is_set():
            raise InterruptedError("Training stopped by user.")
        update_status(
            "pipa-svs-data",
            "Finalizing the pronunciation-window training set...",
            (
                f"Prepared {len(guided_svs_entries)} pronunciation slice{'s' if len(guided_svs_entries) != 1 else ''} "
                f"for direct vocal regeneration."
            ),
            29,
        )
        guided_svs_dataset = self.guided_svs.finalize_training_dataset(
            dataset_dir=guided_svs_dataset_dir,
            sample_entries=guided_svs_entries,
        )

        if cancel_event is not None and cancel_event.is_set():
            raise InterruptedError("Training stopped by user.")
        update_status(
            "pipa-profile",
            "Building the pronunciation profile and rebuild package metadata...",
            (
                f"Reference bank ready: {len(word_entries)} words | "
                f"{len(phrase_entries)} phrases"
            ),
            29,
        )
        reference_bank_index_path = output_dir / "reference_bank_index.json"
        reference_bank_index_path.write_text(
            json.dumps(
                {
                    "created_at": _utc_now_iso(),
                    "package_id": package_id,
                    "words": word_entries,
                    "phrases": phrase_entries,
                },
                indent=2,
            ),
            encoding="utf-8",
        )

        phoneme_profile = self._build_phoneme_profile(
            package_id=package_id,
            clip_reports=clip_reports,
            word_entries=word_entries,
            phrase_entries=phrase_entries,
            alignment_tolerance=str(alignment_tolerance or "balanced"),
        )
        phoneme_profile_path = output_dir / "phoneme_profile.json"
        phoneme_profile_path.write_text(
            json.dumps(phoneme_profile, indent=2),
            encoding="utf-8",
        )

        rebuild_clip_reports_path = output_dir / "rebuild_clip_reports.json"
        rebuild_clip_reports_path.write_text(
            json.dumps(
                {
                    "package_id": package_id,
                    "created_at": _utc_now_iso(),
                    "clip_analyses": rebuild_clip_analyses,
                },
                indent=2,
            ),
            encoding="utf-8",
        )

        rebuild_profile = self.rebuild_builder.build_package_profile(
            package_id=package_id,
            clip_analyses=rebuild_clip_analyses,
            word_entries=word_entries,
            phrase_entries=phrase_entries,
            alignment_tolerance=str(alignment_tolerance or "balanced"),
        )
        rebuild_profile["created_at"] = _utc_now_iso()
        rebuild_profile_path = output_dir / "rebuild_profile.json"
        rebuild_profile_path.write_text(
            json.dumps(rebuild_profile, indent=2),
            encoding="utf-8",
        )

        training_report_path = output_dir / "training_report.json"
        training_report_path.write_text(
            json.dumps(
                {
                    "package_id": package_id,
                    "created_at": _utc_now_iso(),
                    "alignment_tolerance": str(alignment_tolerance or "balanced"),
                    "transcript_report": transcript_report,
                    "clip_reports": clip_reports,
                    "reference_word_count": len(word_entries),
                    "reference_phrase_count": len(phrase_entries),
                    "guided_regeneration_dataset": {
                        "sample_count": int(guided_svs_dataset.get("sample_count", 0)),
                        "total_frames": int(guided_svs_dataset.get("total_frames", 0)),
                        "total_seconds": float(guided_svs_dataset.get("total_seconds", 0.0)),
                        "stats_path": str(guided_svs_dataset.get("stats_path", "")),
                        "report_path": str(guided_svs_dataset.get("report_path", "")),
                    },
                },
                indent=2,
            ),
            encoding="utf-8",
        )

        summary_lines = [
            f"Matched transcripts: {int(transcript_report['matched_audio_files'])}/{int(transcript_report['total_audio_files'])}",
            f"Reference words: {len(word_entries)}",
            f"Reference phrases: {len(phrase_entries)}",
            f"Pronunciation slices: {int(guided_svs_dataset.get('sample_count', 0))}",
        ]
        update_status(
            "pipa-profile",
            "Finished the pronunciation + rebuild prep pass.",
            " | ".join(summary_lines),
            30,
        )
        return {
            "phoneme_profile_path": str(phoneme_profile_path),
            "rebuild_profile_path": str(rebuild_profile_path),
            "rebuild_clip_reports_path": str(rebuild_clip_reports_path),
            "training_report_path": str(training_report_path),
            "reference_bank_index_path": str(reference_bank_index_path),
            "transcript_manifest_path": str(transcript_manifest_path),
            "reference_word_count": len(word_entries),
            "reference_phrase_count": len(phrase_entries),
            "matched_audio_files": int(transcript_report["matched_audio_files"]),
            "total_audio_files": int(transcript_report["total_audio_files"]),
            "skipped_audio_files": int(transcript_report["skipped_audio_files"]),
            "alignment_tolerance": str(alignment_tolerance or "balanced"),
            "phoneme_mode": "approx-pronunciation-units",
            "guided_svs_dataset_dir": str(guided_svs_dataset.get("dataset_dir", "")),
            "guided_svs_stats_path": str(guided_svs_dataset.get("stats_path", "")),
            "guided_svs_report_path": str(guided_svs_dataset.get("report_path", "")),
            "guided_svs_sample_count": int(guided_svs_dataset.get("sample_count", 0)),
            "guided_svs_total_frames": int(guided_svs_dataset.get("total_frames", 0)),
            "guided_svs_total_seconds": float(guided_svs_dataset.get("total_seconds", 0.0)),
            "build_dir": str(output_dir),
        }

    def _analyze_clip_pronunciation(
        self,
        *,
        scorer: LetterAwarePronunciationScorer,
        audio: np.ndarray,
        sample_rate: int,
        clip_text: str,
        clip_name: str = "",
        progress_callback: Optional[Callable[[float, str, str], None]] = None,
        cancel_event: Optional[threading.Event] = None,
    ) -> Tuple[Dict[str, object], Dict[str, object]]:
        words = lyrics_to_words(clip_text)
        total_seconds = float(audio.shape[0]) / float(max(sample_rate, 1))
        if not words:
            return (
                scorer.build_analysis_result([], []),
                {"strategy": "empty-lyrics", "chunk_count": 0, "chunk_reports": []},
            )

        # Long-form vocals align much more reliably in chunked passes than in one giant
        # CTC decode. This keeps full-song datasets usable instead of collapsing to 0%.
        if total_seconds >= 90.0 or len(words) >= 360:
            chunked = self._analyze_long_form_pronunciation(
                scorer=scorer,
                audio=audio,
                sample_rate=sample_rate,
                words=words,
                clip_name=clip_name,
                progress_callback=progress_callback,
                cancel_event=cancel_event,
            )
            return chunked["analysis"], {
                "strategy": "chunked-long-form",
                "chunk_count": int(chunked.get("chunk_count", 0)),
                "chunk_reports": list(chunked.get("chunk_reports", [])),
            }

        if cancel_event is not None and cancel_event.is_set():
            raise InterruptedError("Training stopped by user.")
        if progress_callback is not None:
            progress_callback(
                0.08,
                f"Scoring clip {clip_name or 'audio'} against its transcript...",
                f"Short-form analysis | {len(words)} words | {total_seconds:.1f}s",
            )
        baseline = scorer.analyze_audio(audio, sample_rate, clip_text)
        baseline_similarity = float(baseline.get("similarity_score", 0.0))
        if progress_callback is not None:
            progress_callback(
                1.0,
                f"Finished transcript scoring for {clip_name or 'audio'}.",
                (
                    f"Short-form analysis | similarity {baseline_similarity:.1f}% | "
                    f"scored words {len(baseline.get('word_scores', []))}"
                ),
            )
        if baseline_similarity >= 12.0:
            return baseline, {"strategy": "full-audio", "chunk_count": 0, "chunk_reports": []}

        chunked = self._analyze_long_form_pronunciation(
            scorer=scorer,
            audio=audio,
            sample_rate=sample_rate,
            words=words,
            clip_name=clip_name,
            progress_callback=progress_callback,
            cancel_event=cancel_event,
        )
        chunked_similarity = float(chunked["analysis"].get("similarity_score", 0.0))
        if chunked_similarity >= baseline_similarity:
            return chunked["analysis"], {
                "strategy": "chunked-fallback",
                "chunk_count": int(chunked.get("chunk_count", 0)),
                "chunk_reports": list(chunked.get("chunk_reports", [])),
            }
        return baseline, {
            "strategy": "full-audio-fallback-kept",
            "chunk_count": int(chunked.get("chunk_count", 0)),
            "chunk_reports": list(chunked.get("chunk_reports", [])),
        }

    def _analyze_long_form_pronunciation(
        self,
        *,
        scorer: LetterAwarePronunciationScorer,
        audio: np.ndarray,
        sample_rate: int,
        words: List[str],
        clip_name: str = "",
        progress_callback: Optional[Callable[[float, str, str], None]] = None,
        cancel_event: Optional[threading.Event] = None,
    ) -> Dict[str, object]:
        total_words = len(words)
        total_seconds = float(audio.shape[0]) / float(max(sample_rate, 1))
        if total_words <= 0 or total_seconds <= 0.0:
            return {
                "analysis": scorer.build_analysis_result([], []),
                "chunk_count": 0,
                "chunk_reports": [],
            }

        words_per_second = total_words / max(total_seconds, 1.0)
        words_per_chunk = int(np.clip(round(words_per_second * 14.0), 28, 72))
        overlap_words = max(4, min(12, words_per_chunk // 6))
        step = max(8, words_per_chunk - overlap_words)

        best_word_scores: Dict[int, Dict[str, object]] = {}
        best_letter_scores: Dict[Tuple[int, str, float], Dict[str, object]] = {}
        chunk_reports: List[Dict[str, object]] = []
        chunk_starts = list(range(0, total_words, step))
        total_chunks = max(1, len(chunk_starts))
        running_similarity_sum = 0.0
        running_similarity_count = 0
        running_best_similarity = 0.0

        for chunk_number, start_index in enumerate(chunk_starts, start=1):
            if cancel_event is not None and cancel_event.is_set():
                raise InterruptedError("Training stopped by user.")
            end_index = min(total_words, start_index + words_per_chunk)
            chunk_words = words[start_index:end_index]
            if not chunk_words:
                continue
            chunk_duration_estimate = total_seconds * ((end_index - start_index) / float(total_words))
            estimated_start = total_seconds * (start_index / float(total_words))
            estimated_end = total_seconds * (end_index / float(total_words))
            if progress_callback is not None:
                progress_callback(
                    max(0.01, (chunk_number - 1) / float(total_chunks)),
                    f"Segmenting {clip_name or 'audio'}: chunk {chunk_number}/{total_chunks}",
                    (
                        f"Preparing chunk {chunk_number}/{total_chunks} | "
                        f"words {start_index + 1}-{end_index} | "
                        f"approx {estimated_start:.1f}s-{estimated_end:.1f}s"
                    ),
                )
            best_chunk_result = None
            best_chunk_similarity = -1.0
            paddings = [
                max(1.5, min(5.0, chunk_duration_estimate * 0.35)),
                max(3.0, min(10.0, chunk_duration_estimate * 0.75)),
            ]
            for padding in paddings:
                chunk_start = max(0.0, estimated_start - padding)
                chunk_end = min(total_seconds, estimated_end + padding)
                start_sample = max(0, int(round(chunk_start * sample_rate)))
                end_sample = min(int(audio.shape[0]), int(round(chunk_end * sample_rate)))
                if end_sample <= start_sample:
                    end_sample = min(int(audio.shape[0]), start_sample + sample_rate)
                chunk_audio = np.asarray(audio[start_sample:end_sample], dtype=np.float32)
                result = scorer.analyze_segment(
                    chunk_audio,
                    sample_rate,
                    chunk_words,
                    global_start_seconds=(start_sample / float(max(sample_rate, 1))),
                    absolute_word_indices=list(range(start_index, end_index)),
                    letter_focus_limit=min(10, len(chunk_words)),
                )
                similarity = float(result.get("similarity_score", 0.0))
                nonzero_words = sum(
                    1 for entry in result.get("word_scores", []) if float(entry.get("similarity", 0.0)) > 0.0
                )
                ranking_score = similarity + (nonzero_words * 0.05)
                if ranking_score > best_chunk_similarity:
                    best_chunk_similarity = ranking_score
                    best_chunk_result = result

            if best_chunk_result is None:
                if progress_callback is not None:
                    progress_callback(
                        chunk_number / float(total_chunks),
                        f"Segmenting {clip_name or 'audio'}: chunk {chunk_number}/{total_chunks}",
                        (
                            f"No usable alignment for chunk {chunk_number}/{total_chunks} | "
                            f"words {start_index + 1}-{end_index}"
                        ),
                    )
                continue

            chunk_similarity = float(best_chunk_result.get("similarity_score", 0.0))
            running_similarity_sum += chunk_similarity
            running_similarity_count += 1
            running_best_similarity = max(running_best_similarity, chunk_similarity)
            usable_words = sum(
                1
                for entry in best_chunk_result.get("word_scores", [])
                if float(entry.get("similarity", 0.0)) >= 24.0
            )
            chunk_reports.append(
                {
                    "start_word_index": int(start_index),
                    "end_word_index": int(end_index - 1),
                    "word_count": int(end_index - start_index),
                    "estimated_start_seconds": round(float(estimated_start), 3),
                    "estimated_end_seconds": round(float(estimated_end), 3),
                    "similarity_score": round(chunk_similarity, 2),
                    "usable_word_count": int(usable_words),
                }
            )
            if progress_callback is not None:
                average_similarity = running_similarity_sum / max(running_similarity_count, 1)
                progress_callback(
                    chunk_number / float(total_chunks),
                    f"Segmenting {clip_name or 'audio'}: chunk {chunk_number}/{total_chunks}",
                    (
                        f"Avg similarity {average_similarity:.1f}% | "
                        f"best chunk {running_best_similarity:.1f}% | "
                        f"usable words {usable_words}/{len(chunk_words)}"
                    ),
                )

            for entry in best_chunk_result.get("word_scores", []):
                if not isinstance(entry, dict):
                    continue
                absolute_index = int(entry.get("index", -1))
                if absolute_index < 0:
                    continue
                current_best = best_word_scores.get(absolute_index)
                if current_best is None or float(entry.get("similarity", 0.0)) >= float(current_best.get("similarity", 0.0)):
                    best_word_scores[absolute_index] = dict(entry)

            for entry in best_chunk_result.get("letter_scores", []):
                if not isinstance(entry, dict):
                    continue
                key = (
                    int(entry.get("word_index", -1)),
                    str(entry.get("letter", "")),
                    round(float(entry.get("start", 0.0)), 3),
                )
                current_best = best_letter_scores.get(key)
                if current_best is None or float(entry.get("similarity", 0.0)) >= float(current_best.get("similarity", 0.0)):
                    best_letter_scores[key] = dict(entry)

            if end_index >= total_words:
                break
            if chunk_number % 2 == 0:
                time.sleep(0)

        merged_word_scores: List[Dict[str, object]] = []
        for absolute_index, word in enumerate(words):
            merged_word_scores.append(
                dict(
                    best_word_scores.get(
                        absolute_index,
                        {
                            "index": absolute_index,
                            "word": word,
                            "start": 0.0,
                            "end": 0.0,
                            "confidence": -7.0,
                            "similarity": 0.0,
                        },
                    )
                )
            )

        merged_letter_scores = [dict(entry) for entry in best_letter_scores.values()]
        if progress_callback is not None:
            final_similarity = float(
                scorer.build_analysis_result(merged_word_scores, merged_letter_scores).get(
                    "similarity_score", 0.0
                )
            )
            usable_total = sum(
                1
                for entry in merged_word_scores
                if float(entry.get("similarity", 0.0)) >= 24.0
            )
            progress_callback(
                1.0,
                f"Finished segmenting {clip_name or 'audio'}.",
                (
                    f"Chunked similarity {final_similarity:.1f}% | "
                    f"usable aligned words {usable_total}/{len(merged_word_scores)} | "
                    f"chunks {len(chunk_reports)}"
                ),
            )
        return {
            "analysis": scorer.build_analysis_result(merged_word_scores, merged_letter_scores),
            "chunk_count": len(chunk_reports),
            "chunk_reports": chunk_reports,
        }

    def train_guided_regenerator(
        self,
        *,
        dataset_dir: Path,
        output_dir: Path,
        total_epochs: int,
        save_every_epoch: int,
        batch_size: int,
        update_status: Callable[[str, str, str, int], None],
        cancel_event: Optional[threading.Event] = None,
    ) -> Dict[str, object]:
        return self.guided_svs.train_guided_regenerator(
            dataset_dir=dataset_dir,
            output_dir=output_dir,
            total_epochs=total_epochs,
            save_every_epoch=save_every_epoch,
            batch_size=batch_size,
            update_status=update_status,
            cancel_event=cancel_event,
        )

    def finalize_training_package(
        self,
        *,
        package_id: str,
        build_dir: Path,
        label: str,
        model_path: str = "",
        index_path: str = "",
        settings: Dict[str, object],
        prep_metadata: Dict[str, object],
        regeneration_metadata: Optional[Dict[str, object]] = None,
    ) -> Dict[str, object]:
        final_dir = self.root / package_id
        if final_dir.exists():
            shutil.rmtree(final_dir, ignore_errors=True)
        shutil.copytree(build_dir, final_dir)

        model_path_value = str(model_path or "").strip()
        index_path_value = str(index_path or "").strip()
        output_mode = str(settings.get("output_mode", "pipa-full") or "pipa-full")
        rvc_model_name = Path(model_path_value).name if model_path_value else ""
        regen = dict(regeneration_metadata or {})

        def map_build_path(path_value: str) -> str:
            cleaned = str(path_value or "").strip()
            if not cleaned:
                return ""
            candidate = Path(cleaned)
            try:
                relative = candidate.relative_to(build_dir)
                return str(final_dir / relative)
            except Exception:
                return str(candidate) if candidate.exists() else ""

        manifest = {
            "package_version": 1,
            "kind": "pipa",
            "id": package_id,
            "selection_name": f"pipa:{package_id}",
            "label": label,
            "created_at": _utc_now_iso(),
            "package_mode": output_mode,
            "rvc_model_name": rvc_model_name,
            "model_path": model_path_value,
            "index_path": index_path_value,
            "phoneme_profile_path": str(final_dir / "phoneme_profile.json"),
            "rebuild_profile_path": str(final_dir / "rebuild_profile.json"),
            "rebuild_clip_reports_path": str(final_dir / "rebuild_clip_reports.json"),
            "training_report_path": str(final_dir / "training_report.json"),
            "transcript_manifest_path": str(final_dir / "transcript_manifest.json"),
            "reference_bank_index_path": str(final_dir / "reference_bank_index.json"),
            "guided_regeneration_path": map_build_path(str(regen.get("checkpoint_path", "") or "")),
            "guided_regeneration_config_path": map_build_path(str(regen.get("config_path", "") or "")),
            "guided_regeneration_stats_path": map_build_path(str(prep_metadata.get("guided_svs_stats_path", "") or "")),
            "guided_regeneration_report_path": map_build_path(str(regen.get("report_path", "") or "")),
            "guided_vocoder_path": map_build_path(str(regen.get("vocoder_checkpoint_path", "") or "")),
            "guided_vocoder_config_path": map_build_path(str(regen.get("vocoder_config_path", "") or "")),
            "guided_vocoder_report_path": map_build_path(str(regen.get("vocoder_report_path", "") or "")),
            "guided_vocoder_history_path": map_build_path(str(regen.get("vocoder_history_path", "") or "")),
            "guided_regeneration_preview_path": map_build_path(str(regen.get("preview_path", "") or "")),
            "guided_regeneration_target_preview_path": map_build_path(str(regen.get("target_preview_path", "") or "")),
            "pronunciation_strategy": {
                "unit_mode": str(prep_metadata.get("phoneme_mode", "approx-pronunciation-units")),
                "alignment_tolerance": str(prep_metadata.get("alignment_tolerance", "balanced")),
                "scoring_model": "wav2vec2_asr_base_960h_letter_alignment",
                "transcript_fallback": "forgiving_skip_not_fail",
            },
            "rebuild_strategy": {
                "guide_conditioning": True,
                "uses_pitch_energy": True,
                "uses_phrase_templates": True,
                "plan_version": "guide-conditioned-resynthesis-plan-v1",
            },
            "guided_regeneration_strategy": {
                "enabled": bool(str(regen.get("checkpoint_path", "") or "")),
                "model_type": "guide-conditioned-persona-regenerator-v3",
                "conditioning": ["guide_mel", "pronunciation_units", "pitch", "energy", "voiced_unvoiced", "target_voice_prototype"],
                "vocoder": str(regen.get("render_mode", "griffinlim_preview_only") or "griffinlim_preview_only"),
                "best_val_l1": float(regen.get("best_val_l1", 0.0)),
                "best_val_total": float(regen.get("best_val_total", 0.0)),
                "best_epoch": int(regen.get("best_epoch", 0)),
                "best_phone_accuracy": float(regen.get("best_phone_accuracy", 0.0)),
                "best_lyric_phone_accuracy": float(regen.get("best_lyric_phone_accuracy", 0.0)),
                "best_vuv_accuracy": float(regen.get("best_vuv_accuracy", 0.0)),
                "vocoder_best_val_total": float(regen.get("vocoder_best_val_total", 0.0)),
                "vocoder_best_epoch": int(regen.get("vocoder_best_epoch", 0)),
                "vocoder_quality_summary": str(regen.get("vocoder_quality_summary", "") or ""),
                "vocoder_hardware_summary": str(regen.get("vocoder_hardware_summary", "") or ""),
                "hardware_summary": str(regen.get("hardware_summary", "") or ""),
                "quality_summary": str(regen.get("quality_summary", "") or ""),
                "search_mode": str(regen.get("search_mode", "mc-dropout-target-voice-rerank") or "mc-dropout-target-voice-rerank"),
                "last_epoch": int(regen.get("last_epoch", 0)),
                "sample_count": int(regen.get("sample_count", 0)),
            },
            "backbone": {
                "sample_rate": str(settings.get("sample_rate", "")),
                "version": str(settings.get("version", "")),
                "f0_method": str(settings.get("f0_method", "")),
                "total_epochs": int(settings.get("total_epochs", 0)),
                "save_every_epoch": int(settings.get("save_every_epoch", 0)),
                "batch_size": int(settings.get("batch_size", 0)),
                "build_index": bool(settings.get("build_index", False)),
                "output_mode": output_mode,
            },
            "dataset": {
                "total_audio_files": int(prep_metadata.get("total_audio_files", 0)),
                "matched_audio_files": int(prep_metadata.get("matched_audio_files", 0)),
                "skipped_audio_files": int(prep_metadata.get("skipped_audio_files", 0)),
                "reference_word_count": int(prep_metadata.get("reference_word_count", 0)),
                "reference_phrase_count": int(prep_metadata.get("reference_phrase_count", 0)),
            },
        }
        manifest_path = final_dir / "manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        self._reference_index_cache.pop(str(final_dir / "reference_bank_index.json"), None)
        return {
            "selection_name": str(manifest["selection_name"]),
            "manifest_path": str(manifest_path),
            "phoneme_profile_path": str(final_dir / "phoneme_profile.json"),
            "rebuild_profile_path": str(final_dir / "rebuild_profile.json"),
            "rebuild_clip_reports_path": str(final_dir / "rebuild_clip_reports.json"),
            "training_report_path": str(final_dir / "training_report.json"),
            "reference_bank_index_path": str(final_dir / "reference_bank_index.json"),
            "guided_regeneration_path": str(manifest.get("guided_regeneration_path", "") or ""),
            "guided_regeneration_config_path": str(manifest.get("guided_regeneration_config_path", "") or ""),
            "guided_regeneration_stats_path": str(manifest.get("guided_regeneration_stats_path", "") or ""),
            "guided_regeneration_report_path": str(manifest.get("guided_regeneration_report_path", "") or ""),
            "guided_vocoder_path": str(manifest.get("guided_vocoder_path", "") or ""),
            "guided_vocoder_config_path": str(manifest.get("guided_vocoder_config_path", "") or ""),
            "guided_vocoder_report_path": str(manifest.get("guided_vocoder_report_path", "") or ""),
            "guided_regeneration_preview_path": str(manifest.get("guided_regeneration_preview_path", "") or ""),
            "guided_regeneration_target_preview_path": str(manifest.get("guided_regeneration_target_preview_path", "") or ""),
        }

    def _match_transcripts(
        self,
        *,
        audio_paths: List[Path],
        transcript_paths: List[Path],
    ) -> Tuple[Dict[str, str], Dict[str, object]]:
        transcript_records: Dict[str, str] = {}
        transcript_sources: Dict[str, str] = {}
        unparsed_transcript_files: List[str] = []

        for transcript_path in transcript_paths:
            suffix = transcript_path.suffix.lower()
            parser = {
                ".txt": self._parse_text_transcript,
                ".json": self._parse_json_transcript,
                ".jsonl": self._parse_jsonl_transcript,
                ".csv": lambda path: self._parse_delimited_transcript(path, delimiter=","),
                ".tsv": lambda path: self._parse_delimited_transcript(path, delimiter="\t"),
            }.get(suffix)
            if parser is None:
                unparsed_transcript_files.append(transcript_path.name)
                continue
            parsed = parser(transcript_path)
            if not parsed:
                continue
            for key, value in parsed.items():
                clean_value = normalize_lyrics(value)
                if not clean_value:
                    continue
                transcript_records[key] = clean_value
                transcript_sources[key] = transcript_path.name

        matched: Dict[str, str] = {}
        unmatched_audio_files: List[str] = []
        for audio_path in audio_paths:
            exact_keys = [
                normalize_match_key(audio_path.name),
                normalize_match_key(audio_path.stem),
            ]
            selected_text = ""
            for key in exact_keys:
                if key in transcript_records:
                    selected_text = transcript_records[key]
                    break
            if not selected_text and len(audio_paths) == 1 and len(transcript_records) == 1:
                selected_text = next(iter(transcript_records.values()))
            if selected_text:
                matched[str(audio_path.resolve())] = selected_text
            else:
                unmatched_audio_files.append(audio_path.name)

        return matched, {
            "total_audio_files": len(audio_paths),
            "matched_audio_files": len(matched),
            "skipped_audio_files": len(unmatched_audio_files),
            "unmatched_audio_files": unmatched_audio_files,
            "transcript_files": [path.name for path in transcript_paths],
            "unparsed_transcript_files": unparsed_transcript_files,
            "matched_keys": [
                {
                    "audio_file": audio_path.name,
                    "transcript_source": transcript_sources.get(
                        normalize_match_key(audio_path.name),
                        transcript_sources.get(normalize_match_key(audio_path.stem), ""),
                    ),
                }
                for audio_path in audio_paths
                if str(audio_path.resolve()) in matched
            ],
        }

    def _parse_text_transcript(self, transcript_path: Path) -> Dict[str, str]:
        text = transcript_path.read_text(encoding="utf-8", errors="ignore").strip()
        if not text:
            return {}
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        if len(lines) > 1 and all(("|" in line or "\t" in line) for line in lines):
            records: Dict[str, str] = {}
            for line in lines:
                delimiter = "|" if "|" in line else "\t"
                key, value = line.split(delimiter, 1)
                records[normalize_match_key(key)] = value.strip()
            return records
        return {normalize_match_key(transcript_path.stem): text}

    def _parse_json_transcript(self, transcript_path: Path) -> Dict[str, str]:
        loaded = json.loads(transcript_path.read_text(encoding="utf-8"))
        return self._normalize_transcript_payload(loaded)

    def _parse_jsonl_transcript(self, transcript_path: Path) -> Dict[str, str]:
        records: Dict[str, str] = {}
        for raw_line in transcript_path.read_text(encoding="utf-8", errors="ignore").splitlines():
            line = raw_line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            records.update(self._normalize_transcript_payload(payload))
        return records

    def _parse_delimited_transcript(self, transcript_path: Path, *, delimiter: str) -> Dict[str, str]:
        content = transcript_path.read_text(encoding="utf-8", errors="ignore")
        reader = csv.reader(content.splitlines(), delimiter=delimiter)
        rows = [row for row in reader if row]
        if not rows:
            return {}
        lower_headers = [cell.strip().lower() for cell in rows[0]]
        header_like = {"filename", "file", "name", "clip", "audio"} & set(lower_headers)
        if header_like:
            dict_reader = csv.DictReader(content.splitlines(), delimiter=delimiter)
            records: Dict[str, str] = {}
            for row in dict_reader:
                if not row:
                    continue
                key = (
                    row.get("filename")
                    or row.get("file")
                    or row.get("name")
                    or row.get("clip")
                    or row.get("audio")
                    or ""
                )
                value = (
                    row.get("transcript")
                    or row.get("lyrics")
                    or row.get("text")
                    or row.get("caption")
                    or ""
                )
                if key and value:
                    records[normalize_match_key(key)] = value.strip()
            return records

        records = {}
        for row in rows:
            if len(row) < 2:
                continue
            records[normalize_match_key(row[0])] = " ".join(
                cell.strip() for cell in row[1:] if cell.strip()
            )
        return records

    def _normalize_transcript_payload(self, payload: object) -> Dict[str, str]:
        if isinstance(payload, dict):
            if any(key in payload for key in ("clips", "items", "entries")):
                items = payload.get("clips") or payload.get("items") or payload.get("entries") or []
                return self._normalize_transcript_payload(items)
            if all(isinstance(value, str) for value in payload.values()):
                return {
                    normalize_match_key(key): str(value).strip()
                    for key, value in payload.items()
                    if str(value).strip()
                }
            key = (
                payload.get("filename")
                or payload.get("file")
                or payload.get("name")
                or payload.get("clip")
                or payload.get("audio")
            )
            value = (
                payload.get("transcript")
                or payload.get("lyrics")
                or payload.get("text")
                or payload.get("caption")
            )
            if key and value:
                return {normalize_match_key(str(key)): str(value).strip()}
        if isinstance(payload, list):
            records: Dict[str, str] = {}
            for item in payload:
                records.update(self._normalize_transcript_payload(item))
            return records
        return {}

    def _extract_word_candidates(
        self,
        *,
        audio_path: Path,
        audio: np.ndarray,
        sample_rate: int,
        word_scores: List[Dict[str, object]],
        threshold: float,
        performance_lookup: Optional[Dict[int, Dict[str, object]]] = None,
    ) -> List[Dict[str, object]]:
        candidates: List[Dict[str, object]] = []
        total_samples = int(audio.shape[0])
        for entry_index, entry in enumerate(word_scores, start=1):
            word = normalize_lyrics(str(entry.get("word", "")))
            similarity = float(entry.get("similarity", 0.0))
            if not word or similarity < float(threshold):
                continue
            absolute_index = int(entry.get("index", -1))
            start, end = self._window_to_samples(
                [entry],
                sample_rate=sample_rate,
                total_samples=total_samples,
                padding_ms=55.0,
            )
            if end <= start:
                continue
            duration_seconds = float(end - start) / float(sample_rate)
            if duration_seconds < 0.045 or duration_seconds > 1.35:
                continue
            candidates.append(
                {
                    "kind": "word",
                    "word": word,
                    "phrase": word,
                    "units": pronunciation_units(word),
                    "similarity": round(similarity, 3),
                    "source_path": str(audio_path),
                    "start_sample": int(start),
                    "end_sample": int(end),
                    "duration_seconds": round(duration_seconds, 4),
                    "performance": dict((performance_lookup or {}).get(absolute_index, {})),
                }
            )
            if entry_index % 256 == 0:
                time.sleep(0)
        return candidates

    def _extract_phrase_candidates(
        self,
        *,
        audio_path: Path,
        audio: np.ndarray,
        sample_rate: int,
        word_scores: List[Dict[str, object]],
        threshold: float,
        performance_lookup: Optional[Dict[int, Dict[str, object]]] = None,
    ) -> List[Dict[str, object]]:
        candidates: List[Dict[str, object]] = []
        total_samples = int(audio.shape[0])
        usable = [
            dict(entry)
            for entry in word_scores
            if normalize_lyrics(str(entry.get("word", "")))
        ]
        for window_size in (2, 3):
            for start_index in range(0, max(0, len(usable) - window_size + 1)):
                window = usable[start_index : start_index + window_size]
                phrase_words = [normalize_lyrics(str(entry.get("word", ""))) for entry in window]
                if any(not word for word in phrase_words):
                    continue
                window_scores = [float(entry.get("similarity", 0.0)) for entry in window]
                if min(window_scores) < float(threshold):
                    continue
                start, end = self._window_to_samples(
                    window,
                    sample_rate=sample_rate,
                    total_samples=total_samples,
                    padding_ms=85.0,
                )
                if end <= start:
                    continue
                duration_seconds = float(end - start) / float(sample_rate)
                if duration_seconds < 0.12 or duration_seconds > 2.6:
                    continue
                phrase = " ".join(phrase_words).strip()
                if not phrase:
                    continue
                window_indices = [int(entry.get("index", -1)) for entry in window]
                window_performance = [
                    dict((performance_lookup or {}).get(word_index, {}))
                    for word_index in window_indices
                    if int(word_index) >= 0 and dict((performance_lookup or {}).get(word_index, {}))
                ]
                candidates.append(
                    {
                        "kind": "phrase",
                        "phrase": phrase,
                        "word": phrase_words[0],
                        "words": phrase_words,
                        "units": [unit for word in phrase_words for unit in pronunciation_units(word)],
                        "similarity": round(float(np.mean(window_scores)), 3),
                        "source_path": str(audio_path),
                        "start_sample": int(start),
                        "end_sample": int(end),
                        "duration_seconds": round(duration_seconds, 4),
                        "performance": self._summarize_candidate_performance(window_performance),
                    }
                )
                if start_index % 256 == 0:
                    time.sleep(0)
        return candidates

    def _window_to_samples(
        self,
        word_scores: List[Dict[str, object]],
        *,
        sample_rate: int,
        total_samples: int,
        padding_ms: float,
    ) -> Tuple[int, int]:
        starts = [float(entry.get("start", 0.0)) for entry in word_scores]
        ends = [float(entry.get("end", 0.0)) for entry in word_scores]
        if not starts or not ends:
            return 0, 0
        pad = int((float(padding_ms) / 1000.0) * sample_rate)
        start = max(0, int(min(starts) * sample_rate) - pad)
        end = min(total_samples, int(max(ends) * sample_rate) + pad)
        return start, max(start + 1, end)

    def _select_best_candidates(
        self,
        candidates: List[Dict[str, object]],
        *,
        key_name: str,
        per_key_limit: int,
        overall_limit: int,
    ) -> List[Dict[str, object]]:
        grouped: Dict[str, List[Dict[str, object]]] = defaultdict(list)
        for candidate in candidates:
            key = str(candidate.get(key_name, "")).strip()
            if not key:
                continue
            grouped[key].append(candidate)

        selected: List[Dict[str, object]] = []
        for key, group in grouped.items():
            ordered = sorted(
                group,
                key=lambda entry: (
                    -float(entry.get("similarity", 0.0)),
                    float(entry.get("duration_seconds", 999.0)),
                    str(entry.get("source_path", "")),
                ),
            )
            selected.extend(ordered[: max(1, int(per_key_limit))])
        selected.sort(
            key=lambda entry: (
                -float(entry.get("similarity", 0.0)),
                float(entry.get("duration_seconds", 999.0)),
                str(entry.get(key_name, "")),
            )
        )
        return selected[: max(1, int(overall_limit))]

    def _materialize_reference_candidates(
        self,
        candidates: List[Dict[str, object]],
        *,
        target_dir: Path,
        text_key: str,
        kind: str,
        progress_callback: Optional[Callable[[int, int, str], None]] = None,
        cancel_event: Optional[threading.Event] = None,
    ) -> List[Dict[str, object]]:
        target_dir.mkdir(parents=True, exist_ok=True)
        entries: List[Dict[str, object]] = []
        audio_cache: Dict[str, Tuple[np.ndarray, int]] = {}
        total_candidates = max(1, len(candidates))
        for index, candidate in enumerate(candidates, start=1):
            if cancel_event is not None and cancel_event.is_set():
                raise InterruptedError("Training stopped by user.")
            source_path = Path(str(candidate.get("source_path", "")))
            if not source_path.exists():
                continue
            cache_key = str(source_path.resolve())
            cached = audio_cache.get(cache_key)
            if cached is None:
                cached = self._load_audio(source_path, sample_rate=44100)
                audio_cache[cache_key] = cached
            source_audio, sample_rate = cached
            start = max(0, int(candidate.get("start_sample", 0)))
            end = min(int(source_audio.shape[0]), int(candidate.get("end_sample", 0)))
            if end <= start:
                continue
            snippet = np.asarray(source_audio[start:end], dtype=np.float32)
            if snippet.size == 0:
                continue
            label = slugify_name(str(candidate.get(text_key, f"{kind}_{index}")))
            target_path = target_dir / f"{label}_{index:04d}.wav"
            sf.write(target_path, snippet, sample_rate, subtype="PCM_24")
            performance = dict(candidate.get("performance", {}))
            if not performance:
                performance = self.rebuild_builder.summarize_segment(
                    snippet,
                    sample_rate,
                )
            entries.append(
                {
                    "kind": kind,
                    text_key: str(candidate.get(text_key, "")),
                    "phrase": str(candidate.get("phrase", "")),
                    "word": str(candidate.get("word", "")),
                    "words": list(candidate.get("words", [])),
                    "units": list(candidate.get("units", [])),
                    "similarity": float(candidate.get("similarity", 0.0)),
                    "duration_seconds": float(candidate.get("duration_seconds", 0.0)),
                    "source_file": source_path.name,
                    "performance": performance,
                    "relative_path": target_path.relative_to(target_dir.parent).as_posix(),
                }
            )
            if progress_callback is not None and (
                index == 1 or index == total_candidates or index % 12 == 0
            ):
                progress_callback(index, total_candidates, kind)
            if index % 24 == 0:
                time.sleep(0)
        return entries

    def _summarize_candidate_performance(
        self,
        word_performance: List[Dict[str, object]],
    ) -> Dict[str, float]:
        if not word_performance:
            return {}
        pitch_values = [
            float(item.get("pitch_median_hz", 0.0))
            for item in word_performance
            if float(item.get("pitch_median_hz", 0.0)) > 0.0
        ]
        energy_values = [
            float(item.get("energy_mean", 0.0))
            for item in word_performance
            if float(item.get("energy_mean", 0.0)) > 0.0
        ]
        onset_values = [
            float(item.get("onset_mean", 0.0))
            for item in word_performance
            if float(item.get("onset_mean", 0.0)) > 0.0
        ]
        duration_values = [
            float(item.get("duration_seconds", 0.0))
            for item in word_performance
            if float(item.get("duration_seconds", 0.0)) > 0.0
        ]
        voiced_values = [
            float(item.get("voiced_ratio", 0.0))
            for item in word_performance
            if float(item.get("voiced_ratio", 0.0)) >= 0.0
        ]
        return {
            "duration_seconds": round(float(np.sum(duration_values)) if duration_values else 0.0, 4),
            "pitch_median_hz": round(float(np.median(pitch_values)) if pitch_values else 0.0, 3),
            "pitch_mean_hz": round(float(np.mean(pitch_values)) if pitch_values else 0.0, 3),
            "energy_mean": round(float(np.mean(energy_values)) if energy_values else 0.0, 6),
            "energy_peak": round(float(np.max(energy_values)) if energy_values else 0.0, 6),
            "onset_mean": round(float(np.mean(onset_values)) if onset_values else 0.0, 6),
            "onset_peak": round(float(np.max(onset_values)) if onset_values else 0.0, 6),
            "voiced_ratio": round(float(np.mean(voiced_values)) if voiced_values else 0.0, 4),
        }

    def _build_phoneme_profile(
        self,
        *,
        package_id: str,
        clip_reports: List[Dict[str, object]],
        word_entries: List[Dict[str, object]],
        phrase_entries: List[Dict[str, object]],
        alignment_tolerance: str,
    ) -> Dict[str, object]:
        unit_counter: Counter[str] = Counter()
        word_counter: Counter[str] = Counter()
        similarity_values: List[float] = []
        for entry in word_entries:
            word = normalize_lyrics(str(entry.get("word", "")))
            if word:
                word_counter[word] += 1
            for unit in entry.get("units", []):
                unit_counter[str(unit)] += 1
            similarity_values.append(float(entry.get("similarity", 0.0)))

        clip_similarity_values = [
            float(report.get("word_similarity", 0.0))
            for report in clip_reports
            if bool(report.get("used_for_profile"))
        ]
        return {
            "package_id": package_id,
            "created_at": _utc_now_iso(),
            "alignment_tolerance": alignment_tolerance,
            "unit_mode": "approx-pronunciation-units",
            "average_reference_similarity": round(
                float(np.mean(similarity_values)) if similarity_values else 0.0,
                2,
            ),
            "average_clip_similarity": round(
                float(np.mean(clip_similarity_values)) if clip_similarity_values else 0.0,
                2,
            ),
            "reference_word_count": len(word_entries),
            "reference_phrase_count": len(phrase_entries),
            "top_units": [
                {"unit": unit, "count": int(count)}
                for unit, count in unit_counter.most_common(96)
            ],
            "top_words": [
                {"word": word, "count": int(count)}
                for word, count in word_counter.most_common(96)
            ],
        }
