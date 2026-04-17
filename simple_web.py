from __future__ import annotations

import argparse
import base64
import json
import mimetypes
import os
import re
import shutil
import subprocess
import threading
import time
import traceback
import webbrowser
from datetime import datetime
from dataclasses import dataclass, field
from itertools import count
from pathlib import Path
from typing import Dict, Iterator, List, Optional
from urllib import error as urlerror
from urllib import parse as urlparse
from urllib import request as urlrequest
from uuid import uuid4

import uvicorn
import soundfile as sf
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles

from simple_backend import (
    SimpleRVCBackend,
    create_zip,
    reset_directory,
    sanitize_filename,
)
from simple_detag import SimpleDetagger
from simple_master_conversion import MasterConversionEngine
from simple_mastering import SimpleMasteringEngine
from simple_optimize import VoiceSuitabilityOptimizer
from simple_rebuild import RebuildFeatureBuilder
from simple_touchup import NeuralClarityRepairEngine
from simple_training import SimpleTrainer


REPO_ROOT = Path(__file__).resolve().parent
STATIC_ROOT = REPO_ROOT / "simple_webui"
JOBS_ROOT = REPO_ROOT / "audio-outputs" / "simple-web" / "jobs"
DETAG_ROOT = REPO_ROOT / "audio-outputs" / "simple-web" / "detag"
ISOLATOR_ROOT = REPO_ROOT / "audio-outputs" / "simple-web" / "isolator"
PREVIEW_ROOT = REPO_ROOT / "audio-outputs" / "simple-web" / "previews"
TRAINING_ROOT = REPO_ROOT / "training-runs"
MASTERING_ROOT = REPO_ROOT / "audio-outputs" / "simple-web" / "mastering"
TOUCHUP_ROOT = REPO_ROOT / "audio-outputs" / "simple-web" / "touchup"
OPTIMIZE_ROOT = REPO_ROOT / "audio-outputs" / "simple-web" / "optimize"
API_COMPOSE_ROOT = REPO_ROOT / "audio-outputs" / "simple-web" / "api-compose"
ALBUMS_ROOT = REPO_ROOT / "audio-outputs" / "simple-web" / "albums"
GENERATE_ROOT = REPO_ROOT / "audio-outputs" / "simple-web" / "generate"
MASTER_CONVERSION_ROOT = REPO_ROOT / "audio-outputs" / "simple-web" / "master-conversion"
REBUILD_ROOT = REPO_ROOT / "audio-outputs" / "simple-web" / "rebuild"

# The original RVC codebase relies on many repo-relative paths like "rmvpe.pt".
# Force the working directory to the repo root so those legacy paths resolve
# correctly even when this server is launched from somewhere else.
os.chdir(REPO_ROOT)

QUALITY_PRESETS = {
    "fast": {
        "label": "Fast",
        "description": "Quickest turnaround with lighter pitch analysis.",
        "f0_method": "pm",
        "index_rate": 0.05,
        "filter_radius": 3,
        "rms_mix_rate": 0.35,
        "protect": 0.40,
        "crepe_hop_length": 120,
    },
    "balanced": {
        "label": "Balanced",
        "description": "Good default for most clips.",
        "f0_method": "rmvpe",
        "index_rate": 0.10,
        "filter_radius": 3,
        "rms_mix_rate": 0.25,
        "protect": 0.33,
        "crepe_hop_length": 120,
    },
    "clean": {
        "label": "High Quality",
        "description": "Leans toward clarity and smoother pitch tracking.",
        "f0_method": "rmvpe",
        "index_rate": 0.15,
        "filter_radius": 5,
        "rms_mix_rate": 0.20,
        "protect": 0.35,
        "crepe_hop_length": 120,
    },
}

GENERATE_REPAIR_PROFILES = {
    "fast": {
        "strength": 46,
        "variants_per_batch": 2,
        "max_batches": 1,
        "max_target_words": 4,
    },
    "balanced": {
        "strength": 64,
        "variants_per_batch": 3,
        "max_batches": 2,
        "max_target_words": 6,
    },
    "clean": {
        "strength": 78,
        "variants_per_batch": 4,
        "max_batches": 3,
        "max_target_words": 8,
    },
}

GENERATE_NOTE_ORDER = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
GENERATE_NOTE_MAP = {
    "C": "C",
    "B#": "C",
    "C#": "C#",
    "DB": "C#",
    "D": "D",
    "D#": "D#",
    "EB": "D#",
    "E": "E",
    "FB": "E",
    "F": "F",
    "E#": "F",
    "F#": "F#",
    "GB": "F#",
    "G": "G",
    "G#": "G#",
    "AB": "G#",
    "A": "A",
    "A#": "A#",
    "BB": "A#",
    "B": "B",
    "CB": "B",
}


@dataclass
class JobResult:
    name: str
    url: str
    download_name: str
    sample_rate: int
    timings: Dict[str, float]
    index_path: str
    preprocess_applied: bool
    preprocess_mode: str
    output_mode: str = "single"
    secondary_model_name: str = ""
    blend_percentage: int = 100


@dataclass
class JobState:
    id: str
    status: str = "queued"
    message: str = "Waiting to start..."
    total_files: int = 0
    completed_files: int = 0
    current_file: str = ""
    created_at: float = field(default_factory=time.time)
    results: List[JobResult] = field(default_factory=list)
    zip_url: str = ""
    error: str = ""

    def to_dict(self) -> Dict[str, object]:
        return {
            "id": self.id,
            "status": self.status,
            "message": self.message,
            "total_files": self.total_files,
            "completed_files": self.completed_files,
            "current_file": self.current_file,
            "created_at": self.created_at,
            "zip_url": self.zip_url,
            "error": self.error,
            "results": [
                {
                    "name": result.name,
                    "url": result.url,
                    "download_name": result.download_name,
                    "sample_rate": result.sample_rate,
                    "timings": result.timings,
                    "index_path": result.index_path,
                    "preprocess_applied": result.preprocess_applied,
                    "preprocess_mode": result.preprocess_mode,
                    "output_mode": result.output_mode,
                    "secondary_model_name": result.secondary_model_name,
                    "blend_percentage": result.blend_percentage,
                }
                for result in self.results
            ],
        }


@dataclass
class MasterConversionJobState:
    id: str
    model_name: str
    source_name: str
    status: str = "queued"
    stage: str = "queued"
    message: str = "Waiting to start..."
    progress: int = 0
    created_at: float = field(default_factory=time.time)
    error: str = ""
    lyrics_preview: str = ""
    quality_preset: str = "balanced"
    master_profile: str = "studio"
    preferred_pipeline: str = ""
    output_mode: str = "single"
    secondary_model_name: str = ""
    blend_percentage: int = 50
    sample_rate: int = 0
    best_similarity_score: float = 0.0
    best_word_report: str = ""
    best_letter_report: str = ""
    repair_attempts: int = 0
    repaired_word_count: int = 0
    candidate_reports: List[Dict[str, object]] = field(default_factory=list)
    phrase_choices: List[Dict[str, object]] = field(default_factory=list)
    final_url: str = ""
    final_download_name: str = ""
    reconstructed_lead_url: str = ""
    reconstructed_lead_download_name: str = ""
    reconstructed_removed_url: str = ""
    reconstructed_removed_download_name: str = ""
    raw_conversion_url: str = ""
    raw_conversion_download_name: str = ""
    repaired_url: str = ""
    repaired_download_name: str = ""
    final_removed_url: str = ""
    final_removed_download_name: str = ""
    metadata_url: str = ""
    metadata_download_name: str = ""
    zip_url: str = ""
    timings: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, object]:
        return {
            "id": self.id,
            "model_name": self.model_name,
            "source_name": self.source_name,
            "status": self.status,
            "stage": self.stage,
            "message": self.message,
            "progress": self.progress,
            "created_at": self.created_at,
            "error": self.error,
            "lyrics_preview": self.lyrics_preview,
            "quality_preset": self.quality_preset,
            "master_profile": self.master_profile,
            "preferred_pipeline": self.preferred_pipeline,
            "output_mode": self.output_mode,
            "secondary_model_name": self.secondary_model_name,
            "blend_percentage": self.blend_percentage,
            "sample_rate": self.sample_rate,
            "best_similarity_score": self.best_similarity_score,
            "best_word_report": self.best_word_report,
            "best_letter_report": self.best_letter_report,
            "repair_attempts": self.repair_attempts,
            "repaired_word_count": self.repaired_word_count,
            "candidate_reports": self.candidate_reports,
            "phrase_choices": self.phrase_choices,
            "final_url": self.final_url,
            "final_download_name": self.final_download_name,
            "reconstructed_lead_url": self.reconstructed_lead_url,
            "reconstructed_lead_download_name": self.reconstructed_lead_download_name,
            "reconstructed_removed_url": self.reconstructed_removed_url,
            "reconstructed_removed_download_name": self.reconstructed_removed_download_name,
            "raw_conversion_url": self.raw_conversion_url,
            "raw_conversion_download_name": self.raw_conversion_download_name,
            "repaired_url": self.repaired_url,
            "repaired_download_name": self.repaired_download_name,
            "final_removed_url": self.final_removed_url,
            "final_removed_download_name": self.final_removed_download_name,
            "metadata_url": self.metadata_url,
            "metadata_download_name": self.metadata_download_name,
            "zip_url": self.zip_url,
            "timings": self.timings,
        }


@dataclass
class GenerateJobState:
    id: str
    model_name: str
    guide_name: str
    status: str = "queued"
    message: str = "Waiting to start..."
    progress: int = 0
    created_at: float = field(default_factory=time.time)
    error: str = ""
    lyrics_preview: str = ""
    guide_key: str = ""
    target_key: str = ""
    guide_bpm: float = 0.0
    target_bpm: float = 0.0
    transpose: int = 0
    quality_preset: str = "balanced"
    preprocess_mode: str = "off"
    repair_mode: str = "pronunciation-repair"
    repair_strength: int = 0
    repair_attempts: int = 0
    repaired_word_count: int = 0
    best_similarity_score: float = 0.0
    best_word_report: str = ""
    best_letter_report: str = ""
    detected_word_indices: List[int] = field(default_factory=list)
    regeneration_available: bool = False
    regeneration_reason: str = ""
    sample_rate: int = 0
    tempo_ratio: float = 1.0
    tempo_adjusted: bool = False
    result_url: str = ""
    download_name: str = ""
    repair_source_url: str = ""
    repair_source_download_name: str = ""
    metadata_url: str = ""
    metadata_download_name: str = ""
    timings: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, object]:
        return {
            "id": self.id,
            "model_name": self.model_name,
            "guide_name": self.guide_name,
            "status": self.status,
            "message": self.message,
            "progress": self.progress,
            "created_at": self.created_at,
            "error": self.error,
            "lyrics_preview": self.lyrics_preview,
            "guide_key": self.guide_key,
            "target_key": self.target_key,
            "guide_bpm": self.guide_bpm,
            "target_bpm": self.target_bpm,
            "transpose": self.transpose,
            "quality_preset": self.quality_preset,
            "preprocess_mode": self.preprocess_mode,
            "repair_mode": self.repair_mode,
            "repair_strength": self.repair_strength,
            "repair_attempts": self.repair_attempts,
            "repaired_word_count": self.repaired_word_count,
            "best_similarity_score": self.best_similarity_score,
            "best_word_report": self.best_word_report,
            "best_letter_report": self.best_letter_report,
            "detected_word_indices": self.detected_word_indices,
            "regeneration_available": self.regeneration_available,
            "regeneration_reason": self.regeneration_reason,
            "sample_rate": self.sample_rate,
            "tempo_ratio": self.tempo_ratio,
            "tempo_adjusted": self.tempo_adjusted,
            "result_url": self.result_url,
            "download_name": self.download_name,
            "repair_source_url": self.repair_source_url,
            "repair_source_download_name": self.repair_source_download_name,
            "metadata_url": self.metadata_url,
            "metadata_download_name": self.metadata_download_name,
            "timings": self.timings,
        }


@dataclass
class TrainingJobState:
    id: str
    experiment_name: str
    build_index: bool = False
    output_mode: str = "persona-v1"
    epoch_mode: str = "manual-stop"
    status: str = "queued"
    stage: str = "queued"
    message: str = "Waiting to start..."
    progress: int = 0
    created_at: float = field(default_factory=time.time)
    log_tail: str = ""
    log_history: List[str] = field(default_factory=list)
    error: str = ""
    stop_requested: bool = False
    stopped_early: bool = False
    model_path: str = ""
    index_path: str = ""
    pipa_selection_name: str = ""
    pipa_manifest_path: str = ""
    phoneme_profile_path: str = ""
    rebuild_profile_path: str = ""
    rebuild_clip_reports_path: str = ""
    training_report_path: str = ""
    transcript_manifest_path: str = ""
    reference_bank_index_path: str = ""
    guided_regeneration_path: str = ""
    guided_regeneration_config_path: str = ""
    guided_regeneration_report_path: str = ""
    guided_regeneration_preview_path: str = ""
    guided_regeneration_target_preview_path: str = ""
    guided_regeneration_best_val_l1: float = 0.0
    guided_regeneration_best_val_total: float = 0.0
    guided_regeneration_best_epoch: int = 0
    guided_regeneration_best_phone_accuracy: float = 0.0
    guided_regeneration_best_lyric_phone_accuracy: float = 0.0
    guided_regeneration_best_vuv_accuracy: float = 0.0
    guided_regeneration_plateau_epochs: int = 0
    guided_regeneration_hardware_summary: str = ""
    guided_regeneration_quality_summary: str = ""
    guided_regeneration_last_epoch: int = 0
    guided_regeneration_sample_count: int = 0
    alignment_tolerance: str = "forgiving"
    phoneme_mode: str = ""
    matched_audio_files: int = 0
    total_audio_files: int = 0
    skipped_audio_files: int = 0
    reference_word_count: int = 0
    reference_phrase_count: int = 0
    training_plan_path: str = ""
    base_voice_clip_count: int = 0
    paired_song_count: int = 0
    depersonafied_variant_count: int = 0

    def to_dict(self) -> Dict[str, object]:
        return {
            "id": self.id,
            "experiment_name": self.experiment_name,
            "build_index": self.build_index,
            "output_mode": self.output_mode,
            "epoch_mode": self.epoch_mode,
            "status": self.status,
            "stage": self.stage,
            "message": self.message,
            "progress": self.progress,
            "created_at": self.created_at,
            "log_tail": self.log_tail,
            "log_history": list(self.log_history),
            "error": self.error,
            "stop_requested": self.stop_requested,
            "stopped_early": self.stopped_early,
            "model_path": self.model_path,
            "index_path": self.index_path,
            "pipa_selection_name": self.pipa_selection_name,
            "pipa_manifest_path": self.pipa_manifest_path,
            "phoneme_profile_path": self.phoneme_profile_path,
            "rebuild_profile_path": self.rebuild_profile_path,
            "rebuild_clip_reports_path": self.rebuild_clip_reports_path,
            "training_report_path": self.training_report_path,
            "transcript_manifest_path": self.transcript_manifest_path,
            "reference_bank_index_path": self.reference_bank_index_path,
            "guided_regeneration_path": self.guided_regeneration_path,
            "guided_regeneration_config_path": self.guided_regeneration_config_path,
            "guided_regeneration_report_path": self.guided_regeneration_report_path,
            "guided_regeneration_preview_path": self.guided_regeneration_preview_path,
            "guided_regeneration_target_preview_path": self.guided_regeneration_target_preview_path,
            "guided_regeneration_best_val_l1": self.guided_regeneration_best_val_l1,
            "guided_regeneration_best_val_total": self.guided_regeneration_best_val_total,
            "guided_regeneration_best_epoch": self.guided_regeneration_best_epoch,
            "guided_regeneration_best_phone_accuracy": self.guided_regeneration_best_phone_accuracy,
            "guided_regeneration_best_lyric_phone_accuracy": self.guided_regeneration_best_lyric_phone_accuracy,
            "guided_regeneration_best_vuv_accuracy": self.guided_regeneration_best_vuv_accuracy,
            "guided_regeneration_plateau_epochs": self.guided_regeneration_plateau_epochs,
            "guided_regeneration_hardware_summary": self.guided_regeneration_hardware_summary,
            "guided_regeneration_quality_summary": self.guided_regeneration_quality_summary,
            "guided_regeneration_last_epoch": self.guided_regeneration_last_epoch,
            "guided_regeneration_sample_count": self.guided_regeneration_sample_count,
            "alignment_tolerance": self.alignment_tolerance,
            "phoneme_mode": self.phoneme_mode,
            "matched_audio_files": self.matched_audio_files,
            "total_audio_files": self.total_audio_files,
            "skipped_audio_files": self.skipped_audio_files,
            "reference_word_count": self.reference_word_count,
            "reference_phrase_count": self.reference_phrase_count,
            "training_plan_path": self.training_plan_path,
            "base_voice_clip_count": self.base_voice_clip_count,
            "paired_song_count": self.paired_song_count,
            "depersonafied_variant_count": self.depersonafied_variant_count,
        }


@dataclass
class DetagJobState:
    id: str
    voice_id: str
    status: str = "queued"
    message: str = "Waiting to start..."
    progress: int = 0
    created_at: float = field(default_factory=time.time)
    error: str = ""
    result_url: str = ""
    download_name: str = ""
    threshold: float = 0.0
    kept_ratio: float = 0.0

    def to_dict(self) -> Dict[str, object]:
        return {
            "id": self.id,
            "voice_id": self.voice_id,
            "status": self.status,
            "message": self.message,
            "progress": self.progress,
            "created_at": self.created_at,
            "error": self.error,
            "result_url": self.result_url,
            "download_name": self.download_name,
            "threshold": self.threshold,
            "kept_ratio": self.kept_ratio,
        }


@dataclass
class IsolatorJobState:
    id: str
    status: str = "queued"
    message: str = "Waiting to start..."
    progress: int = 0
    created_at: float = field(default_factory=time.time)
    error: str = ""
    mode: str = "main-vocal"
    input_type: str = "full-mix"
    strength: int = 10
    deecho: bool = True
    width_focus: bool = True
    clarity_preserve: int = 70
    sample_rate: int = 0
    main_vocal_url: str = ""
    main_vocal_download_name: str = ""
    backing_vocal_url: str = ""
    backing_vocal_download_name: str = ""
    source_files: List[str] = field(default_factory=list)
    current_file: str = ""

    def to_dict(self) -> Dict[str, object]:
        return {
            "id": self.id,
            "status": self.status,
            "message": self.message,
            "progress": self.progress,
            "created_at": self.created_at,
            "error": self.error,
            "mode": self.mode,
            "input_type": self.input_type,
            "strength": self.strength,
            "deecho": self.deecho,
            "width_focus": self.width_focus,
            "clarity_preserve": self.clarity_preserve,
            "sample_rate": self.sample_rate,
            "main_vocal_url": self.main_vocal_url,
            "main_vocal_download_name": self.main_vocal_download_name,
            "backing_vocal_url": self.backing_vocal_url,
            "backing_vocal_download_name": self.backing_vocal_download_name,
            "source_files": self.source_files,
            "current_file": self.current_file,
        }


@dataclass
class MasteringJobState:
    id: str
    status: str = "queued"
    message: str = "Waiting to start..."
    progress: int = 0
    created_at: float = field(default_factory=time.time)
    error: str = ""
    resolution: int = 48
    sample_rate: int = 0
    mastered_url: str = ""
    mastered_download_name: str = ""
    profile_url: str = ""
    profile_download_name: str = ""
    curve_points: List[Dict[str, float]] = field(default_factory=list)
    reference_count: int = 0
    reference_files: List[str] = field(default_factory=list)
    source_rms_db: float = 0.0
    reference_rms_db: float = 0.0
    loudness_gain_db: float = 0.0
    band_summary: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, object]:
        return {
            "id": self.id,
            "status": self.status,
            "message": self.message,
            "progress": self.progress,
            "created_at": self.created_at,
            "error": self.error,
            "resolution": self.resolution,
            "sample_rate": self.sample_rate,
            "mastered_url": self.mastered_url,
            "mastered_download_name": self.mastered_download_name,
            "profile_url": self.profile_url,
            "profile_download_name": self.profile_download_name,
            "curve_points": self.curve_points,
            "reference_count": self.reference_count,
            "reference_files": self.reference_files,
            "source_rms_db": self.source_rms_db,
            "reference_rms_db": self.reference_rms_db,
            "loudness_gain_db": self.loudness_gain_db,
            "band_summary": self.band_summary,
        }


@dataclass
class OptimizeJobState:
    id: str
    status: str = "queued"
    message: str = "Waiting to start..."
    progress: int = 0
    created_at: float = field(default_factory=time.time)
    error: str = ""
    total_files: int = 0
    processed_files: int = 0
    current_file: str = ""
    lyrics: str = ""
    stitch_strength: int = 10
    max_cut_db: float = -24.0
    stitched_url: str = ""
    stitched_download_name: str = ""
    anchor_source_name: str = ""
    replaced_word_count: int = 0
    total_word_count: int = 0
    skipped_by_db_gate: int = 0
    edits_preview: List[Dict[str, object]] = field(default_factory=list)
    rankings: List[Dict[str, object]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, object]:
        return {
            "id": self.id,
            "status": self.status,
            "message": self.message,
            "progress": self.progress,
            "created_at": self.created_at,
            "error": self.error,
            "total_files": self.total_files,
            "processed_files": self.processed_files,
            "current_file": self.current_file,
            "lyrics": self.lyrics,
            "stitch_strength": self.stitch_strength,
            "max_cut_db": self.max_cut_db,
            "stitched_url": self.stitched_url,
            "stitched_download_name": self.stitched_download_name,
            "anchor_source_name": self.anchor_source_name,
            "replaced_word_count": self.replaced_word_count,
            "total_word_count": self.total_word_count,
            "skipped_by_db_gate": self.skipped_by_db_gate,
            "edits_preview": self.edits_preview,
            "rankings": self.rankings,
        }


@dataclass
class ApiComposeJobState:
    id: str
    status: str = "queued"
    message: str = "Waiting to start..."
    progress: int = 0
    created_at: float = field(default_factory=time.time)
    error: str = ""
    endpoint_url: str = ""
    midi_name: str = ""
    beat_name: str = ""
    provider_status_code: int = 0
    task_id: str = ""
    audio_urls: List[str] = field(default_factory=list)
    response_preview: str = ""
    response_json: Dict[str, object] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, object]:
        return {
            "id": self.id,
            "status": self.status,
            "message": self.message,
            "progress": self.progress,
            "created_at": self.created_at,
            "error": self.error,
            "endpoint_url": self.endpoint_url,
            "midi_name": self.midi_name,
            "beat_name": self.beat_name,
            "provider_status_code": self.provider_status_code,
            "task_id": self.task_id,
            "audio_urls": self.audio_urls,
            "response_preview": self.response_preview,
            "response_json": self.response_json,
        }


@dataclass
class TouchUpJobState:
    id: str
    status: str = "queued"
    message: str = "Waiting to start..."
    progress: int = 0
    created_at: float = field(default_factory=time.time)
    error: str = ""
    mode: str = "detect-regenerate"
    source_word: str = ""
    strength: int = 55
    max_target_words: int = 5
    batch_index: int = 0
    variants_tested: int = 0
    repair_attempts: int = 0
    repaired_word_count: int = 0
    best_similarity_score: float = 0.0
    best_word_report: str = ""
    best_letter_report: str = ""
    best_word_scores: List[Dict[str, object]] = field(default_factory=list)
    best_letter_scores: List[Dict[str, object]] = field(default_factory=list)
    detected_word_indices: List[int] = field(default_factory=list)
    regeneration_available: bool = False
    regeneration_reason: str = ""
    detected_only: bool = False
    stop_requested: bool = False
    sample_rate: int = 0
    result_url: str = ""
    download_name: str = ""
    removed_url: str = ""
    removed_download_name: str = ""
    source_rms_db: float = 0.0
    output_rms_db: float = 0.0
    kept_segment_count: int = 0
    kept_duration_seconds: float = 0.0
    removed_duration_seconds: float = 0.0

    def to_dict(self) -> Dict[str, object]:
        return {
            "id": self.id,
            "status": self.status,
            "message": self.message,
            "progress": self.progress,
            "created_at": self.created_at,
            "error": self.error,
            "mode": self.mode,
            "source_word": self.source_word,
            "strength": self.strength,
            "max_target_words": self.max_target_words,
            "batch_index": self.batch_index,
            "variants_tested": self.variants_tested,
            "repair_attempts": self.repair_attempts,
            "repaired_word_count": self.repaired_word_count,
            "best_similarity_score": self.best_similarity_score,
            "best_word_report": self.best_word_report,
            "best_letter_report": self.best_letter_report,
            "best_word_scores": self.best_word_scores,
            "best_letter_scores": self.best_letter_scores,
            "detected_word_indices": self.detected_word_indices,
            "regeneration_available": self.regeneration_available,
            "regeneration_reason": self.regeneration_reason,
            "detected_only": self.detected_only,
            "stop_requested": self.stop_requested,
            "sample_rate": self.sample_rate,
            "result_url": self.result_url,
            "download_name": self.download_name,
            "removed_url": self.removed_url,
            "removed_download_name": self.removed_download_name,
            "source_rms_db": self.source_rms_db,
            "output_rms_db": self.output_rms_db,
            "kept_segment_count": self.kept_segment_count,
            "kept_duration_seconds": self.kept_duration_seconds,
            "removed_duration_seconds": self.removed_duration_seconds,
        }


backend = SimpleRVCBackend(REPO_ROOT)
detagger = SimpleDetagger(REPO_ROOT)
trainer = SimpleTrainer(REPO_ROOT)
mastering_engine = SimpleMasteringEngine(REPO_ROOT)
optimizer_engine = VoiceSuitabilityOptimizer(sample_rate=44100)
touchup_engine = NeuralClarityRepairEngine(REPO_ROOT)
master_conversion_engine = MasterConversionEngine(REPO_ROOT, backend, touchup_engine)
rebuild_feature_builder = RebuildFeatureBuilder(REPO_ROOT)
jobs: Dict[str, JobState] = {}
jobs_lock = threading.Lock()
master_conversion_jobs: Dict[str, MasterConversionJobState] = {}
master_conversion_jobs_lock = threading.Lock()
detag_jobs: Dict[str, DetagJobState] = {}
detag_jobs_lock = threading.Lock()
isolator_jobs: Dict[str, IsolatorJobState] = {}
isolator_jobs_lock = threading.Lock()
training_jobs: Dict[str, TrainingJobState] = {}
training_jobs_lock = threading.Lock()
training_stop_events: Dict[str, threading.Event] = {}
training_stop_events_lock = threading.Lock()
mastering_jobs: Dict[str, MasteringJobState] = {}
mastering_jobs_lock = threading.Lock()
optimize_jobs: Dict[str, OptimizeJobState] = {}
optimize_jobs_lock = threading.Lock()
api_compose_jobs: Dict[str, ApiComposeJobState] = {}
api_compose_jobs_lock = threading.Lock()
touchup_jobs: Dict[str, TouchUpJobState] = {}
touchup_jobs_lock = threading.Lock()
touchup_stop_events: Dict[str, threading.Event] = {}
touchup_stop_events_lock = threading.Lock()
generate_jobs: Dict[str, GenerateJobState] = {}
generate_jobs_lock = threading.Lock()
albums_lock = threading.Lock()
app = FastAPI(title="Mangio Simple Web", docs_url=None, redoc_url=None)
progress_log_lock = threading.Lock()
progress_log_cache: Dict[str, tuple[str, int, str]] = {}

app.mount("/static", StaticFiles(directory=str(STATIC_ROOT)), name="static")
app.mount(
    "/downloads",
    StaticFiles(directory=str(REPO_ROOT / "audio-outputs")),
    name="downloads",
)


@app.middleware("http")
async def disable_cache(request: Request, call_next):
    response = await call_next(request)
    if request.method == "GET":
        response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
        response.headers["Pragma"] = "no-cache"
        response.headers["Expires"] = "0"
    return response


def log_progress_bar(job_kind: str, job_id: str, status: str, progress: int, message: str) -> None:
    normalized_progress = max(0, min(int(progress), 100))
    safe_status = status or "running"
    safe_message = (message or "").strip()
    cache_key = f"{job_kind}:{job_id}"
    signature = (safe_status, normalized_progress, safe_message)

    with progress_log_lock:
        if progress_log_cache.get(cache_key) == signature:
            return
        progress_log_cache[cache_key] = signature

    width = 24
    filled = int(round((normalized_progress / 100.0) * width))
    bar = "#" * filled + "-" * (width - filled)
    try:
        print(
            f"[{job_kind} {job_id}] [{bar}] {normalized_progress:>3}% {safe_status.upper()} {safe_message}",
            flush=True,
        )
    except OSError:
        # Detached/background Windows launches can leave stdout in a state where
        # console printing raises Errno 22. Progress logging is helpful, but it
        # should never break the actual audio job.
        return


def start_progress_heartbeat(
    *,
    job_id: str,
    update_fn,
    start_progress: int,
    end_progress: int,
    message: str,
    interval_seconds: float = 4.0,
) -> tuple[threading.Event, threading.Thread]:
    stop_event = threading.Event()
    progress_span = max(1, end_progress - start_progress)
    step = max(1, progress_span // 12)

    def pump() -> None:
        progress = start_progress
        ticks = 0
        phases = [
            "Still working through the separator...",
            "Crunching the vocal layers...",
            "Refining the stem split...",
            "Holding steady while the model finishes...",
        ]
        while not stop_event.wait(interval_seconds):
            ticks += 1
            progress = min(end_progress, progress + step)
            phase = phases[(ticks - 1) % len(phases)]
            update_fn(
                job_id,
                status="running",
                progress=progress,
                message=f"{message} {phase}",
            )

    thread = threading.Thread(target=pump, daemon=True)
    thread.start()
    return stop_event, thread


def audio_output_url(file_path: Path) -> str:
    relative = file_path.resolve().relative_to((REPO_ROOT / "audio-outputs").resolve()).as_posix()
    return f"/downloads/{relative}"


def get_job(job_id: str) -> JobState:
    with jobs_lock:
        job = jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found.")
    return job


def set_job_state(job_id: str, **updates) -> None:
    with jobs_lock:
        job = jobs[job_id]
        for key, value in updates.items():
            setattr(job, key, value)
        status = job.status
        progress = 100 if job.status == "completed" else 0
        if job.total_files:
            progress = int(round((job.completed_files / max(job.total_files, 1)) * 100))
        progress = max(progress, 8 if status in {"queued", "running"} else progress)
        message = job.message
        if job.current_file:
            message = f"{message} Current file: {job.current_file}."
    log_progress_bar("convert", job_id, status, progress, message)


def get_master_conversion_job(job_id: str) -> MasterConversionJobState:
    with master_conversion_jobs_lock:
        job = master_conversion_jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Master conversion job not found.")
    return job


def set_master_conversion_job_state(job_id: str, **updates) -> None:
    with master_conversion_jobs_lock:
        job = master_conversion_jobs[job_id]
        for key, value in updates.items():
            setattr(job, key, value)
        status = job.status
        progress = int(job.progress or 0)
        message = job.message
        if job.stage:
            message = f"{job.stage}: {message}"
    log_progress_bar("master", job_id, status, progress, message)


def get_training_job(job_id: str) -> TrainingJobState:
    with training_jobs_lock:
        job = training_jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Training job not found.")
    return job


def get_detag_job(job_id: str) -> DetagJobState:
    with detag_jobs_lock:
        job = detag_jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Detag job not found.")
    return job


def get_isolator_job(job_id: str) -> IsolatorJobState:
    with isolator_jobs_lock:
        job = isolator_jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Isolator job not found.")
    return job


def get_mastering_job(job_id: str) -> MasteringJobState:
    with mastering_jobs_lock:
        job = mastering_jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Mastering job not found.")
    return job


def get_optimize_job(job_id: str) -> OptimizeJobState:
    with optimize_jobs_lock:
        job = optimize_jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Optimize job not found.")
    return job


def get_api_compose_job(job_id: str) -> ApiComposeJobState:
    with api_compose_jobs_lock:
        job = api_compose_jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="API compose job not found.")
    return job


def get_touchup_job(job_id: str) -> TouchUpJobState:
    with touchup_jobs_lock:
        job = touchup_jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Touch-up job not found.")
    return job


def set_training_job_state(job_id: str, **updates) -> None:
    with training_jobs_lock:
        job = training_jobs[job_id]
        previous_signature = (
            str(job.stage or ""),
            str(job.message or ""),
            str(job.log_tail or ""),
            str(job.status or ""),
        )
        for key, value in updates.items():
            setattr(job, key, value)
        new_signature = (
            str(job.stage or ""),
            str(job.message or ""),
            str(job.log_tail or ""),
            str(job.status or ""),
        )
        if new_signature != previous_signature:
            timestamp = time.strftime("%H:%M:%S")
            line = f"[{timestamp}] {job.stage or job.status or 'training'} | {job.message}"
            detail = str(job.log_tail or "").strip()
            if detail:
                line = f"{line} | {detail}"
            if not job.log_history or job.log_history[-1] != line:
                job.log_history.append(line)
                if len(job.log_history) > 160:
                    job.log_history = job.log_history[-160:]
        status = job.status
        progress = int(job.progress or 0)
        message = job.message
        if job.stage:
            message = f"{job.stage}: {message}"
    log_progress_bar("training", job_id, status, progress, message)


def set_detag_job_state(job_id: str, **updates) -> None:
    with detag_jobs_lock:
        job = detag_jobs[job_id]
        for key, value in updates.items():
            setattr(job, key, value)
        status = job.status
        progress = int(job.progress or 0)
        message = job.message
    log_progress_bar("detag", job_id, status, progress, message)


def set_isolator_job_state(job_id: str, **updates) -> None:
    with isolator_jobs_lock:
        job = isolator_jobs[job_id]
        for key, value in updates.items():
            setattr(job, key, value)
        status = job.status
        progress = int(job.progress or 0)
        message = job.message
        if job.current_file:
            message = f"{message} Current file: {job.current_file}."
        if job.source_files:
            message = f"{message} Source: {', '.join(job.source_files)}."
    log_progress_bar("isolator", job_id, status, progress, message)


def set_mastering_job_state(job_id: str, **updates) -> None:
    with mastering_jobs_lock:
        job = mastering_jobs[job_id]
        for key, value in updates.items():
            setattr(job, key, value)
        status = job.status
        progress = int(job.progress or 0)
        message = job.message
    log_progress_bar("mastering", job_id, status, progress, message)


def set_optimize_job_state(job_id: str, **updates) -> None:
    with optimize_jobs_lock:
        job = optimize_jobs[job_id]
        for key, value in updates.items():
            setattr(job, key, value)
        status = job.status
        progress = int(job.progress or 0)
        message = job.message
        if job.current_file:
            message = f"{message} Current file: {job.current_file}."
    log_progress_bar("optimize", job_id, status, progress, message)


def set_api_compose_job_state(job_id: str, **updates) -> None:
    with api_compose_jobs_lock:
        job = api_compose_jobs[job_id]
        for key, value in updates.items():
            setattr(job, key, value)
        status = job.status
        progress = int(job.progress or 0)
        message = job.message
    log_progress_bar("api-compose", job_id, status, progress, message)


def set_touchup_job_state(job_id: str, **updates) -> None:
    with touchup_jobs_lock:
        job = touchup_jobs[job_id]
        for key, value in updates.items():
            setattr(job, key, value)
        status = job.status
        progress = int(job.progress or 0)
        message = job.message
    log_progress_bar("touchup", job_id, status, progress, message)


def get_available_model_names() -> set[str]:
    return {model["name"] for model in backend.list_models()}


def get_generate_job(job_id: str) -> GenerateJobState:
    with generate_jobs_lock:
        job = generate_jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Generate job not found.")
    return job


def set_generate_job_state(job_id: str, **updates) -> None:
    with generate_jobs_lock:
        job = generate_jobs[job_id]
        for key, value in updates.items():
            setattr(job, key, value)
        status = job.status
        progress = int(job.progress or 0)
        message = job.message
    log_progress_bar("generate", job_id, status, progress, message)


def normalize_generate_key(key_name: str) -> str:
    normalized = str(key_name or "").strip().upper()
    normalized = normalized.replace("MINOR", "").replace("MAJOR", "").replace("M", "")
    normalized = normalized.replace(" ", "")
    return GENERATE_NOTE_MAP.get(normalized, "")


def compute_generate_transpose(guide_key: str, target_key: str) -> int:
    source = normalize_generate_key(guide_key)
    target = normalize_generate_key(target_key)
    if not source or not target:
        return 0
    source_index = GENERATE_NOTE_ORDER.index(source)
    target_index = GENERATE_NOTE_ORDER.index(target)
    delta = (target_index - source_index + 12) % 12
    if delta > 6:
        delta -= 12
    return int(delta)


def build_conversion_settings(
    *,
    quality_preset: str,
    preprocess_mode: str,
    preprocess_strength: int,
    speaker_id: int,
    transpose: int,
    pitch_method: str,
    index_path: str,
    index_rate: float,
    filter_radius: int,
    resample_sr: int,
    rms_mix_rate: float,
    protect: float,
    crepe_hop_length: int,
) -> Dict[str, object]:
    try:
        normalized_preprocess_mode = backend.normalize_preprocess_mode(preprocess_mode)
    except ValueError:
        raise HTTPException(status_code=400, detail="Unsupported preprocess pipeline.")

    preset = QUALITY_PRESETS.get(quality_preset, QUALITY_PRESETS["balanced"])
    return {
        "speaker_id": speaker_id,
        "transpose": transpose,
        "preprocess_mode": normalized_preprocess_mode,
        "preprocess_strength": max(1, min(int(preprocess_strength), 20)),
        "f0_method": pitch_method or preset["f0_method"],
        "index_path": index_path,
        "index_rate": index_rate if index_rate >= 0 else preset["index_rate"],
        "filter_radius": filter_radius if filter_radius >= 0 else preset["filter_radius"],
        "resample_sr": resample_sr,
        "rms_mix_rate": rms_mix_rate if rms_mix_rate >= 0 else preset["rms_mix_rate"],
        "protect": protect if protect >= 0 else preset["protect"],
        "crepe_hop_length": crepe_hop_length if crepe_hop_length >= 0 else preset["crepe_hop_length"],
    }


def prepare_audio_for_concat(input_path: Path, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            backend._ffmpeg_binary(),
            "-nostdin",
            "-y",
            "-i",
            str(input_path),
            "-vn",
            "-ac",
            "2",
            "-ar",
            "44100",
            "-c:a",
            "pcm_s24le",
            str(output_path),
        ],
        check=True,
        capture_output=True,
    )
    return output_path


def combine_audio_files(
    input_paths: List[Path],
    output_path: Path,
    progress_callback=None,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    prepared_dir = output_path.parent / "prepared"
    reset_directory(prepared_dir)
    prepared_paths: List[Path] = []

    for index, input_path in enumerate(input_paths, start=1):
        if progress_callback is not None:
            progress_callback(index, len(input_paths), input_path)
        prepared_path = prepare_audio_for_concat(
            input_path,
            prepared_dir / f"{index:03d}_{sanitize_filename(input_path.stem)}.wav",
        )
        prepared_paths.append(prepared_path)

    if len(prepared_paths) == 1:
        return prepared_paths[0]

    concat_list_path = output_path.parent / "concat-inputs.txt"
    concat_lines = []
    for path in prepared_paths:
        escaped = str(path.resolve()).replace("'", "'\\''")
        concat_lines.append(f"file '{escaped}'")
    concat_list_path.write_text("\n".join(concat_lines), encoding="utf-8")

    subprocess.run(
        [
            backend._ffmpeg_binary(),
            "-nostdin",
            "-y",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            str(concat_list_path),
            "-vn",
            "-ac",
            "2",
            "-ar",
            "44100",
            "-c:a",
            "pcm_s24le",
            str(output_path),
        ],
        check=True,
        capture_output=True,
    )
    return output_path


def _match_stereo_audio_shape(audio: np.ndarray, target_length: int) -> np.ndarray:
    working = np.asarray(audio, dtype=np.float32)
    if working.ndim == 1:
        working = working[:, np.newaxis]
    if working.shape[1] == 1:
        working = np.repeat(working, 2, axis=1)
    elif working.shape[1] > 2:
        working = working[:, :2]

    if working.shape[0] == target_length:
        return working.astype(np.float32, copy=False)
    if working.shape[0] > target_length:
        return working[:target_length].astype(np.float32, copy=False)

    pad = np.zeros((target_length - working.shape[0], working.shape[1]), dtype=np.float32)
    return np.vstack([working.astype(np.float32, copy=False), pad])


def blend_audio_outputs(
    *,
    primary_path: Path,
    secondary_path: Path,
    output_path: Path,
    output_format: str,
    primary_percentage: int,
) -> int:
    primary_audio, primary_sr = sf.read(str(primary_path), always_2d=True)
    secondary_audio, secondary_sr = sf.read(str(secondary_path), always_2d=True)
    if int(primary_sr) != int(secondary_sr):
        raise RuntimeError("Blend mode requires both model renders to have the same sample rate.")

    target_length = max(int(primary_audio.shape[0]), int(secondary_audio.shape[0]))
    primary_matched = _match_stereo_audio_shape(primary_audio, target_length)
    secondary_matched = _match_stereo_audio_shape(secondary_audio, target_length)

    primary_weight = max(0.0, min(float(primary_percentage) / 100.0, 1.0))
    secondary_weight = 1.0 - primary_weight
    blended = (primary_matched * primary_weight) + (secondary_matched * secondary_weight)
    peak = float(np.max(np.abs(blended)) + 1e-9)
    if peak > 0.995:
        blended = blended * np.float32(0.995 / peak)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp_wav = output_path if output_format.lower() == "wav" else output_path.with_suffix(".wav")
    sf.write(str(temp_wav), blended.astype(np.float32, copy=False), int(primary_sr), subtype="PCM_24")

    if output_format.lower() != "wav":
        backend._transcode_audio(temp_wav, output_path)
        temp_wav.unlink(missing_ok=True)

    return int(primary_sr)


def utc_now_iso() -> str:
    return datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


def album_db_path() -> Path:
    return ALBUMS_ROOT / "projects.json"


def load_album_db() -> Dict[str, object]:
    path = album_db_path()
    if not path.exists():
        return {"projects": []}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {"projects": []}
    projects = payload.get("projects")
    if not isinstance(projects, list):
        return {"projects": []}
    return {"projects": projects}


def save_album_db(payload: Dict[str, object]) -> None:
    ALBUMS_ROOT.mkdir(parents=True, exist_ok=True)
    album_db_path().write_text(json.dumps(payload, indent=2), encoding="utf-8")


def album_project_root(project_id: str) -> Path:
    return ALBUMS_ROOT / sanitize_filename(project_id)


def album_song_key(song_index: int) -> str:
    return f"song_{int(song_index):02d}"


def album_song_directory(project_id: str, song_index: int) -> Path:
    return album_project_root(project_id) / "songs" / album_song_key(song_index)


def album_relative_path(project_id: str, *parts: str) -> str:
    clean_parts = [sanitize_filename(part) for part in parts]
    return "/".join(["simple-web", "albums", sanitize_filename(project_id), *clean_parts])


def album_download_url(relative_path: str) -> str:
    normalized = str(relative_path).replace("\\", "/")
    return f"/downloads/{normalized}"


def resolve_album_file_relative_path(
    project_id: str,
    file_name: str,
    explicit_rel_path: str = "",
) -> str:
    project_id = sanitize_filename(project_id)
    if explicit_rel_path:
        explicit_path = REPO_ROOT / "audio-outputs" / Path(explicit_rel_path)
        if explicit_path.exists():
            return explicit_rel_path

    search_root = album_project_root(project_id) / "songs"
    if file_name and search_root.exists():
        for found_path in search_root.rglob(file_name):
            if found_path.is_file():
                relative = found_path.relative_to(REPO_ROOT / "audio-outputs")
                return str(relative).replace("\\", "/")

    return ""


def resolve_album_version_asset_path(
    project_id: str,
    song_index: int,
    version_entry: Dict[str, object],
    prepared: bool = False,
    storage_key: str = "",
    strict: bool = False,
) -> Optional[Path]:
    project_id = sanitize_filename(project_id)
    song_index = int(song_index)
    version_number = int(version_entry.get("version", 0))
    storage_key = sanitize_filename(storage_key or str(version_entry.get("storage_key", "")))

    explicit_rel = str(version_entry.get("prepared_rel_path" if prepared else "stored_rel_path", ""))
    if explicit_rel:
        explicit_path = REPO_ROOT / "audio-outputs" / Path(explicit_rel)
        if explicit_path.exists():
            if not prepared:
                return explicit_path
            stored_name = str(version_entry.get("stored_file_name", ""))
            if stored_name and (explicit_path.parent / stored_name).exists():
                return explicit_path

    search_root = album_project_root(project_id) / "songs"
    if not search_root.exists():
        return None

    target_name = f"v{version_number}_prepared.wav" if prepared else str(version_entry.get("stored_file_name", ""))

    if prepared:
        stored_path = resolve_album_version_asset_path(
            project_id,
            song_index,
            version_entry,
            prepared=False,
            storage_key=storage_key,
            strict=True,
        )
        if stored_path is not None:
            preferred = stored_path.parent / target_name
            if preferred.exists():
                return preferred
        if strict:
            return None

    candidate_dirs: List[Path] = []
    if storage_key:
        candidate_dirs.append(album_song_storage_directory(project_id, storage_key))
    if not strict and not storage_key:
        candidate_dirs.append(search_root / album_song_key(song_index))

    seen_dirs: set[Path] = set()
    for candidate_dir in candidate_dirs:
        if candidate_dir in seen_dirs:
            continue
        seen_dirs.add(candidate_dir)
        if not candidate_dir.exists():
            continue
        preferred = candidate_dir / target_name
        if target_name and preferred.exists():
            return preferred

    if target_name:
        matches = [found_path for found_path in search_root.rglob(target_name) if found_path.is_file()]
        if len(matches) == 1:
            return matches[0]
        if matches and storage_key:
            for match in matches:
                if match.parent.name == storage_key:
                    return match

    if strict:
        return None

    if storage_key:
        storage_dir = album_song_storage_directory(project_id, storage_key)
        if storage_dir.exists():
            candidates = sorted(
                path
                for path in storage_dir.iterdir()
                if path.is_file() and path.suffix.lower() in {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac"}
            )
            if prepared:
                for candidate in candidates:
                    if candidate.name.endswith("_prepared.wav"):
                        return candidate
            else:
                for candidate in candidates:
                    if not candidate.name.endswith("_prepared.wav"):
                        return candidate

    return None


def resolve_album_mix_asset_path(project: Dict[str, object]) -> Optional[Path]:
    project_id = str(project.get("id", ""))
    mix_rel = str(project.get("latest_mix_rel_path", ""))
    if mix_rel:
        mix_path = REPO_ROOT / "audio-outputs" / Path(mix_rel)
        if mix_path.exists():
            return mix_path

    project_root = album_project_root(project_id) / "mixes"
    if not project_root.exists():
        return None

    candidate_paths = sorted(
        path
        for path in project_root.rglob("mix_v*.wav")
        if path.is_file()
    )
    if candidate_paths:
        return candidate_paths[-1]
    return None


def album_song_play_url(project_id: str, song_index: int, version: Optional[int] = None) -> str:
    route = f"/api/albums/projects/{sanitize_filename(project_id)}/songs/{int(song_index)}/play"
    if version is None:
        return route
    return f"{route}?version={int(version)}"


def album_mix_play_url(project_id: str) -> str:
    return f"/api/albums/projects/{sanitize_filename(project_id)}/mix"


def album_file_token(*parts: object) -> str:
    return "-".join(
        sanitize_filename(str(part)).replace("_", "-")
        for part in parts
        if str(part)
    )


def album_song_storage_directory(project_id: str, storage_key: str) -> Path:
    safe_key = sanitize_filename(storage_key or "")
    if not safe_key:
        safe_key = f"song_{uuid4().hex[:8]}"
    return album_project_root(project_id) / "songs" / safe_key


def next_album_storage_key(existing_keys: set[str]) -> str:
    for index in count(1):
        candidate = f"song_{index:02d}"
        if candidate not in existing_keys:
            return candidate
    return f"song_{uuid4().hex[:8]}"


def ensure_album_song_identity(
    project_id: str,
    song: Dict[str, object],
    existing_storage_keys: set[str],
) -> None:
    song_id = str(song.get("song_id", "")).strip()
    if not song_id:
        song["song_id"] = uuid4().hex[:12]

    storage_key = str(song.get("storage_key", "")).strip()
    versions = song.get("versions", [])
    if not storage_key and isinstance(versions, list) and versions:
        first_version = versions[0]
        if isinstance(first_version, dict):
            stored_rel = str(first_version.get("stored_rel_path", "")).replace("\\", "/")
            if stored_rel:
                storage_key = Path(stored_rel).parent.name
    if not storage_key:
        storage_key = next_album_storage_key(existing_storage_keys)
    song["storage_key"] = sanitize_filename(storage_key)
    existing_storage_keys.add(str(song["storage_key"]))


def normalize_album_project(project: Dict[str, object]) -> bool:
    songs = project.get("songs", [])
    if not isinstance(songs, list):
        project["songs"] = []
        return True

    changed = False
    project_id = str(project.get("id", ""))
    existing_storage_keys: set[str] = set()

    for position, song in enumerate(
        sorted((entry for entry in songs if isinstance(entry, dict)), key=lambda entry: int(entry.get("song_index", 0))),
        start=1,
    ):
        original_song_index = int(song.get("song_index", 0) or 0)
        if original_song_index != position:
            song["song_index"] = position
            changed = True

        before_song_id = str(song.get("song_id", ""))
        before_storage_key = str(song.get("storage_key", ""))
        ensure_album_song_identity(project_id, song, existing_storage_keys)
        if before_song_id != str(song.get("song_id", "")) or before_storage_key != str(song.get("storage_key", "")):
            changed = True

        versions = song.get("versions", [])
        if not isinstance(versions, list):
            song["versions"] = []
            changed = True
            continue

        if not song.get("title") and versions:
            first_version = versions[0]
            if isinstance(first_version, dict):
                song["title"] = Path(str(first_version.get("source_name", f"Song {position}"))).stem
                changed = True

        for version in versions:
            if not isinstance(version, dict):
                continue
            version["song_id"] = str(song.get("song_id", ""))
            version["storage_key"] = str(song.get("storage_key", ""))

            resolved_stored = resolve_album_version_asset_path(
                project_id,
                int(song.get("song_index", 0)),
                version,
                prepared=False,
                storage_key=str(song.get("storage_key", "")),
                strict=True,
            )
            if resolved_stored is not None:
                actual_storage_key = resolved_stored.parent.name
                if str(song.get("storage_key", "")) != actual_storage_key:
                    existing_storage_keys.discard(str(song.get("storage_key", "")))
                    song["storage_key"] = actual_storage_key
                    version["storage_key"] = actual_storage_key
                    existing_storage_keys.add(actual_storage_key)
                    changed = True
                stored_rel = str(resolved_stored.relative_to(REPO_ROOT / "audio-outputs")).replace("\\", "/")
                if str(version.get("stored_rel_path", "")) != stored_rel:
                    version["stored_rel_path"] = stored_rel
                    changed = True

            resolved_prepared = resolve_album_version_asset_path(
                project_id,
                int(song.get("song_index", 0)),
                version,
                prepared=True,
                storage_key=str(song.get("storage_key", "")),
                strict=True,
            )
            if resolved_prepared is not None:
                prepared_rel = str(resolved_prepared.relative_to(REPO_ROOT / "audio-outputs")).replace("\\", "/")
                if str(version.get("prepared_rel_path", "")) != prepared_rel:
                    version["prepared_rel_path"] = prepared_rel
                    changed = True
                duration_seconds = read_audio_duration_seconds(resolved_prepared)
                if abs(float(version.get("duration_seconds", 0.0) or 0.0) - float(duration_seconds)) > 0.01:
                    version["duration_seconds"] = round(float(duration_seconds), 3)
                    changed = True

    project["song_count"] = len([entry for entry in songs if isinstance(entry, dict)])
    return changed


def append_album_log(project: Dict[str, object], message: str) -> None:
    logs = project.setdefault("event_log", [])
    if not isinstance(logs, list):
        logs = []
        project["event_log"] = logs
    entry = {
        "at": utc_now_iso(),
        "message": str(message),
    }
    logs.append(entry)
    if len(logs) > 5000:
        del logs[:-5000]
    try:
        root = album_project_root(str(project.get("id", "")))
        root.mkdir(parents=True, exist_ok=True)
        with (root / "activity.log").open("a", encoding="utf-8") as handle:
            handle.write(f"{entry['at']} {entry['message']}\n")
    except Exception:
        pass


def read_audio_duration_seconds(path: Path) -> float:
    try:
        return float(sf.info(str(path)).duration)
    except Exception:
        return 0.0


def create_album_song_slots(song_count: int) -> List[Dict[str, object]]:
    slots: List[Dict[str, object]] = []
    for song_index in range(1, song_count + 1):
        slots.append(
            {
                "song_index": song_index,
                "title": f"Song {song_index}",
                "versions": [],
            }
        )
    return slots


def get_album_project_by_id(payload: Dict[str, object], project_id: str) -> Dict[str, object]:
    projects = payload.get("projects", [])
    if not isinstance(projects, list):
        raise HTTPException(status_code=500, detail="Album project database is corrupted.")
    for project in projects:
        if str(project.get("id", "")) == project_id:
            return project
    raise HTTPException(status_code=404, detail="Album project not found.")


def ensure_album_song(project: Dict[str, object], song_index: int) -> Dict[str, object]:
    song_index = int(song_index)
    if song_index < 1:
        raise HTTPException(status_code=400, detail="Song index must be 1 or higher.")
    songs = project.get("songs", [])
    if not isinstance(songs, list):
        raise HTTPException(status_code=500, detail="Album song metadata is corrupted.")
    for song in songs:
        if int(song.get("song_index", 0)) == int(song_index):
            return song

    existing_indices = [int(entry.get("song_index", 0)) for entry in songs if isinstance(entry, dict)]
    max_existing = max(existing_indices) if existing_indices else 0
    for idx in range(max_existing + 1, song_index + 1):
        songs.append(
            {
                "song_index": int(idx),
                "title": f"Song {idx}",
                "versions": [],
            }
        )
    project["songs"] = songs
    project["song_count"] = max(int(project.get("song_count", 0)), song_index)
    return songs[-1]


def run_album_crossfade_render(
    prepared_paths: List[Path],
    output_path: Path,
    fade_seconds: float = 0.5,
) -> None:
    if not prepared_paths:
        raise RuntimeError("No prepared songs were available for crossfade rendering.")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if len(prepared_paths) == 1:
        subprocess.run(
            [
                backend._ffmpeg_binary(),
                "-nostdin",
                "-y",
                "-i",
                str(prepared_paths[0]),
                "-vn",
                "-ac",
                "2",
                "-ar",
                "44100",
                "-c:a",
                "pcm_s24le",
                str(output_path),
            ],
            check=True,
            capture_output=True,
        )
        return

    command = [
        backend._ffmpeg_binary(),
        "-nostdin",
        "-y",
    ]
    for path in prepared_paths:
        command.extend(["-i", str(path)])

    filters: List[str] = []
    for idx in range(1, len(prepared_paths)):
        if idx == 1:
            left_label = "[0:a]"
        else:
            left_label = f"[xf{idx - 2}]"
        right_label = f"[{idx}:a]"
        out_label = f"[xf{idx - 1}]"
        filters.append(
            f"{left_label}{right_label}acrossfade=d={fade_seconds:.3f}:c1=tri:c2=tri{out_label}"
        )

    final_label = f"[xf{len(prepared_paths) - 2}]"
    command.extend(
        [
            "-filter_complex",
            ";".join(filters),
            "-map",
            final_label,
            "-vn",
            "-ac",
            "2",
            "-ar",
            "44100",
            "-c:a",
            "pcm_s24le",
            str(output_path),
        ]
    )
    subprocess.run(command, check=True, capture_output=True)


def store_album_song_version(
    project: Dict[str, object],
    song_index: int,
    original_file_name: str,
    payload_bytes: bytes,
) -> Dict[str, object]:
    if not payload_bytes:
        raise HTTPException(status_code=400, detail="Uploaded audio file is empty.")

    song = ensure_album_song(project, song_index)
    versions = song.setdefault("versions", [])
    if not isinstance(versions, list):
        versions = []
        song["versions"] = versions

    version_number = len(versions)
    safe_name = sanitize_filename(original_file_name or f"song_{song_index}.wav")
    if "." not in safe_name:
        safe_name = f"{safe_name}.wav"

    project_id = str(project.get("id", ""))
    existing_keys = {
        sanitize_filename(str(entry.get("storage_key", "")))
        for entry in project.get("songs", [])
        if isinstance(entry, dict) and entry is not song and entry.get("storage_key")
    }
    storage_key = str(song.get("storage_key", "")) or next_album_storage_key(existing_keys)
    song["storage_key"] = sanitize_filename(storage_key)
    if not song.get("song_id"):
        song["song_id"] = uuid4().hex[:12]

    song_dir = album_song_storage_directory(project_id, str(song.get("storage_key", "")))
    song_dir.mkdir(parents=True, exist_ok=True)

    source_name = f"v{version_number}_{safe_name}"
    source_path = song_dir / source_name
    with source_path.open("wb") as handle:
        handle.write(payload_bytes)

    prepared_name = f"v{version_number}_prepared.wav"
    prepared_path = song_dir / prepared_name
    prepare_audio_for_concat(source_path, prepared_path)
    duration_seconds = read_audio_duration_seconds(prepared_path)

    source_rel = album_relative_path(
        project_id,
        "songs",
        str(song.get("storage_key", "")),
        source_name,
    )
    prepared_rel = album_relative_path(
        project_id,
        "songs",
        str(song.get("storage_key", "")),
        prepared_name,
    )
    song["title"] = Path(safe_name).stem

    version_entry = {
        "version": int(version_number),
        "created_at": utc_now_iso(),
        "source_name": safe_name,
        "duration_seconds": round(float(duration_seconds), 3),
        "stored_file_name": source_name,
        "stored_rel_path": source_rel,
        "prepared_rel_path": prepared_rel,
        "song_id": str(song.get("song_id", "")),
        "storage_key": str(song.get("storage_key", "")),
    }
    versions.append(version_entry)
    project["updated_at"] = utc_now_iso()
    append_album_log(
        project,
        f"{song.get('title', f'Song {song_index}')} uploaded as V{version_number} ({safe_name}).",
    )
    return version_entry


def delete_album_song(project: Dict[str, object], song_index: int) -> None:
    songs = project.get("songs", [])
    if not isinstance(songs, list):
        raise HTTPException(status_code=500, detail="Album song metadata is corrupted.")

    target_index = int(song_index)
    if target_index < 1:
        raise HTTPException(status_code=400, detail="Song index must be 1 or higher.")

    target_song: Optional[Dict[str, object]] = None
    remaining: List[Dict[str, object]] = []
    for song in songs:
        if not isinstance(song, dict):
            continue
        if int(song.get("song_index", 0)) == target_index and target_song is None:
            target_song = song
            continue
        remaining.append(song)

    if target_song is None:
        raise HTTPException(status_code=404, detail="Track not found.")

    project_id = str(project.get("id", ""))
    target_dir = album_song_storage_directory(project_id, str(target_song.get("storage_key", "")))
    if target_dir.exists():
        shutil.rmtree(target_dir, ignore_errors=True)

    for song in remaining:
        current_index = int(song.get("song_index", 0))
        if current_index > target_index:
            song["song_index"] = int(current_index - 1)

    project["songs"] = sorted(remaining, key=lambda entry: int(entry.get("song_index", 0)))
    project["song_count"] = len(project["songs"])
    project["updated_at"] = utc_now_iso()
    append_album_log(project, f"Deleted track {target_index} and renumbered the remaining tracks.")


def rebuild_album_preview(project: Dict[str, object], fade_seconds: float = 0.5) -> None:
    project_id = str(project.get("id", ""))
    songs = project.get("songs", [])
    if not isinstance(songs, list):
        raise HTTPException(status_code=500, detail="Album song metadata is corrupted.")

    selected_prepared_paths: List[Path] = []
    selected_song_indices: List[int] = []
    for song in sorted(
        (entry for entry in songs if isinstance(entry, dict)),
        key=lambda entry: int(entry.get("song_index", 0)),
    ):
        versions = song.get("versions", [])
        if not isinstance(versions, list) or not versions:
            continue
        latest = versions[-1]
        prepared_path = resolve_album_version_asset_path(
            project_id,
            int(song.get("song_index", 0)),
            latest,
            prepared=True,
            storage_key=str(song.get("storage_key", "")),
            strict=True,
        )
        if prepared_path is None:
            continue
        if not prepared_path.exists():
            continue
        selected_prepared_paths.append(prepared_path)
        selected_song_indices.append(int(song.get("song_index", 0)))

    if not selected_prepared_paths:
        project["latest_mix_rel_path"] = ""
        project["latest_mix_download_name"] = ""
        project["latest_mix_song_count"] = 0
        return

    mix_versions = project.setdefault("mix_versions", [])
    if not isinstance(mix_versions, list):
        mix_versions = []
        project["mix_versions"] = mix_versions

    mix_version = len(mix_versions)
    mix_name = f"mix_v{mix_version}.wav"
    mix_rel = album_relative_path(project_id, "mixes", mix_name)
    mix_path = REPO_ROOT / "audio-outputs" / Path(mix_rel)

    run_album_crossfade_render(
        selected_prepared_paths,
        mix_path,
        fade_seconds=fade_seconds,
    )

    mix_entry = {
        "version": int(mix_version),
        "created_at": utc_now_iso(),
        "stored_file_name": mix_name,
        "stored_rel_path": mix_rel,
        "crossfade_seconds": float(fade_seconds),
        "song_indices": selected_song_indices,
    }
    mix_versions.append(mix_entry)
    project["latest_mix_rel_path"] = mix_rel
    project["latest_mix_download_name"] = mix_name
    project["latest_mix_song_count"] = len(selected_song_indices)
    project["updated_at"] = utc_now_iso()
    append_album_log(
        project,
        f"Album preview rebuilt with {len(selected_song_indices)} song(s), {fade_seconds:.1f}s crossfade.",
    )


def serialize_album_project(project: Dict[str, object]) -> Dict[str, object]:
    project_id = str(project.get("id", ""))
    normalize_album_project(project)
    songs_payload: List[Dict[str, object]] = []
    songs = project.get("songs", [])
    if isinstance(songs, list):
        for song in sorted(
            (entry for entry in songs if isinstance(entry, dict)),
            key=lambda entry: int(entry.get("song_index", 0)),
        ):
            versions_payload: List[Dict[str, object]] = []
            versions = song.get("versions", [])
            if isinstance(versions, list):
                for version_entry in versions:
                    if not isinstance(version_entry, dict):
                        continue
                    version_number = int(version_entry.get("version", 0))
                    resolved_stored_path = resolve_album_version_asset_path(
                        project_id,
                        int(song.get("song_index", 0)),
                        version_entry,
                        prepared=False,
                        storage_key=str(song.get("storage_key", "")),
                        strict=True,
                    )
                    resolved_prepared_path = resolve_album_version_asset_path(
                        project_id,
                        int(song.get("song_index", 0)),
                        version_entry,
                        prepared=True,
                        storage_key=str(song.get("storage_key", "")),
                        strict=True,
                    )
                    duration_seconds = float(version_entry.get("duration_seconds", 0.0) or 0.0)
                    if duration_seconds <= 0.0 and resolved_prepared_path is not None:
                        duration_seconds = read_audio_duration_seconds(resolved_prepared_path)
                    version_token = album_file_token(
                        song.get("song_id", ""),
                        version_number,
                        version_entry.get("created_at", ""),
                        version_entry.get("stored_file_name", ""),
                    )
                    version_url = (
                        f"{album_song_play_url(project_id, int(song.get('song_index', 0)), version_number)}&t={version_token}"
                        if resolved_stored_path is not None
                        else ""
                    )
                    prepared_url = (
                        f"{album_song_play_url(project_id, int(song.get('song_index', 0)), version_number)}&prepared=1&t={version_token}"
                        if resolved_prepared_path is not None
                        else ""
                    )
                    versions_payload.append(
                        {
                            "version": version_number,
                            "created_at": str(version_entry.get("created_at", "")),
                            "source_name": str(version_entry.get("source_name", "")),
                            "duration_seconds": float(duration_seconds),
                            "stored_file_name": str(version_entry.get("stored_file_name", "")),
                            "url": version_url,
                            "prepared_url": prepared_url,
                            "download_name": str(version_entry.get("stored_file_name", "")),
                            "playable": bool(resolved_stored_path),
                        }
                    )
            latest_version = versions_payload[-1] if versions_payload else None
            latest_play_url = str(latest_version.get("url", "")) if latest_version else ""
            songs_payload.append(
                {
                    "song_index": int(song.get("song_index", 0)),
                    "song_id": str(song.get("song_id", "")),
                    "title": str(song.get("title", f"Song {song.get('song_index', 0)}")),
                    "versions": versions_payload,
                    "latest_version": latest_version,
                    "latest_version_url": latest_play_url,
                    "version_count": len(versions_payload),
                }
            )

    logs = project.get("event_log", [])
    if not isinstance(logs, list):
        logs = []
    song_count = max(
        int(project.get("song_count", 0)),
        len(songs_payload),
    )
    filled_songs = len([entry for entry in songs_payload if int(entry.get("version_count", 0)) > 0])

    return {
        "id": project_id,
        "name": str(project.get("name", "Album Project")),
        "song_count": song_count,
        "filled_songs": filled_songs,
        "created_at": str(project.get("created_at", "")),
        "updated_at": str(project.get("updated_at", "")),
        "crossfade_seconds": float(project.get("crossfade_seconds", 0.5)),
        "songs": songs_payload,
        "latest_mix_url": (
            f"{album_mix_play_url(project_id)}?t={album_file_token(project.get('latest_mix_download_name', ''), project.get('updated_at', ''))}"
            if resolve_album_mix_asset_path(project)
            else ""
        ),
        "latest_mix_download_name": str(project.get("latest_mix_download_name", "")),
        "latest_mix_song_count": int(project.get("latest_mix_song_count", 0)),
        "event_log": logs,
    }


def serialize_album_project_summary(project: Dict[str, object]) -> Dict[str, object]:
    payload = serialize_album_project(project)
    return {
        "id": payload["id"],
        "name": payload["name"],
        "song_count": payload["song_count"],
        "filled_songs": payload["filled_songs"],
        "created_at": payload["created_at"],
        "updated_at": payload["updated_at"],
        "latest_mix_url": payload["latest_mix_url"],
        "latest_mix_download_name": payload["latest_mix_download_name"],
        "latest_mix_song_count": payload["latest_mix_song_count"],
    }


def iter_file_bytes(path: Path, start: int, end: int, chunk_size: int = 1024 * 256) -> Iterator[bytes]:
    with path.open("rb") as handle:
        handle.seek(start)
        remaining = end - start + 1
        while remaining > 0:
            chunk = handle.read(min(chunk_size, remaining))
            if not chunk:
                break
            remaining -= len(chunk)
            yield chunk


def build_audio_stream_response(path: Path, request: Request) -> Response:
    file_size = path.stat().st_size
    media_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
    common_headers = {
        "Accept-Ranges": "bytes",
        "Cache-Control": "no-store, no-cache, must-revalidate, max-age=0",
        "Pragma": "no-cache",
        "Expires": "0",
        "Content-Disposition": f'inline; filename="{path.name}"',
    }
    range_header = request.headers.get("range", "").strip()
    if not range_header:
        return FileResponse(path, media_type=media_type, headers=common_headers)

    match = re.match(r"bytes=(\d*)-(\d*)$", range_header)
    if not match:
        return Response(status_code=416, headers={**common_headers, "Content-Range": f"bytes */{file_size}"})

    start_text, end_text = match.groups()
    if start_text == "" and end_text == "":
        return Response(status_code=416, headers={**common_headers, "Content-Range": f"bytes */{file_size}"})

    if start_text == "":
        length = int(end_text)
        if length <= 0:
            return Response(status_code=416, headers={**common_headers, "Content-Range": f"bytes */{file_size}"})
        start = max(file_size - length, 0)
        end = file_size - 1
    else:
        start = int(start_text)
        end = int(end_text) if end_text else file_size - 1

    if start < 0 or start >= file_size or end < start:
        return Response(status_code=416, headers={**common_headers, "Content-Range": f"bytes */{file_size}"})

    end = min(end, file_size - 1)
    content_length = end - start + 1
    headers = {
        **common_headers,
        "Content-Range": f"bytes {start}-{end}/{file_size}",
        "Content-Length": str(content_length),
    }
    return StreamingResponse(
        iter_file_bytes(path, start, end),
        status_code=206,
        media_type=media_type,
        headers=headers,
    )


def _select_album_song_version(
    project: Dict[str, object],
    song_index: int,
    version: Optional[int] = None,
) -> tuple[Dict[str, object], Dict[str, object]]:
    songs = project.get("songs", [])
    if not isinstance(songs, list):
        raise HTTPException(status_code=500, detail="Album song metadata is corrupted.")

    target_song: Optional[Dict[str, object]] = None
    for song in songs:
        if isinstance(song, dict) and int(song.get("song_index", 0)) == int(song_index):
            target_song = song
            break
    if target_song is None:
        raise HTTPException(status_code=404, detail="Track not found.")

    versions = target_song.get("versions", [])
    if not isinstance(versions, list) or not versions:
        raise HTTPException(status_code=404, detail="That track has no playable versions yet.")

    selected_version: Optional[Dict[str, object]] = None
    if version is None:
        selected_version = versions[-1]
    else:
        for version_entry in versions:
            if not isinstance(version_entry, dict):
                continue
            if int(version_entry.get("version", -1)) == int(version):
                selected_version = version_entry
                break

    if selected_version is None or not isinstance(selected_version, dict):
        raise HTTPException(status_code=404, detail="Requested version not found.")

    return target_song, selected_version


def start_conversion_job(
    job_id: str,
    model_name: str,
    uploads: List[Path],
    output_format: str,
    settings: Dict[str, object],
) -> None:
    try:
        set_job_state(
            job_id,
            status="running",
            message="Loading model and preparing conversion...",
            total_files=len(uploads),
        )

        job_root = JOBS_ROOT / job_id
        output_dir = job_root / "outputs"
        output_dir.mkdir(parents=True, exist_ok=True)

        for index, input_path in enumerate(uploads, start=1):
            output_mode = str(settings.get("output_mode", "single") or "single").strip().lower()
            secondary_model_name = str(settings.get("secondary_model_name", "") or "").strip()
            blend_percentage = max(0, min(int(settings.get("blend_percentage", 50) or 50), 100))
            set_job_state(
                job_id,
                completed_files=index - 1,
                current_file=input_path.name,
                message=(
                    f"Blending {input_path.name} ({index}/{len(uploads)})..."
                    if output_mode == "blend" and secondary_model_name
                    else f"Converting {input_path.name} ({index}/{len(uploads)})..."
                ),
            )

            output_name = f"{input_path.stem}_converted.{output_format}"
            output_path = output_dir / output_name
            if output_mode == "blend" and secondary_model_name:
                blend_work_dir = job_root / "blend-work" / f"{index:03d}"
                blend_work_dir.mkdir(parents=True, exist_ok=True)
                primary_render_path = blend_work_dir / f"{input_path.stem}_primary.wav"
                secondary_render_path = blend_work_dir / f"{input_path.stem}_secondary.wav"

                primary_metadata = backend.convert_file(
                    model_name,
                    input_path,
                    primary_render_path,
                    preprocess_mode=str(settings["preprocess_mode"]),
                    preprocess_strength=int(settings["preprocess_strength"]),
                    work_dir=blend_work_dir / "prep-primary",
                    speaker_id=int(settings["speaker_id"]),
                    transpose=int(settings["transpose"]),
                    f0_method=str(settings["f0_method"]),
                    index_path=str(settings["index_path"]),
                    index_rate=float(settings["index_rate"]),
                    filter_radius=int(settings["filter_radius"]),
                    resample_sr=int(settings["resample_sr"]),
                    rms_mix_rate=float(settings["rms_mix_rate"]),
                    protect=float(settings["protect"]),
                    crepe_hop_length=int(settings["crepe_hop_length"]),
                )
                secondary_metadata = backend.convert_file(
                    secondary_model_name,
                    input_path,
                    secondary_render_path,
                    preprocess_mode=str(settings["preprocess_mode"]),
                    preprocess_strength=int(settings["preprocess_strength"]),
                    work_dir=blend_work_dir / "prep-secondary",
                    speaker_id=int(settings["speaker_id"]),
                    transpose=int(settings["transpose"]),
                    f0_method=str(settings["f0_method"]),
                    index_path="",
                    index_rate=float(settings["index_rate"]),
                    filter_radius=int(settings["filter_radius"]),
                    resample_sr=int(settings["resample_sr"]),
                    rms_mix_rate=float(settings["rms_mix_rate"]),
                    protect=float(settings["protect"]),
                    crepe_hop_length=int(settings["crepe_hop_length"]),
                )
                sample_rate = blend_audio_outputs(
                    primary_path=primary_render_path,
                    secondary_path=secondary_render_path,
                    output_path=output_path,
                    output_format=output_format,
                    primary_percentage=blend_percentage,
                )
                metadata = {
                    "sample_rate": sample_rate,
                    "index_path": str(primary_metadata.get("index_path", "")),
                    "timings": {
                        "npy": round(
                            float(primary_metadata["timings"]["npy"]) + float(secondary_metadata["timings"]["npy"]),
                            2,
                        ),
                        "f0": round(
                            float(primary_metadata["timings"]["f0"]) + float(secondary_metadata["timings"]["f0"]),
                            2,
                        ),
                        "infer": round(
                            float(primary_metadata["timings"]["infer"]) + float(secondary_metadata["timings"]["infer"]),
                            2,
                        ),
                    },
                    "preprocess_applied": bool(primary_metadata["preprocess_applied"]),
                    "preprocess_mode": str(primary_metadata.get("preprocess_mode", "off")),
                }
            else:
                metadata = backend.convert_file(
                    model_name,
                    input_path,
                    output_path,
                    preprocess_mode=str(settings["preprocess_mode"]),
                    preprocess_strength=int(settings["preprocess_strength"]),
                    work_dir=job_root / "prep",
                    speaker_id=int(settings["speaker_id"]),
                    transpose=int(settings["transpose"]),
                    f0_method=str(settings["f0_method"]),
                    index_path=str(settings["index_path"]),
                    index_rate=float(settings["index_rate"]),
                    filter_radius=int(settings["filter_radius"]),
                    resample_sr=int(settings["resample_sr"]),
                    rms_mix_rate=float(settings["rms_mix_rate"]),
                    protect=float(settings["protect"]),
                    crepe_hop_length=int(settings["crepe_hop_length"]),
                )

            relative_url = (
                f"/downloads/simple-web/jobs/{job_id}/outputs/{output_name}"
            )
            result = JobResult(
                name=input_path.name,
                url=relative_url,
                download_name=output_name,
                sample_rate=int(metadata["sample_rate"]),
                timings=metadata["timings"],
                index_path=str(metadata["index_path"]),
                preprocess_applied=bool(metadata["preprocess_applied"]),
                preprocess_mode=str(metadata.get("preprocess_mode", "off")),
                output_mode=output_mode,
                secondary_model_name=secondary_model_name if output_mode == "blend" else "",
                blend_percentage=blend_percentage if output_mode == "blend" else 100,
            )
            with jobs_lock:
                jobs[job_id].results.append(result)

        zip_path = job_root / f"{job_id}.zip"
        create_zip(zip_path, output_dir)
        set_job_state(
            job_id,
            status="completed",
            completed_files=len(uploads),
            current_file="",
            message="All files finished. Ready to preview or download.",
            zip_url=f"/downloads/simple-web/jobs/{job_id}/{zip_path.name}",
        )
    except Exception:
        set_job_state(
            job_id,
            status="failed",
            message="Conversion stopped because something went wrong.",
            error=traceback.format_exc(),
        )


def start_master_conversion_job(
    job_id: str,
    model_name: str,
    source_path: Path,
    output_format: str,
    settings: Dict[str, object],
) -> None:
    try:
        set_master_conversion_job_state(
            job_id,
            status="running",
            stage="analysis",
            progress=6,
            message="Preparing the full song and reading the lyric guide.",
        )

        job_root = MASTER_CONVERSION_ROOT / job_id
        output_dir = job_root / "outputs"
        output_dir.mkdir(parents=True, exist_ok=True)

        quality_preset = str(settings.get("quality_preset", "balanced") or "balanced")
        master_profile = str(settings.get("master_profile", "studio") or "studio")
        preferred_pipeline = "off"
        blend_mode = str(settings.get("output_mode", "single") or "single").strip().lower() == "blend"
        secondary_model_name = str(settings.get("secondary_model_name", "") or "").strip() if blend_mode else ""
        blend_percentage = max(0, min(int(settings.get("blend_percentage", 50) or 50), 100))

        def progress_stage(progress_value: float) -> str:
            progress_number = float(progress_value)
            if progress_number < 34:
                return "source analysis"
            if progress_number < 64:
                return "voice conversion"
            if progress_number < 94:
                return "pronunciation repair"
            return "final cleanup"

        metadata = master_conversion_engine.run(
            model_name=model_name,
            input_path=source_path,
            lyrics=str(settings["lyrics"]),
            output_dir=output_dir,
            settings=settings,
            quality_preset=quality_preset,
            master_profile=master_profile,
            preferred_pipeline=preferred_pipeline,
            candidate_strength=1,
            output_format=output_format,
            secondary_model_name=secondary_model_name,
            blend_percentage=blend_percentage,
            cancel_event=threading.Event(),
            update_status=lambda payload: set_master_conversion_job_state(
                job_id,
                status="running",
                stage=progress_stage(float(payload.get("progress", 0))),
                progress=int(payload.get("progress", 0)),
                message=str(payload.get("message", "")) or "Master Conversion is running.",
                best_similarity_score=float(payload.get("best_similarity_score", 0.0)),
                best_word_report=str(payload.get("best_word_report", "")),
                best_letter_report=str(payload.get("best_letter_report", "")),
                repair_attempts=int(payload.get("repair_attempts", 0)),
                repaired_word_count=int(payload.get("repaired_word_count", 0)),
            ),
        )

        zip_path = job_root / f"{job_id}.zip"
        create_zip(zip_path, output_dir)

        def artifact_url(file_path: Path) -> str:
            relative = file_path.relative_to(REPO_ROOT / "audio-outputs").as_posix()
            return f"/downloads/{relative}"

        set_master_conversion_job_state(
            job_id,
            status="completed",
            stage="done",
            progress=100,
            message="Master Conversion finished. Review the prepared source, repaired vocal, and final output below.",
            sample_rate=int(metadata["sample_rate"]),
            best_similarity_score=float(metadata.get("best_similarity_score", 0.0)),
            best_word_report=str(metadata.get("best_word_report", "")),
            best_letter_report=str(metadata.get("best_letter_report", "")),
            repair_attempts=int(metadata.get("repair_attempts", 0)),
            repaired_word_count=int(metadata.get("repaired_word_count", 0)),
            candidate_reports=list(metadata.get("candidate_reports", [])),
            phrase_choices=list(metadata.get("phrase_choices", [])),
            final_url=artifact_url(Path(str(metadata["output_path"]))),
            final_download_name=Path(str(metadata["output_path"])).name,
            reconstructed_lead_url=artifact_url(Path(str(metadata["reconstructed_lead_path"]))),
            reconstructed_lead_download_name=Path(str(metadata["reconstructed_lead_path"])).name,
            reconstructed_removed_url=artifact_url(Path(str(metadata["reconstructed_removed_path"]))),
            reconstructed_removed_download_name=Path(str(metadata["reconstructed_removed_path"])).name,
            raw_conversion_url=artifact_url(Path(str(metadata["raw_conversion_path"]))),
            raw_conversion_download_name=Path(str(metadata["raw_conversion_path"])).name,
            repaired_url=artifact_url(Path(str(metadata["repaired_path"]))),
            repaired_download_name=Path(str(metadata["repaired_path"])).name,
            final_removed_url=artifact_url(Path(str(metadata["final_removed_path"]))),
            final_removed_download_name=Path(str(metadata["final_removed_path"])).name,
            metadata_url=artifact_url(Path(str(metadata["metadata_path"]))),
            metadata_download_name=Path(str(metadata["metadata_path"])).name,
            zip_url=artifact_url(zip_path),
            timings=dict(metadata.get("timings", {})),
        )
    except Exception:
        set_master_conversion_job_state(
            job_id,
            status="failed",
            stage="failed",
            message="Master Conversion stopped because something went wrong.",
            error=traceback.format_exc(),
        )


def start_generate_job(
    job_id: str,
    model_name: str,
    guide_path: Path,
    settings: Dict[str, object],
) -> None:
    try:
        set_generate_job_state(
            job_id,
            status="running",
            progress=8,
            message="Preparing the reference vocal and saving lyrics...",
        )

        job_root = GENERATE_ROOT / job_id
        output_dir = job_root / "outputs"
        output_dir.mkdir(parents=True, exist_ok=True)

        lyrics_text = str(settings.get("lyrics", "")).strip()
        lyrics_path = job_root / "guide_lyrics.txt"
        lyrics_path.write_text(lyrics_text, encoding="utf-8")

        transpose = compute_generate_transpose(
            str(settings.get("guide_key", "")),
            str(settings.get("target_key", "")),
        )
        quality_preset = str(settings.get("quality_preset", "balanced"))
        repair_profile = GENERATE_REPAIR_PROFILES.get(
            quality_preset,
            GENERATE_REPAIR_PROFILES["balanced"],
        )
        set_generate_job_state(
            job_id,
            progress=18,
            transpose=transpose,
            repair_strength=int(repair_profile["strength"]),
            message="Converting the reference performance into the selected voice...",
        )

        raw_output_path = output_dir / f"{guide_path.stem}_generated_raw.wav"
        metadata = backend.convert_file(
            model_name,
            guide_path,
            raw_output_path,
            preprocess_mode=str(settings["preprocess_mode"]),
            preprocess_strength=int(settings["preprocess_strength"]),
            work_dir=job_root / "prep",
            speaker_id=int(settings["speaker_id"]),
            transpose=transpose,
            f0_method=str(settings["f0_method"]),
            index_path=str(settings["index_path"]),
            index_rate=float(settings["index_rate"]),
            filter_radius=int(settings["filter_radius"]),
            resample_sr=int(settings["resample_sr"]),
            rms_mix_rate=float(settings["rms_mix_rate"]),
            protect=float(settings["protect"]),
            crepe_hop_length=int(settings["crepe_hop_length"]),
        )

        set_generate_job_state(
            job_id,
            progress=40,
            message="Scoring words and repairing weak pronunciation regions...",
        )
        repair_output_dir = output_dir / "repair"
        repair_output_dir.mkdir(parents=True, exist_ok=True)
        repair_metadata = touchup_engine.optimize_pronunciation(
            source_path=raw_output_path,
            intended_lyrics=lyrics_text,
            output_dir=repair_output_dir,
            strength=int(repair_profile["strength"]),
            variants_per_batch=int(repair_profile["variants_per_batch"]),
            parallel_variants=1,
            max_batches=int(repair_profile["max_batches"]),
            max_target_words=int(repair_profile["max_target_words"]),
            cancel_event=threading.Event(),
            mode="repair",
            update_status=lambda payload: set_generate_job_state(
                job_id,
                progress=min(
                    86,
                    40 + int(round(max(0.0, min(float(payload.get("progress", 0.0)), 100.0)) * 0.46)),
                ),
                message=str(payload.get("message", "")) or "Repairing pronunciation...",
                repair_attempts=int(payload.get("repair_attempts", payload.get("variants_tested", 0))),
                repaired_word_count=int(payload.get("repaired_word_count", 0)),
                best_similarity_score=float(payload.get("best_similarity_score", 0.0)),
                best_word_report=str(payload.get("best_word_report", "")),
                best_letter_report=str(payload.get("best_letter_report", "")),
                detected_word_indices=list(payload.get("detected_word_indices", [])),
                regeneration_available=bool(payload.get("regeneration_available", False)),
                regeneration_reason=str(payload.get("regeneration_reason", "")),
            ),
        )
        repaired_output_path = Path(str(repair_metadata["output_path"]))

        set_generate_job_state(
            job_id,
            progress=72,
            message="Re-rendering the weakest lyric regions from the reference vocal...",
        )

        def render_reference_phrase_patch(
            reference_segment_path: Path,
            rendered_patch_path: Path,
            phrase_text: str,
            patch_info: Dict[str, object],
        ) -> Path:
            group_index = int(patch_info.get("group_index", 0))
            patch_work_dir = job_root / "reference-phrase-patch" / f"group_{group_index:02d}"
            patch_work_dir.mkdir(parents=True, exist_ok=True)
            patch_output_path = Path(rendered_patch_path)
            backend.convert_file(
                model_name,
                Path(reference_segment_path),
                patch_output_path,
                preprocess_mode="off",
                preprocess_strength=1,
                work_dir=patch_work_dir,
                speaker_id=int(settings["speaker_id"]),
                transpose=transpose,
                f0_method="rmvpe",
                index_path=str(settings["index_path"]),
                index_rate=(
                    min(float(settings["index_rate"]), 0.05)
                    if str(settings["index_path"]).strip()
                    else 0.0
                ),
                filter_radius=max(5, int(settings["filter_radius"])),
                resample_sr=0,
                rms_mix_rate=min(float(settings["rms_mix_rate"]), 0.18),
                protect=min(float(settings["protect"]), 0.12),
                crepe_hop_length=int(settings["crepe_hop_length"]),
            )
            return patch_output_path

        reference_phrase_metadata = touchup_engine.repair_with_reference_phrase_patches(
            source_path=repaired_output_path,
            reference_path=guide_path,
            intended_lyrics=lyrics_text,
            output_dir=output_dir / "reference-phrase-repair",
            cancel_event=threading.Event(),
            max_target_words=int(repair_profile["max_target_words"]),
            patch_renderer=render_reference_phrase_patch,
            update_status=lambda payload: set_generate_job_state(
                job_id,
                progress=min(
                    94,
                    72 + int(round(max(0.0, min(float(payload.get("progress", 0.0)), 100.0)) * 0.22)),
                ),
                message=str(payload.get("message", "")) or "Repairing weak lyric regions...",
                repair_attempts=int(repair_metadata.get("repair_attempts", 0))
                + int(payload.get("repair_attempts", payload.get("variants_tested", 0))),
                repaired_word_count=max(
                    int(repair_metadata.get("repaired_word_count", 0)),
                    int(payload.get("repaired_word_count", 0)),
                ),
                best_similarity_score=float(payload.get("best_similarity_score", 0.0)),
                best_word_report=str(payload.get("best_word_report", "")),
                best_letter_report=str(payload.get("best_letter_report", "")),
                detected_word_indices=list(payload.get("detected_word_indices", [])),
                regeneration_available=bool(repair_metadata.get("regeneration_available", False)),
                regeneration_reason=str(repair_metadata.get("regeneration_reason", "")),
            ),
            blend_strength=0.86 if quality_preset == "clean" else 0.82,
            padding_ms=95.0,
        )
        repaired_output_path = Path(str(reference_phrase_metadata["output_path"]))

        final_output_path = output_dir / f"{guide_path.stem}_generated.wav"
        guide_bpm = float(settings.get("guide_bpm", 0.0) or 0.0)
        target_bpm = float(settings.get("target_bpm", 0.0) or 0.0)
        tempo_ratio = 1.0
        tempo_adjusted = False
        if guide_bpm > 0 and target_bpm > 0 and abs(guide_bpm - target_bpm) >= 0.01:
            set_generate_job_state(
                job_id,
                progress=82,
                message="Matching the repaired vocal to the target BPM...",
            )
            tempo_ratio = backend.conform_audio_tempo(
                repaired_output_path,
                final_output_path,
                source_bpm=guide_bpm,
                target_bpm=target_bpm,
            )
            tempo_adjusted = True
        else:
            shutil.copy2(repaired_output_path, final_output_path)

        metadata_download_name = f"{guide_path.stem}_generation_metadata.json"
        metadata_path = output_dir / metadata_download_name
        metadata_payload = {
            "job_id": job_id,
            "model_name": model_name,
            "guide_name": guide_path.name,
            "lyrics": lyrics_text,
            "guide_key": str(settings.get("guide_key", "")),
            "target_key": str(settings.get("target_key", "")),
            "guide_bpm": guide_bpm,
            "target_bpm": target_bpm,
            "transpose": transpose,
            "tempo_ratio": tempo_ratio,
            "tempo_adjusted": tempo_adjusted,
            "quality_preset": quality_preset,
            "preprocess_mode": str(settings.get("preprocess_mode", "off")),
            "repair_mode": "pronunciation-repair",
            "repair_strength": int(repair_profile["strength"]),
            "repair_attempts": int(repair_metadata.get("repair_attempts", 0))
            + int(reference_phrase_metadata.get("repair_attempts", 0)),
            "repaired_word_count": max(
                int(repair_metadata.get("repaired_word_count", 0)),
                int(reference_phrase_metadata.get("repaired_word_count", 0)),
            ),
            "best_similarity_score": float(reference_phrase_metadata.get("best_similarity_score", 0.0)),
            "best_word_report": str(reference_phrase_metadata.get("best_word_report", "")),
            "best_letter_report": str(reference_phrase_metadata.get("best_letter_report", "")),
            "detected_word_indices": list(reference_phrase_metadata.get("detected_word_indices", [])),
            "regeneration_available": bool(repair_metadata.get("regeneration_available", False)),
            "regeneration_reason": str(repair_metadata.get("regeneration_reason", "")),
            "reference_phrase_patch_attempts": int(reference_phrase_metadata.get("repair_attempts", 0)),
            "repair_source_name": raw_output_path.name,
            "sample_rate": int(metadata.get("sample_rate", 0)),
            "timings": metadata.get("timings", {}),
        }
        metadata_path.write_text(json.dumps(metadata_payload, indent=2), encoding="utf-8")

        set_generate_job_state(
            job_id,
            status="completed",
            progress=100,
            message="Pronunciation repair finished. Your cleaned vocal stem is ready.",
            sample_rate=int(metadata["sample_rate"]),
            preprocess_mode=str(metadata.get("preprocess_mode", "off")),
            repair_mode="pronunciation-repair",
            repair_strength=int(repair_profile["strength"]),
            repair_attempts=int(repair_metadata.get("repair_attempts", 0))
            + int(reference_phrase_metadata.get("repair_attempts", 0)),
            repaired_word_count=max(
                int(repair_metadata.get("repaired_word_count", 0)),
                int(reference_phrase_metadata.get("repaired_word_count", 0)),
            ),
            best_similarity_score=float(reference_phrase_metadata.get("best_similarity_score", 0.0)),
            best_word_report=str(reference_phrase_metadata.get("best_word_report", "")),
            best_letter_report=str(reference_phrase_metadata.get("best_letter_report", "")),
            detected_word_indices=list(reference_phrase_metadata.get("detected_word_indices", [])),
            regeneration_available=bool(repair_metadata.get("regeneration_available", False)),
            regeneration_reason=str(repair_metadata.get("regeneration_reason", "")),
            result_url=f"/downloads/simple-web/generate/{job_id}/outputs/{final_output_path.name}",
            download_name=final_output_path.name,
            repair_source_url=f"/downloads/simple-web/generate/{job_id}/outputs/{raw_output_path.name}",
            repair_source_download_name=raw_output_path.name,
            metadata_url=f"/downloads/simple-web/generate/{job_id}/outputs/{metadata_download_name}",
            metadata_download_name=metadata_download_name,
            timings=metadata.get("timings", {}),
            tempo_ratio=float(tempo_ratio),
            tempo_adjusted=bool(tempo_adjusted),
        )
    except Exception:
        set_generate_job_state(
            job_id,
            status="failed",
            progress=0,
            message="Generate stopped because something went wrong.",
            error=traceback.format_exc(),
        )


def start_training_job(
    job_id: str,
    experiment_name: str,
    dataset_dir: Path,
    transcript_dir: Path,
    plan_dir: Path,
    job_root: Path,
    settings: Dict[str, object],
) -> None:
    with training_stop_events_lock:
        cancel_event = training_stop_events.setdefault(job_id, threading.Event())
    requested_output_mode = str(
        settings.get("output_mode", "persona-v1") or "persona-v1"
    )
    output_mode = (
        "pipa-logic-only"
        if requested_output_mode in {"persona-v1", "pipa-logic-only"}
        else requested_output_mode
    )
    resume_selection_name = str(settings.get("resume_selection_name", "") or "").strip()
    start_phase = str(settings.get("start_phase", "auto") or "auto").strip()
    guided_total_epochs = max(
        1,
        int(
            settings.get(
                "guided_regeneration_epochs",
                settings.get("requested_total_epochs", settings.get("total_epochs", 200)),
            )
        ),
    )

    def map_guided_training_progress(raw_progress: int) -> int:
        normalized = max(0, min(int(raw_progress), 100))
        start = 32
        end = 94 if output_mode == "pipa-logic-only" else 66
        return int(round(start + ((normalized / 100.0) * (end - start))))
    try:
        set_training_job_state(
            job_id,
            status="running",
            stage="queued",
            message="Preparing the training pipeline...",
            progress=4,
        )

        audio_paths = [
            path
            for path in sorted(dataset_dir.iterdir())
            if path.is_file()
        ]
        transcript_paths = [
            path
            for path in sorted(transcript_dir.iterdir())
            if path.is_file()
        ]
        plan_paths = [
            path
            for path in sorted(plan_dir.iterdir())
            if path.is_file()
        ]
        resume_bundle = (
            backend.pipa_store.resolve_bundle(resume_selection_name)
            if resume_selection_name
            else None
        )
        if resume_selection_name and resume_bundle is None:
            raise FileNotFoundError(
                f"Resume package '{resume_selection_name}' was not found on this instance."
            )
        resume_checkpoint_path = None
        resume_report_path = None
        if resume_bundle is not None:
            latest_checkpoint = Path(
                str(
                    resume_bundle.get("guided_regeneration_latest_path", "")
                    or resume_bundle.get("guided_regeneration_path", "")
                    or ""
                )
            )
            if latest_checkpoint.exists():
                resume_checkpoint_path = latest_checkpoint
            else:
                raise FileNotFoundError(
                    f"Resume checkpoint for '{resume_selection_name}' is missing."
                )
            raw_resume_report = str(resume_bundle.get("guided_regeneration_report_path", "") or "").strip()
            if raw_resume_report:
                candidate_report = Path(raw_resume_report)
                if candidate_report.exists():
                    resume_report_path = candidate_report
        pipa_build_dir = job_root / "pipa-build"
        reset_directory(pipa_build_dir)
        prep_metadata = backend.pipa_store.prepare_training_assets(
            package_id=experiment_name,
            audio_paths=audio_paths,
            transcript_paths=transcript_paths,
            plan_paths=plan_paths,
            output_dir=pipa_build_dir,
            alignment_tolerance=str(settings["alignment_tolerance"]),
            update_status=lambda stage, message, log_tail, progress: set_training_job_state(
                job_id,
                stage=stage,
                message=message,
                log_tail=log_tail,
                progress=max(6, min(int(progress), 30)),
            ),
            cancel_event=cancel_event,
        )

        guided_regeneration_dir = pipa_build_dir / "guided_regeneration"
        guided_metadata = backend.pipa_store.train_guided_regenerator(
            dataset_dir=Path(str(prep_metadata["guided_svs_dataset_dir"])),
            output_dir=guided_regeneration_dir,
            total_epochs=guided_total_epochs,
            save_every_epoch=int(settings["save_every_epoch"]),
            batch_size=int(settings["batch_size"]),
            update_status=lambda stage, message, log_tail, progress: set_training_job_state(
                job_id,
                stage=stage,
                message=message,
                log_tail=log_tail,
                progress=map_guided_training_progress(progress),
            ),
            cancel_event=cancel_event,
            resume_checkpoint_path=resume_checkpoint_path,
            resume_report_path=resume_report_path,
            start_phase=start_phase,
        )

        guided_dataset_features_dir = Path(str(prep_metadata["guided_svs_dataset_dir"])) / "features"
        if guided_dataset_features_dir.exists():
            shutil.rmtree(guided_dataset_features_dir, ignore_errors=True)

        skip_backbone = output_mode == "pipa-logic-only" or cancel_event.is_set()
        if skip_backbone:
            set_training_job_state(
                job_id,
                stage="logic-only",
                message=(
                    "Skipping all legacy backbone stages and packaging the Persona v1.0 voice-builder assets."
                    if output_mode == "pipa-logic-only"
                    else "Stop requested before any legacy backbone stage. Packaging the Persona v1.0 voice-builder assets now."
                ),
                progress=94,
                log_tail=(
                    f"Best epoch {int(guided_metadata.get('best_epoch', 0))} | "
                    f"best total {float(guided_metadata.get('best_val_total', 0.0)):.4f} | "
                    f"lyric phone acc {float(guided_metadata.get('best_lyric_phone_accuracy', 0.0)) * 100.0:.1f}% | "
                    f"plateau {int(guided_metadata.get('plateau_epochs', 0))} epochs | "
                    "No legacy .pth or index assets will be created for this package."
                ),
            )
            metadata = {
                "experiment_name": experiment_name,
                "model_path": "",
                "index_path": "",
                "index_built": "0",
                "index_summary": "",
                "train_log_path": "",
                "stopped_early": "1" if cancel_event.is_set() or bool(guided_metadata.get("stopped_early", False)) else "0",
            }
        else:
            metadata = trainer.run_training(
                experiment_name=experiment_name,
                dataset_dir=dataset_dir,
                sample_rate=str(settings["sample_rate"]),
                version=str(settings["version"]),
                f0_method=str(settings["f0_method"]),
                total_epochs=int(settings["total_epochs"]),
                save_every_epoch=int(settings["save_every_epoch"]),
                batch_size=int(settings["batch_size"]),
                crepe_hop_length=int(settings["crepe_hop_length"]),
                build_index=bool(settings["build_index"]),
                update_status=lambda stage, message, log_tail, progress: set_training_job_state(
                    job_id,
                    stage=stage,
                    message=message,
                    log_tail=log_tail,
                    progress=progress,
                ),
                cancel_event=cancel_event,
            )
        set_training_job_state(
            job_id,
            stage="package",
            message="Packaging the Persona v1.0 bundle and writing the final profile files...",
            progress=96 if output_mode == "pipa-logic-only" else 98,
        )
        package_metadata = backend.pipa_store.finalize_training_package(
            package_id=experiment_name,
            build_dir=pipa_build_dir,
            label=experiment_name.replace("_", " "),
            model_path=str(metadata["model_path"]),
            index_path=str(metadata["index_path"]),
            settings=settings,
            prep_metadata=prep_metadata,
            regeneration_metadata=guided_metadata,
        )
        stopped_early = bool(int(str(metadata.get("stopped_early", "0") or "0"))) or bool(
            guided_metadata.get("stopped_early", False)
        )
        guided_hardware_summary = str(guided_metadata.get("hardware_summary", "")).strip()

        set_training_job_state(
            job_id,
            status="completed",
            stage="done",
            message=(
                "Persona v1.0 build finished. The guide-conditioned voice-builder package is ready."
                if output_mode == "pipa-logic-only"
                else (
                    "Stopped cleanly. The latest saved assets were packaged into a usable Persona v1.0 bundle."
                    if stopped_early
                    else (
                        "Training finished. Your package now includes the guide-conditioned voice-builder regenerator, neural waveform decoder, and legacy backbone assets."
                        if bool(settings["build_index"])
                        else "Training finished. Your package now includes the guide-conditioned voice-builder regenerator and neural waveform decoder without an index."
                    )
                )
            ),
            progress=100,
            log_tail=(
                f"Best epoch {int(guided_metadata.get('best_epoch', 0))} | "
                f"best total {float(guided_metadata.get('best_val_total', 0.0)):.4f} | "
                f"lyric phone acc {float(guided_metadata.get('best_lyric_phone_accuracy', 0.0)) * 100.0:.1f}% | "
                f"voicing {float(guided_metadata.get('best_vuv_accuracy', 0.0)) * 100.0:.1f}% | "
                f"plateau {int(guided_metadata.get('plateau_epochs', 0))} epochs | "
                f"render {str(guided_metadata.get('render_mode', 'griffinlim_preview_only'))}"
                f"{f' | {guided_hardware_summary}' if guided_hardware_summary else ''}"
            ),
            stop_requested=cancel_event.is_set(),
            stopped_early=stopped_early,
            model_path=str(metadata["model_path"]),
            index_path=str(metadata["index_path"]),
            pipa_selection_name=str(package_metadata["selection_name"]),
            pipa_manifest_path=str(package_metadata["manifest_path"]),
            phoneme_profile_path=str(package_metadata["phoneme_profile_path"]),
            rebuild_profile_path=str(package_metadata.get("rebuild_profile_path", "")),
            rebuild_clip_reports_path=str(package_metadata.get("rebuild_clip_reports_path", "")),
            training_report_path=str(package_metadata["training_report_path"]),
            reference_bank_index_path=str(package_metadata["reference_bank_index_path"]),
            guided_regeneration_path=str(package_metadata.get("guided_regeneration_path", "")),
            guided_regeneration_config_path=str(package_metadata.get("guided_regeneration_config_path", "")),
            guided_regeneration_report_path=str(package_metadata.get("guided_regeneration_report_path", "")),
            guided_regeneration_preview_path=str(package_metadata.get("guided_regeneration_preview_path", "")),
            guided_regeneration_target_preview_path=str(package_metadata.get("guided_regeneration_target_preview_path", "")),
            guided_regeneration_best_val_l1=float(guided_metadata.get("best_val_l1", 0.0)),
            guided_regeneration_best_val_total=float(guided_metadata.get("best_val_total", 0.0)),
            guided_regeneration_best_epoch=int(guided_metadata.get("best_epoch", 0)),
            guided_regeneration_best_phone_accuracy=float(guided_metadata.get("best_phone_accuracy", 0.0)),
            guided_regeneration_best_lyric_phone_accuracy=float(guided_metadata.get("best_lyric_phone_accuracy", 0.0)),
            guided_regeneration_best_vuv_accuracy=float(guided_metadata.get("best_vuv_accuracy", 0.0)),
            guided_regeneration_plateau_epochs=int(guided_metadata.get("plateau_epochs", 0)),
            guided_regeneration_hardware_summary=str(guided_metadata.get("hardware_summary", "")),
            guided_regeneration_quality_summary=str(guided_metadata.get("quality_summary", "")),
            guided_regeneration_last_epoch=int(guided_metadata.get("last_epoch", 0)),
            guided_regeneration_sample_count=int(guided_metadata.get("sample_count", 0)),
            transcript_manifest_path=str(prep_metadata["transcript_manifest_path"]),
            alignment_tolerance=str(prep_metadata["alignment_tolerance"]),
            phoneme_mode=str(prep_metadata["phoneme_mode"]),
            matched_audio_files=int(prep_metadata["matched_audio_files"]),
            total_audio_files=int(prep_metadata["total_audio_files"]),
            skipped_audio_files=int(prep_metadata["skipped_audio_files"]),
            reference_word_count=int(prep_metadata["reference_word_count"]),
            reference_phrase_count=int(prep_metadata["reference_phrase_count"]),
            training_plan_path=str(prep_metadata.get("training_plan_path", "")),
            base_voice_clip_count=int(prep_metadata.get("base_voice_clip_count", 0)),
            paired_song_count=int(prep_metadata.get("paired_song_count", 0)),
            depersonafied_variant_count=int(prep_metadata.get("depersonafied_variant_count", 0)),
        )
    except InterruptedError as exc:
        set_training_job_state(
            job_id,
            status="stopped",
            stage="stopped",
            stop_requested=True,
            message=str(exc) or "Training stopped by user.",
            error="",
        )
    except Exception:
        set_training_job_state(
            job_id,
            status="failed",
            stage="failed",
            message="Training stopped because something went wrong.",
            error=traceback.format_exc(),
        )
    finally:
        with training_stop_events_lock:
            training_stop_events.pop(job_id, None)


def start_detag_job(
    job_id: str,
    voice_id: str,
    input_path: Path,
    strength: int,
) -> None:
    try:
        set_detag_job_state(
            job_id,
            status="running",
            message="Loading the selected voice profile...",
            progress=6,
        )

        output_dir = DETAG_ROOT / job_id / "outputs"
        output_dir.mkdir(parents=True, exist_ok=True)
        output_name = f"{input_path.stem}_detagged.wav"
        output_path = output_dir / output_name

        metadata = detagger.detag_file(
            voice_id=voice_id,
            input_path=input_path,
            output_path=output_path,
            strength=strength,
            update_progress=lambda message, progress: set_detag_job_state(
                job_id,
                message=message,
                progress=progress,
            ),
        )

        set_detag_job_state(
            job_id,
            status="completed",
            message="Done. Only the selected voice was kept in the output.",
            progress=100,
            result_url=f"/downloads/simple-web/detag/{job_id}/outputs/{output_name}",
            download_name=output_name,
            threshold=float(metadata["threshold"]),
            kept_ratio=float(metadata["kept_ratio"]),
        )
    except Exception:
        set_detag_job_state(
            job_id,
            status="failed",
            message="Detagging stopped because something went wrong.",
            error=traceback.format_exc(),
        )


def start_isolator_job(
    job_id: str,
    input_paths: List[Path],
    mode: str,
    input_type: str,
    strength: int,
    deecho: bool,
    width_focus: bool,
    clarity_preserve: int,
) -> None:
    try:
        set_isolator_job_state(
            job_id,
            status="running",
            message="Preparing the vocal isolation pass...",
            progress=6,
        )

        output_dir = ISOLATOR_ROOT / job_id / "outputs"
        output_dir.mkdir(parents=True, exist_ok=True)
        isolated_main_paths: List[Path] = []
        isolated_backing_paths: List[Path] = []
        final_metadata: Optional[Dict[str, object]] = None
        total_files = max(len(input_paths), 1)

        for index, input_path in enumerate(input_paths, start=1):
            file_work_dir = ISOLATOR_ROOT / job_id / "per-file" / f"{index:03d}"
            file_progress_start = 8 + int(round(((index - 1) / total_files) * 76))
            file_progress_end = 8 + int(round((index / total_files) * 76))

            set_isolator_job_state(
                job_id,
                message=f"Starting isolation for file {index}/{total_files}...",
                progress=file_progress_start,
                current_file=input_path.name,
            )

            metadata = backend.isolate_vocals(
                input_path=input_path,
                output_dir=file_work_dir,
                mode=mode,
                input_type=input_type,
                strength=strength,
                deecho=deecho,
                width_focus=width_focus,
                clarity_preserve=clarity_preserve,
                update_progress=lambda message, progress, current_index=index, current_name=input_path.name, start=file_progress_start, end=file_progress_end: set_isolator_job_state(
                    job_id,
                    message=f"{message} ({current_index}/{total_files})",
                    progress=min(
                        end,
                        start + int(round((max(0, min(progress, 100)) / 100.0) * max(1, end - start))),
                    ),
                    current_file=current_name,
                ),
            )
            isolated_main_paths.append(Path(str(metadata["main_vocal_path"])))
            isolated_backing_paths.append(Path(str(metadata["backing_vocal_path"])))
            final_metadata = metadata

        if final_metadata is None:
            raise RuntimeError("No files were isolated.")

        set_isolator_job_state(
            job_id,
            message="Combining the isolated main stems into one file...",
            progress=88,
            current_file="main vocal output",
        )
        main_output_path = output_dir / "main_vocal.wav"
        combined_main_path = combine_audio_files(isolated_main_paths, main_output_path)
        if combined_main_path != main_output_path:
            prepare_audio_for_concat(combined_main_path, main_output_path)

        set_isolator_job_state(
            job_id,
            message="Combining the isolated backing stems into one file...",
            progress=94,
            current_file="backing vocal output",
        )
        backing_output_path = output_dir / "backing_vocal.wav"
        combined_backing_path = combine_audio_files(isolated_backing_paths, backing_output_path)
        if combined_backing_path != backing_output_path:
            prepare_audio_for_concat(combined_backing_path, backing_output_path)

        main_name = main_output_path.name
        backing_name = backing_output_path.name
        set_isolator_job_state(
            job_id,
            status="completed",
            message="Done. Each file was isolated first, then the stems were combined.",
            progress=100,
            mode=str(final_metadata["mode"]),
            input_type=str(final_metadata["input_type"]),
            strength=int(final_metadata["strength"]),
            deecho=bool(final_metadata["deecho"]),
            width_focus=bool(final_metadata["width_focus"]),
            clarity_preserve=int(final_metadata["clarity_preserve"]),
            sample_rate=int(final_metadata["sample_rate"]),
            main_vocal_url=f"/downloads/simple-web/isolator/{job_id}/outputs/{main_name}",
            main_vocal_download_name=main_name,
            backing_vocal_url=f"/downloads/simple-web/isolator/{job_id}/outputs/{backing_name}",
            backing_vocal_download_name=backing_name,
            current_file="combined outputs",
        )
    except Exception:
        set_isolator_job_state(
            job_id,
            status="failed",
            message="Isolation stopped because something went wrong.",
            error=traceback.format_exc(),
        )


def start_mastering_job(
    job_id: str,
    source_path: Path,
    reference_paths: List[Path],
    resolution: int,
) -> None:
    try:
        set_mastering_job_state(
            job_id,
            status="running",
            message="Analyzing both tracks and learning the EQ profile...",
            progress=10,
        )

        output_dir = MASTERING_ROOT / job_id / "outputs"
        output_dir.mkdir(parents=True, exist_ok=True)

        set_mastering_job_state(
            job_id,
            message=(
                "Matching the source EQ curve to the blended mastered reference profile..."
                if len(reference_paths) > 1
                else "Matching the source EQ curve to the mastered reference..."
            ),
            progress=48,
        )
        metadata = mastering_engine.match_reference_eq(
            source_path=source_path,
            reference_paths=reference_paths,
            output_dir=output_dir,
            resolution=resolution,
        )

        mastered_name = Path(str(metadata["mastered_path"])).name
        profile_name = Path(str(metadata["profile_path"])).name
        set_mastering_job_state(
            job_id,
            status="completed",
            message="Done. The EQ profile and matched master are ready.",
            progress=100,
            resolution=int(metadata["resolution"]),
            sample_rate=int(metadata["sample_rate"]),
            mastered_url=f"/downloads/simple-web/mastering/{job_id}/outputs/{mastered_name}",
            mastered_download_name=mastered_name,
            profile_url=f"/downloads/simple-web/mastering/{job_id}/outputs/{profile_name}",
            profile_download_name=profile_name,
            curve_points=list(metadata["curve_points"]),
            reference_count=int(metadata["reference_count"]),
            reference_files=list(metadata["reference_files"]),
            source_rms_db=float(metadata["source_rms_db"]),
            reference_rms_db=float(metadata["reference_rms_db"]),
            loudness_gain_db=float(metadata["loudness_gain_db"]),
            band_summary=dict(metadata["band_summary"]),
        )
    except Exception:
        set_mastering_job_state(
            job_id,
            status="failed",
            message="Mastering stopped because something went wrong.",
            error=traceback.format_exc(),
        )


def start_optimize_job(
    job_id: str,
    input_paths: List[Path],
    lyrics: str,
    max_cut_db: float,
) -> None:
    try:
        set_optimize_job_state(
            job_id,
            status="running",
            message="Analyzing each take against your lyrics...",
            progress=6,
            total_files=len(input_paths),
            processed_files=0,
        )

        outputs_dir = OPTIMIZE_ROOT / job_id / "outputs"
        outputs_dir.mkdir(parents=True, exist_ok=True)
        rankings: List[Dict[str, object]] = []
        total_files = max(len(input_paths), 1)
        prepared_paths: List[Path] = []
        analyses: List[Dict[str, object]] = []

        for index, input_path in enumerate(input_paths, start=1):
            start_progress = 8 + int(round(((index - 1) / total_files) * 58))
            end_progress = 8 + int(round((index / total_files) * 58))
            set_optimize_job_state(
                job_id,
                message=f"Scoring take {index}/{total_files} against lyrics...",
                current_file=input_path.name,
                progress=start_progress,
                processed_files=index - 1,
            )

            prepared_path = prepare_audio_for_concat(
                input_path,
                outputs_dir / f"{index:02d}_{sanitize_filename(input_path.stem)}_prepared.wav",
            )
            prepared_paths.append(prepared_path)
            analysis = optimizer_engine.analyze_candidate(prepared_path, lyrics)
            analyses.append(analysis)
            prepared_name = prepared_path.name
            weak_words = list(analysis.get("weak_words", []))[:5]
            weak_word_text = ", ".join(
                f"{entry.get('word', '')} ({float(entry.get('similarity', 0.0)):.0f}%)"
                for entry in weak_words
            )
            if not weak_word_text:
                weak_word_text = "No obvious weak words were found."

            rankings.append(
                {
                    "rank": index,
                    "source_name": input_path.name,
                    "score": float(analysis.get("score", 0.0)),
                    "summary": str(analysis.get("summary", "")),
                    "issues": list(analysis.get("issues", [])),
                    "duration_seconds": float(analysis.get("duration_seconds", 0.0)),
                    "weak_words": list(analysis.get("weak_words", [])),
                    "weak_words_summary": weak_word_text,
                    "prepared_vocal_url": f"/downloads/simple-web/optimize/{job_id}/outputs/{prepared_name}",
                    "prepared_vocal_download_name": prepared_name,
                }
            )

            set_optimize_job_state(
                job_id,
                message=f"Analyzed {input_path.name}.",
                progress=end_progress,
                processed_files=index,
                current_file=input_path.name,
            )

        rankings_sorted = sorted(rankings, key=lambda item: float(item["score"]), reverse=True)
        for rank, entry in enumerate(rankings_sorted, start=1):
            entry["rank"] = rank

        set_optimize_job_state(
            job_id,
            message="Stitching strongest lyric regions across all takes...",
            progress=78,
            current_file="",
        )
        stitched_output_name = "stitched_best_acapella.wav"
        stitched_output_path = outputs_dir / stitched_output_name
        stitched_metadata = optimizer_engine.stitch_best_parts(
            analyses=analyses,
            lyrics=lyrics,
            output_path=stitched_output_path,
            max_cut_db=max_cut_db,
        )

        set_optimize_job_state(
            job_id,
            status="completed",
            message="Done. Best lyric regions were stitched into one acapella.",
            progress=100,
            processed_files=len(input_paths),
            current_file="",
            stitched_url=f"/downloads/simple-web/optimize/{job_id}/outputs/{stitched_output_name}",
            stitched_download_name=stitched_output_name,
            anchor_source_name=str(stitched_metadata.get("anchor_source_name", "")),
            replaced_word_count=int(stitched_metadata.get("replaced_word_count", 0)),
            total_word_count=int(stitched_metadata.get("total_word_count", 0)),
            max_cut_db=float(stitched_metadata.get("max_cut_db", max_cut_db)),
            skipped_by_db_gate=int(stitched_metadata.get("skipped_by_db_gate", 0)),
            edits_preview=list(stitched_metadata.get("edits_preview", [])),
            rankings=rankings_sorted,
        )
    except Exception:
        set_optimize_job_state(
            job_id,
            status="failed",
            message="Optimization stopped because something went wrong.",
            error=traceback.format_exc(),
        )


def start_api_compose_job(
    job_id: str,
    endpoint_url: str,
    api_key: str,
    auth_header: str,
    lyrics: str,
    midi_path: Path,
    beat_path: Optional[Path],
    extra_json: str,
) -> None:
    release_task_url = ""
    try:
        set_api_compose_job_state(
            job_id,
            status="running",
            message="Preparing ACE-Step local request...",
            progress=12,
        )

        def _to_bool(value: object, default: bool) -> bool:
            if value is None:
                return default
            if isinstance(value, bool):
                return value
            text = str(value).strip().lower()
            if text in {"1", "true", "yes", "y", "on"}:
                return True
            if text in {"0", "false", "no", "n", "off"}:
                return False
            return default

        def _normalize_base_url(raw_url: str) -> str:
            base = (raw_url or "").strip().rstrip("/")
            if base.endswith("/release_task"):
                base = base[: -len("/release_task")]
            if base.endswith("/query_result"):
                base = base[: -len("/query_result")]
            return base

        def _encode_multipart(
            fields: Dict[str, str],
            files: List[tuple[str, str, bytes, str]],
        ) -> tuple[bytes, str]:
            boundary = f"----AceStepBoundary{uuid4().hex}"
            chunks: List[bytes] = []
            for key, value in fields.items():
                chunks.append(f"--{boundary}\r\n".encode("utf-8"))
                chunks.append(
                    f'Content-Disposition: form-data; name="{key}"\r\n\r\n'.encode("utf-8")
                )
                chunks.append(str(value).encode("utf-8"))
                chunks.append(b"\r\n")
            for field_name, file_name, file_bytes, content_type in files:
                chunks.append(f"--{boundary}\r\n".encode("utf-8"))
                chunks.append(
                    (
                        f'Content-Disposition: form-data; name="{field_name}"; '
                        f'filename="{file_name}"\r\n'
                    ).encode("utf-8")
                )
                chunks.append(f"Content-Type: {content_type}\r\n\r\n".encode("utf-8"))
                chunks.append(file_bytes)
                chunks.append(b"\r\n")
            chunks.append(f"--{boundary}--\r\n".encode("utf-8"))
            return b"".join(chunks), f"multipart/form-data; boundary={boundary}"

        extras: Dict[str, object] = {}
        if extra_json.strip():
            parsed = json.loads(extra_json)
            if not isinstance(parsed, dict):
                raise RuntimeError("Extra JSON must be an object.")
            extras = parsed

        base_url = _normalize_base_url(endpoint_url)
        if not base_url.startswith(("http://", "https://")):
            raise RuntimeError("ACE-Step URL must start with http:// or https://")
        release_task_url = f"{base_url}/release_task"
        query_result_url = f"{base_url}/query_result"

        prompt = str(extras.get("prompt", "clean modern vocal, studio quality")).strip()
        task_type = str(extras.get("task_type", "text2music")).strip() or "text2music"
        model = str(extras.get("model", "acestep-v15-turbo")).strip()
        thinking = _to_bool(extras.get("thinking", True), True)
        vocal_language = str(extras.get("vocal_language", "en")).strip() or "en"
        inference_steps = max(4, min(int(extras.get("inference_steps", 8)), 200))
        audio_format = str(extras.get("audio_format", "wav")).strip() or "wav"
        poll_interval = max(0.5, min(float(extras.get("poll_interval_seconds", 2.0)), 10.0))
        max_wait_seconds = max(15, min(int(extras.get("max_wait_seconds", 720)), 3600))

        request_fields: Dict[str, str] = {
            "prompt": prompt,
            "lyrics": lyrics,
            "task_type": task_type,
            "thinking": "true" if thinking else "false",
            "vocal_language": vocal_language,
            "inference_steps": str(inference_steps),
            "audio_format": audio_format,
            # Keep MIDI in the task metadata for your own tracking, even though current
            # ACE-Step API does not directly consume MIDI as a generation input.
            "midi_file_name": midi_path.name,
        }
        if model:
            request_fields["model"] = model
        if api_key.strip():
            request_fields["ai_token"] = api_key.strip()

        multipart_files: List[tuple[str, str, bytes, str]] = []
        if beat_path is not None:
            beat_bytes = beat_path.read_bytes()
            multipart_files.append(
                ("reference_audio", beat_path.name, beat_bytes, "audio/wav")
            )
        midi_bytes = midi_path.read_bytes()
        midi_b64 = base64.b64encode(midi_bytes).decode("ascii")
        request_fields["midi_base64"] = midi_b64

        set_api_compose_job_state(
            job_id,
            message="Submitting ACE-Step task...",
            progress=36,
            endpoint_url=base_url,
            midi_name=midi_path.name,
            beat_name=beat_path.name if beat_path is not None else "",
        )
        request_headers: Dict[str, str] = {}
        safe_auth_header = (auth_header or "Authorization").strip() or "Authorization"
        if api_key.strip():
            request_headers[safe_auth_header] = api_key.strip()
        multipart_body, multipart_content_type = _encode_multipart(
            request_fields,
            multipart_files,
        )
        request_headers["Content-Type"] = multipart_content_type
        release_req = urlrequest.Request(
            url=release_task_url,
            data=multipart_body,
            headers=request_headers,
            method="POST",
        )
        with urlrequest.urlopen(release_req, timeout=180) as response:
            release_status = int(getattr(response, "status", response.getcode()))
            release_text = response.read().decode("utf-8", errors="replace")
        release_json = json.loads(release_text)
        release_data = release_json.get("data", {})
        task_id = (
            str(release_data.get("task_id", "")).strip()
            if isinstance(release_data, dict)
            else ""
        )
        if not task_id:
            raise RuntimeError("ACE-Step did not return task_id.")

        set_api_compose_job_state(
            job_id,
            message=f"Task submitted. Waiting for completion ({task_id[:8]}...)",
            progress=56,
            provider_status_code=release_status,
            task_id=task_id,
            response_preview=release_text[:5000],
            response_json=release_json if isinstance(release_json, dict) else {},
        )

        deadline = time.time() + float(max_wait_seconds)
        last_query_json: Dict[str, object] = {}
        audio_urls: List[str] = []
        while time.time() < deadline:
            query_payload = {"task_id_list": [task_id]}
            query_headers = {"Content-Type": "application/json"}
            if api_key.strip():
                query_headers[safe_auth_header] = api_key.strip()
            query_req = urlrequest.Request(
                url=query_result_url,
                data=json.dumps(query_payload).encode("utf-8"),
                headers=query_headers,
                method="POST",
            )
            with urlrequest.urlopen(query_req, timeout=120) as query_response:
                query_status_code = int(
                    getattr(query_response, "status", query_response.getcode())
                )
                query_text = query_response.read().decode("utf-8", errors="replace")
            query_json = json.loads(query_text)
            last_query_json = query_json if isinstance(query_json, dict) else {}
            query_data = query_json.get("data", [])
            task_state = query_data[0] if isinstance(query_data, list) and query_data else {}
            status_value = int(task_state.get("status", 0))
            result_payload = task_state.get("result", "[]")
            parsed_results: List[Dict[str, object]] = []
            if isinstance(result_payload, str):
                try:
                    loaded = json.loads(result_payload)
                    if isinstance(loaded, list):
                        parsed_results = [item for item in loaded if isinstance(item, dict)]
                except json.JSONDecodeError:
                    parsed_results = []
            elif isinstance(result_payload, list):
                parsed_results = [item for item in result_payload if isinstance(item, dict)]

            if status_value == 1:
                for item in parsed_results:
                    file_url = str(item.get("file", "")).strip()
                    if not file_url:
                        continue
                    if file_url.startswith("http://") or file_url.startswith("https://"):
                        audio_urls.append(file_url)
                    elif file_url.startswith("/"):
                        audio_urls.append(f"{base_url}{file_url}")
                    else:
                        quoted = urlparse.quote(file_url, safe="")
                        audio_urls.append(f"{base_url}/v1/audio?path={quoted}")
                set_api_compose_job_state(
                    job_id,
                    status="completed",
                    message="ACE-Step generation finished.",
                    progress=100,
                    provider_status_code=query_status_code,
                    task_id=task_id,
                    audio_urls=audio_urls,
                    response_preview=query_text[:5000],
                    response_json=last_query_json,
                )
                return

            if status_value == 2:
                set_api_compose_job_state(
                    job_id,
                    status="failed",
                    message="ACE-Step task failed.",
                    provider_status_code=query_status_code,
                    task_id=task_id,
                    response_preview=query_text[:5000],
                    response_json=last_query_json,
                    error=query_text[:1200],
                )
                return

            elapsed = max(0.0, time.time() - (deadline - max_wait_seconds))
            fraction = min(1.0, elapsed / max(float(max_wait_seconds), 1.0))
            progress = min(95, 56 + int(round(fraction * 36)))
            set_api_compose_job_state(
                job_id,
                message="Task running on local ACE-Step server...",
                progress=progress,
                provider_status_code=query_status_code,
                task_id=task_id,
                response_preview=query_text[:2000],
                response_json=last_query_json,
            )
            time.sleep(poll_interval)

        set_api_compose_job_state(
            job_id,
            status="failed",
            message="Timed out waiting for ACE-Step task result.",
            error="ACE-Step task timed out before completion.",
        )
    except urlerror.HTTPError as exc:
        error_text = exc.read().decode("utf-8", errors="replace")
        set_api_compose_job_state(
            job_id,
            status="failed",
            message="ACE-Step API returned an HTTP error.",
            provider_status_code=int(exc.code or 0),
            response_preview=error_text[:5000],
            error=f"HTTP {exc.code}: {error_text[:1200]}",
        )
    except urlerror.URLError as exc:
        reason = getattr(exc, "reason", exc)
        reason_text = str(reason)
        lower_reason = reason_text.lower()
        if "10061" in reason_text or isinstance(reason, ConnectionRefusedError):
            message = "ACE-Step local server is offline or endpoint is wrong."
            detail = (
                f"Connection refused at {release_task_url or endpoint_url}. "
                "Start ACE-Step server, then retry."
            )
        elif "timed out" in lower_reason:
            message = "ACE-Step request timed out."
            detail = (
                f"Timed out while calling {release_task_url or endpoint_url}. "
                "Check server load and endpoint URL."
            )
        else:
            message = "Could not reach ACE-Step endpoint."
            detail = f"{release_task_url or endpoint_url}: {reason_text}"
        set_api_compose_job_state(
            job_id,
            status="failed",
            message=message,
            response_preview=reason_text[:5000],
            error=detail[:1200],
        )
    except Exception:
        set_api_compose_job_state(
            job_id,
            status="failed",
            message="API compose job failed.",
            error=traceback.format_exc(),
        )


def start_touchup_job(
    job_id: str,
    source_path: Path,
    source_word: str,
    mode: str,
    strength: int,
    max_target_words: int,
) -> None:
    try:
        with touchup_stop_events_lock:
            cancel_event = touchup_stop_events.setdefault(job_id, threading.Event())

        set_touchup_job_state(
            job_id,
            status="running",
            message=(
                "Scoring the vocal and detecting lyric windows..."
                if mode == "smart-removal"
                else "Scoring the vocal and detecting weak lyric regions..."
            ),
            progress=6,
        )

        output_dir = TOUCHUP_ROOT / job_id / "outputs"
        output_dir.mkdir(parents=True, exist_ok=True)

        if mode == "smart-removal":
            metadata = touchup_engine.smart_remove_non_lyrics(
                source_path=source_path,
                intended_lyrics=source_word,
                output_dir=output_dir,
                strength=strength,
                cancel_event=cancel_event,
                update_status=lambda payload: set_touchup_job_state(
                    job_id,
                    progress=int(payload.get("progress", 0)),
                    message=str(payload.get("message", "")),
                    best_similarity_score=float(payload.get("best_similarity_score", 0.0)),
                    best_word_report=str(payload.get("best_word_report", "")),
                    best_letter_report=str(payload.get("best_letter_report", "")),
                    variants_tested=int(payload.get("variants_tested", 0)),
                    repair_attempts=int(payload.get("repair_attempts", payload.get("variants_tested", 0))),
                    repaired_word_count=int(payload.get("repaired_word_count", 0)),
                    batch_index=int(payload.get("batch_index", 0)),
                    detected_word_indices=list(payload.get("detected_word_indices", [])),
                    stop_requested=cancel_event.is_set(),
                ),
            )
        else:
            metadata = touchup_engine.optimize_pronunciation(
                source_path=source_path,
                intended_lyrics=source_word,
                output_dir=output_dir,
                mode=mode,
                strength=strength,
                variants_per_batch=1,
                parallel_variants=1,
                max_batches=0,
                max_target_words=max_target_words,
                cancel_event=cancel_event,
                update_status=lambda payload: set_touchup_job_state(
                    job_id,
                    progress=int(payload.get("progress", 0)),
                    message=str(payload.get("message", "")),
                    best_similarity_score=float(payload.get("best_similarity_score", 0.0)),
                    best_word_report=str(payload.get("best_word_report", "")),
                    best_letter_report=str(payload.get("best_letter_report", "")),
                    variants_tested=int(payload.get("variants_tested", 0)),
                    repair_attempts=int(payload.get("repair_attempts", payload.get("variants_tested", 0))),
                    repaired_word_count=int(payload.get("repaired_word_count", 0)),
                    batch_index=int(payload.get("batch_index", 0)),
                    detected_word_indices=list(payload.get("detected_word_indices", [])),
                    regeneration_available=bool(payload.get("regeneration_available", False)),
                    regeneration_reason=str(payload.get("regeneration_reason", "")),
                    detected_only=bool(payload.get("detected_only", False)),
                    stop_requested=cancel_event.is_set(),
                ),
            )

        output_name = Path(str(metadata["output_path"])).name
        set_touchup_job_state(
            job_id,
            status="completed",
            message=(
                "Stopped. The best pronunciation result found so far is ready."
                if bool(metadata.get("stopped_early"))
                else (
                    "Smart removal finished. Only lyric-aligned vocal regions were kept."
                    if mode == "smart-removal"
                    else (
                        "Detection finished. Regeneration is not enabled yet, so this is still the original vocal."
                        if bool(metadata.get("detected_only"))
                        else "Done. The best regenerated pronunciation result is ready."
                    )
                )
            ),
            progress=100,
            batch_index=int(metadata.get("batch_index", 0)),
            variants_tested=int(metadata.get("variants_tested", 0)),
            repair_attempts=int(metadata.get("repair_attempts", metadata.get("variants_tested", 0))),
            repaired_word_count=int(metadata.get("repaired_word_count", 0)),
            best_similarity_score=float(metadata.get("best_similarity_score", 0.0)),
            best_word_report=str(metadata.get("best_word_report", "")),
            best_letter_report=str(metadata.get("best_letter_report", "")),
            best_word_scores=list(metadata.get("best_word_scores", [])),
            best_letter_scores=list(metadata.get("best_letter_scores", [])),
            detected_word_indices=list(metadata.get("detected_word_indices", [])),
            regeneration_available=bool(metadata.get("regeneration_available", False)),
            regeneration_reason=str(metadata.get("regeneration_reason", "")),
            detected_only=bool(metadata.get("detected_only", False)),
            stop_requested=bool(metadata.get("stopped_early")),
            sample_rate=int(metadata["sample_rate"]),
            result_url=f"/downloads/simple-web/touchup/{job_id}/outputs/{output_name}",
            download_name=output_name,
            removed_url=(
                f"/downloads/simple-web/touchup/{job_id}/outputs/{Path(str(metadata['removed_path'])).name}"
                if metadata.get("removed_path")
                else ""
            ),
            removed_download_name=(
                Path(str(metadata["removed_path"])).name if metadata.get("removed_path") else ""
            ),
            source_rms_db=float(metadata["source_rms_db"]),
            output_rms_db=float(metadata["output_rms_db"]),
            kept_segment_count=int(metadata.get("kept_segment_count", 0)),
            kept_duration_seconds=float(metadata.get("kept_duration_seconds", 0.0)),
            removed_duration_seconds=float(metadata.get("removed_duration_seconds", 0.0)),
        )
    except Exception:
        set_touchup_job_state(
            job_id,
            status="failed",
            message="Touch-up optimization stopped because something went wrong.",
            error=traceback.format_exc(),
        )
    finally:
        with touchup_stop_events_lock:
            touchup_stop_events.pop(job_id, None)


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC_ROOT / "index.html")


@app.get("/api/health")
def health() -> Dict[str, object]:
    return {
        "ok": True,
        "models_found": len(backend.list_models()),
    }


@app.get("/api/models")
def models() -> Dict[str, object]:
    preprocess_options = backend.get_preprocess_options()
    return {
        "models": backend.list_models(),
        "pipa_packages": backend.pipa_store.list_bundles(),
        "quality_presets": QUALITY_PRESETS,
        "preprocess_pipelines": preprocess_options["pipelines"],
        "preprocess_defaults": preprocess_options["defaults"],
    }


@app.get("/api/master-conversion/options")
def master_conversion_options() -> Dict[str, object]:
    return master_conversion_engine.get_options()


@app.get("/api/generate/options")
def generate_options() -> Dict[str, object]:
    return {
        "keys": [{"id": "", "label": "Leave as guide"}]
        + [{"id": note, "label": note} for note in GENERATE_NOTE_ORDER],
        "defaults": {
            "quality_preset": "balanced",
            "preprocess_mode": backend.DEFAULT_PREPROCESS_PIPELINE,
            "preprocess_strength": 9,
        },
        "description": (
            "Convert the reference vocal into the selected voice, then score the "
            "render against the pasted lyrics and repair weak pronunciation regions."
        ),
    }


@app.get("/api/jobs/{job_id}")
def job_status(job_id: str) -> Dict[str, object]:
    return get_job(job_id).to_dict()


@app.get("/api/master-conversion/jobs/{job_id}")
def master_conversion_job_status(job_id: str) -> Dict[str, object]:
    return get_master_conversion_job(job_id).to_dict()


@app.get("/api/generate/jobs/{job_id}")
def generate_job_status(job_id: str) -> Dict[str, object]:
    return get_generate_job(job_id).to_dict()


@app.get("/api/detag/options")
def detag_options() -> Dict[str, object]:
    return {
        "voices": detagger.list_voices(),
    }


@app.get("/api/isolator/options")
def isolator_options() -> Dict[str, object]:
    return backend.get_isolator_options()


@app.get("/api/mastering/options")
def mastering_options() -> Dict[str, object]:
    return mastering_engine.get_options()


@app.get("/api/optimize/options")
def optimize_options() -> Dict[str, object]:
    return {
        "defaults": {
            "max_cut_db": -24,
        },
        "description": "Upload multiple vocal takes plus lyrics. The engine scores lyric intelligibility and only cuts between takes when both seam sides are below your max dB cut threshold.",
    }


@app.get("/api/api-compose/options")
def api_compose_options() -> Dict[str, object]:
    return {
        "defaults": {
            "endpoint_url": "http://127.0.0.1:8001",
            "auth_header": "",
            "extra_json": json.dumps(
                {
                    "prompt": "clean modern pop vocal, studio quality",
                    "model": "acestep-v15-turbo",
                    "thinking": True,
                    "task_type": "text2music",
                    "vocal_language": "en",
                    "inference_steps": 8,
                    "audio_format": "wav",
                },
                ensure_ascii=False,
            ),
        },
        "required_files": ["midi_file"],
        "optional_files": ["beat_file"],
        "description": "ACE-Step local mode. Upload MIDI + lyrics and optionally beat reference. Backend submits release_task and polls query_result.",
    }


@app.get("/api/albums/options")
def albums_options() -> Dict[str, object]:
    return {
        "defaults": {
            "crossfade_seconds": 0.5,
        },
        "description": "Create album projects with persistent local version history. Upload tracks in any amount; song slots auto-expand and a 0.5 second crossfade album preview is rebuilt each update.",
    }


@app.get("/api/albums/projects")
def albums_projects() -> Dict[str, object]:
    with albums_lock:
        payload = load_album_db()
        projects = payload.get("projects", [])
        if not isinstance(projects, list):
            projects = []
        changed = False
        for project in projects:
            if isinstance(project, dict) and normalize_album_project(project):
                changed = True
        if changed:
            save_album_db(payload)
        serialized = [
            serialize_album_project_summary(project)
            for project in projects
            if isinstance(project, dict)
        ]
    serialized.sort(key=lambda item: str(item.get("updated_at", "")), reverse=True)
    return {"projects": serialized}


@app.get("/api/albums/projects/{project_id}")
def albums_project(project_id: str) -> Dict[str, object]:
    with albums_lock:
        payload = load_album_db()
        project = get_album_project_by_id(payload, project_id)
        if normalize_album_project(project):
            save_album_db(payload)
        serialized = serialize_album_project(project)
    return {"project": serialized}


@app.post("/api/albums/projects")
async def create_album_project(
    name: str = Form("Album Project"),
) -> Dict[str, object]:
    clean_name = str(name or "").strip() or "Album Project"
    project_id = uuid4().hex[:10]
    root = album_project_root(project_id)
    reset_directory(root)

    project = {
        "id": project_id,
        "name": clean_name,
        "song_count": 0,
        "crossfade_seconds": 0.5,
        "created_at": utc_now_iso(),
        "updated_at": utc_now_iso(),
        "songs": [],
        "mix_versions": [],
        "latest_mix_rel_path": "",
        "latest_mix_download_name": "",
        "latest_mix_song_count": 0,
        "event_log": [],
    }
    append_album_log(
        project,
        f"Created project \"{clean_name}\".",
    )

    with albums_lock:
        payload = load_album_db()
        projects = payload.setdefault("projects", [])
        if not isinstance(projects, list):
            projects = []
            payload["projects"] = projects
        projects.append(project)
        save_album_db(payload)

    return {"project": serialize_album_project(project)}


@app.post("/api/albums/projects/{project_id}/songs/{song_index}/versions")
async def upload_album_song_version(
    project_id: str,
    song_index: int,
    file: UploadFile = File(...),
) -> Dict[str, object]:
    if file is None or not file.filename:
        raise HTTPException(status_code=400, detail="Please upload one audio file.")

    payload_bytes = await file.read()
    clean_name = sanitize_filename(file.filename or f"song_{song_index}.wav")

    with albums_lock:
        payload = load_album_db()
        project = get_album_project_by_id(payload, project_id)
        normalize_album_project(project)
        version_entry = store_album_song_version(
            project=project,
            song_index=song_index,
            original_file_name=clean_name,
            payload_bytes=payload_bytes,
        )
        rebuild_album_preview(
            project=project,
            fade_seconds=float(project.get("crossfade_seconds", 0.5)),
        )
        save_album_db(payload)
        serialized = serialize_album_project(project)

    return {
        "project": serialized,
        "song_index": int(song_index),
        "version": int(version_entry.get("version", 0)),
    }


@app.post("/api/albums/projects/{project_id}/songs/bulk")
async def upload_album_songs_bulk(
    project_id: str,
    files: List[UploadFile] = File(...),
) -> Dict[str, object]:
    if not files:
        raise HTTPException(status_code=400, detail="Please upload one or more audio files.")

    prepared_uploads: List[tuple[str, bytes]] = []
    for upload in files:
        if upload is None or not upload.filename:
            prepared_uploads.append(("", b""))
            continue
        prepared_uploads.append(
            (
                sanitize_filename(upload.filename or "song.wav"),
                await upload.read(),
            )
        )

    with albums_lock:
        payload = load_album_db()
        project = get_album_project_by_id(payload, project_id)
        normalize_album_project(project)
        songs = project.get("songs", [])
        existing_track_count = len(songs) if isinstance(songs, list) else 0
        starting_slot = existing_track_count + 1

        uploaded_count = 0
        ignored_count = 0
        for offset, (upload_name, payload_bytes) in enumerate(prepared_uploads):
            if not upload_name:
                ignored_count += 1
                continue
            if not payload_bytes:
                ignored_count += 1
                continue
            slot_index = starting_slot + offset
            store_album_song_version(
                project=project,
                song_index=slot_index,
                original_file_name=upload_name or f"song_{slot_index}.wav",
                payload_bytes=payload_bytes,
            )
            uploaded_count += 1

        if uploaded_count <= 0:
            raise HTTPException(status_code=400, detail="No usable audio files were uploaded.")

        rebuild_album_preview(
            project=project,
            fade_seconds=float(project.get("crossfade_seconds", 0.5)),
        )
        append_album_log(
            project,
            (
                f"Added {uploaded_count} track(s) into slots "
                f"{starting_slot}-{starting_slot + uploaded_count - 1}."
            ),
        )
        if ignored_count > 0:
            append_album_log(
                project,
                f"Ignored {ignored_count} extra/invalid file(s) during bulk mapping.",
            )
        save_album_db(payload)
        serialized = serialize_album_project(project)

    return {
        "project": serialized,
        "uploaded_count": uploaded_count,
        "ignored_count": ignored_count,
    }


@app.post("/api/albums/projects/{project_id}/reorder")
async def reorder_album_tracks(project_id: str, request: Request) -> Dict[str, object]:
    payload = await request.json()
    ordered_indices = payload.get("song_indices", [])
    if not isinstance(ordered_indices, list) or not ordered_indices:
        raise HTTPException(status_code=400, detail="Provide a non-empty song_indices array.")

    with albums_lock:
        db_payload = load_album_db()
        project = get_album_project_by_id(db_payload, project_id)
        normalize_album_project(project)
        songs = project.get("songs", [])
        if not isinstance(songs, list) or not songs:
            raise HTTPException(status_code=400, detail="This album has no tracks to reorder.")

        existing_by_index = {
            int(song.get("song_index", 0)): song
            for song in songs
            if isinstance(song, dict)
        }
        requested = [int(value) for value in ordered_indices]
        existing_keys = sorted(existing_by_index.keys())
        if sorted(requested) != existing_keys:
            raise HTTPException(status_code=400, detail="song_indices must include every current track exactly once.")

        reordered: List[Dict[str, object]] = []
        for new_position, old_index in enumerate(requested, start=1):
            song = existing_by_index[old_index]
            song["song_index"] = int(new_position)
            reordered.append(song)

        project["songs"] = reordered
        project["song_count"] = len(reordered)
        project["updated_at"] = utc_now_iso()
        append_album_log(
            project,
            f"Reordered {len(reordered)} track(s).",
        )
        rebuild_album_preview(
            project=project,
            fade_seconds=float(project.get("crossfade_seconds", 0.5)),
        )
        save_album_db(db_payload)
        serialized = serialize_album_project(project)

    return {"project": serialized}


@app.delete("/api/albums/projects/{project_id}/songs/{song_index}")
async def delete_album_song_route(project_id: str, song_index: int) -> Dict[str, object]:
    with albums_lock:
        db_payload = load_album_db()
        project = get_album_project_by_id(db_payload, project_id)
        normalize_album_project(project)
        delete_album_song(project, song_index)
        rebuild_album_preview(
            project=project,
            fade_seconds=float(project.get("crossfade_seconds", 0.5)),
        )
        save_album_db(db_payload)
        serialized = serialize_album_project(project)

    return {"project": serialized}


@app.get("/api/albums/projects/{project_id}/songs/{song_index}/play")
def play_album_song_route(
    request: Request,
    project_id: str,
    song_index: int,
    version: Optional[int] = None,
    prepared: bool = False,
) -> Response:
    with albums_lock:
        payload = load_album_db()
        project = get_album_project_by_id(payload, project_id)
        if normalize_album_project(project):
            save_album_db(payload)
        _, selected_version = _select_album_song_version(project, song_index, version)
        asset_path = resolve_album_version_asset_path(
            project_id,
            song_index,
            selected_version,
            prepared=prepared,
            storage_key=str(selected_version.get("storage_key", "")),
            strict=True,
        )
        if asset_path is None or not asset_path.exists():
            raise HTTPException(status_code=404, detail="Audio file not found on disk.")

    return build_audio_stream_response(asset_path, request)


@app.get("/api/albums/projects/{project_id}/mix")
def play_album_mix_route(request: Request, project_id: str) -> Response:
    with albums_lock:
        payload = load_album_db()
        project = get_album_project_by_id(payload, project_id)
        if normalize_album_project(project):
            save_album_db(payload)
        asset_path = resolve_album_mix_asset_path(project)
        if asset_path is None or not asset_path.exists():
            raise HTTPException(status_code=404, detail="Album mix not found on disk.")

    return build_audio_stream_response(asset_path, request)


@app.get("/api/detag/jobs/{job_id}")
def detag_job_status(job_id: str) -> Dict[str, object]:
    return get_detag_job(job_id).to_dict()


@app.get("/api/isolator/jobs/{job_id}")
def isolator_job_status(job_id: str) -> Dict[str, object]:
    return get_isolator_job(job_id).to_dict()


@app.get("/api/mastering/jobs/{job_id}")
def mastering_job_status(job_id: str) -> Dict[str, object]:
    return get_mastering_job(job_id).to_dict()


@app.get("/api/optimize/jobs/{job_id}")
def optimize_job_status(job_id: str) -> Dict[str, object]:
    return get_optimize_job(job_id).to_dict()


@app.get("/api/api-compose/jobs/{job_id}")
def api_compose_job_status(job_id: str) -> Dict[str, object]:
    return get_api_compose_job(job_id).to_dict()


@app.get("/api/api-compose/provider-health")
def api_compose_provider_health(endpoint_url: str) -> Dict[str, object]:
    base_url = endpoint_url.strip().rstrip("/")
    if base_url.endswith("/release_task"):
        base_url = base_url[: -len("/release_task")]
    if base_url.endswith("/query_result"):
        base_url = base_url[: -len("/query_result")]
    if not base_url.startswith(("http://", "https://")):
        raise HTTPException(status_code=400, detail="Endpoint URL must start with http:// or https://.")

    health_url = f"{base_url}/health"
    try:
        req = urlrequest.Request(url=health_url, method="GET")
        with urlrequest.urlopen(req, timeout=8) as response:
            status_code = int(getattr(response, "status", response.getcode()))
            body = response.read().decode("utf-8", errors="replace")
        return {
            "ok": status_code == 200,
            "status_code": status_code,
            "body_preview": body[:500],
            "url": health_url,
        }
    except Exception as exc:
        return {
            "ok": False,
            "status_code": 0,
            "body_preview": str(exc),
            "url": health_url,
        }


@app.get("/api/touchup/jobs/{job_id}")
def touchup_job_status(job_id: str) -> Dict[str, object]:
    return get_touchup_job(job_id).to_dict()


@app.post("/api/touchup/jobs/{job_id}/stop")
def stop_touchup_job(job_id: str) -> Dict[str, object]:
    get_touchup_job(job_id)
    with touchup_stop_events_lock:
        stop_event = touchup_stop_events.get(job_id)
        if stop_event is None:
            raise HTTPException(status_code=409, detail="That touch-up job is not currently running.")
        stop_event.set()
    set_touchup_job_state(
        job_id,
        stop_requested=True,
        message="Stop requested. Finishing the current word region...",
    )
    return {"ok": True}


@app.get("/api/training/options")
def training_options() -> Dict[str, object]:
    return trainer.get_options()


@app.get("/api/training/packages")
def training_packages() -> Dict[str, object]:
    packages: List[Dict[str, object]] = []
    for bundle in backend.pipa_store.list_bundles():
        manifest_path = Path(str(bundle.get("manifest_path", "") or "").strip())
        if not manifest_path.exists():
            continue
        package_dir = manifest_path.parent
        packages.append(
            {
                "name": str(bundle.get("name", "")),
                "label": str(bundle.get("label", bundle.get("name", ""))),
                "package_mode": str(bundle.get("package_mode", "persona-v1") or "persona-v1"),
                "manifest_path": str(manifest_path),
                "package_dir": str(package_dir),
            }
        )
    return {"packages": packages}


@app.get("/api/training/packages/download")
def download_training_package(selection_name: str) -> FileResponse:
    bundle = backend.pipa_store.resolve_bundle(str(selection_name or "").strip())
    if bundle is None:
        raise HTTPException(status_code=404, detail="Selected Persona package was not found.")

    manifest_path = Path(str(bundle.get("manifest_path", "") or "").strip())
    if not manifest_path.exists():
        raise HTTPException(status_code=404, detail="Selected Persona package is missing its manifest.")

    package_dir = manifest_path.parent
    safe_label = sanitize_filename(str(bundle.get("label", bundle.get("name", package_dir.name))) or package_dir.name)
    download_root = TRAINING_ROOT / "_package_downloads"
    download_root.mkdir(parents=True, exist_ok=True)
    zip_path = download_root / f"{safe_label or package_dir.name}.zip"
    create_zip(zip_path, package_dir)
    return FileResponse(
        zip_path,
        media_type="application/zip",
        filename=zip_path.name,
    )


@app.get("/api/rebuild/options")
def rebuild_options() -> Dict[str, object]:
    packages = []
    for bundle in backend.pipa_store.list_bundles():
        reference_bank_path = str(bundle.get("reference_bank_path", "") or "").strip()
        if not reference_bank_path:
            continue
        rebuild_profile_path = str(bundle.get("rebuild_profile_path", "") or "").strip()
        packages.append(
            {
                "name": str(bundle.get("name", "")),
                "label": str(bundle.get("label", bundle.get("name", ""))),
                "package_mode": str(bundle.get("package_mode", "pipa-full") or "pipa-full"),
                "has_backing_model": bool(bundle.get("has_backing_model", False)),
                "rvc_model_name": str(bundle.get("rvc_model_name", "") or ""),
                "reference_word_count": int(bundle.get("reference_word_count", 0) or 0),
                "reference_phrase_count": int(bundle.get("reference_phrase_count", 0) or 0),
                "alignment_tolerance": str(bundle.get("alignment_tolerance", "balanced") or "balanced"),
                "rebuild_profile_path": rebuild_profile_path,
                "reference_bank_path": reference_bank_path,
            }
        )
    return {
        "packages": packages,
        "note": (
            "These packages contain lyric-aware rebuild references. The rebuild plan uses your "
            "guide vocal plus lyrics to map phrases and words onto the trained package style."
        ),
    }


@app.get("/api/training/jobs/{job_id}")
def training_job_status(job_id: str) -> Dict[str, object]:
    return get_training_job(job_id).to_dict()


@app.post("/api/training/jobs/{job_id}/stop")
def stop_training_job(job_id: str) -> Dict[str, object]:
    job = get_training_job(job_id)
    if job.status not in {"queued", "running"}:
        return job.to_dict()
    with training_stop_events_lock:
        stop_event = training_stop_events.get(job_id)
        if stop_event is None:
            raise HTTPException(status_code=404, detail="Training stop control not found.")
        stop_event.set()
    set_training_job_state(
        job_id,
        stop_requested=True,
        message=(
            "Stop requested. Finishing the current paired voice-builder chunk..."
            if job.stage in {"pipa-svs-data", "persona-base", "persona-paired", "persona-dataset"}
            else (
                "Stop requested. Finishing the current alignment chunk..."
                if job.output_mode in {"pipa-logic-only", "persona-v1"}
                else "Stop requested. Finishing the current training checkpoint..."
            )
        ),
    )
    return get_training_job(job_id).to_dict()


@app.post("/api/rebuild/plan")
async def create_rebuild_plan(
    pipa_package_name: str = Form(...),
    lyrics: str = Form(...),
    guide_file: UploadFile = File(...),
    top_k: int = Form(3),
) -> Dict[str, object]:
    package_name = str(pipa_package_name or "").strip()
    if not package_name:
        raise HTTPException(status_code=400, detail="Pick a PIPA package first.")
    bundle = backend.pipa_store.resolve_bundle(package_name)
    if bundle is None:
        raise HTTPException(status_code=400, detail="Selected PIPA package does not exist.")
    reference_bank_path = Path(str(bundle.get("reference_bank_path", "") or "").strip())
    if not reference_bank_path.exists():
        raise HTTPException(
            status_code=400,
            detail="Selected PIPA package is missing its reference bank.",
        )
    clean_lyrics = str(lyrics or "").strip()
    if not clean_lyrics:
        raise HTTPException(status_code=400, detail="Paste the intended lyrics first.")
    if guide_file is None or not guide_file.filename:
        raise HTTPException(status_code=400, detail="Upload a guide vocal first.")

    top_k = max(1, min(int(top_k), 8))
    job_id = uuid4().hex[:10]
    job_root = REBUILD_ROOT / job_id
    upload_dir = job_root / "uploads"
    output_dir = job_root / "outputs"
    reset_directory(upload_dir)
    reset_directory(output_dir)

    safe_name = sanitize_filename(guide_file.filename or "guide.wav")
    guide_path = upload_dir / safe_name
    with guide_path.open("wb") as handle:
        handle.write(await guide_file.read())

    try:
        guide_analysis = rebuild_feature_builder.analyze_file(
            guide_path=guide_path,
            lyrics=clean_lyrics,
            scorer=touchup_engine.scorer,
        )
    except Exception as exc:
        raise HTTPException(
            status_code=400,
            detail=f"Could not analyze the guide vocal: {exc}",
        ) from exc

    reference_bank = backend.pipa_store.load_reference_index(str(reference_bank_path))
    plan = rebuild_feature_builder.build_guide_plan(
        guide_analysis=guide_analysis,
        reference_bank=reference_bank,
        package_label=str(bundle.get("label", bundle.get("name", package_name))),
        top_k=top_k,
    )

    rebuild_profile_payload: Dict[str, object] = {}
    rebuild_profile_path = Path(str(bundle.get("rebuild_profile_path", "") or "").strip())
    if rebuild_profile_path.exists():
        try:
            loaded_profile = json.loads(rebuild_profile_path.read_text(encoding="utf-8"))
            if isinstance(loaded_profile, dict):
                rebuild_profile_payload = loaded_profile
        except Exception:
            rebuild_profile_payload = {}

    guide_analysis_path = output_dir / "guide_analysis.json"
    rebuild_plan_path = output_dir / "rebuild_plan.json"
    request_manifest_path = output_dir / "request_manifest.json"
    guide_analysis_path.write_text(json.dumps(guide_analysis, indent=2), encoding="utf-8")
    rebuild_plan_path.write_text(json.dumps(plan, indent=2), encoding="utf-8")
    request_manifest_path.write_text(
        json.dumps(
            {
                "created_at": datetime.utcnow().replace(microsecond=0).isoformat() + "Z",
                "job_id": job_id,
                "pipa_package_name": package_name,
                "guide_file": safe_name,
                "top_k": top_k,
                "lyrics_preview": clean_lyrics[:240],
                "package_mode": str(bundle.get("package_mode", "pipa-full") or "pipa-full"),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    guide_summary = dict(guide_analysis.get("style_summary", {}))
    voice_style = dict(rebuild_profile_payload.get("voice_style", {}))
    summary = dict(plan.get("summary", {}))
    return {
        "job_id": job_id,
        "package": {
            "name": str(bundle.get("name", package_name)),
            "label": str(bundle.get("label", bundle.get("name", package_name))),
            "package_mode": str(bundle.get("package_mode", "pipa-full") or "pipa-full"),
            "has_backing_model": bool(bundle.get("has_backing_model", False)),
            "rvc_model_name": str(bundle.get("rvc_model_name", "") or ""),
        },
        "guide_summary": guide_summary,
        "voice_style": voice_style,
        "summary": summary,
        "guide_analysis_url": audio_output_url(guide_analysis_path),
        "guide_analysis_download_name": guide_analysis_path.name,
        "rebuild_plan_url": audio_output_url(rebuild_plan_path),
        "rebuild_plan_download_name": rebuild_plan_path.name,
        "request_manifest_url": audio_output_url(request_manifest_path),
        "request_manifest_download_name": request_manifest_path.name,
    }


@app.post("/api/master-conversion/jobs")
async def create_master_conversion_job(
    model_name: str = Form(...),
    lyrics: str = Form(...),
    quality_preset: str = Form("balanced"),
    output_format: str = Form("wav"),
    source_file: UploadFile = File(...),
) -> Dict[str, object]:
    available_models = get_available_model_names()
    if not available_models:
        raise HTTPException(
            status_code=400,
            detail="No Persona v1.0 voices were found. Train or add a persona package first.",
        )
    if model_name not in available_models:
        raise HTTPException(status_code=400, detail="Selected persona voice does not exist.")
    primary_model_bundle = backend.resolve_model_reference(model_name)
    selected_pipa_bundle = backend.pipa_store.resolve_bundle(
        str(primary_model_bundle.get("selection_name", ""))
    )
    if not str(primary_model_bundle.get("guided_regeneration_path", "") or "").strip():
        raise HTTPException(
            status_code=400,
            detail="Selected voice is missing its Persona v1.0 builder checkpoint.",
        )
    if source_file is None or not source_file.filename:
        raise HTTPException(status_code=400, detail="Please upload one lead vocal file.")
    clean_lyrics = str(lyrics or "").strip()
    if not clean_lyrics:
        raise HTTPException(status_code=400, detail="Paste the intended lyrics first.")

    output_format = output_format.lower()
    if output_format != "wav":
        output_format = "wav"
    settings = build_conversion_settings(
        quality_preset=quality_preset,
        preprocess_mode="off",
        preprocess_strength=1,
        speaker_id=0,
        transpose=0,
        pitch_method="",
        index_path="",
        index_rate=-1,
        filter_radius=-1,
        resample_sr=0,
        rms_mix_rate=-1,
        protect=-1,
        crepe_hop_length=-1,
    )

    job_id = uuid4().hex[:10]
    job_root = MASTER_CONVERSION_ROOT / job_id
    upload_dir = job_root / "uploads"
    reset_directory(upload_dir)

    safe_name = sanitize_filename(source_file.filename or "song.wav")
    source_target = upload_dir / safe_name
    with source_target.open("wb") as handle:
        handle.write(await source_file.read())

    settings["lyrics"] = clean_lyrics
    settings["quality_preset"] = quality_preset
    settings["master_profile"] = "studio"
    settings["output_mode"] = "persona-v1"
    settings["secondary_model_name"] = ""
    settings["blend_percentage"] = 100
    settings["preprocess_mode"] = "off"
    settings["preprocess_strength"] = 1
    settings["pipa_reference_bank_path"] = str(
        (selected_pipa_bundle or {}).get("reference_bank_path", "") or ""
    )
    settings["pipa_rebuild_profile_path"] = str(
        (selected_pipa_bundle or {}).get("rebuild_profile_path", "") or ""
    )
    settings["pipa_manifest_path"] = str(
        (selected_pipa_bundle or {}).get("manifest_path", "") or ""
    )
    settings["pipa_phoneme_profile_path"] = str(
        (selected_pipa_bundle or {}).get("phoneme_profile_path", "") or ""
    )
    settings["pipa_manifest_path"] = str(
        (selected_pipa_bundle or {}).get("manifest_path", "") or ""
    )

    preview_lyrics = clean_lyrics.replace("\r", " ").replace("\n", " ").strip()
    if len(preview_lyrics) > 180:
        preview_lyrics = f"{preview_lyrics[:177]}..."

    job = MasterConversionJobState(
        id=job_id,
        model_name=str(primary_model_bundle.get("label", model_name)),
        source_name=safe_name,
        lyrics_preview=preview_lyrics,
        quality_preset=quality_preset,
        master_profile="studio",
        preferred_pipeline=str(settings.get("preprocess_mode", "")),
        output_mode="persona-v1",
        secondary_model_name="",
        blend_percentage=100,
    )
    with master_conversion_jobs_lock:
        master_conversion_jobs[job_id] = job

    worker = threading.Thread(
        target=start_master_conversion_job,
        args=(job_id, model_name, source_target, output_format, settings),
        daemon=True,
    )
    worker.start()
    return {"job_id": job_id}


@app.post("/api/generate/jobs")
async def create_generate_job(
    model_name: str = Form(...),
    lyrics: str = Form(...),
    guide_key: str = Form(""),
    target_key: str = Form(""),
    guide_bpm: float = Form(0),
    target_bpm: float = Form(0),
    quality_preset: str = Form("balanced"),
    preprocess_mode: str = Form("off"),
    preprocess_strength: int = Form(9),
    guide_file: UploadFile = File(...),
) -> Dict[str, object]:
    available_models = get_available_model_names()
    if not available_models:
        raise HTTPException(
            status_code=400,
            detail="No Persona v1.0 voices were found. Train or add a persona package first.",
        )
    if model_name not in available_models:
        raise HTTPException(status_code=400, detail="Selected voice model does not exist.")
    primary_model_bundle = backend.resolve_model_reference(model_name)
    if guide_file is None:
        raise HTTPException(status_code=400, detail="Please upload one reference vocal file.")

    lyrics = str(lyrics or "").strip()
    if not lyrics:
        raise HTTPException(
            status_code=400,
            detail="Paste the intended lyrics so the repair system can score weak words correctly.",
        )

    settings = build_conversion_settings(
        quality_preset=quality_preset,
        preprocess_mode=preprocess_mode,
        preprocess_strength=preprocess_strength,
        speaker_id=0,
        transpose=0,
        pitch_method="",
        index_path="",
        index_rate=-1,
        filter_radius=-1,
        resample_sr=0,
        rms_mix_rate=-1,
        protect=-1,
        crepe_hop_length=-1,
    )
    settings["lyrics"] = lyrics
    settings["guide_key"] = normalize_generate_key(guide_key)
    settings["target_key"] = normalize_generate_key(target_key)
    settings["guide_bpm"] = float(guide_bpm or 0.0)
    settings["target_bpm"] = float(target_bpm or 0.0)
    settings["quality_preset"] = quality_preset
    settings["pipa_reference_bank_path"] = str(
        primary_model_bundle.get("reference_bank_path", "") or ""
    )
    settings["pipa_phoneme_profile_path"] = str(
        primary_model_bundle.get("phoneme_profile_path", "") or ""
    )
    settings["pipa_manifest_path"] = str(
        primary_model_bundle.get("manifest_path", "") or ""
    )

    job_id = uuid4().hex[:10]
    job_root = GENERATE_ROOT / job_id
    upload_dir = job_root / "uploads"
    output_dir = job_root / "outputs"
    reset_directory(upload_dir)
    reset_directory(output_dir)

    safe_name = sanitize_filename(guide_file.filename or "guide.wav")
    guide_path = upload_dir / safe_name
    with guide_path.open("wb") as handle:
        handle.write(await guide_file.read())

    preview_lyrics = lyrics.replace("\r", " ").replace("\n", " ").strip()
    if len(preview_lyrics) > 180:
        preview_lyrics = f"{preview_lyrics[:177]}..."
    repair_profile = GENERATE_REPAIR_PROFILES.get(
        quality_preset,
        GENERATE_REPAIR_PROFILES["balanced"],
    )

    job = GenerateJobState(
        id=job_id,
        model_name=str(primary_model_bundle.get("label", model_name)),
        guide_name=safe_name,
        lyrics_preview=preview_lyrics,
        guide_key=str(settings["guide_key"]),
        target_key=str(settings["target_key"]),
        guide_bpm=float(settings["guide_bpm"]),
        target_bpm=float(settings["target_bpm"]),
        quality_preset=quality_preset,
        preprocess_mode=str(settings["preprocess_mode"]),
        repair_mode="pronunciation-repair",
        repair_strength=int(repair_profile["strength"]),
    )
    with generate_jobs_lock:
        generate_jobs[job_id] = job

    worker = threading.Thread(
        target=start_generate_job,
        args=(job_id, model_name, guide_path, settings),
        daemon=True,
    )
    worker.start()
    return {"job_id": job_id}


@app.post("/api/jobs")
async def create_job(
    model_name: str = Form(...),
    output_mode: str = Form("single"),
    secondary_model_name: str = Form(""),
    blend_percentage: int = Form(50),
    transpose: int = Form(0),
    quality_preset: str = Form("balanced"),
    output_format: str = Form("wav"),
    preprocess_mode: str = Form("off"),
    preprocess_strength: int = Form(10),
    speaker_id: int = Form(0),
    pitch_method: str = Form(""),
    index_path: str = Form(""),
    index_rate: float = Form(-1),
    filter_radius: int = Form(-1),
    resample_sr: int = Form(0),
    rms_mix_rate: float = Form(-1),
    protect: float = Form(-1),
    crepe_hop_length: int = Form(-1),
    files: List[UploadFile] = File(...),
    index_file: Optional[UploadFile] = File(None),
) -> Dict[str, object]:
    available_models = get_available_model_names()
    if not available_models:
        raise HTTPException(
            status_code=400,
            detail="No Persona v1.0 voices were found. Train or add a persona package first.",
        )
    if model_name not in available_models:
        raise HTTPException(status_code=400, detail="Selected model does not exist.")
    if not files:
        raise HTTPException(status_code=400, detail="Please upload at least one audio file.")
    output_mode = str(output_mode or "single").strip().lower()
    if output_mode not in {"single", "blend"}:
        output_mode = "single"
    secondary_model_name = str(secondary_model_name or "").strip()
    blend_percentage = max(0, min(int(blend_percentage), 100))
    if output_mode == "blend":
        if not secondary_model_name:
            raise HTTPException(status_code=400, detail="Pick a second model for blend mode.")
        if secondary_model_name not in available_models:
            raise HTTPException(status_code=400, detail="Second blend model does not exist.")

    output_format = output_format.lower()
    if output_format not in {"wav", "mp3", "flac"}:
        raise HTTPException(status_code=400, detail="Unsupported output format.")
    settings = build_conversion_settings(
        quality_preset=quality_preset,
        preprocess_mode=preprocess_mode,
        preprocess_strength=preprocess_strength,
        speaker_id=speaker_id,
        transpose=transpose,
        pitch_method=pitch_method,
        index_path=index_path,
        index_rate=index_rate,
        filter_radius=filter_radius,
        resample_sr=resample_sr,
        rms_mix_rate=rms_mix_rate,
        protect=protect,
        crepe_hop_length=crepe_hop_length,
    )

    job_id = uuid4().hex[:10]
    job_root = JOBS_ROOT / job_id
    upload_dir = job_root / "uploads"
    output_dir = job_root / "outputs"
    index_upload_dir = job_root / "indexes"
    reset_directory(upload_dir)
    reset_directory(output_dir)
    reset_directory(index_upload_dir)

    saved_uploads = []
    for upload in files:
        safe_name = sanitize_filename(upload.filename or "audio.wav")
        target = upload_dir / safe_name
        with target.open("wb") as handle:
            handle.write(await upload.read())
        saved_uploads.append(target)

    if index_file is not None and index_file.filename:
        safe_index_name = sanitize_filename(index_file.filename)
        index_target = index_upload_dir / safe_index_name
        with index_target.open("wb") as handle:
            handle.write(await index_file.read())
        settings["index_path"] = str(index_target)
    settings["output_mode"] = output_mode
    settings["secondary_model_name"] = secondary_model_name
    settings["blend_percentage"] = blend_percentage

    job = JobState(id=job_id, total_files=len(saved_uploads))
    with jobs_lock:
        jobs[job_id] = job

    worker = threading.Thread(
        target=start_conversion_job,
        args=(job_id, model_name, saved_uploads, output_format, settings),
        daemon=True,
    )
    worker.start()
    return {"job_id": job_id}


@app.post("/api/preview/source")
async def upload_preview_source(file: UploadFile = File(...)) -> Dict[str, object]:
    if file is None:
        raise HTTPException(status_code=400, detail="Please upload one audio file.")

    source_id = uuid4().hex[:12]
    source_dir = PREVIEW_ROOT / "sources" / source_id
    reset_directory(source_dir)
    safe_name = sanitize_filename(file.filename or "preview-source.wav")
    target = source_dir / safe_name
    with target.open("wb") as handle:
        handle.write(await file.read())

    return {
        "source_id": source_id,
        "name": safe_name,
    }


@app.post("/api/preview")
async def generate_preview(
    source_id: str = Form(...),
    model_name: str = Form(...),
    output_mode: str = Form("single"),
    secondary_model_name: str = Form(""),
    blend_percentage: int = Form(50),
    transpose: int = Form(0),
    quality_preset: str = Form("balanced"),
    preprocess_mode: str = Form("off"),
    preprocess_strength: int = Form(10),
    speaker_id: int = Form(0),
    pitch_method: str = Form(""),
    index_path: str = Form(""),
    index_rate: float = Form(-1),
    filter_radius: int = Form(-1),
    resample_sr: int = Form(0),
    rms_mix_rate: float = Form(-1),
    protect: float = Form(-1),
    crepe_hop_length: int = Form(-1),
    index_file: Optional[UploadFile] = File(None),
) -> Dict[str, object]:
    available_models = get_available_model_names()
    if not available_models:
        raise HTTPException(
            status_code=400,
            detail="No Persona v1.0 voices were found. Train or add a persona package first.",
        )
    if model_name not in available_models:
        raise HTTPException(status_code=400, detail="Selected model does not exist.")
    output_mode = str(output_mode or "single").strip().lower()
    if output_mode not in {"single", "blend"}:
        output_mode = "single"
    secondary_model_name = str(secondary_model_name or "").strip()
    blend_percentage = max(0, min(int(blend_percentage), 100))
    if output_mode == "blend":
        if not secondary_model_name:
            raise HTTPException(status_code=400, detail="Pick a second model for blend mode.")
        if secondary_model_name not in available_models:
            raise HTTPException(status_code=400, detail="Second blend model does not exist.")

    source_dir = PREVIEW_ROOT / "sources" / sanitize_filename(source_id)
    source_files = [path for path in source_dir.iterdir()] if source_dir.exists() else []
    source_files = [path for path in source_files if path.is_file()]
    if not source_files:
        raise HTTPException(status_code=400, detail="Preview source is missing. Re-add the audio file.")

    settings = build_conversion_settings(
        quality_preset=quality_preset,
        preprocess_mode=preprocess_mode,
        preprocess_strength=preprocess_strength,
        speaker_id=speaker_id,
        transpose=transpose,
        pitch_method=pitch_method,
        index_path=index_path,
        index_rate=index_rate,
        filter_radius=filter_radius,
        resample_sr=resample_sr,
        rms_mix_rate=rms_mix_rate,
        protect=protect,
        crepe_hop_length=crepe_hop_length,
    )

    preview_id = uuid4().hex[:10]
    preview_dir = PREVIEW_ROOT / "runs" / preview_id
    reset_directory(preview_dir)
    clip_path = preview_dir / "preview_input.wav"
    clip_metadata = backend.extract_middle_preview_clip(
        source_files[0],
        clip_path,
        clip_duration=5.0,
    )

    index_upload_dir = preview_dir / "indexes"
    reset_directory(index_upload_dir)
    if index_file is not None and index_file.filename:
        safe_index_name = sanitize_filename(index_file.filename)
        index_target = index_upload_dir / safe_index_name
        with index_target.open("wb") as handle:
            handle.write(await index_file.read())
        settings["index_path"] = str(index_target)
    output_path = preview_dir / "preview_converted.wav"
    if output_mode == "blend" and secondary_model_name:
        primary_render_path = preview_dir / "preview_primary.wav"
        secondary_render_path = preview_dir / "preview_secondary.wav"
        primary_metadata = backend.convert_file(
            model_name,
            clip_path,
            primary_render_path,
            preprocess_mode=str(settings["preprocess_mode"]),
            preprocess_strength=int(settings["preprocess_strength"]),
            work_dir=preview_dir / "prep-primary",
            speaker_id=int(settings["speaker_id"]),
            transpose=int(settings["transpose"]),
            f0_method=str(settings["f0_method"]),
            index_path=str(settings["index_path"]),
            index_rate=float(settings["index_rate"]),
            filter_radius=int(settings["filter_radius"]),
            resample_sr=int(settings["resample_sr"]),
            rms_mix_rate=float(settings["rms_mix_rate"]),
            protect=float(settings["protect"]),
            crepe_hop_length=int(settings["crepe_hop_length"]),
        )
        secondary_metadata = backend.convert_file(
            secondary_model_name,
            clip_path,
            secondary_render_path,
            preprocess_mode=str(settings["preprocess_mode"]),
            preprocess_strength=int(settings["preprocess_strength"]),
            work_dir=preview_dir / "prep-secondary",
            speaker_id=int(settings["speaker_id"]),
            transpose=int(settings["transpose"]),
            f0_method=str(settings["f0_method"]),
            index_path="",
            index_rate=float(settings["index_rate"]),
            filter_radius=int(settings["filter_radius"]),
            resample_sr=int(settings["resample_sr"]),
            rms_mix_rate=float(settings["rms_mix_rate"]),
            protect=float(settings["protect"]),
            crepe_hop_length=int(settings["crepe_hop_length"]),
        )
        sample_rate = blend_audio_outputs(
            primary_path=primary_render_path,
            secondary_path=secondary_render_path,
            output_path=output_path,
            output_format="wav",
            primary_percentage=blend_percentage,
        )
        metadata = {
            "sample_rate": sample_rate,
            "index_path": str(primary_metadata.get("index_path", "")),
            "timings": {
                "npy": round(
                    float(primary_metadata["timings"]["npy"]) + float(secondary_metadata["timings"]["npy"]),
                    2,
                ),
                "f0": round(
                    float(primary_metadata["timings"]["f0"]) + float(secondary_metadata["timings"]["f0"]),
                    2,
                ),
                "infer": round(
                    float(primary_metadata["timings"]["infer"]) + float(secondary_metadata["timings"]["infer"]),
                    2,
                ),
            },
            "preprocess_applied": bool(primary_metadata["preprocess_applied"]),
            "preprocess_mode": str(primary_metadata.get("preprocess_mode", "off")),
        }
    else:
        metadata = backend.convert_file(
            model_name,
            clip_path,
            output_path,
            preprocess_mode=str(settings["preprocess_mode"]),
            preprocess_strength=int(settings["preprocess_strength"]),
            work_dir=preview_dir / "prep",
            speaker_id=int(settings["speaker_id"]),
            transpose=int(settings["transpose"]),
            f0_method=str(settings["f0_method"]),
            index_path=str(settings["index_path"]),
            index_rate=float(settings["index_rate"]),
            filter_radius=int(settings["filter_radius"]),
            resample_sr=int(settings["resample_sr"]),
            rms_mix_rate=float(settings["rms_mix_rate"]),
            protect=float(settings["protect"]),
            crepe_hop_length=int(settings["crepe_hop_length"]),
        )

    return {
        "preview_url": f"/downloads/simple-web/previews/runs/{preview_id}/{output_path.name}",
        "download_name": output_path.name,
        "sample_rate": int(metadata["sample_rate"]),
        "timings": metadata["timings"],
        "clip_start": float(clip_metadata["start"]),
        "clip_duration": float(clip_metadata["duration"]),
        "source_duration": float(clip_metadata["total_duration"]),
        "cleanup_mode": str(settings["preprocess_mode"]),
        "preprocess_mode": str(settings["preprocess_mode"]),
        "preprocess_label": backend.get_preprocess_label(str(settings["preprocess_mode"])),
        "output_mode": output_mode,
        "secondary_model_name": secondary_model_name if output_mode == "blend" else "",
        "blend_percentage": blend_percentage if output_mode == "blend" else 100,
    }


@app.post("/api/detag/jobs")
async def create_detag_job(
    voice_id: str = Form(...),
    strength: int = Form(65),
    file: UploadFile = File(...),
) -> Dict[str, object]:
    voices = detagger.list_voices()
    available_voices = {voice["id"] for voice in voices}
    if not available_voices:
        raise HTTPException(
            status_code=400,
            detail="No detag voice references were found. You need a logs/<voice>/0_gt_wavs folder first.",
        )
    if voice_id not in available_voices:
        raise HTTPException(status_code=400, detail="Selected voice does not exist.")
    selected_voice = next(voice for voice in voices if voice["id"] == voice_id)
    if not bool(selected_voice.get("ready")):
        raise HTTPException(
            status_code=400,
            detail="That voice appears in weights, but detag still needs matching reference clips in logs/<voice>/0_gt_wavs.",
        )
    if file is None:
        raise HTTPException(status_code=400, detail="Please upload one audio file.")

    strength = max(1, min(int(strength), 100))
    job_id = uuid4().hex[:10]
    job_root = DETAG_ROOT / job_id
    upload_dir = job_root / "uploads"
    reset_directory(upload_dir)

    safe_name = sanitize_filename(file.filename or "audio.wav")
    target = upload_dir / safe_name
    with target.open("wb") as handle:
        handle.write(await file.read())

    job = DetagJobState(id=job_id, voice_id=voice_id)
    with detag_jobs_lock:
        detag_jobs[job_id] = job

    worker = threading.Thread(
        target=start_detag_job,
        args=(job_id, voice_id, target, strength),
        daemon=True,
    )
    worker.start()
    return {"job_id": job_id}


@app.post("/api/isolator/jobs")
async def create_isolator_job(
    mode: str = Form("main-vocal"),
    input_type: str = Form("full-mix"),
    strength: int = Form(10),
    deecho: bool = Form(True),
    width_focus: bool = Form(True),
    clarity_preserve: int = Form(70),
    files: List[UploadFile] = File(...),
) -> Dict[str, object]:
    options = backend.get_isolator_options()
    available_modes = {str(item["id"]) for item in options.get("modes", [])}
    available_input_types = {
        str(item["id"]) for item in options.get("input_types", [])
    }
    if mode not in available_modes:
        raise HTTPException(status_code=400, detail="Unsupported isolation mode.")
    if input_type not in available_input_types:
        raise HTTPException(status_code=400, detail="Unsupported input type.")
    if not files:
        raise HTTPException(status_code=400, detail="Please upload at least one audio file.")

    strength = max(1, min(int(strength), 20))
    clarity_preserve = max(0, min(int(clarity_preserve), 100))
    job_id = uuid4().hex[:10]
    job_root = ISOLATOR_ROOT / job_id
    upload_dir = job_root / "uploads"
    reset_directory(upload_dir)

    saved_uploads = []
    for upload in files:
        safe_name = sanitize_filename(upload.filename or "vocal.wav")
        target = upload_dir / safe_name
        with target.open("wb") as handle:
            handle.write(await upload.read())
        saved_uploads.append(target)

    job = IsolatorJobState(
        id=job_id,
        mode=mode,
        input_type=input_type,
        strength=strength,
        deecho=bool(deecho),
        width_focus=bool(width_focus),
        clarity_preserve=clarity_preserve,
        source_files=[path.name for path in saved_uploads],
        current_file=saved_uploads[0].name if saved_uploads else "",
    )
    with isolator_jobs_lock:
        isolator_jobs[job_id] = job

    worker = threading.Thread(
        target=start_isolator_job,
        args=(
            job_id,
            saved_uploads,
            mode,
            input_type,
            strength,
            bool(deecho),
            bool(width_focus),
            clarity_preserve,
        ),
        daemon=True,
    )
    worker.start()
    return {"job_id": job_id}


@app.post("/api/mastering/jobs")
async def create_mastering_job(
    source_file: UploadFile = File(...),
    reference_files: List[UploadFile] = File(...),
    resolution: int = Form(48),
) -> Dict[str, object]:
    if source_file is None or not source_file.filename:
        raise HTTPException(status_code=400, detail="Please upload the file you want to master.")
    if not reference_files:
        raise HTTPException(status_code=400, detail="Please upload at least one mastered reference file.")

    resolution = max(8, min(int(resolution), 160))
    job_id = uuid4().hex[:10]
    job_root = MASTERING_ROOT / job_id
    upload_dir = job_root / "uploads"
    output_dir = job_root / "outputs"
    reset_directory(upload_dir)
    reset_directory(output_dir)

    source_name = sanitize_filename(source_file.filename or "source.wav")
    source_path = upload_dir / f"source_{source_name}"
    with source_path.open("wb") as handle:
        handle.write(await source_file.read())

    saved_reference_paths: List[Path] = []
    for index, reference_file in enumerate(reference_files, start=1):
        if reference_file is None or not reference_file.filename:
            continue
        reference_name = sanitize_filename(reference_file.filename or f"reference-{index}.wav")
        reference_path = upload_dir / f"reference_{index:02d}_{reference_name}"
        with reference_path.open("wb") as handle:
            handle.write(await reference_file.read())
        saved_reference_paths.append(reference_path)

    if not saved_reference_paths:
        raise HTTPException(status_code=400, detail="Please upload at least one usable mastered reference file.")

    job = MasteringJobState(
        id=job_id,
        resolution=resolution,
        reference_count=len(saved_reference_paths),
        reference_files=[path.name for path in saved_reference_paths],
    )
    with mastering_jobs_lock:
        mastering_jobs[job_id] = job

    worker = threading.Thread(
        target=start_mastering_job,
        args=(job_id, source_path, saved_reference_paths, resolution),
        daemon=True,
    )
    worker.start()
    return {"job_id": job_id}


@app.post("/api/optimize/jobs")
async def create_optimize_job(
    lyrics: str = Form(...),
    max_cut_db: Optional[float] = Form(None),
    stitch_strength: Optional[int] = Form(None),
    files: List[UploadFile] = File(...),
) -> Dict[str, object]:
    if not files:
        raise HTTPException(
            status_code=400,
            detail="Please upload at least one vocal file.",
        )
    cleaned_lyrics = lyrics.strip()
    if not cleaned_lyrics:
        raise HTTPException(
            status_code=400,
            detail="Please paste the intended lyrics.",
        )

    if max_cut_db is None:
        # Backward compatibility for older cached frontends that still send stitch_strength.
        strength_value = 10 if stitch_strength is None else max(1, min(int(stitch_strength), 20))
        # strength=1 -> stricter quiet seams, strength=20 -> looser seam dB gate.
        max_cut_db = -38.0 + ((float(strength_value) - 1.0) / 19.0) * 26.0
    max_cut_db = max(-60.0, min(float(max_cut_db), -3.0))

    job_id = uuid4().hex[:10]
    job_root = OPTIMIZE_ROOT / job_id
    upload_dir = job_root / "uploads"
    reset_directory(upload_dir)

    saved_uploads: List[Path] = []
    for upload in files:
        safe_name = sanitize_filename(upload.filename or "audio.wav")
        target = upload_dir / safe_name
        with target.open("wb") as handle:
            handle.write(await upload.read())
        saved_uploads.append(target)

    job = OptimizeJobState(
        id=job_id,
        lyrics=cleaned_lyrics,
        stitch_strength=(10 if stitch_strength is None else max(1, min(int(stitch_strength), 20))),
        max_cut_db=max_cut_db,
        total_files=len(saved_uploads),
        processed_files=0,
        current_file=saved_uploads[0].name if saved_uploads else "",
    )
    with optimize_jobs_lock:
        optimize_jobs[job_id] = job

    worker = threading.Thread(
        target=start_optimize_job,
        args=(job_id, saved_uploads, cleaned_lyrics, max_cut_db),
        daemon=True,
    )
    worker.start()
    return {"job_id": job_id}


@app.post("/api/api-compose/jobs")
async def create_api_compose_job(
    endpoint_url: str = Form(...),
    lyrics: str = Form(...),
    auth_header: str = Form("Authorization"),
    api_key: str = Form(""),
    extra_json: str = Form(""),
    midi_file: UploadFile = File(...),
    beat_file: Optional[UploadFile] = File(None),
) -> Dict[str, object]:
    cleaned_endpoint = endpoint_url.strip()
    if not cleaned_endpoint:
        raise HTTPException(status_code=400, detail="Please enter a provider endpoint URL.")
    if not (cleaned_endpoint.startswith("http://") or cleaned_endpoint.startswith("https://")):
        raise HTTPException(status_code=400, detail="Endpoint URL must start with http:// or https://.")

    cleaned_lyrics = lyrics.strip()
    if not cleaned_lyrics:
        raise HTTPException(status_code=400, detail="Please enter lyrics.")
    if midi_file is None or not midi_file.filename:
        raise HTTPException(status_code=400, detail="Please upload one MIDI file.")

    midi_name = sanitize_filename(midi_file.filename or "input.mid")
    beat_name = sanitize_filename(beat_file.filename) if beat_file and beat_file.filename else ""
    job_id = uuid4().hex[:10]
    job_root = API_COMPOSE_ROOT / job_id
    upload_dir = job_root / "uploads"
    reset_directory(upload_dir)

    midi_path = upload_dir / midi_name
    with midi_path.open("wb") as handle:
        handle.write(await midi_file.read())

    beat_path: Optional[Path] = None
    if beat_file is not None and beat_file.filename:
        beat_path = upload_dir / beat_name
        with beat_path.open("wb") as handle:
            handle.write(await beat_file.read())

    job = ApiComposeJobState(
        id=job_id,
        endpoint_url=cleaned_endpoint,
        midi_name=midi_name,
        beat_name=beat_name,
    )
    with api_compose_jobs_lock:
        api_compose_jobs[job_id] = job

    worker = threading.Thread(
        target=start_api_compose_job,
        args=(
            job_id,
            cleaned_endpoint,
            api_key,
            auth_header,
            cleaned_lyrics,
            midi_path,
            beat_path,
            extra_json,
        ),
        daemon=True,
    )
    worker.start()
    return {"job_id": job_id}


@app.post("/api/touchup/jobs")
async def create_touchup_job(
    source_word: str = Form(...),
    mode: str = Form("smart-removal"),
    strength: int = Form(55),
    max_target_words: int = Form(5),
    source_file: UploadFile = File(...),
) -> Dict[str, object]:
    if source_file is None or not source_file.filename:
        raise HTTPException(status_code=400, detail="Please upload the AI vocal file.")

    source_word = source_word.strip()
    if not source_word:
        raise HTTPException(status_code=400, detail="Please enter the intended AI lyrics.")

    mode = str(mode or "detect-regenerate").strip().lower()
    if mode not in {"detect-only", "detect-regenerate", "repair", "auto-repair", "smart-removal"}:
        mode = "detect-regenerate"
    strength = max(1, min(int(strength), 100))
    max_target_words = max(1, min(int(max_target_words), 12))
    job_id = uuid4().hex[:10]
    job_root = TOUCHUP_ROOT / job_id
    upload_dir = job_root / "uploads"
    output_dir = job_root / "outputs"
    reset_directory(upload_dir)
    reset_directory(output_dir)

    source_name = sanitize_filename(source_file.filename or "ai-word.wav")
    source_path = upload_dir / f"source_{source_name}"
    with source_path.open("wb") as handle:
        handle.write(await source_file.read())

    job = TouchUpJobState(
        id=job_id,
        mode=mode,
        source_word=source_word,
        strength=strength,
        max_target_words=max_target_words,
    )
    with touchup_jobs_lock:
        touchup_jobs[job_id] = job
    with touchup_stop_events_lock:
        touchup_stop_events[job_id] = threading.Event()

    worker = threading.Thread(
        target=start_touchup_job,
        args=(
            job_id,
            source_path,
            source_word,
            mode,
            strength,
            max_target_words,
        ),
        daemon=True,
    )
    worker.start()
    return {"job_id": job_id}


@app.post("/api/training/jobs")
async def create_training_job(
    experiment_name: str = Form("voice-model"),
    sample_rate: str = Form("40k"),
    version: str = Form("v2"),
    f0_method: str = Form("rmvpe"),
    output_mode: str = Form("persona-v1"),
    epoch_mode: str = Form("manual-stop"),
    alignment_tolerance: str = Form("forgiving"),
    total_epochs: int = Form(200),
    save_every_epoch: int = Form(25),
    batch_size: int = Form(4),
    crepe_hop_length: int = Form(128),
    resume_selection_name: str = Form(""),
    start_phase: str = Form("auto"),
    files: List[UploadFile] = File(...),
    transcript_files: Optional[List[UploadFile]] = File(None),
    plan_files: Optional[List[UploadFile]] = File(None),
) -> Dict[str, object]:
    transcript_files = list(transcript_files or [])
    plan_files = list(plan_files or [])
    if not files:
        raise HTTPException(
            status_code=400, detail="Please upload at least one audio file to train on."
        )
    if not transcript_files and not plan_files:
        raise HTTPException(
            status_code=400,
            detail="Please upload either transcripts or a persona training plan manifest to build a Persona v1.0 model.",
        )
    options = trainer.get_options()
    if sample_rate not in set(options["sample_rates"]):
        raise HTTPException(status_code=400, detail="Unsupported sample rate.")
    if version not in set(options["versions"]):
        raise HTTPException(status_code=400, detail="Unsupported model version.")
    if f0_method not in set(options["f0_methods"]):
        raise HTTPException(status_code=400, detail="Unsupported pitch extraction method.")
    if output_mode not in {"persona-v1", "pipa-full", "pipa-lite", "pipa-logic-only"}:
        raise HTTPException(status_code=400, detail="Unsupported training system.")
    if epoch_mode not in {"fixed", "manual-stop"}:
        raise HTTPException(status_code=400, detail="Unsupported training length mode.")
    if alignment_tolerance not in {"forgiving", "balanced", "strict"}:
        raise HTTPException(status_code=400, detail="Unsupported transcript alignment tolerance.")
    start_phase = str(start_phase or "auto").strip().lower()
    start_phase = {
        "warmup": "warm-up",
        "bridge": "curriculum-bridge",
        "full": "full-diversity",
    }.get(start_phase, start_phase)
    if start_phase not in {"auto", "warm-up", "curriculum-bridge", "full-diversity"}:
        raise HTTPException(status_code=400, detail="Unsupported training start phase.")

    requested_total_epochs = max(1, min(int(total_epochs), 10000))
    total_epochs = requested_total_epochs
    guided_regeneration_epochs = requested_total_epochs
    persona_only_mode = output_mode in {"persona-v1", "pipa-logic-only"}
    if epoch_mode == "manual-stop":
        if persona_only_mode:
            guided_regeneration_epochs = max(requested_total_epochs, 20000)
        else:
            # Full/lite mode still needs to reach the RVC backbone, so keep the
            # regeneration pass substantial without trapping the run there forever.
            guided_regeneration_epochs = max(requested_total_epochs, 600)
            total_epochs = max(requested_total_epochs, 20000)
    save_every_epoch = max(1, min(int(save_every_epoch), total_epochs))
    batch_size = max(1, min(int(batch_size), 128))
    crepe_hop_length = max(1, min(int(crepe_hop_length), 512))
    chosen_name = trainer.make_unique_experiment_name(experiment_name)

    job_id = uuid4().hex[:10]
    job_root = TRAINING_ROOT / job_id
    upload_dir = job_root / "uploads"
    transcript_dir = job_root / "transcripts"
    plan_dir = job_root / "plans"
    reset_directory(upload_dir)
    reset_directory(transcript_dir)
    reset_directory(plan_dir)

    for upload in files:
        safe_name = sanitize_filename(upload.filename or "clip.wav")
        target = upload_dir / safe_name
        with target.open("wb") as handle:
            handle.write(await upload.read())

    for upload in transcript_files:
        safe_name = sanitize_filename(upload.filename or "transcript.txt")
        target = transcript_dir / safe_name
        with target.open("wb") as handle:
            handle.write(await upload.read())

    for upload in plan_files:
        safe_name = sanitize_filename(upload.filename or "training-plan.json")
        target = plan_dir / safe_name
        with target.open("wb") as handle:
            handle.write(await upload.read())

    build_index = output_mode == "pipa-full"
    job = TrainingJobState(
        id=job_id,
        experiment_name=chosen_name,
        build_index=bool(build_index),
        output_mode=output_mode,
        epoch_mode=epoch_mode,
        alignment_tolerance=alignment_tolerance,
    )
    with training_jobs_lock:
        training_jobs[job_id] = job
    with training_stop_events_lock:
        training_stop_events[job_id] = threading.Event()

    settings = {
        "sample_rate": sample_rate,
        "version": version,
        "f0_method": f0_method,
        "output_mode": output_mode,
        "epoch_mode": epoch_mode,
        "requested_total_epochs": requested_total_epochs,
        "total_epochs": total_epochs,
        "guided_regeneration_epochs": guided_regeneration_epochs,
        "save_every_epoch": save_every_epoch,
        "batch_size": batch_size,
        "crepe_hop_length": crepe_hop_length,
        "build_index": bool(build_index),
        "alignment_tolerance": alignment_tolerance,
        "resume_selection_name": str(resume_selection_name or "").strip(),
        "start_phase": start_phase,
    }
    worker = threading.Thread(
        target=start_training_job,
        args=(job_id, chosen_name, upload_dir, transcript_dir, plan_dir, job_root, settings),
        daemon=True,
    )
    worker.start()
    return {"job_id": job_id, "experiment_name": chosen_name}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=3001, help="Port for the simple web UI.")
    parser.add_argument(
        "--noautoopen",
        action="store_true",
        help="Do not open the browser automatically.",
    )
    args = parser.parse_args()

    if not args.noautoopen:
        threading.Timer(
            1.0, lambda: webbrowser.open(f"http://127.0.0.1:{args.port}")
        ).start()

    uvicorn.run(app, host="0.0.0.0", port=args.port, log_level="info")


if __name__ == "__main__":
    main()
