from __future__ import annotations

import json
import os
import shutil
import subprocess
import threading
import zipfile
from pathlib import Path
from typing import Callable, Dict, List, Optional
from urllib.request import urlretrieve

import ffmpeg
import numpy as np
import soundfile as sf
import torch

from config import Config
from lib.infer_pack.models import (
    SynthesizerTrnMs256NSFsid,
    SynthesizerTrnMs256NSFsid_nono,
    SynthesizerTrnMs768NSFsid,
    SynthesizerTrnMs768NSFsid_nono,
)
from simple_pipa import PIPAModelStore

DIRECT_GUIDED_PACKAGE_MODES = {
    "concert-remaster-paired",
}
ALIGNED_PTH_PACKAGE_MODES = {
    "persona-aligned-pth",
}
CLASSIC_SUPPORT_PACKAGE_MODES = {
    "classic-rvc-support",
}


class SimpleRVCBackend:
    HIGH_END_ROFORMER_MODEL = "model_bs_roformer_ep_317_sdr_12.9755.ckpt"
    PREPROCESS_ROFORMER_PREFERRED_MODELS = [
        "BS-Roformer-Viper-4.ckpt",
        "BS-Roformer-Viper-1.ckpt",
        "model_bs_roformer_ep_317_sdr_12.9755.ckpt",
    ]
    PREPROCESS_ROFORMER_SEGMENT_SIZE = 256
    PREPROCESS_ROFORMER_OVERLAP = 0.75
    PREPROCESS_ROFORMER_DENOISE = True
    PREPROCESS_KARAOKE_AGGRESSION = 5
    PREPROCESS_DEECHO_AGGRESSION = 5
    DEFAULT_PREPROCESS_PIPELINE = "fullness-first"
    PREPROCESS_PIPELINES = {
        "off": {
            "label": "Off",
            "description": "Skip vocal pre-processing and convert the file as-is.",
        },
        "extract-only": {
            "label": "Extract only",
            "description": "RoFormer vocal extraction only. Best when the file is already close to clean and you want the most body.",
        },
        "fullness-first": {
            "label": "Fullness first",
            "description": "Recommended. Removes some stacks while blending body back so the lead stays loud and natural.",
        },
        "reverb-polish": {
            "label": "Reverb polish",
            "description": "For already-clean vocals that mostly need de-echo and de-reverb without heavy backing suppression.",
        },
        "balanced-clean": {
            "label": "Balanced clean",
            "description": "Stronger cleanup for doubles, adlibs, and echo while still restoring some chest and level.",
        },
        "max-clean": {
            "label": "Max clean",
            "description": "Most aggressive cleanup path for messy vocals. Usually cleanest, but most likely to sound thinner.",
        },
        "legacy-lead": {
            "label": "Legacy lead focus",
            "description": "Older UVR-only lead vocal cleanup path kept for compatibility.",
            "hidden": True,
        },
    }
    UVR_MODELS = {
        "lead-vocal": {
            "filename": "HP5-\u4e3b\u65cb\u5f8b\u4eba\u58f0vocals+\u5176\u4ed6instrumentals.pth",
            "url": "https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/uvr5_weights/HP5-%E4%B8%BB%E6%97%8B%E5%BE%8B%E4%BA%BA%E5%A3%B0vocals+%E5%85%B6%E4%BB%96instrumentals.pth",
            "deecho": False,
        },
        "main-vocal": {
            "filename": "HP5_only_main_vocal.pth",
            "url": "https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/uvr5_weights/HP5_only_main_vocal.pth",
            "deecho": False,
        },
        "karaoke-body": {
            "filename": "5_HP-Karaoke-UVR.pth",
            "url": "https://github.com/TRvlvr/model_repo/releases/download/all_public_uvr_models/5_HP-Karaoke-UVR.pth",
            "deecho": False,
        },
        "deecho-normal": {
            "filename": "VR-DeEchoNormal.pth",
            "url": "https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/uvr5_weights/VR-DeEchoNormal.pth",
            "deecho": True,
        },
        "deecho-aggressive": {
            "filename": "VR-DeEchoAggressive.pth",
            "url": "https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/uvr5_weights/VR-DeEchoAggressive.pth",
            "deecho": True,
        },
        "deecho-dereverb": {
            "filename": "VR-DeEchoDeReverb.pth",
            "url": "https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/uvr5_weights/VR-DeEchoDeReverb.pth",
            "deecho": True,
        },
    }
    ISOLATOR_MODES = {
        "main-vocal": {
            "label": "Main vocal AI",
            "description": "Best for splitting a vocal stem into lead and backing layers.",
            "model_key": "main-vocal",
        },
        "lead-vocal": {
            "label": "Lead vocal AI",
            "description": "A wider split that can help when the backing stack is messy.",
            "model_key": "lead-vocal",
        },
        "reverb-echo-only": {
            "label": "Reverb & echo only",
            "description": "For vocal files only. Focuses on cleaning reverb/echo tails while keeping the main vocal.",
            "model_key": "main-vocal",
        },
    }

    def __init__(self, repo_root: Path):
        self.repo_root = repo_root
        self.weights_root = self.repo_root / "weights"
        self.logs_root = self.repo_root / "logs"
        self.uvr5_root = self.repo_root / "uvr5_weights"
        self.separator_env_root = self.repo_root / ".venv-separator311"
        self.separator_model_root = (
            self.repo_root / "pretrained_models" / "audio-separator-models"
        )
        self.separator_runner = self.repo_root / "tools" / "run_roformer_separator.py"
        self.config = Config()
        self.lock = threading.Lock()
        self.hubert_model = None
        self.current_model_name: Optional[str] = None
        self.net_g = None
        self.vc: Optional[object] = None
        self.cpt = None
        self.tgt_sr: Optional[int] = None
        self.version = "v1"
        self.n_spk = 0
        self.separator_cache: Dict[tuple[str, int], object] = {}
        self.pipa_store = PIPAModelStore(self.repo_root)
        self._model_listing_cache: Optional[List[Dict[str, object]]] = None
        self._model_listing_signature: tuple = ((), (), ())

    def _build_model_listing_signature(self) -> tuple:
        weight_signature = tuple(
            sorted(
                (
                    path.name,
                    int(path.stat().st_mtime_ns),
                )
                for path in self.weights_root.glob("*.pth")
                if path.is_file()
            )
        )
        index_signature = tuple(
            sorted(
                (
                    file.as_posix(),
                    int(file.stat().st_mtime_ns),
                )
                for file in self.logs_root.rglob("*.index")
                if file.is_file() and "trained" not in file.name.lower()
            )
        )
        pipa_signature = tuple(
            sorted(
                (
                    file.as_posix(),
                    int(file.stat().st_mtime_ns),
                )
                for file in self.pipa_store.root.glob("*/manifest.json")
                if file.is_file()
            )
        )
        return weight_signature, index_signature, pipa_signature

    def _find_default_index_from_list(
        self,
        model_name: str,
        index_files: List[Path],
    ) -> str:
        stem = Path(model_name).stem
        candidates = [
            stem,
            stem.lower(),
            stem.split("_")[0],
            stem.split("_")[0].lower(),
            stem.split("-")[0],
            stem.split("-")[0].lower(),
        ]
        candidate_set = {candidate for candidate in candidates if candidate}
        exact_folder_matches = [
            file
            for file in index_files
            if file.parent.name in candidate_set or file.parent.name.lower() in candidate_set
        ]
        if exact_folder_matches:
            return sorted(exact_folder_matches)[0].as_posix()

        stem_lower = stem.lower()
        fuzzy_matches = [
            file
            for file in index_files
            if stem_lower in file.parent.name.lower() or stem_lower in file.name.lower()
        ]
        if fuzzy_matches:
            return sorted(fuzzy_matches)[0].as_posix()
        return ""

    def _find_classic_weight_path(self, requested: str) -> Optional[Path]:
        target = str(requested or "").strip()
        if not target:
            return None

        normalized = target.lower()
        candidates = sorted(
            (path for path in self.weights_root.glob("*.pth") if path.is_file()),
            key=lambda path: (int(path.stat().st_mtime_ns), path.name),
            reverse=True,
        )
        exact_matches = [
            path
            for path in candidates
            if path.stem.lower() == normalized
            or path.name.lower() == normalized
            or path.name.lower() == f"{normalized}.pth"
        ]
        if exact_matches:
            return exact_matches[0]

        fuzzy_matches = [
            path
            for path in candidates
            if normalized in path.stem.lower() or normalized in path.name.lower()
        ]
        if fuzzy_matches:
            return fuzzy_matches[0]
        return None

    def list_models(self) -> List[Dict[str, object]]:
        signature = self._build_model_listing_signature()
        if (
            self._model_listing_cache is not None
            and self._model_listing_signature == signature
        ):
            return [dict(model) for model in self._model_listing_cache]

        models: List[Dict[str, object]] = []
        index_files = [
            file
            for file in self.logs_root.rglob("*.index")
            if file.is_file() and "trained" not in file.name.lower()
        ]
        pipa_models = self.pipa_store.list_bundles()
        for bundle in pipa_models:
            manifest_path = str(bundle.get("manifest_path", "") or "").strip()
            guided_regeneration_path = str(
                bundle.get("guided_regeneration_path", "") or ""
            ).strip()
            package_mode = str(bundle.get("package_mode", "persona-v1") or "persona-v1")
            model_path = str(bundle.get("model_path", "") or "").strip()
            default_index = str(bundle.get("default_index", "") or "").strip()
            if not manifest_path:
                continue
            if package_mode in DIRECT_GUIDED_PACKAGE_MODES:
                if not guided_regeneration_path:
                    continue
                model_label = (
                    "Concert remaster"
                    if package_mode == "concert-remaster-paired"
                    else "Paired aligned conversion"
                )
                models.append(
                    {
                        "name": str(bundle["name"]),
                        "label": str(bundle.get("label", bundle["name"])),
                        "default_index": default_index,
                        "has_index": bool(default_index),
                        "kind": "model",
                        "system": model_label,
                        "rvc_model_name": Path(guided_regeneration_path).name,
                        "model_path": "",
                        "phoneme_profile_path": str(bundle.get("phoneme_profile_path", "") or ""),
                        "rebuild_profile_path": str(bundle.get("rebuild_profile_path", "") or ""),
                        "reference_bank_path": str(bundle.get("reference_bank_path", "") or ""),
                        "manifest_path": manifest_path,
                        "guided_regeneration_path": guided_regeneration_path,
                        "guided_regeneration_report_path": str(bundle.get("guided_regeneration_report_path", "") or ""),
                        "guided_regeneration_preview_path": str(bundle.get("guided_regeneration_preview_path", "") or ""),
                        "package_mode": package_mode,
                    }
                )
                continue
            if package_mode in ALIGNED_PTH_PACKAGE_MODES:
                if not model_path:
                    continue
                models.append(
                    {
                        "name": str(bundle["name"]),
                        "label": str(bundle.get("label", bundle["name"])),
                        "default_index": default_index,
                        "has_index": bool(default_index),
                        "kind": "model",
                        "system": "Paired aligned conversion",
                        "rvc_model_name": str(bundle.get("rvc_model_name", "") or Path(model_path).name),
                        "model_path": model_path,
                        "phoneme_profile_path": str(bundle.get("phoneme_profile_path", "") or ""),
                        "rebuild_profile_path": str(bundle.get("rebuild_profile_path", "") or ""),
                        "reference_bank_path": str(bundle.get("reference_bank_path", "") or ""),
                        "manifest_path": manifest_path,
                        "guided_regeneration_path": guided_regeneration_path,
                        "guided_regeneration_report_path": str(bundle.get("guided_regeneration_report_path", "") or ""),
                        "guided_regeneration_preview_path": str(bundle.get("guided_regeneration_preview_path", "") or ""),
                        "package_mode": package_mode,
                    }
                )
                continue
            if package_mode in CLASSIC_SUPPORT_PACKAGE_MODES:
                if not model_path:
                    continue
                models.append(
                    {
                        "name": str(bundle["name"]),
                        "label": str(bundle.get("label", bundle["name"])),
                        "default_index": default_index,
                        "has_index": bool(default_index),
                        "kind": "model",
                        "system": "Classic RVC + SUNO audition",
                        "rvc_model_name": str(bundle.get("rvc_model_name", "") or Path(model_path).name),
                        "model_path": model_path,
                        "phoneme_profile_path": str(bundle.get("phoneme_profile_path", "") or ""),
                        "rebuild_profile_path": str(bundle.get("rebuild_profile_path", "") or ""),
                        "reference_bank_path": str(bundle.get("reference_bank_path", "") or ""),
                        "manifest_path": manifest_path,
                        "guided_regeneration_path": "",
                        "guided_regeneration_report_path": str(bundle.get("guided_regeneration_report_path", "") or ""),
                        "guided_regeneration_preview_path": str(bundle.get("guided_regeneration_preview_path", "") or ""),
                        "package_mode": package_mode,
                    }
                )
                continue
            if not guided_regeneration_path:
                continue
            models.append(
                {
                    "name": str(bundle["name"]),
                    "label": str(bundle.get("label", bundle["name"])),
                    "default_index": "",
                    "has_index": False,
                    "kind": "persona",
                    "system": "Persona v1.1" if package_mode == "persona-v1.1" else "Persona v1.0",
                    "rvc_model_name": "",
                    "phoneme_profile_path": str(bundle.get("phoneme_profile_path", "") or ""),
                    "rebuild_profile_path": str(bundle.get("rebuild_profile_path", "") or ""),
                    "reference_bank_path": str(bundle.get("reference_bank_path", "") or ""),
                    "manifest_path": manifest_path,
                    "guided_regeneration_path": guided_regeneration_path,
                    "guided_regeneration_report_path": str(bundle.get("guided_regeneration_report_path", "") or ""),
                    "guided_regeneration_preview_path": str(bundle.get("guided_regeneration_preview_path", "") or ""),
                    "package_mode": package_mode,
                }
            )

        classic_weight_paths = sorted(
            (path for path in self.weights_root.glob("*.pth") if path.is_file()),
            key=lambda path: (int(path.stat().st_mtime_ns), path.name),
            reverse=True,
        )
        for path in classic_weight_paths:
            default_index = self._find_default_index_from_list(path.name, index_files)
            models.append(
                {
                    "name": path.stem,
                    "label": path.stem,
                    "default_index": default_index,
                    "has_index": bool(default_index),
                    "kind": "classic",
                    "system": "Classic RVC",
                    "rvc_model_name": path.name,
                    "model_path": path.as_posix(),
                    "phoneme_profile_path": "",
                    "rebuild_profile_path": "",
                    "reference_bank_path": "",
                    "manifest_path": "",
                    "guided_regeneration_path": "",
                    "guided_regeneration_report_path": "",
                    "guided_regeneration_preview_path": "",
                    "package_mode": "classic-rvc",
                }
            )

        self._model_listing_signature = signature
        self._model_listing_cache = [dict(model) for model in models]
        return models

    def resolve_model_reference(self, model_name: str) -> Dict[str, object]:
        requested = str(model_name or "").strip()
        if not requested:
            raise FileNotFoundError("No model was selected.")
        bundle = self.pipa_store.resolve_bundle(requested)
        if bundle is not None:
            package_mode = str(bundle.get("package_mode", "persona-v1") or "persona-v1")
            manifest_path = str(bundle.get("manifest_path", "") or "").strip()
            if package_mode in DIRECT_GUIDED_PACKAGE_MODES:
                guided_regeneration_path = str(bundle.get("guided_regeneration_path", "") or "").strip()
                if not guided_regeneration_path or not manifest_path:
                    raise FileNotFoundError(
                        "That direct paired package is missing its guided checkpoint files."
                    )
                return {
                    "selection_name": str(bundle["name"]),
                    "label": str(bundle["label"]),
                    "kind": "model",
                    "system": "Concert remaster" if package_mode == "concert-remaster-paired" else "Paired aligned conversion",
                    "rvc_model_name": Path(guided_regeneration_path).name,
                    "model_path": "",
                    "default_index": str(bundle.get("default_index", "") or ""),
                    "phoneme_profile_path": str(bundle.get("phoneme_profile_path", "") or ""),
                    "rebuild_profile_path": str(bundle.get("rebuild_profile_path", "") or ""),
                    "reference_bank_path": str(bundle.get("reference_bank_path", "") or ""),
                    "manifest_path": manifest_path,
                    "guided_regeneration_path": guided_regeneration_path,
                    "guided_regeneration_report_path": str(bundle.get("guided_regeneration_report_path", "") or ""),
                    "guided_regeneration_preview_path": str(bundle.get("guided_regeneration_preview_path", "") or ""),
                    "package_mode": package_mode,
                }
            if package_mode in ALIGNED_PTH_PACKAGE_MODES:
                model_path = str(bundle.get("model_path", "") or "").strip()
                if not model_path or not manifest_path:
                    raise FileNotFoundError(
                        "That aligned PTH package is missing its backbone model files."
                    )
                return {
                    "selection_name": str(bundle["name"]),
                    "label": str(bundle["label"]),
                    "kind": "model",
                    "system": "Paired aligned conversion",
                    "rvc_model_name": str(bundle.get("rvc_model_name", "") or Path(model_path).name),
                    "model_path": model_path,
                    "default_index": str(bundle.get("default_index", "") or ""),
                    "phoneme_profile_path": str(bundle.get("phoneme_profile_path", "") or ""),
                    "rebuild_profile_path": str(bundle.get("rebuild_profile_path", "") or ""),
                    "reference_bank_path": str(bundle.get("reference_bank_path", "") or ""),
                    "manifest_path": manifest_path,
                    "guided_regeneration_path": str(bundle.get("guided_regeneration_path", "") or ""),
                    "guided_regeneration_report_path": str(bundle.get("guided_regeneration_report_path", "") or ""),
                    "guided_regeneration_preview_path": str(bundle.get("guided_regeneration_preview_path", "") or ""),
                    "package_mode": package_mode,
                }
            if package_mode in CLASSIC_SUPPORT_PACKAGE_MODES:
                model_path = str(bundle.get("model_path", "") or "").strip()
                if not model_path or not manifest_path:
                    raise FileNotFoundError(
                        "That classic RVC support package is missing its backbone model files."
                    )
                return {
                    "selection_name": str(bundle["name"]),
                    "label": str(bundle["label"]),
                    "kind": "model",
                    "system": "Classic RVC + SUNO audition",
                    "rvc_model_name": str(bundle.get("rvc_model_name", "") or Path(model_path).name),
                    "model_path": model_path,
                    "default_index": str(bundle.get("default_index", "") or ""),
                    "phoneme_profile_path": str(bundle.get("phoneme_profile_path", "") or ""),
                    "rebuild_profile_path": str(bundle.get("rebuild_profile_path", "") or ""),
                    "reference_bank_path": str(bundle.get("reference_bank_path", "") or ""),
                    "manifest_path": manifest_path,
                    "guided_regeneration_path": "",
                    "guided_regeneration_report_path": str(bundle.get("guided_regeneration_report_path", "") or ""),
                    "guided_regeneration_preview_path": str(bundle.get("guided_regeneration_preview_path", "") or ""),
                    "package_mode": package_mode,
                }
            guided_regeneration_path = str(
                bundle.get("guided_regeneration_path", "") or ""
            ).strip()
            if not guided_regeneration_path or not manifest_path:
                raise FileNotFoundError(
                    "That Persona package is missing its builder files."
                )
            return {
                "selection_name": str(bundle["name"]),
                "label": str(bundle["label"]),
                "kind": "persona",
                "system": (
                    "Persona v1.1"
                    if package_mode == "persona-v1.1"
                    else "Persona v1.0"
                ),
                "rvc_model_name": "",
                "default_index": "",
                "phoneme_profile_path": str(bundle.get("phoneme_profile_path", "") or ""),
                "rebuild_profile_path": str(bundle.get("rebuild_profile_path", "") or ""),
                "reference_bank_path": str(bundle.get("reference_bank_path", "") or ""),
                "manifest_path": manifest_path,
                "guided_regeneration_path": guided_regeneration_path,
                "guided_regeneration_report_path": str(bundle.get("guided_regeneration_report_path", "") or ""),
                "guided_regeneration_preview_path": str(bundle.get("guided_regeneration_preview_path", "") or ""),
                "package_mode": str(bundle.get("package_mode", "persona-v1") or "persona-v1"),
            }

        classic_model_path = self._find_classic_weight_path(requested)
        if classic_model_path is not None:
            index_files = [
                file
                for file in self.logs_root.rglob("*.index")
                if file.is_file() and "trained" not in file.name.lower()
            ]
            default_index = self._find_default_index_from_list(classic_model_path.name, index_files)
            return {
                "selection_name": classic_model_path.stem,
                "label": classic_model_path.stem,
                "kind": "classic",
                "system": "Classic RVC",
                "rvc_model_name": classic_model_path.name,
                "model_path": classic_model_path.as_posix(),
                "default_index": default_index,
                "phoneme_profile_path": "",
                "rebuild_profile_path": "",
                "reference_bank_path": "",
                "manifest_path": "",
                "guided_regeneration_path": "",
                "guided_regeneration_report_path": "",
                "guided_regeneration_preview_path": "",
                "package_mode": "classic-rvc",
            }

        raise FileNotFoundError(f"Persona package not found: {requested}")

    def get_preprocess_options(self) -> Dict[str, object]:
        return {
            "pipelines": [
                {
                    "id": pipeline_id,
                    "label": str(info["label"]),
                    "description": str(info["description"]),
                }
                for pipeline_id, info in self.PREPROCESS_PIPELINES.items()
                if not bool(info.get("hidden"))
            ],
            "defaults": {
                "mode": self.DEFAULT_PREPROCESS_PIPELINE,
                "strength": 9,
            },
        }

    def normalize_preprocess_mode(self, preprocess_mode: str) -> str:
        normalized = str(preprocess_mode or "").strip().lower()
        aliases = {
            "": self.DEFAULT_PREPROCESS_PIPELINE,
            "on": "balanced-clean",
            "adlib-ai": "balanced-clean",
            "lead-only": "legacy-lead",
        }
        normalized = aliases.get(normalized, normalized)
        if normalized not in self.PREPROCESS_PIPELINES:
            raise ValueError(f"Unsupported preprocess pipeline: {preprocess_mode}")
        return normalized

    def get_preprocess_label(self, preprocess_mode: str) -> str:
        normalized = self.normalize_preprocess_mode(preprocess_mode)
        return str(self.PREPROCESS_PIPELINES[normalized]["label"])

    def _strength_ratio(self, strength: int) -> float:
        return float(np.clip((int(strength) - 1) / 19.0, 0.0, 1.0))

    def _interpolate_strength(self, strength: int, low: float, high: float) -> float:
        ratio = self._strength_ratio(strength)
        return float(low + ((high - low) * ratio))

    def _interpolate_inverse_strength(
        self, strength: int, high_value: float, low_value: float
    ) -> float:
        ratio = self._strength_ratio(strength)
        return float(high_value + ((low_value - high_value) * ratio))

    def find_default_index(self, model_name: str) -> str:
        try:
            resolved = self.resolve_model_reference(model_name)
            if str(resolved.get("default_index", "")).strip():
                return str(resolved.get("default_index", ""))
            model_name = str(resolved.get("rvc_model_name", model_name))
        except FileNotFoundError:
            pass
        index_files = [
            file
            for file in self.logs_root.rglob("*.index")
            if file.is_file() and "trained" not in file.name.lower()
        ]
        return self._find_default_index_from_list(model_name, index_files)

    def _ffmpeg_binary(self) -> str:
        local = self.repo_root / ("ffmpeg.exe" if os.name == "nt" else "ffmpeg")
        return str(local) if local.exists() else "ffmpeg"

    def _ffprobe_binary(self) -> str:
        local = self.repo_root / ("ffprobe.exe" if os.name == "nt" else "ffprobe")
        return str(local) if local.exists() else "ffprobe"

    def _separator_python(self) -> Path:
        python_name = "python.exe" if os.name == "nt" else "python"
        return self.separator_env_root / "Scripts" / python_name if os.name == "nt" else self.separator_env_root / "bin" / python_name

    def _ensure_hubert_loaded(self) -> None:
        if self.hubert_model is not None:
            return

        hubert_path = self.repo_root / "hubert_base.pt"
        try:
            from fairseq import checkpoint_utils
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "fairseq is required only for the legacy HuBERT/RVC path and is not installed in the current Python environment."
            ) from exc
        models, _, _ = checkpoint_utils.load_model_ensemble_and_task(
            [str(hubert_path)],
            suffix="",
        )
        self.hubert_model = models[0].to(self.config.device)
        if self.config.is_half:
            self.hubert_model = self.hubert_model.half()
        else:
            self.hubert_model = self.hubert_model.float()
        self.hubert_model.eval()

    def _ensure_model_loaded(self, model_name: str) -> None:
        resolved = self.resolve_model_reference(model_name)
        resolved_model_name = str(resolved["rvc_model_name"])
        resolved_model_path = str(resolved.get("model_path", "") or "")
        if not resolved_model_path and not resolved_model_name:
            raise FileNotFoundError(
                "Selected Persona package does not contain a legacy .pth backbone model."
            )
        model_path = Path(resolved_model_path) if resolved_model_path else (self.weights_root / resolved_model_name)
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")
        model_mtime_ns = int(model_path.stat().st_mtime_ns)
        cache_key = f"{(resolved_model_path or resolved_model_name)}::{model_mtime_ns}"
        if self.current_model_name == cache_key and self.net_g is not None:
            return

        cpt = torch.load(model_path, map_location="cpu")
        tgt_sr = cpt["config"][-1]
        cpt["config"][-3] = cpt["weight"]["emb_g.weight"].shape[0]
        if_f0 = cpt.get("f0", 1)
        version = cpt.get("version", "v1")

        if version == "v1":
            if if_f0 == 1:
                net_g = SynthesizerTrnMs256NSFsid(
                    *cpt["config"], is_half=self.config.is_half
                )
            else:
                net_g = SynthesizerTrnMs256NSFsid_nono(*cpt["config"])
        else:
            if if_f0 == 1:
                net_g = SynthesizerTrnMs768NSFsid(
                    *cpt["config"], is_half=self.config.is_half
                )
            else:
                net_g = SynthesizerTrnMs768NSFsid_nono(*cpt["config"])

        del net_g.enc_q
        net_g.load_state_dict(cpt["weight"], strict=False)
        net_g.eval().to(self.config.device)
        if self.config.is_half:
            net_g = net_g.half()
        else:
            net_g = net_g.float()

        try:
            from vc_infer_pipeline import VC
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "The legacy RVC conversion path requires optional dependencies that are not installed in the current Python environment."
            ) from exc

        self.current_model_name = cache_key
        self.net_g = net_g
        self.cpt = cpt
        self.tgt_sr = tgt_sr
        self.version = version
        self.n_spk = cpt["config"][-3]
        self.vc = VC(tgt_sr, self.config)

    def _load_audio_file(self, file_path: Path, sr: int) -> np.ndarray:
        cleaned = str(file_path).strip().strip('"')
        out, _ = (
            ffmpeg.input(cleaned, threads=0)
            .output("-", format="f32le", acodec="pcm_f32le", ac=1, ar=sr)
            .run(
                cmd=[self._ffmpeg_binary(), "-nostdin"],
                capture_stdout=True,
                capture_stderr=True,
            )
        )
        return np.frombuffer(out, np.float32).flatten()

    def _normalize_index_path(self, index_path: str, model_name: str) -> str:
        normalized = (index_path or "").strip().strip('"')
        if not normalized:
            resolved = self.resolve_model_reference(model_name)
            normalized = str(resolved.get("default_index", "") or "")
            if not normalized:
                normalized = self.find_default_index(str(resolved.get("rvc_model_name", model_name)))
        return normalized.replace("trained", "added")

    def _transcode_audio(self, input_wav: Path, output_path: Path) -> None:
        command = [
            self._ffmpeg_binary(),
            "-y",
            "-i",
            str(input_wav),
            str(output_path),
        ]
        subprocess.run(command, check=True, capture_output=True)

    def _build_atempo_filter(self, tempo_ratio: float) -> str:
        ratio = float(tempo_ratio)
        if ratio <= 0:
            raise ValueError("Tempo ratio must be greater than zero.")

        filters: List[str] = []
        while ratio > 2.0:
            filters.append("atempo=2.0")
            ratio /= 2.0
        while ratio < 0.5:
            filters.append("atempo=0.5")
            ratio /= 0.5
        filters.append(f"atempo={ratio:.6f}")
        return ",".join(filters)

    def conform_audio_tempo(
        self,
        input_path: Path,
        output_path: Path,
        *,
        source_bpm: float,
        target_bpm: float,
    ) -> float:
        source_value = float(source_bpm)
        target_value = float(target_bpm)
        if source_value <= 0 or target_value <= 0:
            raise ValueError("Both source and target BPM must be greater than zero.")

        tempo_ratio = target_value / source_value
        command = [
            self._ffmpeg_binary(),
            "-y",
            "-i",
            str(input_path),
            "-filter:a",
            self._build_atempo_filter(tempo_ratio),
            str(output_path),
        ]
        subprocess.run(command, check=True, capture_output=True)
        return float(tempo_ratio)

    def get_audio_duration(self, input_path: Path) -> float:
        try:
            probe = ffmpeg.probe(str(input_path), cmd=self._ffprobe_binary())
        except ffmpeg.Error:
            return 0.0
        streams = probe.get("streams", [])
        format_info = probe.get("format", {})

        durations = []
        for stream in streams:
            duration_value = stream.get("duration")
            if duration_value is None:
                continue
            try:
                durations.append(float(duration_value))
            except (TypeError, ValueError):
                continue

        if durations:
            return max(durations)

        try:
            return float(format_info.get("duration", 0.0))
        except (TypeError, ValueError):
            return 0.0

    def extract_middle_preview_clip(
        self,
        input_path: Path,
        output_path: Path,
        *,
        clip_duration: float = 5.0,
    ) -> Dict[str, float]:
        total_duration = max(0.0, self.get_audio_duration(input_path))
        actual_duration = clip_duration if total_duration <= 0 else min(
            float(clip_duration), total_duration
        )
        start_time = max(0.0, (total_duration - actual_duration) / 2.0)

        output_path.parent.mkdir(parents=True, exist_ok=True)
        command = [
            self._ffmpeg_binary(),
            "-y",
            "-ss",
            f"{start_time:.3f}",
            "-i",
            str(input_path),
            "-t",
            f"{actual_duration:.3f}",
            "-vn",
            "-acodec",
            "pcm_s16le",
            "-ac",
            "2",
            "-ar",
            "44100",
            str(output_path),
        ]
        subprocess.run(command, check=True, capture_output=True)
        return {
            "start": round(start_time, 2),
            "duration": round(actual_duration, 2),
            "total_duration": round(total_duration, 2),
        }

    def _ensure_uvr_model_file(self, model_key: str) -> Path:
        model_info = self.UVR_MODELS[model_key]
        model_path = self.uvr5_root / str(model_info["filename"])
        if model_path.exists():
            return model_path

        self.uvr5_root.mkdir(parents=True, exist_ok=True)
        urlretrieve(str(model_info["url"]), model_path)
        return model_path

    def _select_preprocess_roformer_model(self) -> str:
        available_models = {
            path.name
            for path in self.separator_model_root.glob("*.ckpt")
            if path.is_file()
        }
        for model_name in self.PREPROCESS_ROFORMER_PREFERRED_MODELS:
            if model_name in available_models:
                return model_name
        return self.HIGH_END_ROFORMER_MODEL

    def _extract_vocals_high_end(
        self,
        input_path: Path,
        output_dir: Path,
        *,
        model_name: Optional[str] = None,
        segment_size: int = 256,
        overlap: float = 0.75,
        enable_denoise: bool = True,
    ) -> Dict[str, str]:
        separator_python = self._separator_python()
        if not separator_python.exists():
            raise RuntimeError(
                "The high-end separator environment is missing. Expected "
                f"{separator_python}"
            )
        if not self.separator_runner.exists():
            raise RuntimeError(
                f"High-end separator runner is missing: {self.separator_runner}"
            )

        chosen_model = model_name or self._select_preprocess_roformer_model()
        output_dir.mkdir(parents=True, exist_ok=True)
        command = [
            str(separator_python),
            str(self.separator_runner),
            "--input",
            str(input_path),
            "--output-dir",
            str(output_dir),
            "--model-dir",
            str(self.separator_model_root),
            "--model",
            chosen_model,
            "--vocals-name",
            "vocals",
            "--instrumental-name",
            "instrumental",
            "--segment-size",
            str(int(segment_size)),
            "--overlap",
            f"{float(overlap):.2f}",
        ]
        if enable_denoise:
            command.append("--denoise")
        try:
            completed = subprocess.run(
                command,
                check=True,
                capture_output=True,
                text=True,
                cwd=str(self.repo_root),
            )
        except subprocess.CalledProcessError as exc:
            stderr = (exc.stderr or "").strip()
            stdout = (exc.stdout or "").strip()
            detail = stderr or stdout or str(exc)
            raise RuntimeError(
                "RoFormer high-end extraction failed. "
                f"Command: {' '.join(command)}. "
                f"Details: {detail[:2000]}"
            ) from exc
        stdout = (completed.stdout or "").strip().splitlines()
        payload_line = stdout[-1] if stdout else "{}"
        payload = json.loads(payload_line)
        outputs = [Path(item) for item in payload.get("outputs", [])]

        vocals_path = next(
            (path for path in outputs if "vocals" in path.stem.lower()),
            None,
        )
        instrumental_path = next(
            (path for path in outputs if "instrumental" in path.stem.lower()),
            None,
        )
        if vocals_path is None or instrumental_path is None:
            raise RuntimeError(
                "High-end separator did not return both vocals and instrumental stems."
            )

        return {
            "vocals": str(vocals_path),
            "instrumental": str(instrumental_path),
            "model": str(payload.get("model", chosen_model)),
            "segment_size": int(payload.get("segment_size", int(segment_size))),
            "overlap": float(payload.get("overlap", float(overlap))),
            "denoise": bool(payload.get("denoise", enable_denoise)),
        }

    def _ensure_uvr_separator(self, model_key: str, strength: int = 10):
        cache_key = (model_key, int(strength))
        separator = self.separator_cache.get(cache_key)
        if separator is not None:
            return separator

        model_path = self._ensure_uvr_model_file(model_key)
        model_info = self.UVR_MODELS[model_key]

        if bool(model_info["deecho"]):
            from infer_uvr5 import _audio_pre_new

            separator = _audio_pre_new(
                agg=int(strength),
                model_path=str(model_path),
                device=self.config.device,
                is_half=self.config.is_half,
            )
        else:
            from infer_uvr5 import _audio_pre_

            separator = _audio_pre_(
                agg=int(strength),
                model_path=str(model_path),
                device=self.config.device,
                is_half=self.config.is_half,
            )

        self.separator_cache[cache_key] = separator
        return separator

    def _ensure_best_deecho_separator(self, strength: int):
        preferred_model_key = "deecho-dereverb"
        try:
            return self._ensure_uvr_separator(preferred_model_key, strength), preferred_model_key
        except Exception:
            fallback_model_key = (
                "deecho-aggressive" if int(strength) >= 12 else "deecho-normal"
            )
            return self._ensure_uvr_separator(fallback_model_key, strength), fallback_model_key

    def _prepare_uvr_input(self, input_path: Path, work_dir: Path) -> Path:
        reset_directory(work_dir)
        prepared_path = work_dir / f"{input_path.stem}_uvr_input.wav"
        command = [
            self._ffmpeg_binary(),
            "-y",
            "-i",
            str(input_path),
            "-vn",
            "-acodec",
            "pcm_s16le",
            "-ac",
            "2",
            "-ar",
            "44100",
            str(prepared_path),
        ]
        subprocess.run(command, check=True, capture_output=True)
        return prepared_path

    def _create_width_focused_vocal_input(
        self, prepared_path: Path, work_dir: Path, strength: int
    ) -> Path:
        focused_path = work_dir / f"{prepared_path.stem}_focused.wav"
        reset_directory(work_dir)

        audio, sample_rate = sf.read(str(prepared_path), always_2d=True)
        if audio.shape[1] < 2:
            sf.write(str(focused_path), audio, sample_rate)
            return focused_path

        left = audio[:, 0].astype(np.float32)
        right = audio[:, 1].astype(np.float32)
        mid = 0.5 * (left + right)
        side = 0.5 * (left - right)

        strength_ratio = np.clip(float(strength) / 20.0, 0.0, 1.0)
        window = max(128, int(sample_rate * 0.028))
        kernel = np.ones(window, dtype=np.float32) / float(window)
        mid_energy = np.convolve(np.abs(mid), kernel, mode="same")
        side_energy = np.convolve(np.abs(side), kernel, mode="same")
        width_ratio = side_energy / (mid_energy + side_energy + 1e-6)

        width_trigger = 0.26 - (0.13 * strength_ratio)
        width_trigger = float(np.clip(width_trigger, 0.08, 0.28))
        width_mask = np.clip(
            (width_ratio - width_trigger) / max(0.08, 0.55 - width_trigger),
            0.0,
            1.0,
        ).astype(np.float32)

        side_scale = 1.0 - ((0.58 + (0.32 * strength_ratio)) * width_mask)
        side_scale = np.clip(side_scale, 0.06, 1.0)
        mono_pull = (0.18 + (0.52 * strength_ratio)) * width_mask

        focused_left = mid + (side * side_scale)
        focused_right = mid - (side * side_scale)
        focused_left = ((1.0 - mono_pull) * focused_left) + (mono_pull * mid)
        focused_right = ((1.0 - mono_pull) * focused_right) + (mono_pull * mid)

        focused_audio = np.stack([focused_left, focused_right], axis=1)
        peak = float(np.max(np.abs(focused_audio)) + 1e-9)
        if peak > 1.0:
            focused_audio = focused_audio / peak

        sf.write(str(focused_path), focused_audio.astype(np.float32), sample_rate)
        return focused_path

    def _find_uvr_output(self, root: Path, prefix: str) -> Path:
        matches = sorted(root.glob(f"{prefix}_*.wav"))
        if not matches:
            raise RuntimeError(f"UVR did not create a {prefix} output file.")
        return matches[-1]

    def _score_vocal_likelihood(self, file_path: Path) -> float:
        try:
            audio, sample_rate = sf.read(str(file_path), always_2d=True)
        except Exception:
            return float("-inf")
        if audio.size == 0:
            return float("-inf")

        mono = audio.mean(axis=1).astype(np.float32, copy=False)
        if mono.size < 8:
            return float("-inf")

        rms = float(np.sqrt(np.mean(np.square(mono)) + 1e-9))
        peak = float(np.max(np.abs(mono)) + 1e-9)
        if peak <= 1e-7:
            return float("-inf")

        silence_ratio = float(np.mean(np.abs(mono) < (peak * 0.03)))
        spectrum = np.abs(np.fft.rfft(mono))
        freqs = np.fft.rfftfreq(mono.shape[0], d=1.0 / max(int(sample_rate), 1))
        total_energy = float(np.sum(spectrum) + 1e-9)
        vocal_band = (freqs >= 120.0) & (freqs <= 5500.0)
        low_band = freqs < 110.0
        vocal_ratio = float(np.sum(spectrum[vocal_band]) / total_energy) if np.any(vocal_band) else 0.0
        low_ratio = float(np.sum(spectrum[low_band]) / total_energy) if np.any(low_band) else 0.0

        score = (
            (rms * 7.5)
            + (vocal_ratio * 4.0)
            - (low_ratio * 2.2)
            - (silence_ratio * 1.8)
        )
        return float(score)

    def _describe_output_candidate(
        self,
        file_path: Path,
        reference_path: Optional[Path] = None,
    ) -> Dict[str, float]:
        try:
            audio, sample_rate = sf.read(str(file_path), always_2d=True)
        except Exception:
            return {
                "vocal_score": float("-inf"),
                "rms": 0.0,
                "width_ratio": 1.0,
                "correlation": 0.0,
                "rms_closeness": 0.0,
                "final_score": float("-inf"),
            }
        if audio.size == 0:
            return {
                "vocal_score": float("-inf"),
                "rms": 0.0,
                "width_ratio": 1.0,
                "correlation": 0.0,
                "rms_closeness": 0.0,
                "final_score": float("-inf"),
            }

        fixed = self._ensure_2d_audio(audio)
        mono = fixed.mean(axis=1).astype(np.float32, copy=False)
        rms = float(np.sqrt(np.mean(np.square(mono)) + 1e-9))

        if fixed.shape[1] >= 2:
            mid = 0.5 * (fixed[:, 0] + fixed[:, 1])
            side = 0.5 * (fixed[:, 0] - fixed[:, 1])
            mid_rms = float(np.sqrt(np.mean(np.square(mid)) + 1e-9))
            side_rms = float(np.sqrt(np.mean(np.square(side)) + 1e-9))
            width_ratio = float(side_rms / max(mid_rms, 1e-6))
        else:
            width_ratio = 0.0

        correlation = 0.0
        rms_closeness = 0.0
        if reference_path is not None and reference_path.exists():
            try:
                reference_audio, reference_sr = sf.read(str(reference_path), always_2d=True)
                reference_fixed = self._ensure_2d_audio(reference_audio)
                if int(reference_sr) == int(sample_rate) and reference_fixed.size > 0:
                    reference_mono = reference_fixed.mean(axis=1).astype(np.float32, copy=False)
                    usable = min(reference_mono.shape[0], mono.shape[0])
                    if usable > 256:
                        step = max(1, usable // 4096)
                        reference_env = np.abs(reference_mono[:usable:step])
                        candidate_env = np.abs(mono[:usable:step])
                        if reference_env.size > 8 and candidate_env.size > 8:
                            reference_centered = reference_env - np.mean(reference_env)
                            candidate_centered = candidate_env - np.mean(candidate_env)
                            reference_std = float(np.std(reference_centered))
                            candidate_std = float(np.std(candidate_centered))
                            if reference_std > 1e-7 and candidate_std > 1e-7:
                                correlation = float(
                                    np.clip(
                                        np.corrcoef(reference_centered, candidate_centered)[0, 1],
                                        -1.0,
                                        1.0,
                                    )
                                )
                                correlation = max(0.0, correlation)
                    reference_rms = float(np.sqrt(np.mean(np.square(reference_mono)) + 1e-9))
                    if reference_rms > 1e-7 and rms > 1e-7:
                        rms_closeness = float(
                            np.clip(
                                1.0 - (abs(np.log(rms / reference_rms)) / 2.6),
                                0.0,
                                1.0,
                            )
                        )
            except Exception:
                correlation = 0.0
                rms_closeness = 0.0

        vocal_score = self._score_vocal_likelihood(file_path)
        center_score = float(np.clip(1.0 - (width_ratio / 1.25), 0.0, 1.0))
        final_score = (
            float(vocal_score)
            + (correlation * 4.2)
            + (rms_closeness * 2.4)
            + (center_score * 1.4)
        )
        return {
            "vocal_score": float(vocal_score),
            "rms": float(rms),
            "width_ratio": float(width_ratio),
            "correlation": float(correlation),
            "rms_closeness": float(rms_closeness),
            "final_score": float(final_score),
        }

    def _choose_best_vocal_like_output(
        self,
        preferred_path: Path,
        alternate_path: Path,
        *,
        reference_path: Optional[Path] = None,
    ) -> Path:
        preferred_stats = self._describe_output_candidate(preferred_path, reference_path=reference_path)
        alternate_stats = self._describe_output_candidate(alternate_path, reference_path=reference_path)

        # If the preferred stem is at least plausibly vocal-like, keep it unless
        # the alternate is clearly better. This avoids accidentally routing the
        # removed reverb/noise layer in place of the lead.
        if preferred_stats["final_score"] == float("-inf"):
            return alternate_path

        preferred_rms = float(preferred_stats["rms"])
        alternate_rms = float(alternate_stats["rms"])
        if preferred_rms <= 1e-5 and alternate_rms > (preferred_rms * 4.0):
            return alternate_path

        margin = 1.8
        if alternate_stats["final_score"] > (preferred_stats["final_score"] + margin):
            return alternate_path
        return preferred_path

    def _remove_backing_vocals_body_preserving(
        self, input_path: Path, work_dir: Path, aggression: Optional[int] = None
    ) -> Path:
        chosen_aggression = (
            self.PREPROCESS_KARAOKE_AGGRESSION
            if aggression is None
            else max(1, min(int(aggression), 20))
        )
        separator = self._ensure_uvr_separator("karaoke-body", chosen_aggression)
        lead_vocal_dir = work_dir / "lead-vocal"
        removed_backing_dir = work_dir / "removed-backing"
        reset_directory(lead_vocal_dir)
        reset_directory(removed_backing_dir)

        # For karaoke UVR, the instrumental output is the preserved lead vocal pass.
        separator._path_audio_(
            str(input_path),
            ins_root=str(lead_vocal_dir),
            vocal_root=str(removed_backing_dir),
            format="wav",
        )
        lead_path = self._find_uvr_output(lead_vocal_dir, "instrument")
        removed_path = self._find_uvr_output(removed_backing_dir, "vocal")
        return self._choose_best_vocal_like_output(
            lead_path,
            removed_path,
            reference_path=input_path,
        )

    def _polish_lead_vocal_dereverb(
        self,
        input_path: Path,
        work_dir: Path,
        *,
        model_key: str = "deecho-normal",
        strength: Optional[int] = None,
    ) -> Path:
        chosen_strength = (
            self.PREPROCESS_DEECHO_AGGRESSION
            if strength is None
            else max(1, min(int(strength), 20))
        )
        separator = self._ensure_uvr_separator(model_key, chosen_strength)
        cleaned_dir = work_dir / "cleaned"
        removed_dir = work_dir / "removed"
        reset_directory(cleaned_dir)
        reset_directory(removed_dir)

        # FoxJoy's De-Echo models return the cleaned vocal in the instrumental output.
        separator._path_audio_(
            str(input_path),
            vocal_root=str(removed_dir),
            ins_root=str(cleaned_dir),
            format="wav",
        )
        cleaned_path = self._find_uvr_output(cleaned_dir, "instrument")
        removed_path = self._find_uvr_output(removed_dir, "vocal")
        return self._choose_best_vocal_like_output(
            cleaned_path,
            removed_path,
            reference_path=input_path,
        )

    def get_isolator_options(self) -> Dict[str, object]:
        return {
            "modes": [
                {
                    "id": mode_id,
                    "label": str(info["label"]),
                    "description": str(info["description"]),
                }
                for mode_id, info in self.ISOLATOR_MODES.items()
            ],
            "input_types": [
                {
                    "id": "vocal-stem",
                    "label": "Vocal stem",
                    "description": "Use this when the file is already mostly vocals with little or no beat left.",
                },
                {
                    "id": "full-mix",
                    "label": "Song with instrumental",
                    "description": "Use this when vocals are still sitting on top of music or a beat.",
                },
            ],
            "defaults": {
                "mode": "main-vocal",
                "input_type": "full-mix",
                "strength": 10,
                "deecho": True,
                "width_focus": True,
                "clarity_preserve": 70,
            },
        }

    def _ensure_2d_audio(self, audio: np.ndarray) -> np.ndarray:
        if audio.ndim == 1:
            return audio[:, None].astype(np.float32)
        return audio.astype(np.float32)

    def _match_audio_shape(
        self, audio: np.ndarray, target_shape: tuple[int, int]
    ) -> np.ndarray:
        fixed = self._ensure_2d_audio(audio)
        target_frames, target_channels = target_shape

        if fixed.shape[1] < target_channels:
            fixed = np.repeat(fixed, target_channels, axis=1)
        elif fixed.shape[1] > target_channels:
            fixed = fixed[:, :target_channels]

        if fixed.shape[0] < target_frames:
            pad = np.zeros(
                (target_frames - fixed.shape[0], fixed.shape[1]), dtype=np.float32
            )
            fixed = np.vstack([fixed, pad])
        elif fixed.shape[0] > target_frames:
            fixed = fixed[:target_frames]

        return fixed.astype(np.float32)

    def _write_normalized_audio(
        self, output_path: Path, audio: np.ndarray, sample_rate: int
    ) -> None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        final_audio = np.asarray(audio, dtype=np.float32)
        peak = float(np.max(np.abs(final_audio)) + 1e-9)
        if peak > 1.0:
            final_audio = final_audio / peak
        sf.write(str(output_path), final_audio, sample_rate)

    def _rms_level(self, audio: np.ndarray) -> float:
        fixed = self._ensure_2d_audio(audio)
        return float(np.sqrt(np.mean(np.square(fixed), dtype=np.float64) + 1e-9))

    def _restore_stem_presence(
        self,
        processed_audio: np.ndarray,
        dry_reference_audio: np.ndarray,
        *,
        preserve_amount: int,
        strength: int,
    ) -> np.ndarray:
        processed = self._ensure_2d_audio(processed_audio)
        reference = self._match_audio_shape(dry_reference_audio, processed.shape)
        preserve_ratio = np.clip(float(preserve_amount) / 100.0, 0.0, 1.0)
        strength_ratio = np.clip(float(strength) / 20.0, 0.0, 1.0)

        dry_blend = 0.08 + (0.34 * preserve_ratio) - (0.10 * strength_ratio)
        dry_blend = float(np.clip(dry_blend, 0.08, 0.42))
        combined = ((1.0 - dry_blend) * processed) + (dry_blend * reference)

        processed_rms = self._rms_level(combined)
        reference_rms = self._rms_level(reference)
        if processed_rms > 1e-6 and reference_rms > 1e-6:
            target_gain = reference_rms / processed_rms
            max_gain = 1.10 + (0.90 * preserve_ratio)
            gain = float(np.clip(target_gain, 0.90, max_gain))
            combined = combined * gain

        peak = float(np.max(np.abs(combined)) + 1e-9)
        if peak > 1.0:
            combined = combined / peak
        return combined.astype(np.float32)

    def _blend_preprocess_presence(
        self,
        processed_path: Path,
        reference_path: Path,
        output_path: Path,
        *,
        preserve_amount: int,
        strength: int,
    ) -> Path:
        processed_audio, sample_rate = sf.read(str(processed_path), always_2d=True)
        reference_audio, reference_sample_rate = sf.read(
            str(reference_path), always_2d=True
        )
        if int(reference_sample_rate) != int(sample_rate):
            raise RuntimeError(
                "Preprocess stage sample rates do not match for body restoration."
            )
        restored_audio = self._restore_stem_presence(
            processed_audio,
            reference_audio,
            preserve_amount=preserve_amount,
            strength=strength,
        )
        self._write_normalized_audio(output_path, restored_audio, sample_rate)
        return output_path

    def preprocess_lead_vocals(
        self, input_path: Path, work_dir: Path, strength: int = 10
    ) -> Path:
        separator = self._ensure_uvr_separator("lead-vocal", strength)
        prepared_path = self._prepare_uvr_input(input_path, work_dir / "prepared")
        vocal_dir = work_dir / "lead-vocal"
        instrumental_dir = work_dir / "backing-removed"
        reset_directory(vocal_dir)
        reset_directory(instrumental_dir)
        separator._path_audio_(
            str(prepared_path),
            ins_root=str(instrumental_dir),
            vocal_root=str(vocal_dir),
            format="wav",
        )
        return self._find_uvr_output(vocal_dir, "vocal")

    def preprocess_adlib_cleanup(
        self, input_path: Path, work_dir: Path, strength: int = 10
    ) -> Path:
        return self.preprocess_for_conversion_pipeline(
            input_path,
            work_dir,
            preprocess_mode="balanced-clean",
            strength=strength,
        )

    def preprocess_for_conversion_pipeline(
        self,
        input_path: Path,
        work_dir: Path,
        *,
        preprocess_mode: str,
        strength: int = 10,
    ) -> Path:
        pipeline_id = self.normalize_preprocess_mode(preprocess_mode)
        strength = max(1, min(int(strength), 20))

        if pipeline_id == "off":
            return input_path
        if pipeline_id == "legacy-lead":
            return self.preprocess_lead_vocals(input_path, work_dir, strength=strength)

        prepared_path = self._prepare_uvr_input(input_path, work_dir / "prepared")
        extracted_vocal_path = prepared_path
        extracted_ok = False

        if pipeline_id in {
            "extract-only",
            "fullness-first",
            "reverb-polish",
            "balanced-clean",
            "max-clean",
        }:
            stage1_dir = work_dir / "step1-high-fidelity"
            try:
                extracted = self._extract_vocals_high_end(
                    prepared_path,
                    stage1_dir,
                    model_name=self._select_preprocess_roformer_model(),
                    segment_size=self.PREPROCESS_ROFORMER_SEGMENT_SIZE,
                    overlap=self.PREPROCESS_ROFORMER_OVERLAP,
                    enable_denoise=self.PREPROCESS_ROFORMER_DENOISE,
                )
                extracted_vocal_path = Path(str(extracted["vocals"]))
                extracted_ok = True
            except Exception as exc:
                print(
                    "[preprocess] RoFormer extraction failed, using the normalized input "
                    f"for pipeline {pipeline_id}. Reason: {exc}"
                )
                extracted_vocal_path = prepared_path

        if pipeline_id == "extract-only":
            return extracted_vocal_path

        if pipeline_id == "reverb-polish":
            polish_strength = int(
                round(self._interpolate_strength(strength, 4.0, 7.0))
            )
            try:
                polished_path = self._polish_lead_vocal_dereverb(
                    extracted_vocal_path,
                    work_dir / "step2-reverb-polish",
                    model_key="deecho-normal",
                    strength=polish_strength,
                )
            except Exception as exc:
                print(
                    "[preprocess] Reverb polish failed, using the extracted vocal "
                    f"directly. Reason: {exc}"
                )
                return extracted_vocal_path

            preserve_amount = int(
                round(self._interpolate_inverse_strength(strength, 90.0, 72.0))
            )
            reference_path = extracted_vocal_path if extracted_ok else prepared_path
            return self._blend_preprocess_presence(
                polished_path,
                reference_path,
                work_dir / "final" / "reverb-polish.wav",
                preserve_amount=preserve_amount,
                strength=strength,
            )

        karaoke_strength = int(round(self._interpolate_strength(strength, 4.0, 6.0)))
        try:
            lead_vocal_path = self._remove_backing_vocals_body_preserving(
                extracted_vocal_path,
                work_dir / "step2-backing-removal",
                aggression=karaoke_strength,
            )
        except Exception as exc:
            print(
                "[preprocess] Backing-vocal suppression failed, using the extracted vocal "
                f"directly for pipeline {pipeline_id}. Reason: {exc}"
            )
            lead_vocal_path = extracted_vocal_path

        if pipeline_id == "fullness-first":
            preserve_amount = int(
                round(self._interpolate_inverse_strength(strength, 92.0, 76.0))
            )
            reference_path = extracted_vocal_path if extracted_ok else prepared_path
            return self._blend_preprocess_presence(
                lead_vocal_path,
                reference_path,
                work_dir / "final" / "fullness-first.wav",
                preserve_amount=preserve_amount,
                strength=strength,
            )

        deecho_model = "deecho-normal"
        deecho_strength_low = 4.0
        deecho_strength_high = 7.0
        preserve_high = 82.0
        preserve_low = 62.0
        final_name = "balanced-clean.wav"
        reference_path = lead_vocal_path
        if pipeline_id == "max-clean":
            deecho_model = "deecho-dereverb"
            deecho_strength_low = 7.0
            deecho_strength_high = 10.0
            preserve_high = 66.0
            preserve_low = 46.0
            final_name = "max-clean.wav"

        deecho_strength = int(
            round(self._interpolate_strength(strength, deecho_strength_low, deecho_strength_high))
        )
        try:
            polished_path = self._polish_lead_vocal_dereverb(
                lead_vocal_path,
                work_dir / "step3-dereverb",
                model_key=deecho_model,
                strength=deecho_strength,
            )
        except Exception as exc:
            print(
                "[preprocess] De-echo / de-reverb failed, using the lead-vocal pass "
                f"directly for pipeline {pipeline_id}. Reason: {exc}"
            )
            polished_path = lead_vocal_path

        preserve_amount = int(
            round(self._interpolate_inverse_strength(strength, preserve_high, preserve_low))
        )
        return self._blend_preprocess_presence(
            polished_path,
            reference_path,
            work_dir / "final" / final_name,
            preserve_amount=preserve_amount,
            strength=strength,
        )

    def isolate_vocals(
        self,
        input_path: Path,
        output_dir: Path,
        *,
        mode: str = "main-vocal",
        input_type: str = "full-mix",
        strength: int = 10,
        deecho: bool = False,
        width_focus: bool = True,
        clarity_preserve: int = 70,
        update_progress: Optional[Callable[[str, int], None]] = None,
    ) -> Dict[str, object]:
        chosen_mode = mode if mode in self.ISOLATOR_MODES else "main-vocal"
        is_reverb_echo_only = chosen_mode == "reverb-echo-only"
        model_key = str(self.ISOLATOR_MODES[chosen_mode]["model_key"])
        strength = max(1, min(int(strength), 20))
        chosen_input_type = (
            input_type if input_type in {"vocal-stem", "full-mix"} else "full-mix"
        )
        if is_reverb_echo_only:
            # This cleanup mode is meant for vocal-only files.
            chosen_input_type = "vocal-stem"
            deecho = True
        clarity_preserve = max(0, min(int(clarity_preserve), 100))

        working_dir = output_dir / "work"
        if update_progress is not None:
            update_progress("Normalizing the input audio for separation...", 32)
        prepared_path = self._prepare_uvr_input(input_path, working_dir / "prepared")
        source_path = prepared_path

        high_end_dir = (
            working_dir / "high-end-full-mix"
            if chosen_input_type == "full-mix"
            else working_dir / "high-end-vocal-stem"
        )
        reset_directory(high_end_dir)
        high_end_used = True
        high_end_warning = ""
        if update_progress is not None:
            update_progress("Running RoFormer vocal isolation...", 46)
        try:
            extracted = self._extract_vocals_high_end(prepared_path, high_end_dir)
            source_path = Path(str(extracted["vocals"]))
        except Exception as exc:
            high_end_used = False
            high_end_warning = str(exc)
            print(
                "[isolator] RoFormer extraction failed, falling back to UVR path. "
                f"Reason: {exc}"
            )
            if chosen_input_type == "full-mix":
                if update_progress is not None:
                    update_progress(
                        "RoFormer unavailable. Falling back to UVR vocal extraction...",
                        52,
                    )
                fallback_main_dir = working_dir / "fallback-main-vocal"
                fallback_backing_dir = working_dir / "fallback-backing"
                reset_directory(fallback_main_dir)
                reset_directory(fallback_backing_dir)
                fallback_separator = self._ensure_uvr_separator("lead-vocal", strength)
                fallback_separator._path_audio_(
                    str(prepared_path),
                    ins_root=str(fallback_backing_dir),
                    vocal_root=str(fallback_main_dir),
                    format="wav",
                )
                source_path = self._find_uvr_output(fallback_main_dir, "vocal")
            else:
                if update_progress is not None:
                    update_progress(
                        "RoFormer unavailable. Continuing with the uploaded vocal stem...",
                        52,
                    )
                source_path = prepared_path

        if width_focus:
            focus_dir = (
                working_dir / "focused-vocals"
                if chosen_input_type == "full-mix"
                else working_dir / "focused"
            )
            if update_progress is not None:
                update_progress("Analyzing stereo width and focusing the lead...", 58)
            source_path = self._create_width_focused_vocal_input(
                source_path, focus_dir, strength
            )

        if is_reverb_echo_only:
            if update_progress is not None:
                update_progress("Cleaning reverb and echo tails from the vocal...", 74)
            deecho_separator, _ = self._ensure_best_deecho_separator(strength)
            deecho_clean_dir = working_dir / "deecho-clean"
            deecho_removed_dir = working_dir / "deecho-removed"
            reset_directory(deecho_clean_dir)
            reset_directory(deecho_removed_dir)
            deecho_separator._path_audio_(
                str(source_path),
                vocal_root=str(deecho_removed_dir),
                ins_root=str(deecho_clean_dir),
                format="wav",
            )

            clean_main_path = self._find_uvr_output(deecho_clean_dir, "instrument")
            removed_echo_path = self._find_uvr_output(deecho_removed_dir, "vocal")

            if update_progress is not None:
                update_progress("Finalizing cleaned vocal and removed-echo layer...", 94)
            source_audio, sample_rate = sf.read(str(source_path), always_2d=True)
            cleaned_audio, _ = sf.read(str(clean_main_path), always_2d=True)
            removed_echo_audio, _ = sf.read(str(removed_echo_path), always_2d=True)
            reference_shape = self._ensure_2d_audio(source_audio).shape
            matched_clean = self._match_audio_shape(cleaned_audio, reference_shape)
            matched_source = self._match_audio_shape(source_audio, reference_shape)
            matched_removed_echo = self._match_audio_shape(
                removed_echo_audio, reference_shape
            )
            restored_main = self._restore_stem_presence(
                matched_clean,
                matched_source,
                preserve_amount=clarity_preserve,
                strength=strength,
            )

            main_output_path = output_dir / "main_vocal.wav"
            backing_output_path = output_dir / "backing_vocal.wav"
            self._write_normalized_audio(main_output_path, restored_main, sample_rate)
            self._write_normalized_audio(
                backing_output_path, matched_removed_echo, sample_rate
            )

            return {
                "main_vocal_path": str(main_output_path),
                "backing_vocal_path": str(backing_output_path),
                "sample_rate": sample_rate,
                "mode": chosen_mode,
                "input_type": chosen_input_type,
                "strength": strength,
                "deecho": bool(deecho),
                "width_focus": bool(width_focus),
                "clarity_preserve": clarity_preserve,
                "high_end_used": high_end_used,
                "high_end_warning": high_end_warning,
            }

        if update_progress is not None:
            update_progress("Splitting the lead and backing layers...", 72)
        separator = self._ensure_uvr_separator(model_key, strength)
        raw_main_dir = working_dir / "raw-main"
        raw_backing_dir = working_dir / "raw-backing"
        reset_directory(raw_main_dir)
        reset_directory(raw_backing_dir)
        separator._path_audio_(
            str(source_path),
            ins_root=str(raw_backing_dir),
            vocal_root=str(raw_main_dir),
            format="wav",
        )

        raw_main_path = self._find_uvr_output(raw_main_dir, "vocal")
        raw_backing_path = self._find_uvr_output(raw_backing_dir, "instrument")
        final_main_path = raw_main_path

        if deecho:
            if update_progress is not None:
                update_progress("Cleaning lingering echo and repeated tails...", 84)
            deecho_separator, _ = self._ensure_best_deecho_separator(strength)
            deecho_clean_dir = working_dir / "deecho-clean"
            deecho_removed_dir = working_dir / "deecho-removed"
            reset_directory(deecho_clean_dir)
            reset_directory(deecho_removed_dir)
            deecho_separator._path_audio_(
                str(raw_main_path),
                vocal_root=str(deecho_removed_dir),
                ins_root=str(deecho_clean_dir),
                format="wav",
            )
            final_main_path = self._find_uvr_output(deecho_clean_dir, "instrument")

        if update_progress is not None:
            update_progress("Finalizing both stems and restoring vocal presence...", 94)
        split_input_audio, sample_rate = sf.read(str(source_path), always_2d=True)
        main_audio, _ = sf.read(str(final_main_path), always_2d=True)
        raw_main_audio, _ = sf.read(str(raw_main_path), always_2d=True)
        backing_audio, _ = sf.read(str(raw_backing_path), always_2d=True)

        reference_shape = self._ensure_2d_audio(split_input_audio).shape
        matched_main = self._match_audio_shape(main_audio, reference_shape)
        matched_raw_main = self._match_audio_shape(raw_main_audio, reference_shape)
        matched_backing = self._match_audio_shape(backing_audio, reference_shape)
        restored_main = self._restore_stem_presence(
            matched_main,
            matched_raw_main,
            preserve_amount=clarity_preserve,
            strength=strength,
        )

        residual_backing = self._ensure_2d_audio(split_input_audio) - restored_main
        blend_ratio = 0.72 if deecho else 0.84
        final_backing = (blend_ratio * matched_backing) + (
            (1.0 - blend_ratio) * residual_backing
        )
        backing_reference = self._ensure_2d_audio(split_input_audio) - matched_raw_main
        backing_rms = self._rms_level(final_backing)
        reference_backing_rms = self._rms_level(backing_reference)
        if backing_rms > 1e-6 and reference_backing_rms > 1e-6:
            final_backing = final_backing * float(
                np.clip(reference_backing_rms / backing_rms, 0.9, 1.8)
            )

        main_output_path = output_dir / "main_vocal.wav"
        backing_output_path = output_dir / "backing_vocal.wav"
        self._write_normalized_audio(main_output_path, restored_main, sample_rate)
        self._write_normalized_audio(backing_output_path, final_backing, sample_rate)

        return {
            "main_vocal_path": str(main_output_path),
            "backing_vocal_path": str(backing_output_path),
            "sample_rate": sample_rate,
            "mode": chosen_mode,
            "input_type": chosen_input_type,
            "strength": strength,
            "deecho": bool(deecho),
            "width_focus": bool(width_focus),
            "clarity_preserve": clarity_preserve,
            "high_end_used": high_end_used,
            "high_end_warning": high_end_warning,
        }

    def convert_file(
        self,
        model_name: str,
        input_path: Path,
        output_path: Path,
        *,
        preprocess_mode: str = "off",
        preprocess_strength: int = 10,
        work_dir: Optional[Path] = None,
        speaker_id: int = 0,
        transpose: int = 0,
        f0_method: str = "rmvpe",
        index_path: str = "",
        index_rate: float = 0.75,
        filter_radius: int = 3,
        resample_sr: int = 0,
        rms_mix_rate: float = 0.25,
        protect: float = 0.33,
        crepe_hop_length: int = 120,
    ) -> Dict[str, object]:
        with self.lock:
            resolved = self.resolve_model_reference(model_name)
            package_mode = str(resolved.get("package_mode", "") or "").strip().lower()
            if package_mode in DIRECT_GUIDED_PACKAGE_MODES:
                guided_regeneration_path = str(resolved.get("guided_regeneration_path", "") or "").strip()
                guided_config_path = str(resolved.get("guided_regeneration_config_path", "") or "").strip()
                guided_manifest_path = str(resolved.get("manifest_path", "") or "").strip()
                guided_report_path = str(resolved.get("guided_regeneration_report_path", "") or "").strip()
                if not guided_regeneration_path or not guided_manifest_path:
                    raise FileNotFoundError(
                        "Selected direct paired package is missing its guided checkpoint files."
                    )
                output_path.parent.mkdir(parents=True, exist_ok=True)
                temp_wav = (
                    output_path
                    if output_path.suffix.lower() == ".wav"
                    else output_path.with_suffix(".wav")
                )
                synthesis_metadata = self.pipa_store.guided_svs.synthesize_direct_guide_v11(
                    checkpoint_path=Path(guided_regeneration_path),
                    guide_audio_path=input_path,
                    output_path=temp_wav,
                    config_path=(Path(guided_config_path) if guided_config_path else None),
                    manifest_path=(Path(guided_manifest_path) if guided_manifest_path else None),
                    training_report_path=(Path(guided_report_path) if guided_report_path else None),
                )
                if output_path.suffix.lower() != ".wav":
                    self._transcode_audio(temp_wav, output_path)
                    temp_wav.unlink(missing_ok=True)
                return {
                    "sample_rate": int(synthesis_metadata.get("sample_rate", 44100)),
                    "index_path": "",
                    "timings": {
                        "npy": 0.0,
                        "f0": 0.0,
                        "infer": round(float(synthesis_metadata.get("synthesis_seconds", 0.0)), 2),
                    },
                    "preprocess_applied": False,
                    "preprocess_mode": "off",
                }

            self._ensure_hubert_loaded()
            self._ensure_model_loaded(model_name)

            normalized_preprocess_mode = self.normalize_preprocess_mode(preprocess_mode)
            source_path = input_path
            preprocess_applied = False
            if normalized_preprocess_mode != "off":
                if work_dir is None:
                    raise RuntimeError("A work directory is required for preprocessing.")
                source_path = self.preprocess_for_conversion_pipeline(
                    input_path,
                    work_dir,
                    preprocess_mode=normalized_preprocess_mode,
                    strength=preprocess_strength,
                )
                preprocess_applied = True

            audio = self._load_audio_file(source_path, 16000)
            audio_max = np.abs(audio).max() / 0.95 if audio.size else 0
            if audio_max > 1:
                audio /= audio_max

            normalized_index = self._normalize_index_path(index_path, model_name)
            times = [0, 0, 0]
            if_f0 = self.cpt.get("f0", 1)
            audio_opt = self.vc.pipeline(
                self.hubert_model,
                self.net_g,
                speaker_id,
                audio,
                str(source_path),
                times,
                int(transpose),
                f0_method,
                normalized_index,
                float(index_rate),
                if_f0,
                int(filter_radius),
                self.tgt_sr,
                int(resample_sr),
                float(rms_mix_rate),
                self.version,
                float(protect),
                int(crepe_hop_length),
            )

            final_sr = self.tgt_sr
            if self.tgt_sr != int(resample_sr) and int(resample_sr) >= 16000:
                final_sr = int(resample_sr)

            output_path.parent.mkdir(parents=True, exist_ok=True)
            temp_wav = (
                output_path
                if output_path.suffix.lower() == ".wav"
                else output_path.with_suffix(".wav")
            )
            sf.write(str(temp_wav), audio_opt, final_sr)

            if output_path.suffix.lower() != ".wav":
                self._transcode_audio(temp_wav, output_path)
                temp_wav.unlink(missing_ok=True)

            return {
                "sample_rate": final_sr,
                "index_path": normalized_index
                if Path(normalized_index).exists()
                else "",
                "timings": {
                    "npy": round(times[0], 2),
                    "f0": round(times[1], 2),
                    "infer": round(times[2], 2),
                },
                "preprocess_applied": preprocess_applied,
                "preprocess_mode": normalized_preprocess_mode,
            }


def sanitize_filename(name: str) -> str:
    cleaned = Path(name).name
    safe = []
    for char in cleaned:
        if char.isalnum() or char in ("-", "_", ".", " "):
            safe.append(char)
        else:
            safe.append("_")
    return "".join(safe).strip() or "audio.wav"


def create_zip(zip_path: Path, source_dir: Path) -> None:
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zip_file:
        for file_path in sorted(source_dir.rglob("*")):
            if file_path.is_dir():
                continue
            zip_file.write(file_path, arcname=file_path.relative_to(source_dir))


def reset_directory(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path, ignore_errors=True)
    path.mkdir(parents=True, exist_ok=True)
