from __future__ import annotations

import math
import os
import re
import subprocess
import sys
import threading
import time
import shutil
from pathlib import Path
from typing import Callable, Dict, List, Optional
from urllib.request import urlretrieve

import faiss
import numpy as np
import torch

from simple_modes import ALIGNED_SUNO_MODE, NORMAL_RVC_MODE
from sklearn.cluster import MiniBatchKMeans
from scipy.io import wavfile


class StageInterruptedError(InterruptedError):
    def __init__(self, stage_name: str, message: str):
        super().__init__(message)
        self.stage_name = stage_name


class SimpleTrainer:
    AUDIO_EXTENSIONS = {
        ".wav",
        ".mp3",
        ".flac",
        ".ogg",
        ".m4a",
        ".aac",
        ".opus",
        ".wma",
    }
    SAMPLE_RATE_MAP = {
        "32k": 32000,
        "40k": 40000,
        "48k": 48000,
    }
    PRETRAIN_BASE_URL = (
        "https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main"
    )

    def __init__(self, repo_root: Path):
        self.repo_root = repo_root
        self.logs_root = self.repo_root / "logs"
        self.weights_root = self.repo_root / "weights"
        self.pretrained_root = self.repo_root / "pretrained"
        self.pretrained_v2_root = self.repo_root / "pretrained_v2"
        self.lock = threading.Lock()

    def _make_run_capture_path(self, exp_dir: Path, stage_name: str) -> Path:
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        return exp_dir / f"_{stage_name}_{timestamp}.capture.log"

    def _get_classic_rvc_hardware_profile(
        self,
        *,
        training_slice_count: int,
        sample_rate: str,
        version: str,
        use_f0: bool,
    ) -> Dict[str, object]:
        cpu_count = max(1, os.cpu_count() or 1)
        gpu_memory_gb = 0.0
        if torch.cuda.is_available():
            try:
                gpu_memory_gb = float(torch.cuda.get_device_properties(0).total_memory) / float(1024**3)
            except Exception:
                gpu_memory_gb = 0.0
        worker_cap = 12 if os.name == "nt" else 10
        if not torch.cuda.is_available():
            return {
                "gpu_memory_gb": round(gpu_memory_gb, 2),
                "max_batch_size": min(4, max(1, training_slice_count)),
                "num_workers": min(4, cpu_count),
                "prefetch_factor": 2,
                "pin_memory": False,
                "persistent_workers": False,
                "cache_data_in_gpu": False,
                "enable_cudnn": False,
                "enable_tf32": False,
            }

        base_max_batch = 8 if use_f0 else 10
        if gpu_memory_gb >= 46.0:
            base_max_batch = 20 if use_f0 else 24
        elif gpu_memory_gb >= 30.0:
            base_max_batch = 16 if use_f0 else 20
        elif gpu_memory_gb >= 24.0:
            base_max_batch = 12 if use_f0 else 16

        sample_rate_penalty = 0
        if str(sample_rate) == "48k":
            sample_rate_penalty = 2
        elif str(sample_rate) == "40k":
            sample_rate_penalty = 0
        else:
            sample_rate_penalty = 1
        if str(version) == "v2" and use_f0:
            sample_rate_penalty += 0

        max_batch_size = max(2, base_max_batch - sample_rate_penalty)
        if training_slice_count > 0:
            max_batch_size = min(max_batch_size, max(2, training_slice_count // 4))

        cache_data_in_gpu = bool(
            gpu_memory_gb >= 24.0
            and training_slice_count > 0
            and training_slice_count <= 1800
            and max_batch_size <= 16
        )
        if gpu_memory_gb >= 46.0 and training_slice_count <= 3200:
            cache_data_in_gpu = True

        return {
            "gpu_memory_gb": round(gpu_memory_gb, 2),
            "max_batch_size": int(max_batch_size),
            "num_workers": min(worker_cap, cpu_count),
            "prefetch_factor": 4 if gpu_memory_gb >= 24.0 else 3,
            "pin_memory": True,
            "persistent_workers": True,
            "cache_data_in_gpu": cache_data_in_gpu,
            "enable_cudnn": True,
            "enable_tf32": True,
        }

    def _can_use_gpu_rmvpe(self) -> bool:
        if not torch.cuda.is_available():
            return False
        try:
            total_memory_gb = float(torch.cuda.get_device_properties(0).total_memory) / float(1024**3)
        except Exception:
            return False
        return total_memory_gb >= 20.0

    def get_options(self) -> Dict[str, object]:
        cuda_available = bool(torch.cuda.is_available())
        warning = ""
        if not cuda_available:
            warning = (
                "PyTorch CUDA is not available in this environment, so training will "
                "run on CPU fallback and be much slower."
            )
        return {
            "sample_rates": list(self.SAMPLE_RATE_MAP.keys()),
            "versions": ["v2", "v1"],
            "f0_methods": ["rmvpe", "mangio-crepe", "crepe", "harvest", "dio", "pm"],
            "output_modes": [
                {
                    "id": NORMAL_RVC_MODE,
                    "label": "Normal RVC",
                    "description": "Builds a standard RVC .pth and index from real target-voice clips. Use this when you want a normal voice model for standard conversion.",
                },
                {
                    "id": ALIGNED_SUNO_MODE,
                    "label": "Aligned SUNO to Target",
                    "description": "Uses BASE identity clips plus tightly aligned TARGET and SUNO clips to train a direct SUNO-to-target vocal mapper with DTW-warped pairs, balanced identity sampling, and focused detail windows.",
                },
            ],
            "alignment_tolerances": [
                {
                    "id": "forgiving",
                    "label": "Forgiving",
                    "description": "Most tolerant. Small naming or duration mismatches are accepted so more candidate pairs survive dataset assembly.",
                },
                {
                    "id": "balanced",
                    "label": "Balanced",
                    "description": "Recommended default. Keeps cleaner aligned matches without becoming so strict that the paired dataset collapses.",
                },
                {
                    "id": "strict",
                    "label": "Strict",
                    "description": "Requires stronger aligned-pair confidence before the run is allowed to continue.",
                },
            ],
            "transcript_formats": [
                "Matching .txt files named like each audio clip",
                "CSV or TSV with filename + transcript columns",
                "JSON/JSONL with filename + transcript objects",
            ],
            "phoneme_note": (
                "Normal RVC learns from true target-voice audio only. Aligned SUNO to Target learns a direct SUNO -> "
                "target mapping from matched TARGET/SUNO pairs plus a smaller BASE identity set."
            ),
            "rebuild_note": (
                "Normal RVC packages store a standard .pth and index. Aligned SUNO to Target packages store a direct "
                "guide-conditioned mapper plus metadata describing the BASE/TARGET/SUNO pair set used for training."
            ),
            "training_plan_note": (
                "These modes do not need transcripts or a persona plan. For aligned training, use BASE clips for the "
                "target voice identity plus matching TARGET_xxx and SUNO_xxx clips for paired supervision."
            ),
            "cuda_available": cuda_available,
            "warning": warning,
            "defaults": {
                "output_mode": NORMAL_RVC_MODE,
            },
        }

    def make_unique_experiment_name(self, raw_name: str) -> str:
        cleaned = []
        for char in raw_name.strip():
            if char.isalnum() or char in ("-", "_", " "):
                cleaned.append(char)
            else:
                cleaned.append("_")
        base_name = ("".join(cleaned).strip() or "voice-model").replace(" ", "_")

        candidate = base_name
        counter = 2
        while (
            (self.logs_root / candidate).exists()
            or (self.weights_root / f"{candidate}.pth").exists()
        ):
            candidate = f"{base_name}-{counter}"
            counter += 1
        return candidate

    def run_training(
        self,
        *,
        experiment_name: str,
        dataset_dir: Path,
        sample_rate: str,
        version: str,
        f0_method: str,
        total_epochs: int,
        save_every_epoch: int,
        batch_size: int,
        crepe_hop_length: int = 128,
        build_index: bool = True,
        use_f0: bool = True,
        speaker_id: int = 0,
        update_status: Callable[[str, str, str, int], None],
        cancel_event: Optional[threading.Event] = None,
        checkpoint_callback: Optional[Callable[[Dict[str, object]], None]] = None,
    ) -> Dict[str, str]:
        with self.lock:
            exp_dir = self.logs_root / experiment_name
            exp_dir.mkdir(parents=True, exist_ok=True)

            sr_hz = self.SAMPLE_RATE_MAP[sample_rate]
            cpu_threads = max(1, min(12, os.cpu_count() or 1))
            rmvpe_uses_gpu = bool(use_f0 and f0_method == "rmvpe" and self._can_use_gpu_rmvpe())
            if use_f0 and f0_method == "rmvpe":
                if rmvpe_uses_gpu:
                    cpu_threads = 1
                else:
                    # CPU RMVPE can scale to a few workers, but we still cap it
                    # to keep the legacy extractor stable.
                    cpu_threads = max(1, min(4, cpu_threads))

            extract_capture = self._make_run_capture_path(exp_dir, "extract")
            train_capture = self._make_run_capture_path(exp_dir, "train")
            preprocess_capture = self._make_run_capture_path(exp_dir, "preprocess")

            update_status(
                "prepare-dataset",
                "Preparing the dataset with the classic RVC slicer and chunking pipeline...",
                "",
                32,
            )
            self._prepare_dataset_with_classic_preprocess(
                dataset_dir=dataset_dir,
                exp_dir=exp_dir,
                sample_rate=sr_hz,
                worker_count=cpu_threads,
                update_status=update_status,
                start_progress=32,
                end_progress=46,
                capture_path=preprocess_capture,
                cancel_event=cancel_event,
            )
            training_slice_count = self._count_training_slices(exp_dir)
            classic_profile = self._get_classic_rvc_hardware_profile(
                training_slice_count=training_slice_count,
                sample_rate=sample_rate,
                version=version,
                use_f0=use_f0,
            )
            effective_batch_size = self._effective_batch_size(
                requested_batch_size=batch_size,
                use_f0=use_f0,
                version=version,
                training_slice_count=training_slice_count,
                max_batch_size=int(classic_profile["max_batch_size"]),
            )
            if effective_batch_size != batch_size:
                update_status(
                    "prepare-dataset",
                    (
                        f"Using effective batch size {effective_batch_size} instead of {batch_size} "
                        f"for {training_slice_count} classic RVC slices on this GPU."
                    ),
                    "",
                    47,
                )
            batch_size = effective_batch_size
            update_status(
                "prepare-dataset",
                (
                    f"Classic RVC slices {training_slice_count} | batch {batch_size} | "
                    f"workers {int(classic_profile['num_workers'])} | "
                    f"gpu cache {'on' if bool(classic_profile['cache_data_in_gpu']) else 'off'}"
                ),
                "",
                48,
            )

            extract_log = exp_dir / "extract_f0_feature.log"
            extract_log.write_text("", encoding="utf-8")
            train_log = exp_dir / "train.log"
            train_log.write_text("", encoding="utf-8")

            try:
                if use_f0:
                    update_status(
                        "extract-f0",
                        "Extracting pitch information from the clips...",
                        "",
                        48,
                    )
                    self._run_stage(
                        [
                            self._python_cmd(),
                            "extract_f0_print.py",
                            str(exp_dir),
                            str(cpu_threads),
                            f0_method,
                            str(crepe_hop_length),
                        ],
                        stage_name="extract-f0",
                        progress=56,
                        log_path=extract_log,
                        capture_path=extract_capture,
                        update_status=update_status,
                        cancel_event=cancel_event,
                        env_overrides={
                            "RVC_FORCE_RMVPE_CPU": (
                                "0" if (f0_method == "rmvpe" and rmvpe_uses_gpu) else "1"
                            ),
                            "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
                        },
                    )

                update_status(
                    "extract-features",
                    "Extracting HuBERT features for training...",
                    "",
                    60,
                )
                self._run_stage(
                    [
                        self._python_cmd(),
                        "extract_feature_print.py",
                        "cpu",
                        "1",
                        "0",
                        str(exp_dir),
                        version,
                    ],
                    stage_name="extract-features",
                    progress=70,
                    log_path=extract_log,
                    capture_path=extract_capture,
                    update_status=update_status,
                    cancel_event=cancel_event,
                    env_overrides={
                        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
                    },
                )

                self._write_filelist(
                    experiment_name=experiment_name,
                    version=version,
                    sample_rate=sample_rate,
                    use_f0=use_f0,
                    speaker_id=speaker_id,
                )
                pretrained_g, pretrained_d = self._ensure_pretrained_weights(
                    sample_rate=sample_rate,
                    version=version,
                    use_f0=use_f0,
                )

                update_status(
                    "train",
                    "Training the voice model. This is the long step.",
                    "",
                    76,
                )
                self._run_training_stage(
                    experiment_name=experiment_name,
                    exp_dir=exp_dir,
                    sample_rate=sample_rate,
                    version=version,
                    use_f0=use_f0,
                    batch_size=batch_size,
                    total_epochs=total_epochs,
                    save_every_epoch=save_every_epoch,
                    pretrained_g=pretrained_g,
                    pretrained_d=pretrained_d,
                    progress=88,
                    log_path=train_log,
                    capture_path=train_capture,
                    classic_profile=classic_profile,
                    update_status=update_status,
                    cancel_event=cancel_event,
                    checkpoint_callback=checkpoint_callback,
                )
                stopped_early = False
            except StageInterruptedError as exc:
                if exc.stage_name != "train":
                    raise InterruptedError(str(exc)) from exc
                update_status(
                    "train",
                    "Stop requested. Finalizing the latest saved checkpoint...",
                    self._tail_file(train_log),
                    90,
                )
                stopped_early = True

            checkpoint_files = sorted(exp_dir.glob("G_*.pth"))
            model_path = self.weights_root / f"{experiment_name}.pth"
            if not model_path.exists():
                latest_saved_model = self._find_latest_saved_weight(experiment_name)
                if latest_saved_model is not None:
                    model_path = latest_saved_model
                else:
                    if not checkpoint_files:
                        raise RuntimeError(
                            self._summarize_training_failure(train_log, train_capture)
                            or "Training did not produce any usable checkpoints."
                        )
                    raise RuntimeError(
                        "Training finished but no usable model file was written."
                    )

            index_path: Optional[Path] = None
            index_summary = ""
            if build_index:
                update_status(
                    "index",
                    "Building the search index for inference...",
                    "",
                    92,
                )
                index_summary = self._train_index(experiment_name, version)
                update_status(
                    "index",
                    "Finishing up the model and index...",
                    index_summary,
                    97,
                )
                index_path = self._find_added_index(exp_dir)
            else:
                update_status(
                    "finalize",
                    "Finishing up the voice model without building an index...",
                    "",
                    96,
                )

            return {
                "experiment_name": experiment_name,
                "model_path": str(model_path),
                "index_path": str(index_path) if index_path else "",
                "index_built": "1" if build_index and index_path else "0",
                "index_summary": index_summary,
                "train_log_path": str(train_log),
                "stopped_early": "1" if stopped_early else "0",
            }

    def _run_training_stage(
        self,
        *,
        experiment_name: str,
        exp_dir: Path,
        sample_rate: str,
        version: str,
        use_f0: bool,
        batch_size: int,
        total_epochs: int,
        save_every_epoch: int,
        pretrained_g: Path,
        pretrained_d: Path,
        progress: int,
        log_path: Optional[Path],
        capture_path: Path,
        classic_profile: Dict[str, object],
        update_status: Callable[[str, str, str, int], None],
        cancel_event: Optional[threading.Event] = None,
        checkpoint_callback: Optional[Callable[[Dict[str, object]], None]] = None,
    ) -> None:
        command = [
            self._python_cmd(),
            "train_nsf_sim_cache_sid_load_pretrain.py",
            "-e",
            experiment_name,
            "-sr",
            sample_rate,
            "-f0",
            "1" if use_f0 else "0",
            "-bs",
            str(batch_size),
            "-te",
            str(total_epochs),
            "-se",
            str(save_every_epoch),
            "-pg",
            str(pretrained_g),
            "-pd",
            str(pretrained_d),
            "-l",
            "1",
            "-c",
            "1" if bool(classic_profile.get("cache_data_in_gpu", False)) else "0",
            "-sw",
            "1",
            "-v",
            version,
            "-li",
            str(self._set_log_interval(exp_dir, batch_size)),
        ]
        capture_path.parent.mkdir(parents=True, exist_ok=True)
        capture_path.write_text("", encoding="utf-8")
        metric_state: Dict[str, float] = {}
        last_preview_epoch = 0
        with capture_path.open("a", encoding="utf-8", errors="ignore") as capture:
            capture.write(f"\n\n[train] {' '.join(command)}\n")
            capture.flush()
            env = os.environ.copy()
            env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
            env["RVC_TRAIN_NUM_WORKERS"] = str(int(classic_profile.get("num_workers", 4)))
            env["RVC_TRAIN_PREFETCH_FACTOR"] = str(int(classic_profile.get("prefetch_factor", 2)))
            env["RVC_TRAIN_PIN_MEMORY"] = "1" if bool(classic_profile.get("pin_memory", True)) else "0"
            env["RVC_TRAIN_PERSISTENT_WORKERS"] = (
                "1" if bool(classic_profile.get("persistent_workers", True)) else "0"
            )
            env["RVC_TRAIN_ENABLE_CUDNN"] = "1" if bool(classic_profile.get("enable_cudnn", True)) else "0"
            env["RVC_TRAIN_ENABLE_TF32"] = "1" if bool(classic_profile.get("enable_tf32", True)) else "0"
            process = subprocess.Popen(
                command,
                cwd=str(self.repo_root),
                stdout=capture,
                stderr=subprocess.STDOUT,
                env=env,
            )
            while process.poll() is None:
                if cancel_event is not None and cancel_event.is_set():
                    self._terminate_process(process)
                    raise StageInterruptedError(
                        "train",
                        "Training stopped by user.",
                    )
                source = log_path if log_path and log_path.exists() else capture_path
                tail_text = self._tail_file(source)
                metric_summary = self._summarize_stage_metrics(
                    "train",
                    tail_text,
                    metric_state=metric_state,
                )
                message = "Training is running..."
                if metric_summary:
                    message = f"{message} {metric_summary}"
                update_status(
                    "train",
                    message,
                    tail_text,
                    progress,
                )
                if checkpoint_callback is not None:
                    latest_checkpoint_epoch = self._find_latest_checkpoint_epoch(exp_dir)
                    if latest_checkpoint_epoch > last_preview_epoch:
                        latest_saved_model = self._find_latest_saved_weight(experiment_name)
                        if latest_saved_model is not None and latest_saved_model.exists():
                            try:
                                checkpoint_callback(
                                    {
                                        "epoch": latest_checkpoint_epoch,
                                        "model_path": str(latest_saved_model),
                                    }
                                )
                                last_preview_epoch = latest_checkpoint_epoch
                            except Exception:
                                pass
                time.sleep(0.35)
            return_code = process.wait()

        if checkpoint_callback is not None:
            latest_checkpoint_epoch = self._find_latest_checkpoint_epoch(exp_dir)
            if latest_checkpoint_epoch > last_preview_epoch:
                latest_saved_model = self._find_latest_saved_weight(experiment_name)
                if latest_saved_model is not None and latest_saved_model.exists():
                    try:
                        checkpoint_callback(
                            {
                                "epoch": latest_checkpoint_epoch,
                                "model_path": str(latest_saved_model),
                            }
                        )
                    except Exception:
                        pass

        source = log_path if log_path and log_path.exists() else capture_path
        final_log = self._tail_file(source)
        if return_code != 0:
            capture_log = self._tail_file(capture_path)
            failure_summary = self._summarize_training_failure(log_path, capture_path)
            raise RuntimeError(
                failure_summary
                or capture_log
                or final_log
                or "Training failed."
            )

    def _python_cmd(self) -> str:
        candidate_paths = [
            self.repo_root / ".venv" / "Scripts" / "python.exe",
            self.repo_root / ".venv" / "bin" / "python",
        ]
        for candidate in candidate_paths:
            if candidate.exists():
                return str(candidate)

        active_python = Path(sys.executable) if sys.executable else None
        if active_python is not None and active_python.exists():
            return str(active_python)

        path_python = shutil.which("python3") or shutil.which("python")
        if path_python:
            return path_python

        return "python"

    def _run_stage(
        self,
        command: List[str],
        *,
        stage_name: str,
        progress: int,
        log_path: Optional[Path],
        capture_path: Path,
        update_status: Callable[[str, str, str, int], None],
        cancel_event: Optional[threading.Event] = None,
        env_overrides: Optional[Dict[str, str]] = None,
    ) -> None:
        capture_path.parent.mkdir(parents=True, exist_ok=True)
        capture_path.write_text("", encoding="utf-8")
        metric_state: Dict[str, float] = {}
        with capture_path.open("a", encoding="utf-8", errors="ignore") as capture:
            capture.write(f"\n\n[{stage_name}] {' '.join(command)}\n")
            capture.flush()
            env = os.environ.copy()
            if env_overrides:
                env.update(env_overrides)
            process = subprocess.Popen(
                command,
                cwd=str(self.repo_root),
                stdout=capture,
                stderr=subprocess.STDOUT,
                env=env,
            )
            while process.poll() is None:
                if cancel_event is not None and cancel_event.is_set():
                    self._terminate_process(process)
                    raise StageInterruptedError(
                        stage_name,
                        f"{self._human_stage_name(stage_name)} stopped by user.",
                    )
                source = log_path if log_path and log_path.exists() else capture_path
                tail_text = self._tail_file(source)
                metric_summary = self._summarize_stage_metrics(
                    stage_name,
                    tail_text,
                    metric_state=metric_state,
                )
                message = f"{self._human_stage_name(stage_name)} is running..."
                if metric_summary:
                    message = f"{message} {metric_summary}"
                update_status(
                    stage_name,
                    message,
                    tail_text,
                    progress,
                )
                time.sleep(1)
            return_code = process.wait()

        source = log_path if log_path and log_path.exists() else capture_path
        final_log = self._tail_file(source)
        if return_code != 0:
            capture_log = self._tail_file(capture_path)
            failure_summary = self._summarize_training_failure(log_path, capture_path)
            raise RuntimeError(
                failure_summary
                or capture_log
                or final_log
                or f"{self._human_stage_name(stage_name)} failed."
            )

    def _terminate_process(self, process: subprocess.Popen) -> None:
        try:
            if os.name == "nt":
                subprocess.run(
                    ["taskkill", "/PID", str(process.pid), "/T", "/F"],
                    check=False,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
                process.wait(timeout=8)
                return
            process.terminate()
            process.wait(timeout=8)
        except Exception:
            try:
                process.kill()
            except Exception:
                pass

    def _summarize_stage_metrics(
        self,
        stage_name: str,
        log_text: str,
        *,
        metric_state: Dict[str, float],
    ) -> str:
        if stage_name != "train":
            return ""

        epoch_match = None
        loss_match = None
        for line in reversed((log_text or "").splitlines()):
            if epoch_match is None:
                epoch_match = re.search(r"Train Epoch:\s*(\d+)\s*\[(\d+)%\]", line)
            if loss_match is None:
                loss_match = re.search(
                    r"loss_disc=([0-9.]+),\s*loss_gen=([0-9.]+),\s*loss_fm=([0-9.]+),loss_mel=([0-9.]+),\s*loss_kl=([0-9.]+)",
                    line,
                )
            if epoch_match and loss_match:
                break

        parts: List[str] = []
        if epoch_match:
            metric_state["epoch"] = float(epoch_match.group(1))
            metric_state["percent"] = float(epoch_match.group(2))
            parts.append(
                f"epoch {int(metric_state['epoch'])} ({int(metric_state['percent'])}%)"
            )
        if loss_match:
            metric_state["loss_disc"] = float(loss_match.group(1))
            metric_state["loss_gen"] = float(loss_match.group(2))
            metric_state["loss_fm"] = float(loss_match.group(3))
            metric_state["loss_mel"] = float(loss_match.group(4))
            metric_state["loss_kl"] = float(loss_match.group(5))
            best_mel = float(metric_state.get("best_loss_mel", metric_state["loss_mel"]))
            metric_state["best_loss_mel"] = min(best_mel, metric_state["loss_mel"])
            parts.append(
                f"mel {metric_state['loss_mel']:.2f} (best {metric_state['best_loss_mel']:.2f})"
            )
            parts.append(f"gen {metric_state['loss_gen']:.2f}")
            parts.append(f"kl {metric_state['loss_kl']:.2f}")
        return " | ".join(parts)

    def _ensure_pretrained_weights(
        self, *, sample_rate: str, version: str, use_f0: bool
    ) -> tuple[Path, Path]:
        prefix = "f0" if use_f0 else ""
        root_name = "pretrained" if version == "v1" else "pretrained_v2"
        local_root = self.repo_root / root_name
        local_root.mkdir(parents=True, exist_ok=True)

        generator_name = f"{prefix}G{sample_rate}.pth"
        discriminator_name = f"{prefix}D{sample_rate}.pth"
        generator_path = local_root / generator_name
        discriminator_path = local_root / discriminator_name

        if not generator_path.exists():
            urlretrieve(
                f"{self.PRETRAIN_BASE_URL}/{root_name}/{generator_name}",
                generator_path,
            )
        if not discriminator_path.exists():
            urlretrieve(
                f"{self.PRETRAIN_BASE_URL}/{root_name}/{discriminator_name}",
                discriminator_path,
            )
        return generator_path, discriminator_path

    def _summarize_training_failure(
        self,
        train_log: Optional[Path],
        capture_log: Optional[Path],
    ) -> str:
        combined = []
        if train_log and train_log.exists():
            combined.extend(train_log.read_text(encoding="utf-8", errors="ignore").splitlines())
        if capture_log and capture_log.exists():
            combined.extend(capture_log.read_text(encoding="utf-8", errors="ignore").splitlines())

        interesting = [
            line.strip()
            for line in combined
            if any(
                token in line
                for token in (
                    "RuntimeError:",
                    "torch.OutOfMemoryError:",
                    "AssertionError:",
                    "AttributeError:",
                    "FileNotFoundError:",
                    "ModuleNotFoundError:",
                    "PermissionError:",
                    "Error(s) in loading state_dict",
                    "size mismatch",
                    "Torch not compiled with CUDA enabled",
                    "saving final ckpt",
                )
            )
        ]
        if not interesting:
            return self._tail_file(train_log) or self._tail_file(capture_log)
        return "\n".join(interesting[-8:])

    def _prepare_dataset_with_classic_preprocess(
        self,
        *,
        dataset_dir: Path,
        exp_dir: Path,
        sample_rate: int,
        worker_count: int,
        update_status: Callable[[str, str, str, int], None],
        start_progress: int,
        end_progress: int,
        capture_path: Path,
        cancel_event: Optional[threading.Event] = None,
    ) -> None:
        if not dataset_dir.exists():
            raise RuntimeError("The selected dataset folder does not exist.")

        source_files = sorted(
            path
            for path in dataset_dir.rglob("*")
            if path.is_file() and path.suffix.lower() in self.AUDIO_EXTENSIONS
        )
        if not source_files:
            raise RuntimeError("No supported audio files were found in the dataset folder.")

        stage_dir = exp_dir / "_source_audio"
        if stage_dir.exists():
            shutil.rmtree(stage_dir, ignore_errors=True)
        stage_dir.mkdir(parents=True, exist_ok=True)

        for target_dir in (exp_dir / "0_gt_wavs", exp_dir / "1_16k_wavs"):
            if target_dir.exists():
                shutil.rmtree(target_dir, ignore_errors=True)

        total_files = len(source_files)
        for index, source_path in enumerate(source_files, start=1):
            if cancel_event is not None and cancel_event.is_set():
                raise InterruptedError("Training stopped by user.")
            target_name = f"{index:04d}_{self._safe_dataset_stem(source_path.stem)}{source_path.suffix.lower()}"
            shutil.copy2(source_path, stage_dir / target_name)

            progress = start_progress + int(
                round(((index / total_files) * max(0, (end_progress - start_progress) // 2)))
            )
            update_status(
                "prepare-dataset",
                f"Staged {index}/{total_files} source files for classic RVC preprocessing.",
                source_path.name,
                min(end_progress - 2, progress),
            )

        preprocess_log = exp_dir / "preprocess.log"
        preprocess_log.write_text("", encoding="utf-8")
        self._run_stage(
            [
                self._python_cmd(),
                "trainset_preprocess_pipeline_print.py",
                str(stage_dir),
                str(sample_rate),
                str(max(1, min(12, worker_count))),
                str(exp_dir),
                "False",
            ],
            stage_name="prepare-dataset",
            progress=max(start_progress + 1, end_progress - 1),
            log_path=preprocess_log,
            capture_path=capture_path,
            update_status=update_status,
            cancel_event=cancel_event,
        )

        slice_count = self._count_training_slices(exp_dir)
        if slice_count <= 0:
            failure_summary = self._summarize_preprocess_zero_output(
                preprocess_log=preprocess_log,
                stage_dir=stage_dir,
            )
            raise RuntimeError(
                failure_summary
                or "Classic RVC preprocessing finished, but it produced zero usable slices."
            )
        update_status(
            "prepare-dataset",
            f"Classic RVC preprocessing produced {slice_count} training slices.",
            "",
            end_progress,
        )

    def _count_training_slices(self, exp_dir: Path) -> int:
        gt_wavs_dir = exp_dir / "0_gt_wavs"
        if not gt_wavs_dir.exists():
            return 0
        return len(list(gt_wavs_dir.glob("*.wav")))

    def _effective_batch_size(
        self,
        *,
        requested_batch_size: int,
        use_f0: bool,
        version: str,
        training_slice_count: int,
        max_batch_size: int,
    ) -> int:
        batch_size = max(1, int(requested_batch_size))
        if training_slice_count > 0:
            batch_size = min(batch_size, max(2, training_slice_count // 4))

        batch_size = min(batch_size, max(1, int(max_batch_size)))

        if not torch.cuda.is_available():
            return min(batch_size, 4)
        return batch_size

    def _summarize_preprocess_zero_output(
        self,
        *,
        preprocess_log: Optional[Path],
        stage_dir: Optional[Path] = None,
    ) -> str:
        details: List[str] = []
        if stage_dir is not None and stage_dir.exists():
            staged_count = len(list(stage_dir.iterdir()))
            details.append(f"Staged source clips: {staged_count}.")
        if preprocess_log is None or not preprocess_log.exists():
            if details:
                return "Classic RVC preprocessing produced zero slices. " + " ".join(details)
            return "Classic RVC preprocessing produced zero slices."

        log_lines = preprocess_log.read_text(encoding="utf-8", errors="ignore").splitlines()
        interesting = [
            line.strip()
            for line in log_lines
            if any(
                token in line
                for token in (
                    "Traceback",
                    "FileNotFoundError",
                    "ModuleNotFoundError",
                    "RuntimeError",
                    "Failed to load audio",
                    "->",
                )
            )
        ]
        if interesting:
            excerpt = "\n".join(interesting[-12:])
            prefix = "Classic RVC preprocessing produced zero slices."
            if details:
                prefix = f"{prefix} {' '.join(details)}"
            return f"{prefix}\n{excerpt}"

        fallback_tail = self._tail_file(preprocess_log)
        if fallback_tail:
            prefix = "Classic RVC preprocessing produced zero slices."
            if details:
                prefix = f"{prefix} {' '.join(details)}"
            return f"{prefix}\n{fallback_tail}"
        if details:
            return "Classic RVC preprocessing produced zero slices. " + " ".join(details)
        return "Classic RVC preprocessing produced zero slices."

    def _find_latest_saved_weight(self, experiment_name: str) -> Optional[Path]:
        candidates = sorted(
            self.weights_root.glob(f"{experiment_name}*.pth"),
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )
        return candidates[0] if candidates else None

    def _find_latest_checkpoint_epoch(self, exp_dir: Path) -> int:
        latest_epoch = 0
        for checkpoint_path in exp_dir.glob("G_*.pth"):
            match = re.search(r"G_(\d+)$", checkpoint_path.stem)
            if not match:
                continue
            latest_epoch = max(latest_epoch, int(match.group(1)))
        return latest_epoch

    def _safe_dataset_stem(self, stem: str) -> str:
        cleaned = []
        for char in stem:
            if char.isalnum() or char in ("-", "_"):
                cleaned.append(char)
            else:
                cleaned.append("_")
        normalized = "".join(cleaned).strip("_")
        return normalized or "clip"

    def _write_filelist(
        self,
        *,
        experiment_name: str,
        version: str,
        sample_rate: str,
        use_f0: bool,
        speaker_id: int,
    ) -> None:
        exp_dir = self.logs_root / experiment_name
        gt_wavs_dir = exp_dir / "0_gt_wavs"
        feature_dir = exp_dir / ("3_feature256" if version == "v1" else "3_feature768")
        if not gt_wavs_dir.exists() or not feature_dir.exists():
            raise RuntimeError("The dataset has not been prepared for training yet.")

        gt_names = {path.stem for path in gt_wavs_dir.glob("*.wav")}
        feature_names = {path.stem for path in feature_dir.glob("*.npy")}
        names = gt_names & feature_names

        if use_f0:
            f0_dir = exp_dir / "2a_f0"
            f0nsf_dir = exp_dir / "2b-f0nsf"
            f0_names = {path.name.replace(".wav.npy", "") for path in f0_dir.glob("*.wav.npy")}
            f0nsf_names = {
                path.name.replace(".wav.npy", "") for path in f0nsf_dir.glob("*.wav.npy")
            }
            names &= f0_names & f0nsf_names

        if not names:
            raise RuntimeError(
                "No usable training slices were found after preprocessing and feature extraction."
            )

        entries = []
        for name in sorted(names):
            if use_f0:
                entries.append(
                    "%s|%s|%s|%s|%s"
                    % (
                        str((gt_wavs_dir / f"{name}.wav")).replace("\\", "\\\\"),
                        str((feature_dir / f"{name}.npy")).replace("\\", "\\\\"),
                        str((exp_dir / "2a_f0" / f"{name}.wav.npy")).replace("\\", "\\\\"),
                        str((exp_dir / "2b-f0nsf" / f"{name}.wav.npy")).replace("\\", "\\\\"),
                        speaker_id,
                    )
                )
            else:
                entries.append(
                    "%s|%s|%s"
                    % (
                        str((gt_wavs_dir / f"{name}.wav")).replace("\\", "\\\\"),
                        str((feature_dir / f"{name}.npy")).replace("\\", "\\\\"),
                        speaker_id,
                    )
                )

        feature_dim = 256 if version == "v1" else 768
        self._ensure_mute_assets(
            sample_rate=sample_rate,
            feature_dim=feature_dim,
            use_f0=use_f0,
        )
        mute_root = self.logs_root / "mute"
        mute_wav = mute_root / "0_gt_wavs" / f"mute{sample_rate}.wav"
        mute_feature = mute_root / f"3_feature{feature_dim}" / "mute.npy"
        mute_f0 = mute_root / "2a_f0" / "mute.wav.npy"
        mute_f0nsf = mute_root / "2b-f0nsf" / "mute.wav.npy"
        if use_f0:
            entries.extend(
                [
                    "%s|%s|%s|%s|%s"
                    % (
                        str(mute_wav).replace("\\", "\\\\"),
                        str(mute_feature).replace("\\", "\\\\"),
                        str(mute_f0).replace("\\", "\\\\"),
                        str(mute_f0nsf).replace("\\", "\\\\"),
                        speaker_id,
                    )
                ]
                * 2
            )
        else:
            entries.extend(
                [
                    "%s|%s|%s"
                    % (
                        str(mute_wav).replace("\\", "\\\\"),
                        str(mute_feature).replace("\\", "\\\\"),
                        speaker_id,
                    )
                ]
                * 2
            )

        (exp_dir / "filelist.txt").write_text("\n".join(entries), encoding="utf-8")

    def _ensure_mute_assets(
        self,
        *,
        sample_rate: str,
        feature_dim: int,
        use_f0: bool,
    ) -> None:
        mute_root = self.logs_root / "mute"
        sample_rate_hz = self.SAMPLE_RATE_MAP[sample_rate]
        silence_seconds = 1.0
        silence_samples = max(1, int(round(sample_rate_hz * silence_seconds)))
        feature_frames = max(1, int(round((16000 * silence_seconds) / 160)))

        wav_dir = mute_root / "0_gt_wavs"
        feature_dir = mute_root / f"3_feature{feature_dim}"
        f0_dir = mute_root / "2a_f0"
        f0nsf_dir = mute_root / "2b-f0nsf"

        wav_dir.mkdir(parents=True, exist_ok=True)
        feature_dir.mkdir(parents=True, exist_ok=True)
        if use_f0:
            f0_dir.mkdir(parents=True, exist_ok=True)
            f0nsf_dir.mkdir(parents=True, exist_ok=True)

        mute_wav_path = wav_dir / f"mute{sample_rate}.wav"
        if not mute_wav_path.exists():
            silence = np.zeros(silence_samples, dtype=np.float32)
            wavfile.write(str(mute_wav_path), sample_rate_hz, silence)

        mute_feature_path = feature_dir / "mute.npy"
        if not mute_feature_path.exists():
            mute_feature = np.zeros((feature_frames, feature_dim), dtype=np.float32)
            np.save(mute_feature_path, mute_feature, allow_pickle=False)

        if use_f0:
            mute_f0 = np.zeros(feature_frames, dtype=np.float32)
            mute_f0_path = f0_dir / "mute.wav.npy"
            mute_f0nsf_path = f0nsf_dir / "mute.wav.npy"
            if not mute_f0_path.exists():
                np.save(mute_f0_path, mute_f0, allow_pickle=False)
            if not mute_f0nsf_path.exists():
                np.save(mute_f0nsf_path, mute_f0, allow_pickle=False)

    def _set_log_interval(self, exp_dir: Path, batch_size: int) -> int:
        wav_dir = exp_dir / "1_16k_wavs"
        wav_count = len(list(wav_dir.glob("*.wav")))
        if not wav_count:
            return 1
        log_interval = math.ceil(wav_count / max(1, batch_size))
        if log_interval > 1:
            log_interval += 1
        return log_interval

    def _train_index(self, experiment_name: str, version: str) -> str:
        exp_dir = self.logs_root / experiment_name
        feature_dir = exp_dir / ("3_feature256" if version == "v1" else "3_feature768")
        feature_files = sorted(feature_dir.glob("*.npy"))
        if not feature_files:
            raise RuntimeError("Feature extraction finished, but no features were found.")

        npys = [np.load(str(path)) for path in feature_files]
        big_npy = np.concatenate(npys, axis=0)
        big_npy_idx = np.arange(big_npy.shape[0])
        np.random.shuffle(big_npy_idx)
        big_npy = big_npy[big_npy_idx]

        if big_npy.shape[0] > 2e5:
            big_npy = (
                MiniBatchKMeans(
                    n_clusters=10000,
                    verbose=False,
                    batch_size=256 * max(1, os.cpu_count() or 1),
                    compute_labels=False,
                    init="random",
                )
                .fit(big_npy)
                .cluster_centers_
            )

        np.save(str(exp_dir / "total_fea.npy"), big_npy)
        n_ivf = min(int(16 * np.sqrt(big_npy.shape[0])), max(1, big_npy.shape[0] // 39))
        dim = 256 if version == "v1" else 768
        index = faiss.index_factory(dim, f"IVF{n_ivf},Flat")
        index_ivf = faiss.extract_index_ivf(index)
        index_ivf.nprobe = 1
        index.train(big_npy)

        trained_path = exp_dir / (
            f"trained_IVF{n_ivf}_Flat_nprobe_{index_ivf.nprobe}_{experiment_name}_{version}.index"
        )
        faiss.write_index(index, str(trained_path))

        batch_size_add = 8192
        for start in range(0, big_npy.shape[0], batch_size_add):
            index.add(big_npy[start : start + batch_size_add])

        added_path = exp_dir / (
            f"added_IVF{n_ivf}_Flat_nprobe_{index_ivf.nprobe}_{experiment_name}_{version}.index"
        )
        faiss.write_index(index, str(added_path))
        return f"Built {added_path.name}"

    def _find_added_index(self, exp_dir: Path) -> Optional[Path]:
        matches = sorted(
            path for path in exp_dir.glob("*.index") if "added" in path.name.lower()
        )
        return matches[-1] if matches else None

    def _tail_file(self, path: Path, max_lines: int = 120) -> str:
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            return ""
        lines = text.splitlines()
        return "\n".join(lines[-max_lines:])

    def _human_stage_name(self, stage_name: str) -> str:
        return stage_name.replace("-", " ").title()
