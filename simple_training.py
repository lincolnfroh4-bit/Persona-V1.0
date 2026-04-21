from __future__ import annotations

import math
import os
import re
import subprocess
import threading
import time
import shutil
from pathlib import Path
from typing import Callable, Dict, List, Optional
from urllib.request import urlretrieve

import faiss
import librosa
import numpy as np
import torch
from sklearn.cluster import MiniBatchKMeans
from scipy.io import wavfile

from my_utils import load_audio


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
                    "id": "classic-rvc-support",
                    "label": "Classic RVC + SUNO audition",
                    "description": "Builds a standard RVC .pth and index from the real BASE target vocals first, with optional extra true-target clips. SUNO clips are kept out of the speaker-truth dataset and used instead for fixed checkpoint audition previews so you can hear how the current model handles that source domain.",
                },
                {
                    "id": "persona-aligned-pth",
                    "label": "Paired aligned conversion",
                    "description": "Uses BASE identity clips plus tightly aligned TARGET/SUNO pairs to train a direct full-vocal .pth converter, and over-samples perceptual detail windows where the target has more edge, articulation, or texture than the aligned SUNO source.",
                },
                {
                    "id": "concert-remaster-paired",
                    "label": "Concert remaster",
                    "description": "Uses matched CONCERT and CD clips to train a direct learned remaster bundle that maps live concert vocals toward CD-quality output without relying on EQ-only processing.",
                },
                {
                    "id": "persona-lyric-repair",
                    "label": "Persona lyric repair",
                    "description": "Builds a lyric-only target vocal package that focuses on clean lead conversion, adlib suppression, and dense pronunciation repair instead of full vocal regeneration.",
                },
                {
                    "id": "persona-v1.1",
                    "label": "Persona v1.1",
                    "description": "Builds the overhauled guide-conditioned package with guide-delta supervision, contextual identity conditioning, stronger temporal coherence losses, and optional legacy-checkpoint reform into the new v1.1 architecture.",
                },
                {
                    "id": "persona-v1",
                    "label": "Persona v1.0",
                    "description": "Builds the direct voice-builder package only: paired-song regenerator, pronunciation assets, and the persona manifest used by the new conversion flow.",
                },
            ],
            "alignment_tolerances": [
                {
                    "id": "forgiving",
                    "label": "Forgiving",
                    "description": "Best default. Small naming or duration mismatches are tolerated instead of failing the aligned package build immediately.",
                },
                {
                    "id": "balanced",
                    "label": "Balanced",
                    "description": "Keeps only cleaner aligned matches when assembling the paired conversion dataset.",
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
                "Classic RVC + SUNO audition keeps BASE clips as the real target-voice truth and re-renders the same "
                "short SUNO source chunk at each checkpoint so you can hear progress. Paired aligned conversion still "
                "uses matched TARGET/SUNO clips for exact source-to-target supervision."
            ),
            "rebuild_note": (
                "The stable classic package stores a standard .pth backbone plus index, along with metadata showing "
                "which BASE, TARGET, and SUNO clips were used for truth audio versus checkpoint auditions."
            ),
            "training_plan_note": (
                "These modes do not need a persona plan. Use BASE for the true target voice, optional TARGET/VOCALP "
                "for extra true-target coverage, and SUNO/PREP/PRE for fixed checkpoint audition sources."
            ),
            "cuda_available": cuda_available,
            "warning": warning,
            "defaults": {
                "output_mode": "classic-rvc-support",
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
            cpu_threads = max(1, min(6, os.cpu_count() or 1))

            update_status(
                "prepare-dataset",
                "Copying your dataset directly into training format with no isolation or slicing...",
                "",
                32,
            )
            self._prepare_dataset_without_preprocessing(
                dataset_dir=dataset_dir,
                exp_dir=exp_dir,
                sample_rate=sr_hz,
                update_status=update_status,
                start_progress=32,
                end_progress=42,
                cancel_event=cancel_event,
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
                        update_status=update_status,
                        cancel_event=cancel_event,
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
                    update_status=update_status,
                    cancel_event=cancel_event,
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
            if not checkpoint_files:
                raise RuntimeError(
                    self._summarize_training_failure(train_log)
                    or "Training did not produce any generator checkpoints."
                )

            model_path = self.weights_root / f"{experiment_name}.pth"
            if not model_path.exists():
                latest_saved_model = self._find_latest_saved_weight(experiment_name)
                if latest_saved_model is not None:
                    model_path = latest_saved_model
                else:
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
            "0",
            "-sw",
            "1",
            "-v",
            version,
            "-li",
            str(self._set_log_interval(exp_dir, batch_size)),
        ]
        capture_path = self.logs_root / "_simple_web_training_capture.log"
        capture_path.parent.mkdir(parents=True, exist_ok=True)
        metric_state: Dict[str, float] = {}
        last_preview_epoch = 0
        with capture_path.open("a", encoding="utf-8", errors="ignore") as capture:
            capture.write(f"\n\n[train] {' '.join(command)}\n")
            capture.flush()
            process = subprocess.Popen(
                command,
                cwd=str(self.repo_root),
                stdout=capture,
                stderr=subprocess.STDOUT,
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
            failure_summary = ""
            if log_path:
                failure_summary = self._summarize_training_failure(log_path)
            raise RuntimeError(
                failure_summary
                or capture_log
                or final_log
                or "Training failed."
            )

    def _python_cmd(self) -> str:
        venv_python = self.repo_root / ".venv" / "Scripts" / "python.exe"
        if venv_python.exists():
            return str(venv_python)
        return "python"

    def _run_stage(
        self,
        command: List[str],
        *,
        stage_name: str,
        progress: int,
        log_path: Optional[Path],
        update_status: Callable[[str, str, str, int], None],
        cancel_event: Optional[threading.Event] = None,
    ) -> None:
        capture_path = self.logs_root / "_simple_web_training_capture.log"
        capture_path.parent.mkdir(parents=True, exist_ok=True)
        metric_state: Dict[str, float] = {}
        with capture_path.open("a", encoding="utf-8", errors="ignore") as capture:
            capture.write(f"\n\n[{stage_name}] {' '.join(command)}\n")
            capture.flush()
            process = subprocess.Popen(
                command,
                cwd=str(self.repo_root),
                stdout=capture,
                stderr=subprocess.STDOUT,
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
            failure_summary = ""
            if stage_name == "train" and log_path:
                failure_summary = self._summarize_training_failure(log_path)
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

    def _summarize_training_failure(self, train_log: Path) -> str:
        combined = []
        if train_log.exists():
            combined.extend(train_log.read_text(encoding="utf-8", errors="ignore").splitlines())
        capture_path = self.logs_root / "_simple_web_training_capture.log"
        if capture_path.exists():
            combined.extend(capture_path.read_text(encoding="utf-8", errors="ignore").splitlines())

        interesting = [
            line.strip()
            for line in combined
            if any(
                token in line
                for token in (
                    "RuntimeError:",
                    "AssertionError:",
                    "Error(s) in loading state_dict",
                    "size mismatch",
                    "Torch not compiled with CUDA enabled",
                    "saving final ckpt",
                )
            )
        ]
        if not interesting:
            return self._tail_file(train_log) or self._tail_file(capture_path)
        return "\n".join(interesting[-8:])

    def _prepare_dataset_without_preprocessing(
        self,
        *,
        dataset_dir: Path,
        exp_dir: Path,
        sample_rate: int,
        update_status: Callable[[str, str, str, int], None],
        start_progress: int,
        end_progress: int,
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

        gt_wavs_dir = exp_dir / "0_gt_wavs"
        wavs16k_dir = exp_dir / "1_16k_wavs"
        for target_dir in (gt_wavs_dir, wavs16k_dir):
            if target_dir.exists():
                shutil.rmtree(target_dir)
            target_dir.mkdir(parents=True, exist_ok=True)

        total_files = len(source_files)
        for index, source_path in enumerate(source_files, start=1):
            if cancel_event is not None and cancel_event.is_set():
                raise InterruptedError("Training stopped by user.")
            try:
                audio = load_audio(str(source_path), sample_rate, False, 1.0, 1.0)
            except Exception as exc:
                raise RuntimeError(
                    f"Could not read training audio '{source_path.name}': {exc}"
                ) from exc

            if audio is None or len(audio) == 0:
                raise RuntimeError(
                    f"Training audio '{source_path.name}' is empty after loading."
                )

            audio = np.asarray(audio, dtype=np.float32)
            target_name = f"{index:04d}_{self._safe_dataset_stem(source_path.stem)}.wav"
            gt_target = gt_wavs_dir / target_name
            wav16_target = wavs16k_dir / target_name

            wavfile.write(str(gt_target), sample_rate, audio.astype(np.float32))
            audio_16k = librosa.resample(audio, orig_sr=sample_rate, target_sr=16000)
            wavfile.write(str(wav16_target), 16000, np.asarray(audio_16k, dtype=np.float32))

            progress = start_progress + int(
                round(((index / total_files) * max(0, end_progress - start_progress)))
            )
            update_status(
                "prepare-dataset",
                f"Prepared {index}/{total_files} training files with no preprocessing.",
                source_path.name,
                min(end_progress, progress),
            )

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
