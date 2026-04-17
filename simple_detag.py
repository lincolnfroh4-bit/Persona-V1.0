from __future__ import annotations

import os
import threading
from pathlib import Path
from typing import Callable, Dict, List, Optional

import ffmpeg
import numpy as np
import soundfile as sf
import torch
import torchaudio


class SimpleDetagger:
    def __init__(self, repo_root: Path):
        self.repo_root = repo_root
        self.logs_root = self.repo_root / "logs"
        self.weights_root = self.repo_root / "weights"
        self.model_cache_root = (
            self.repo_root / "pretrained_models" / "speechbrain-ecapa"
        )
        self.separator_cache_root = (
            self.repo_root / "pretrained_models" / "speechbrain-sepformer-wsj02mix"
        )
        self.classifier: Optional[EncoderClassifier] = None
        self.separator: Optional[SepformerSeparation] = None
        self.reference_cache: Dict[str, Dict[str, object]] = {}
        self.lock = threading.Lock()

    def _load_speechbrain_components(self):
        try:
            from speechbrain.inference.separation import SepformerSeparation
            from speechbrain.inference.speaker import EncoderClassifier
            from speechbrain.utils.fetching import LocalStrategy
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "speechbrain is only required for detagging and is not installed in the current Python environment."
            ) from exc
        return SepformerSeparation, EncoderClassifier, LocalStrategy

    def list_voices(self) -> List[Dict[str, object]]:
        voices = []
        seen_ids = set()
        seen_reference_ids = set()
        reference_lookup = self._build_reference_lookup()

        for weight_path in sorted(self.weights_root.glob("*.pth")):
            voice_id = weight_path.name
            matched_reference = self._match_reference_folder(
                weight_path.stem, reference_lookup
            )
            reference_clips = 0
            reference_id = ""
            if matched_reference is not None:
                reference_id = matched_reference.name
                reference_clips = self._count_reference_clips(matched_reference)
                seen_reference_ids.add(reference_id)
            voices.append(
                {
                    "id": voice_id,
                    "label": weight_path.stem.replace("_", " "),
                    "reference_clips": reference_clips,
                    "has_model": True,
                    "ready": bool(reference_id),
                    "reference_id": reference_id,
                }
            )
            seen_ids.add(voice_id)

        for log_dir in sorted(
            reference_lookup.values(), key=lambda path: path.name.lower()
        ):
            voice_id = log_dir.name
            if voice_id in seen_ids or log_dir.name in seen_reference_ids:
                continue
            voices.append(
                {
                    "id": voice_id,
                    "label": log_dir.name.replace("_", " "),
                    "reference_clips": self._count_reference_clips(log_dir),
                    "has_model": (self.weights_root / f"{log_dir.name}.pth").exists(),
                    "ready": True,
                    "reference_id": log_dir.name,
                }
            )
        return voices

    def detag_file(
        self,
        *,
        voice_id: str,
        input_path: Path,
        output_path: Path,
        strength: int,
        update_progress: Optional[Callable[[str, int], None]] = None,
    ) -> Dict[str, object]:
        with self.lock:
            if update_progress is not None:
                update_progress("Loading the selected voice profile...", 8)
            reference_profile = self._get_reference_profile(voice_id)
            reference_embedding = np.asarray(
                reference_profile["embedding"], dtype=np.float32
            )

            if update_progress is not None:
                update_progress("Scanning the audio for matching speech...", 18)
            analysis_sr = 16000
            analysis_audio = self._load_audio_mono(input_path, analysis_sr)
            if analysis_audio.size == 0:
                raise RuntimeError("The input audio is empty.")

            similarities = self._score_audio_against_reference(
                analysis_audio,
                analysis_sr,
                reference_embedding,
                update_progress,
            )

            strength_ratio = np.clip(float(strength) / 100.0, 0.0, 1.0)
            match_floor = float(reference_profile["match_floor"])
            match_center = float(reference_profile["match_center"])
            match_ceiling = float(reference_profile["match_ceiling"])
            threshold_min = max(0.12, match_floor - 0.05)
            threshold_max = max(threshold_min + 0.04, match_center + 0.02)
            threshold = threshold_min + (
                (threshold_max - threshold_min) * strength_ratio
            )
            threshold = min(threshold, match_ceiling)
            binary_mask = (similarities >= threshold).astype(np.float32)

            # Lower strengths give the chosen voice a little more padding, while
            # higher strengths cut more aggressively around the edges.
            pad_seconds = 0.14 - (0.08 * strength_ratio)
            pad_samples = max(1, int(pad_seconds * analysis_sr))
            if pad_samples > 1:
                kernel = np.ones(pad_samples, dtype=np.float32)
                binary_mask = (
                    np.convolve(binary_mask, kernel, mode="same") > 0
                ).astype(np.float32)

            fade_samples = max(8, int(0.02 * analysis_sr))
            fade_kernel = np.ones(fade_samples, dtype=np.float32) / float(fade_samples)
            smooth_mask = np.convolve(binary_mask, fade_kernel, mode="same")
            smooth_mask = np.clip(smooth_mask, 0.0, 1.0)
            gated_audio = analysis_audio * smooth_mask

            overlap_audio = self._extract_overlap_target(
                analysis_audio,
                reference_profile,
                strength_ratio,
                update_progress,
            )
            combined_audio = self._combine_gate_and_overlap(
                gated_audio,
                overlap_audio,
                analysis_sr,
            )

            if update_progress is not None:
                update_progress("Writing the isolated voice output...", 88)
            original_audio, original_sr = self._load_audio_native(input_path)
            original_length = original_audio.shape[0]
            mask_positions = np.linspace(
                0, original_length - 1, num=combined_audio.shape[0]
            )
            target_positions = np.arange(original_length)
            output_mono = np.interp(
                target_positions, mask_positions, combined_audio
            ).astype(np.float32)
            peak = float(np.max(np.abs(output_mono)) + 1e-9)
            if peak > 1.0:
                output_mono = output_mono / peak
            output_audio = np.repeat(
                output_mono[:, None], original_audio.shape[1], axis=1
            )

            output_path.parent.mkdir(parents=True, exist_ok=True)
            sf.write(str(output_path), output_audio, original_sr)

            kept_ratio = float(np.mean(np.abs(output_mono) > 0.01))
            return {
                "voice_id": voice_id,
                "threshold": round(float(threshold), 3),
                "kept_ratio": round(kept_ratio, 3),
                "sample_rate": original_sr,
            }

    def _load_classifier(self) -> EncoderClassifier:
        if self.classifier is None:
            _, EncoderClassifier, LocalStrategy = self._load_speechbrain_components()
            self.classifier = EncoderClassifier.from_hparams(
                source="speechbrain/spkrec-ecapa-voxceleb",
                savedir=str(self.model_cache_root),
                local_strategy=LocalStrategy.COPY,
            )
        return self.classifier

    def _load_separator(self) -> SepformerSeparation:
        if self.separator is None:
            SepformerSeparation, _, LocalStrategy = self._load_speechbrain_components()
            self.separator = SepformerSeparation.from_hparams(
                source="speechbrain/sepformer-wsj02mix",
                savedir=str(self.separator_cache_root),
                local_strategy=LocalStrategy.COPY,
            )
        return self.separator

    def _get_reference_profile(self, voice_id: str) -> Dict[str, object]:
        cached = self.reference_cache.get(voice_id)
        if cached is not None:
            return cached

        reference_id = self._resolve_reference_id(voice_id)
        wav_dir = self.logs_root / reference_id / "0_gt_wavs"
        wav_files = sorted(wav_dir.glob("*.wav"))
        if not wav_files:
            raise RuntimeError(
                "That voice does not have reference clips available in logs/<voice>/0_gt_wavs."
            )

        selected_files = wav_files[: min(18, len(wav_files))]
        embeddings = []
        window_samples = int(1.2 * 16000)
        hop_samples = int(0.4 * 16000)
        for wav_file in selected_files:
            audio = self._load_audio_mono(wav_file, 16000)
            if audio.shape[0] < 8000:
                continue
            if audio.shape[0] <= window_samples:
                embeddings.append(self._embed_audio(audio))
                continue

            taken = 0
            for start in range(0, audio.shape[0] - window_samples + 1, hop_samples):
                embeddings.append(
                    self._embed_audio(audio[start : start + window_samples])
                )
                taken += 1
                if taken >= 3:
                    break

        if not embeddings:
            raise RuntimeError("No usable reference clips were found for that voice.")

        stacked = np.stack(embeddings, axis=0)
        embedding = np.mean(stacked, axis=0)
        embedding /= np.linalg.norm(embedding) + 1e-9
        similarities = np.dot(stacked, embedding.astype(np.float32))
        profile = {
            "embedding": embedding.astype(np.float32),
            "match_floor": float(np.percentile(similarities, 15)),
            "match_center": float(np.percentile(similarities, 50)),
            "match_ceiling": float(np.percentile(similarities, 85)),
        }
        self.reference_cache[voice_id] = profile
        return profile

    def _resolve_reference_id(self, voice_id: str) -> str:
        for voice in self.list_voices():
            if voice["id"] != voice_id:
                continue
            reference_id = str(voice.get("reference_id") or "")
            if reference_id:
                return reference_id
            break
        raise RuntimeError(
            "That voice is listed, but it does not have usable reference clips yet. Add or keep logs/<voice>/0_gt_wavs to use detag."
        )

    def _build_reference_lookup(self) -> Dict[str, Path]:
        lookup: Dict[str, Path] = {}
        if not self.logs_root.exists():
            return lookup
        for log_dir in sorted(
            path for path in self.logs_root.iterdir() if path.is_dir()
        ):
            if log_dir.name.lower() == "mute":
                continue
            if self._count_reference_clips(log_dir) < 1:
                continue
            lookup[log_dir.name.lower()] = log_dir
        return lookup

    def _match_reference_folder(
        self, voice_name: str, reference_lookup: Dict[str, Path]
    ) -> Optional[Path]:
        for candidate in self._candidate_keys(voice_name):
            match = reference_lookup.get(candidate)
            if match is not None:
                return match

        voice_lower = voice_name.lower()
        for key, path in reference_lookup.items():
            if voice_lower in key or key in voice_lower:
                return path
        return None

    def _candidate_keys(self, value: str) -> List[str]:
        stem = value.strip().replace(".pth", "")
        tokens = [
            stem,
            stem.lower(),
            stem.split("_")[0],
            stem.split("_")[0].lower(),
            stem.split("-")[0],
            stem.split("-")[0].lower(),
        ]
        seen = set()
        ordered = []
        for token in tokens:
            token = token.strip().lower()
            if not token or token in seen:
                continue
            seen.add(token)
            ordered.append(token)
        return ordered

    def _count_reference_clips(self, log_dir: Path) -> int:
        wav_dir = log_dir / "0_gt_wavs"
        if not wav_dir.exists():
            return 0
        return len(list(wav_dir.glob("*.wav")))

    def _score_audio_against_reference(
        self,
        analysis_audio: np.ndarray,
        analysis_sr: int,
        reference_embedding: np.ndarray,
        update_progress: Optional[Callable[[str, int], None]] = None,
    ) -> np.ndarray:
        window_samples = int(1.2 * analysis_sr)
        hop_samples = int(0.25 * analysis_sr)
        if analysis_audio.shape[0] <= window_samples:
            score = self._score_segment(analysis_audio, reference_embedding)
            return np.full(analysis_audio.shape[0], score, dtype=np.float32)

        positions = []
        scores = []
        last_progress = -1
        total_steps = max(
            1, 1 + (analysis_audio.shape[0] - window_samples) // hop_samples
        )
        for step_index, start in enumerate(
            range(0, analysis_audio.shape[0] - window_samples + 1, hop_samples)
        ):
            segment = analysis_audio[start : start + window_samples]
            positions.append(start + (window_samples // 2))
            scores.append(self._score_segment(segment, reference_embedding))

            if update_progress is not None:
                progress = 18 + int(58 * ((step_index + 1) / total_steps))
                if progress != last_progress:
                    update_progress(
                        "Scanning the audio for matching speech...", progress
                    )
                    last_progress = progress

        positions = np.asarray(positions, dtype=np.float32)
        scores = np.asarray(scores, dtype=np.float32)
        sample_positions = np.arange(analysis_audio.shape[0], dtype=np.float32)
        interpolated = np.interp(
            sample_positions,
            positions,
            scores,
            left=float(scores[0]),
            right=float(scores[-1]),
        )

        smooth_window = max(1, int(0.12 * analysis_sr))
        if smooth_window > 1:
            kernel = np.ones(smooth_window, dtype=np.float32) / float(smooth_window)
            interpolated = np.convolve(interpolated, kernel, mode="same")
        return interpolated.astype(np.float32)

    def _score_segment(
        self, audio: np.ndarray, reference_embedding: np.ndarray
    ) -> float:
        if audio.size == 0:
            return 0.0
        energy = float(np.sqrt(np.mean(np.square(audio))) + 1e-9)
        if energy < 0.008:
            return 0.0
        embedding = self._embed_audio(audio)
        return float(np.dot(embedding, reference_embedding))

    def _embed_audio(self, audio: np.ndarray, sample_rate: int = 16000) -> np.ndarray:
        classifier = self._load_classifier()
        waveform = torch.from_numpy(audio).float().unsqueeze(0)
        if sample_rate != 16000:
            waveform = torchaudio.functional.resample(waveform, sample_rate, 16000)
        with torch.no_grad():
            embedding = classifier.encode_batch(waveform).squeeze().cpu().numpy()
        embedding = embedding.astype(np.float32)
        embedding /= np.linalg.norm(embedding) + 1e-9
        return embedding

    def _extract_overlap_target(
        self,
        analysis_audio: np.ndarray,
        reference_profile: Dict[str, object],
        strength_ratio: float,
        update_progress: Optional[Callable[[str, int], None]] = None,
    ) -> np.ndarray:
        separator_sr = 8000
        waveform = torch.from_numpy(analysis_audio).float().unsqueeze(0)
        waveform_8k = torchaudio.functional.resample(waveform, 16000, separator_sr)
        audio_8k = waveform_8k.squeeze(0).cpu().numpy()
        if audio_8k.size < separator_sr:
            return np.zeros_like(analysis_audio, dtype=np.float32)

        if update_progress is not None:
            update_progress("Separating overlapping voices...", 52)

        separator = self._load_separator()
        window_samples = int(10.0 * separator_sr)
        hop_samples = int(8.0 * separator_sr)
        if audio_8k.shape[0] <= window_samples:
            window_samples = audio_8k.shape[0]
            hop_samples = audio_8k.shape[0]

        target_floor = max(0.18, float(reference_profile["match_floor"]) - 0.10)
        accum = np.zeros(audio_8k.shape[0], dtype=np.float32)
        weights = np.zeros(audio_8k.shape[0], dtype=np.float32)
        starts = list(
            range(0, max(1, audio_8k.shape[0] - window_samples + 1), hop_samples)
        )
        if starts[-1] != max(0, audio_8k.shape[0] - window_samples):
            starts.append(max(0, audio_8k.shape[0] - window_samples))
        blend_window = np.hanning(window_samples).astype(np.float32)
        if not np.any(blend_window):
            blend_window = np.ones(window_samples, dtype=np.float32)

        for index, start in enumerate(starts, start=1):
            chunk = audio_8k[start : start + window_samples]
            if chunk.shape[0] < window_samples:
                chunk = np.pad(chunk, (0, window_samples - chunk.shape[0]))
            separated = self._separate_chunk(separator, chunk)
            best_stem = np.zeros(window_samples, dtype=np.float32)
            best_score = -1.0
            for stem_index in range(separated.shape[1]):
                stem = separated[:, stem_index]
                score = self._score_stem_candidate(
                    stem,
                    np.asarray(reference_profile["embedding"], dtype=np.float32),
                    separator_sr,
                )
                if score > best_score:
                    best_score = score
                    best_stem = stem.astype(np.float32)

            confidence = np.clip(
                (best_score - target_floor) / max(0.05, 0.18 + (0.18 * strength_ratio)),
                0.0,
                1.0,
            )
            best_stem = best_stem * confidence
            end = min(audio_8k.shape[0], start + window_samples)
            valid = end - start
            accum[start:end] += best_stem[:valid] * blend_window[:valid]
            weights[start:end] += blend_window[:valid]

            if update_progress is not None:
                progress = 52 + int(28 * (index / max(1, len(starts))))
                update_progress("Separating overlapping voices...", progress)

        separated_audio = accum / np.maximum(weights, 1e-6)
        separated_tensor = torch.from_numpy(separated_audio).float().unsqueeze(0)
        separated_16k = torchaudio.functional.resample(
            separated_tensor, separator_sr, 16000
        )
        overlap_audio = separated_16k.squeeze(0).cpu().numpy().astype(np.float32)
        if overlap_audio.shape[0] != analysis_audio.shape[0]:
            overlap_audio = np.interp(
                np.arange(analysis_audio.shape[0]),
                np.linspace(0, analysis_audio.shape[0] - 1, num=overlap_audio.shape[0]),
                overlap_audio,
            ).astype(np.float32)
        return overlap_audio

    def _separate_chunk(
        self, separator: SepformerSeparation, chunk: np.ndarray
    ) -> np.ndarray:
        mix = torch.from_numpy(chunk).float().unsqueeze(0)
        with torch.no_grad():
            separated = separator.separate_batch(mix).squeeze(0).cpu().numpy()
        return separated.astype(np.float32)

    def _score_stem_candidate(
        self, stem: np.ndarray, reference_embedding: np.ndarray, sample_rate: int
    ) -> float:
        energy = float(np.sqrt(np.mean(np.square(stem))) + 1e-9)
        if energy < 0.003:
            return 0.0
        embedding = self._embed_audio(stem, sample_rate=sample_rate)
        return float(np.dot(embedding, reference_embedding))

    def _combine_gate_and_overlap(
        self,
        gated_audio: np.ndarray,
        overlap_audio: np.ndarray,
        sample_rate: int,
    ) -> np.ndarray:
        if overlap_audio.size == 0:
            return gated_audio.astype(np.float32)

        overlap_presence = (np.abs(overlap_audio) > 0.006).astype(np.float32)
        smooth_window = max(1, int(0.08 * sample_rate))
        if smooth_window > 1:
            kernel = np.ones(smooth_window, dtype=np.float32) / float(smooth_window)
            overlap_presence = np.convolve(overlap_presence, kernel, mode="same")
        overlap_presence = np.clip(overlap_presence, 0.0, 1.0)

        combined = overlap_audio + (gated_audio * (1.0 - overlap_presence))
        peak = float(np.max(np.abs(combined)) + 1e-9)
        if peak > 1.0:
            combined = combined / peak
        return combined.astype(np.float32)

    def _load_audio_mono(self, file_path: Path, sample_rate: int) -> np.ndarray:
        cleaned = str(file_path).strip().strip('"')
        out, _ = (
            ffmpeg.input(cleaned, threads=0)
            .output("-", format="f32le", acodec="pcm_f32le", ac=1, ar=sample_rate)
            .run(
                cmd=[self._ffmpeg_binary(), "-nostdin"],
                capture_stdout=True,
                capture_stderr=True,
            )
        )
        return np.frombuffer(out, np.float32).flatten()

    def _load_audio_native(self, file_path: Path) -> tuple[np.ndarray, int]:
        cleaned = str(file_path).strip().strip('"')
        probe = ffmpeg.probe(cleaned, cmd="ffprobe")
        audio_stream = next(
            stream for stream in probe["streams"] if stream.get("codec_type") == "audio"
        )
        sample_rate = int(audio_stream["sample_rate"])
        channels = int(audio_stream.get("channels", 1))
        out, _ = (
            ffmpeg.input(cleaned, threads=0)
            .output(
                "-",
                format="f32le",
                acodec="pcm_f32le",
                ac=channels,
                ar=sample_rate,
            )
            .run(
                cmd=[self._ffmpeg_binary(), "-nostdin"],
                capture_stdout=True,
                capture_stderr=True,
            )
        )
        audio = np.frombuffer(out, np.float32)
        if channels == 1:
            return audio.reshape(-1, 1), sample_rate
        return audio.reshape(-1, channels), sample_rate

    def _ffmpeg_binary(self) -> str:
        local = self.repo_root / ("ffmpeg.exe" if os.name == "nt" else "ffmpeg")
        return str(local) if local.exists() else "ffmpeg"
