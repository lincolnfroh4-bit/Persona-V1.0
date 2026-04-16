from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, List, Tuple

import ffmpeg
import numpy as np
import soundfile as sf
from scipy.ndimage import gaussian_filter1d
from scipy.signal import istft, stft


class SimpleMasteringEngine:
    def __init__(self, repo_root: Path):
        self.repo_root = repo_root

    def _ffmpeg_binary(self) -> str:
        local = self.repo_root / ("ffmpeg.exe" if os.name == "nt" else "ffmpeg")
        return str(local) if local.exists() else "ffmpeg"

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
        audio = audio.reshape(-1, 2)
        return audio.copy(), sample_rate

    def _average_spectrum(self, mono_audio: np.ndarray, fft_size: int, hop_size: int) -> np.ndarray:
        if mono_audio.size == 0:
            raise ValueError("Audio file was empty.")

        if mono_audio.size < fft_size:
            mono_audio = np.pad(mono_audio, (0, fft_size - mono_audio.size))

        window = np.hanning(fft_size).astype(np.float32)
        frame_starts = range(0, mono_audio.size - fft_size + 1, hop_size)
        magnitudes = []
        for start in frame_starts:
            frame = mono_audio[start : start + fft_size] * window
            spectrum = np.fft.rfft(frame)
            magnitudes.append(np.abs(spectrum) + 1e-8)

        if not magnitudes:
            frame = np.pad(mono_audio, (0, max(0, fft_size - mono_audio.size)))[:fft_size] * window
            magnitudes.append(np.abs(np.fft.rfft(frame)) + 1e-8)

        return np.mean(np.stack(magnitudes, axis=0), axis=0)

    def _build_eq_curve(
        self,
        source_audio: np.ndarray,
        reference_audios: List[np.ndarray],
        sample_rate: int,
        resolution: int,
    ) -> Dict[str, object]:
        fft_size = 4096
        hop_size = fft_size // 4
        freqs = np.fft.rfftfreq(fft_size, d=1.0 / sample_rate)
        positive_freqs = np.maximum(freqs[1:], 20.0)

        source_mono = source_audio.mean(axis=1)
        source_mag = self._average_spectrum(source_mono, fft_size=fft_size, hop_size=hop_size)
        if not reference_audios:
            raise ValueError("At least one reference file is required.")

        reference_mags = []
        reference_rms_values = []
        for reference_audio in reference_audios:
            reference_mono = reference_audio.mean(axis=1)
            reference_mags.append(
                self._average_spectrum(
                    reference_mono,
                    fft_size=fft_size,
                    hop_size=hop_size,
                )
            )
            reference_rms_values.append(
                float(np.sqrt(np.mean(np.square(reference_audio)) + 1e-9))
            )
        reference_mag = np.mean(np.stack(reference_mags, axis=0), axis=0)

        source_db = 20.0 * np.log10(np.maximum(source_mag, 1e-8))
        reference_db = 20.0 * np.log10(np.maximum(reference_mag, 1e-8))

        band_count = int(max(8, min(int(resolution), 256)))
        band_freqs = np.geomspace(20.0, sample_rate / 2.0, num=band_count)
        source_bands = np.interp(band_freqs, positive_freqs, source_db[1:])
        reference_bands = np.interp(band_freqs, positive_freqs, reference_db[1:])

        raw_curve_db = np.clip(reference_bands - source_bands, -12.0, 12.0)
        smoothing_sigma = float(np.interp(band_count, [8, 256], [3.8, 0.9]))
        curve_db = gaussian_filter1d(raw_curve_db, sigma=smoothing_sigma, mode="nearest")
        curve_db = np.clip(curve_db, -10.0, 10.0)

        curve_full_db = np.zeros_like(source_db)
        curve_full_db[1:] = np.interp(
            positive_freqs,
            band_freqs,
            curve_db,
            left=float(curve_db[0]),
            right=float(curve_db[-1]),
        )
        curve_full_db[0] = 0.0

        display_count = min(64, band_count)
        display_freqs = np.geomspace(20.0, sample_rate / 2.0, num=display_count)
        display_curve_db = np.interp(display_freqs, band_freqs, curve_db)

        low_mask = display_freqs < 200.0
        mid_mask = (display_freqs >= 200.0) & (display_freqs < 4000.0)
        air_mask = display_freqs >= 4000.0
        band_summary = {
            "low": float(np.mean(display_curve_db[low_mask])) if np.any(low_mask) else 0.0,
            "mid": float(np.mean(display_curve_db[mid_mask])) if np.any(mid_mask) else 0.0,
            "air": float(np.mean(display_curve_db[air_mask])) if np.any(air_mask) else 0.0,
        }

        source_rms = float(np.sqrt(np.mean(np.square(source_audio)) + 1e-9))
        reference_rms = float(np.mean(np.asarray(reference_rms_values, dtype=np.float32)))
        loudness_gain_db = float(
            np.clip(20.0 * np.log10(max(reference_rms, 1e-8) / max(source_rms, 1e-8)), -6.0, 6.0)
        )

        return {
            "fft_size": fft_size,
            "hop_size": hop_size,
            "curve_full_db": curve_full_db,
            "display_points": [
                {
                    "frequency_hz": float(freq),
                    "gain_db": float(gain),
                }
                for freq, gain in zip(display_freqs, display_curve_db)
            ],
            "band_summary": band_summary,
            "loudness_gain_db": loudness_gain_db,
            "source_rms_db": float(20.0 * np.log10(max(source_rms, 1e-8))),
            "reference_rms_db": float(20.0 * np.log10(max(reference_rms, 1e-8))),
            "resolution": band_count,
            "sample_rate": sample_rate,
        }

    def _apply_eq_curve(
        self,
        source_audio: np.ndarray,
        curve_full_db: np.ndarray,
        fft_size: int,
        hop_size: int,
        loudness_gain_db: float,
    ) -> np.ndarray:
        gain = np.power(10.0, curve_full_db / 20.0).astype(np.float32)
        processed_channels = []
        noverlap = fft_size - hop_size

        for channel_index in range(source_audio.shape[1]):
            _, _, spectrum = stft(
                source_audio[:, channel_index],
                fs=1.0,
                window="hann",
                nperseg=fft_size,
                noverlap=noverlap,
                boundary="zeros",
                padded=True,
            )
            spectrum *= gain[:, np.newaxis]
            _, restored = istft(
                spectrum,
                fs=1.0,
                window="hann",
                nperseg=fft_size,
                noverlap=noverlap,
                input_onesided=True,
                boundary=True,
            )
            restored = restored[: source_audio.shape[0]]
            if restored.shape[0] < source_audio.shape[0]:
                restored = np.pad(restored, (0, source_audio.shape[0] - restored.shape[0]))
            processed_channels.append(restored.astype(np.float32))

        mastered = np.stack(processed_channels, axis=1)
        mastered *= np.float32(np.power(10.0, loudness_gain_db / 20.0))

        peak = float(np.max(np.abs(mastered))) if mastered.size else 0.0
        if peak > 0.995:
            mastered *= np.float32(0.995 / peak)
        return mastered

    def match_reference_eq(
        self,
        source_path: Path,
        reference_paths: List[Path],
        output_dir: Path,
        resolution: int,
    ) -> Dict[str, object]:
        output_dir.mkdir(parents=True, exist_ok=True)
        source_audio, sample_rate = self._load_audio(source_path, sample_rate=44100)
        normalized_reference_paths = [Path(path) for path in reference_paths]
        if not normalized_reference_paths:
            raise ValueError("At least one mastered reference file is required.")
        reference_audios = [
            self._load_audio(reference_path, sample_rate=sample_rate)[0]
            for reference_path in normalized_reference_paths
        ]

        profile = self._build_eq_curve(
            source_audio=source_audio,
            reference_audios=reference_audios,
            sample_rate=sample_rate,
            resolution=resolution,
        )
        mastered_audio = self._apply_eq_curve(
            source_audio=source_audio,
            curve_full_db=np.asarray(profile["curve_full_db"], dtype=np.float32),
            fft_size=int(profile["fft_size"]),
            hop_size=int(profile["hop_size"]),
            loudness_gain_db=float(profile["loudness_gain_db"]),
        )

        mastered_path = output_dir / f"{source_path.stem}_matched_master.wav"
        sf.write(mastered_path, mastered_audio, sample_rate, subtype="PCM_24")

        profile_path = output_dir / f"{source_path.stem}_eq_profile.json"
        profile_payload = {
            "sample_rate": sample_rate,
            "resolution": int(profile["resolution"]),
            "reference_count": len(normalized_reference_paths),
            "reference_files": [path.name for path in normalized_reference_paths],
            "loudness_gain_db": float(profile["loudness_gain_db"]),
            "source_rms_db": float(profile["source_rms_db"]),
            "reference_rms_db": float(profile["reference_rms_db"]),
            "band_summary": profile["band_summary"],
            "points": profile["display_points"],
        }
        profile_path.write_text(json.dumps(profile_payload, indent=2), encoding="utf-8")

        return {
            "sample_rate": sample_rate,
            "mastered_path": mastered_path,
            "profile_path": profile_path,
            "curve_points": profile["display_points"],
            "band_summary": profile["band_summary"],
            "resolution": int(profile["resolution"]),
            "loudness_gain_db": float(profile["loudness_gain_db"]),
            "source_rms_db": float(profile["source_rms_db"]),
            "reference_rms_db": float(profile["reference_rms_db"]),
            "reference_count": len(normalized_reference_paths),
            "reference_files": [path.name for path in normalized_reference_paths],
        }

    def get_options(self) -> Dict[str, object]:
        return {
            "defaults": {
                "resolution": 48,
            },
            "limits": {
                "min_resolution": 8,
                "max_resolution": 160,
            },
            "description": "Learns a smoothed EQ curve from the mastered reference and applies it to the source track.",
        }
