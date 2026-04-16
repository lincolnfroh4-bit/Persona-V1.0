from __future__ import annotations

import argparse
import os
from pathlib import Path

import soundfile as sf
import torch
import torchaudio


def _patch_torch_load_for_coqui() -> None:
    original_torch_load = torch.load

    def patched_torch_load(*args, **kwargs):
        # Coqui XTTS checkpoints currently rely on object pickles that PyTorch 2.6+
        # blocks by default via weights_only=True. This runner is only used for a
        # trusted local model path after the user has explicitly enabled XTTS.
        kwargs.setdefault("weights_only", False)
        return original_torch_load(*args, **kwargs)

    torch.load = patched_torch_load


_patch_torch_load_for_coqui()


def _patch_torchaudio_load_fallback() -> None:
    original_torchaudio_load = torchaudio.load

    def patched_torchaudio_load(uri, *args, **kwargs):
        try:
            return original_torchaudio_load(uri, *args, **kwargs)
        except Exception:
            audio, sample_rate = sf.read(str(uri), dtype="float32", always_2d=True)
            waveform = torch.from_numpy(np.ascontiguousarray(audio.T))
            return waveform, sample_rate

    torchaudio.load = patched_torchaudio_load


import numpy as np

_patch_torchaudio_load_fallback()

from TTS.api import TTS


def _coqui_tos_agreed() -> bool:
    if os.environ.get("COQUI_TOS_AGREED") == "1":
        return True
    if os.name == "nt":
        try:
            import winreg

            with winreg.OpenKey(winreg.HKEY_CURRENT_USER, r"Environment") as key:
                value, _ = winreg.QueryValueEx(key, "COQUI_TOS_AGREED")
            return str(value).strip() == "1"
        except OSError:
            return False
    return False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Regenerate a short word or phrase with XTTS v2.")
    parser.add_argument("--text", required=True, help="Word or phrase to generate.")
    parser.add_argument("--speaker-wav", required=True, help="Reference speaker audio file.")
    parser.add_argument("--output", required=True, help="Output wav path.")
    parser.add_argument("--language", default="en", help="XTTS language code.")
    parser.add_argument(
        "--model-name",
        default="tts_models/multilingual/multi-dataset/xtts_v2",
        help="Coqui model identifier.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not _coqui_tos_agreed():
        raise RuntimeError(
            "XTTS license not accepted. Set COQUI_TOS_AGREED=1 only if you have agreed to Coqui's terms "
            "or purchased a commercial license."
        )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    speaker_wav = Path(args.speaker_wav)
    if not speaker_wav.exists():
        raise FileNotFoundError(f"Speaker reference not found: {speaker_wav}")

    tts = TTS(args.model_name, gpu=torch.cuda.is_available())
    tts.tts_to_file(
        text=args.text,
        speaker_wav=str(speaker_wav),
        language=args.language,
        file_path=str(output_path),
    )
    print(str(output_path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
