from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from audio_separator.separator import Separator


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Input audio file")
    parser.add_argument("--output-dir", required=True, help="Directory for output stems")
    parser.add_argument("--model-dir", required=True, help="Directory for cached separator models")
    parser.add_argument(
        "--model",
        default="model_bs_roformer_ep_317_sdr_12.9755.ckpt",
        help="Model filename from audio-separator's catalog",
    )
    parser.add_argument(
        "--vocals-name",
        default="vocals",
        help="Output stem name to use for vocals",
    )
    parser.add_argument(
        "--instrumental-name",
        default="instrumental",
        help="Output stem name to use for instrumental",
    )
    parser.add_argument(
        "--segment-size",
        type=int,
        default=256,
        help="RoFormer segment size to use for MDX-style chunking.",
    )
    parser.add_argument(
        "--overlap",
        type=float,
        default=0.75,
        help="RoFormer overlap ratio.",
    )
    parser.add_argument(
        "--denoise",
        action="store_true",
        help="Enable denoise during RoFormer separation.",
    )
    args = parser.parse_args()

    input_path = Path(args.input).resolve()
    output_dir = Path(args.output_dir).resolve()
    model_dir = Path(args.model_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)

    separator = Separator(
        log_level=logging.INFO,
        output_dir=str(output_dir),
        output_format="WAV",
        model_file_dir=str(model_dir),
        mdx_params={
            "hop_length": 1024,
            "segment_size": int(args.segment_size),
            "overlap": float(args.overlap),
            "batch_size": 1,
            "enable_denoise": bool(args.denoise),
        },
    )
    separator.load_model(model_filename=args.model)
    output_files = separator.separate(
        str(input_path),
        custom_output_names={
            "Vocals": args.vocals_name,
            "Instrumental": args.instrumental_name,
        },
    )

    resolved_outputs = [str((output_dir / Path(file).name).resolve()) for file in output_files]
    payload = {
        "model": args.model,
        "segment_size": int(args.segment_size),
        "overlap": float(args.overlap),
        "denoise": bool(args.denoise),
        "outputs": resolved_outputs,
    }
    print(json.dumps(payload))


if __name__ == "__main__":
    main()
