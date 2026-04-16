from __future__ import annotations

import argparse
import multiprocessing
import os
import re
import subprocess
import sys
import time
import queue
from pathlib import Path
from queue import Empty
from typing import Optional, TYPE_CHECKING

import numpy as np
import sounddevice as sd

if sys.platform == "darwin":
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

NOW_DIR = os.getcwd()
sys.path.append(NOW_DIR)

if TYPE_CHECKING:
    import torch
    from rvc_for_realtime import RVC

torch = None
F = None
tat = None


class Harvest(multiprocessing.Process):
    def __init__(self, inp_q: multiprocessing.Queue, opt_q: multiprocessing.Queue):
        super().__init__()
        self.inp_q = inp_q
        self.opt_q = opt_q

    def run(self) -> None:
        import pyworld

        while True:
            idx, x, res_f0, n_cpu, ts = self.inp_q.get()
            f0, _ = pyworld.harvest(
                x.astype(np.double),
                fs=16000,
                f0_ceil=1100,
                f0_floor=50,
                frame_period=10,
            )
            res_f0[idx] = f0
            if len(res_f0.keys()) >= n_cpu:
                self.opt_q.put(ts)


class MonitorOutput:
    def __init__(self, samplerate: int, block_frame: int, channels: int, device: int):
        self.samplerate = samplerate
        self.block_frame = block_frame
        self.channels = channels
        self.device = device
        self.queue: "queue.Queue[np.ndarray]" = queue.Queue(maxsize=16)
        self.stream: Optional[sd.OutputStream] = None

    def _callback(self, outdata, frames, times, status) -> None:
        if status:
            print(f"[monitor] {status}")
        try:
            chunk = self.queue.get_nowait()
        except queue.Empty:
            outdata[:] = np.zeros((frames, self.channels), dtype=np.float32)
            return

        if chunk.shape[0] != frames:
            fixed = np.zeros((frames, self.channels), dtype=np.float32)
            limit = min(frames, chunk.shape[0])
            fixed[:limit, :] = chunk[:limit, :]
            outdata[:] = fixed
        else:
            outdata[:] = chunk

    def start(self) -> None:
        self.stream = sd.OutputStream(
            samplerate=self.samplerate,
            blocksize=self.block_frame,
            dtype="float32",
            channels=self.channels,
            device=self.device,
            callback=self._callback,
        )
        self.stream.start()

    def stop(self) -> None:
        if self.stream is not None:
            self.stream.stop()
            self.stream.close()
            self.stream = None

    def push(self, wav: np.ndarray) -> None:
        if wav.ndim == 1:
            chunk = wav[:, None]
        else:
            chunk = wav
        if chunk.shape[1] != self.channels:
            chunk = np.tile(chunk[:, :1], (1, self.channels))
        try:
            self.queue.put_nowait(chunk.astype(np.float32))
        except queue.Full:
            try:
                _ = self.queue.get_nowait()
            except queue.Empty:
                pass
            try:
                self.queue.put_nowait(chunk.astype(np.float32))
            except queue.Full:
                pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Realtime RVC mic voice changer using .pth voices."
    )
    parser.add_argument(
        "--model",
        required=False,
        default="",
        help="Path to .pth model or model filename inside ./weights",
    )
    parser.add_argument(
        "--index",
        default="",
        help="Optional index path. If omitted, tries to auto-find from ./logs",
    )
    parser.add_argument("--pitch", type=int, default=0, help="Semitone shift.")
    parser.add_argument(
        "--f0-method",
        default="rmvpe",
        choices=["pm", "harvest", "crepe", "rmvpe"],
        help="Pitch extraction method.",
    )
    parser.add_argument(
        "--index-rate",
        type=float,
        default=0.08,
        help="Index blend amount. Set 0 to disable index.",
    )
    parser.add_argument(
        "--threshold-db",
        type=float,
        default=-55.0,
        help="Noise gate threshold in dBFS. Lower is more sensitive.",
    )
    parser.add_argument(
        "--block-time",
        type=float,
        default=0.24,
        help="Stream block size in seconds. Lower = lower latency, higher CPU.",
    )
    parser.add_argument(
        "--crossfade-time",
        type=float,
        default=0.04,
        help="Crossfade length in seconds.",
    )
    parser.add_argument(
        "--extra-time",
        type=float,
        default=1.2,
        help="Extra context in seconds to stabilize conversion.",
    )
    parser.add_argument(
        "--n-cpu",
        type=int,
        default=min(multiprocessing.cpu_count(), 8),
        help="Harvest worker processes.",
    )
    parser.add_argument(
        "--input-device",
        default="",
        help="Input device index or name fragment.",
    )
    parser.add_argument(
        "--output-device",
        default="",
        help="Output device index or name fragment (virtual cable recommended).",
    )
    parser.add_argument(
        "--monitor-device",
        default="",
        help="Optional monitor output device (headphones/speakers) to hear yourself live.",
    )
    parser.add_argument(
        "--list-devices",
        action="store_true",
        help="Print audio devices and exit.",
    )
    return parser.parse_args()


def query_devices() -> list[dict]:
    sd._terminate()
    sd._initialize()
    devices = sd.query_devices()
    hostapis = sd.query_hostapis()
    for hostapi in hostapis:
        for device_idx in hostapi["devices"]:
            devices[device_idx]["hostapi_name"] = hostapi["name"]
    return devices


def print_devices(devices: list[dict]) -> None:
    print("\nAudio devices:\n")
    for i, d in enumerate(devices):
        print(
            f"[{i}] {d['name']} ({d.get('hostapi_name', 'Unknown API')}) | "
            f"in:{d['max_input_channels']} out:{d['max_output_channels']} "
            f"default_sr:{int(d['default_samplerate'])}"
        )
    print("")


def resolve_device(
    devices: list[dict], user_value: str, need_input: bool
) -> Optional[int]:
    if not user_value:
        return None
    value = user_value.strip()
    if re.fullmatch(r"\d+", value):
        idx = int(value)
        if idx < 0 or idx >= len(devices):
            raise ValueError(f"Device index out of range: {idx}")
        if need_input and devices[idx]["max_input_channels"] <= 0:
            raise ValueError(f"Device {idx} is not an input device.")
        if not need_input and devices[idx]["max_output_channels"] <= 0:
            raise ValueError(f"Device {idx} is not an output device.")
        return idx

    lowered = value.lower()
    matches = []
    for i, d in enumerate(devices):
        if lowered not in str(d["name"]).lower():
            continue
        if need_input and d["max_input_channels"] <= 0:
            continue
        if not need_input and d["max_output_channels"] <= 0:
            continue
        matches.append(i)
    if not matches:
        kind = "input" if need_input else "output"
        raise ValueError(f"No {kind} device matched: {user_value}")
    return matches[0]


def find_default_index(repo_root: Path, model_path: Path) -> str:
    logs_root = repo_root / "logs"
    stem = model_path.stem
    candidates = [
        stem,
        stem.lower(),
        stem.split("_")[0],
        stem.split("_")[0].lower(),
        stem.split("-")[0],
        stem.split("-")[0].lower(),
    ]
    seen = set()
    for candidate in candidates:
        if not candidate or candidate in seen:
            continue
        seen.add(candidate)
        folder = logs_root / candidate
        if not folder.exists():
            continue
        index_files = sorted(
            file for file in folder.glob("*.index") if "trained" not in file.name.lower()
        )
        if index_files:
            return index_files[0].as_posix()

    stem_lower = stem.lower()
    all_indexes = sorted(logs_root.glob("**/*.index"))
    for file in all_indexes:
        if "trained" in file.name.lower():
            continue
        full = file.as_posix().lower()
        if stem_lower in full:
            return file.as_posix()
    return ""


def resolve_model_path(repo_root: Path, model_arg: str) -> Path:
    candidate = Path(model_arg)
    if candidate.is_file():
        return candidate.resolve()

    weights_root = repo_root / "weights"
    if not model_arg:
        pths = sorted(weights_root.glob("*.pth"))
        if not pths:
            raise FileNotFoundError("No .pth files found in ./weights.")
        return pths[0].resolve()

    if not model_arg.lower().endswith(".pth"):
        candidate = weights_root / f"{model_arg}.pth"
    else:
        candidate = weights_root / model_arg

    if not candidate.exists():
        raise FileNotFoundError(f"Model not found: {candidate}")
    return candidate.resolve()


class RealtimeMicVoiceChanger:
    def __init__(
        self,
        rvc: "RVC",
        device: torch.device,
        samplerate: int,
        input_channels: int,
        output_channels: int,
        block_time: float,
        crossfade_time: float,
        extra_time: float,
        threshold_db: float,
        f0_method: str,
        monitor_output: Optional[MonitorOutput] = None,
    ) -> None:
        self.rvc = rvc
        self.device = device
        self.samplerate = samplerate
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.block_time = block_time
        self.crossfade_time = min(crossfade_time, block_time)
        self.extra_time = extra_time
        self.threshold_db = threshold_db
        self.f0_method = f0_method
        self.monitor_output = monitor_output

        self.block_frame = int(self.block_time * self.samplerate)
        self.crossfade_frame = int(self.crossfade_time * self.samplerate)
        self.sola_search_frame = int(0.01 * self.samplerate)
        self.extra_frame = int(self.extra_time * self.samplerate)
        self.zc = self.samplerate // 100
        total = self.extra_frame + self.crossfade_frame + self.sola_search_frame + self.block_frame
        buf_len = int(np.ceil(total / self.zc) * self.zc)

        self.input_wav = np.zeros(buf_len, dtype="float32")
        self.output_wav_cache = torch.zeros(buf_len, device=self.device, dtype=torch.float32)
        self.pitch = np.zeros(self.input_wav.shape[0] // self.zc, dtype="int32")
        self.pitchf = np.zeros(self.input_wav.shape[0] // self.zc, dtype="float64")
        self.output_wav = torch.zeros(self.block_frame, device=self.device, dtype=torch.float32)
        self.sola_buffer = torch.zeros(self.crossfade_frame, device=self.device, dtype=torch.float32)
        self.fade_in_window = torch.linspace(
            0.0, 1.0, steps=max(1, self.crossfade_frame), device=self.device, dtype=torch.float32
        )
        self.fade_out_window = 1 - self.fade_in_window
        self.resampler = tat.Resample(
            orig_freq=self.samplerate, new_freq=16000, dtype=torch.float32
        ).to(self.device)
        self.block_frame_16k = int(self.block_frame * 16000 / self.samplerate)
        self.last_print = time.time()

    def run(self) -> None:
        print("\nRealtime conversion is running. Press Ctrl+C to stop.\n")
        if self.monitor_output is not None:
            self.monitor_output.start()
        with sd.Stream(
            samplerate=self.samplerate,
            blocksize=self.block_frame,
            dtype="float32",
            channels=(self.input_channels, self.output_channels),
            callback=self.audio_callback,
        ):
            while True:
                time.sleep(0.25)
        if self.monitor_output is not None:
            self.monitor_output.stop()

    def audio_callback(self, indata, outdata, frames, times, status) -> None:
        if status:
            print(f"[stream] {status}")

        start = time.perf_counter()
        mono = np.mean(indata[:, : self.input_channels], axis=1, dtype=np.float32)
        rms = float(np.sqrt(np.mean(np.square(mono), dtype=np.float64) + 1e-9))
        db = 20.0 * np.log10(max(rms, 1e-7))
        if db < self.threshold_db:
            mono[:] = 0.0

        self.input_wav[:] = np.append(self.input_wav[self.block_frame :], mono)
        inp = torch.from_numpy(self.input_wav).to(self.device)
        resampled = self.resampler(inp)

        rate1 = self.block_frame / (
            self.extra_frame + self.crossfade_frame + self.sola_search_frame + self.block_frame
        )
        rate2 = (
            self.crossfade_frame + self.sola_search_frame + self.block_frame
        ) / (
            self.extra_frame + self.crossfade_frame + self.sola_search_frame + self.block_frame
        )
        res2 = self.rvc.infer(
            resampled,
            resampled[-self.block_frame_16k :].detach().cpu().numpy(),
            rate1,
            rate2,
            self.pitch,
            self.pitchf,
            self.f0_method,
        )
        self.output_wav_cache[-res2.shape[0] :] = res2
        infer_wav = self.output_wav_cache[
            -self.crossfade_frame - self.sola_search_frame - self.block_frame :
        ]

        if self.crossfade_frame > 0:
            cor_nom = F.conv1d(
                infer_wav[None, None, : self.crossfade_frame + self.sola_search_frame],
                self.sola_buffer[None, None, :],
            )
            cor_den = torch.sqrt(
                F.conv1d(
                    infer_wav[None, None, : self.crossfade_frame + self.sola_search_frame] ** 2,
                    torch.ones(1, 1, self.crossfade_frame, device=self.device),
                )
                + 1e-8
            )
            if self.device.type == "mps":
                cor_nom = cor_nom.cpu()
                cor_den = cor_den.cpu()
            sola_offset = int(torch.argmax(cor_nom[0, 0] / cor_den[0, 0]))
            self.output_wav[:] = infer_wav[sola_offset : sola_offset + self.block_frame]
            self.output_wav[: self.crossfade_frame] *= self.fade_in_window
            self.output_wav[: self.crossfade_frame] += self.sola_buffer[:]
            if sola_offset < self.sola_search_frame:
                self.sola_buffer[:] = (
                    infer_wav[
                        -self.sola_search_frame
                        - self.crossfade_frame
                        + sola_offset : -self.sola_search_frame
                        + sola_offset
                    ]
                    * self.fade_out_window
                )
            else:
                self.sola_buffer[:] = infer_wav[-self.crossfade_frame :] * self.fade_out_window
        else:
            self.output_wav[:] = infer_wav[-self.block_frame :]

        wav = self.output_wav.detach().cpu().numpy()
        if self.output_channels == 1:
            outdata[:] = wav[:, None]
        else:
            outdata[:] = np.tile(wav[:, None], (1, self.output_channels))

        if self.monitor_output is not None:
            self.monitor_output.push(outdata.copy())

        elapsed_ms = int((time.perf_counter() - start) * 1000)
        now = time.time()
        if now - self.last_print >= 1.0:
            self.last_print = now
            print(f"[rt] infer={elapsed_ms}ms")


def main() -> None:
    global torch, F, tat
    repo_root = Path(__file__).resolve().parent
    preferred_python = repo_root / ".venv" / "Scripts" / "python.exe"
    current_python = Path(sys.executable).resolve()
    if (
        preferred_python.exists()
        and current_python != preferred_python.resolve()
        and os.environ.get("RVC_RT_ALREADY_REEXEC") != "1"
    ):
        print(
            "Switching to project venv interpreter for compatibility:\n"
            f"  {preferred_python}\n"
        )
        env = os.environ.copy()
        env["RVC_RT_ALREADY_REEXEC"] = "1"
        completed = subprocess.run(
            [str(preferred_python), str(Path(__file__).resolve()), *sys.argv[1:]],
            cwd=str(repo_root),
            env=env,
        )
        raise SystemExit(completed.returncode)

    args = parse_args()
    devices = query_devices()

    if args.list_devices:
        print_devices(devices)
        return

    import torch as _torch
    import torch.nn.functional as _F
    import torchaudio.transforms as _tat

    torch = _torch
    F = _F
    tat = _tat

    model_path = resolve_model_path(repo_root, args.model)
    index_path = args.index.strip()
    if index_path:
        index_path = str(Path(index_path).resolve())
    else:
        index_path = find_default_index(repo_root, model_path)

    index_rate = float(np.clip(args.index_rate, 0.0, 1.0))
    if not index_path or not Path(index_path).exists():
        index_rate = 0.0
        index_path = ""

    in_dev = resolve_device(devices, args.input_device, need_input=True)
    out_dev = resolve_device(devices, args.output_device, need_input=False)
    monitor_dev = resolve_device(devices, args.monitor_device, need_input=False)
    if in_dev is not None:
        sd.default.device[0] = in_dev
    if out_dev is not None:
        sd.default.device[1] = out_dev

    input_device_idx = sd.default.device[0]
    output_device_idx = sd.default.device[1]
    in_info = devices[input_device_idx]
    out_info = devices[output_device_idx]
    input_channels = 1 if int(in_info["max_input_channels"]) <= 1 else 2
    output_channels = 1 if int(out_info["max_output_channels"]) <= 1 else 2

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else ("mps" if torch.backends.mps.is_available() else "cpu")
    )
    print(f"\nModel: {model_path.name}")
    print(f"Index: {index_path if index_path else '(disabled)'}")
    print(f"Input device [{input_device_idx}]: {in_info['name']}")
    print(f"Output device [{output_device_idx}]: {out_info['name']}")
    if monitor_dev is not None:
        print(f"Monitor device [{monitor_dev}]: {devices[monitor_dev]['name']}")
    print(f"Torch device: {device}\n")

    inp_q = multiprocessing.Queue()
    opt_q = multiprocessing.Queue()
    n_cpu = max(1, min(int(args.n_cpu), multiprocessing.cpu_count()))
    workers: list[Harvest] = []
    for _ in range(n_cpu):
        worker = Harvest(inp_q, opt_q)
        worker.daemon = True
        worker.start()
        workers.append(worker)

    original_argv = sys.argv[:]
    try:
        # rvc_for_realtime imports Config(), which parses sys.argv.
        # Keep only script name so its parser doesn't choke on our realtime flags.
        sys.argv = [original_argv[0]]
        from rvc_for_realtime import RVC
    finally:
        sys.argv = original_argv

    rvc = RVC(
        args.pitch,
        str(model_path),
        index_path,
        index_rate,
        n_cpu,
        inp_q,
        opt_q,
        device,
    )
    if not hasattr(rvc, "tgt_sr"):
        raise RuntimeError("RVC failed to initialize. Check model and dependencies.")

    monitor_output: Optional[MonitorOutput] = None
    if monitor_dev is not None and monitor_dev != output_device_idx:
        monitor_channels = 1 if int(devices[monitor_dev]["max_output_channels"]) <= 1 else 2
        monitor_output = MonitorOutput(
            samplerate=int(rvc.tgt_sr),
            block_frame=int(max(0.12, float(args.block_time)) * int(rvc.tgt_sr)),
            channels=monitor_channels,
            device=monitor_dev,
        )

    engine = RealtimeMicVoiceChanger(
        rvc=rvc,
        device=device,
        samplerate=int(rvc.tgt_sr),
        input_channels=input_channels,
        output_channels=output_channels,
        block_time=max(0.12, float(args.block_time)),
        crossfade_time=max(0.0, float(args.crossfade_time)),
        extra_time=max(0.05, float(args.extra_time)),
        threshold_db=float(args.threshold_db),
        f0_method=args.f0_method,
        monitor_output=monitor_output,
    )

    try:
        engine.run()
    except KeyboardInterrupt:
        print("\nStopping realtime conversion...")
    except Exception:
        if monitor_output is not None:
            monitor_output.stop()
        raise
    finally:
        for worker in workers:
            if worker.is_alive():
                worker.terminate()
                worker.join(timeout=0.5)
        try:
            while True:
                inp_q.get_nowait()
        except Empty:
            pass
        try:
            while True:
                opt_q.get_nowait()
        except Empty:
            pass
        print("Stopped.")


if __name__ == "__main__":
    main()
