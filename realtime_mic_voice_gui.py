from __future__ import annotations

import subprocess
import threading
import tkinter as tk
from pathlib import Path
from tkinter import scrolledtext
from tkinter import messagebox, ttk
from typing import Dict, Optional

import sounddevice as sd


class RealtimeVoiceGui:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("Realtime Voice Mic")
        self.root.geometry("760x620")
        self.root.resizable(False, False)

        self.repo_root = Path(__file__).resolve().parent
        self.python_exe = self.repo_root / ".venv" / "Scripts" / "python.exe"
        self.runner = self.repo_root / "realtime_mic_voice.py"
        self.weights_root = self.repo_root / "weights"

        self.process: Optional[subprocess.Popen] = None
        self.active_model_name: Optional[str] = None
        self.active_input_device_id: Optional[int] = None
        self.active_output_device_id: Optional[int] = None
        self.active_monitor_device_id: Optional[int] = None
        self.pending_model_name: Optional[str] = None
        self.pending_input_device_id: Optional[int] = None
        self.pending_output_device_id: Optional[int] = None
        self.pending_monitor_device_id: Optional[int] = None

        self.voice_var = tk.StringVar()
        self.input_device_var = tk.StringVar()
        self.output_device_var = tk.StringVar()
        self.monitor_device_var = tk.StringVar()
        self.pitch_var = tk.StringVar(value="0")
        self.index_rate_var = tk.StringVar(value="0.08")
        self.status_var = tk.StringVar(value="Ready")
        self.log_lines = 0
        self.input_device_map: Dict[str, int] = {}
        self.output_device_map: Dict[str, int] = {}

        self._build_ui()
        self.refresh_voices()
        self.refresh_input_devices()
        self.refresh_output_devices()
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

    def _build_ui(self) -> None:
        frame = ttk.Frame(self.root, padding=16)
        frame.pack(fill="both", expand=True)

        ttk.Label(frame, text="Voice model").grid(row=0, column=0, sticky="w")
        self.voice_combo = ttk.Combobox(
            frame,
            textvariable=self.voice_var,
            state="readonly",
            width=56,
        )
        self.voice_combo.grid(row=1, column=0, columnspan=3, sticky="we", pady=(4, 12))
        self.voice_combo.bind("<<ComboboxSelected>>", self.on_voice_selected)

        ttk.Label(frame, text="Microphone").grid(row=2, column=0, sticky="w")
        self.input_device_combo = ttk.Combobox(
            frame,
            textvariable=self.input_device_var,
            state="readonly",
            width=56,
        )
        self.input_device_combo.grid(
            row=3, column=0, columnspan=3, sticky="we", pady=(4, 12)
        )

        ttk.Label(frame, text="Call output (virtual cable)").grid(row=4, column=0, sticky="w")
        self.output_device_combo = ttk.Combobox(
            frame,
            textvariable=self.output_device_var,
            state="readonly",
            width=56,
        )
        self.output_device_combo.grid(
            row=5, column=0, columnspan=3, sticky="we", pady=(4, 12)
        )

        ttk.Label(frame, text="Hear output (optional monitor)").grid(row=6, column=0, sticky="w")
        self.monitor_device_combo = ttk.Combobox(
            frame,
            textvariable=self.monitor_device_var,
            state="readonly",
            width=56,
        )
        self.monitor_device_combo.grid(
            row=7, column=0, columnspan=3, sticky="we", pady=(4, 12)
        )

        ttk.Label(frame, text="Pitch").grid(row=8, column=0, sticky="w")
        ttk.Entry(frame, textvariable=self.pitch_var, width=10).grid(
            row=9, column=0, sticky="w", pady=(4, 12)
        )

        ttk.Label(frame, text="Index rate").grid(row=8, column=1, sticky="w")
        ttk.Entry(frame, textvariable=self.index_rate_var, width=10).grid(
            row=9, column=1, sticky="w", pady=(4, 12)
        )

        self.start_button = ttk.Button(
            frame,
            text="Start Mic",
            command=self.start_mic,
            width=18,
        )
        self.start_button.grid(row=10, column=0, sticky="w")

        self.stop_button = ttk.Button(
            frame,
            text="Stop Mic",
            command=self.stop_mic,
            width=18,
            state="disabled",
        )
        self.stop_button.grid(row=10, column=1, sticky="w")

        ttk.Button(
            frame,
            text="Refresh Voices",
            command=self.refresh_voices,
            width=18,
        ).grid(row=10, column=2, sticky="w")

        ttk.Button(
            frame,
            text="Refresh Devices",
            command=self.refresh_devices,
            width=18,
        ).grid(row=11, column=2, sticky="w")

        ttk.Label(frame, text="Status").grid(row=12, column=0, sticky="w", pady=(16, 4))
        ttk.Label(frame, textvariable=self.status_var).grid(
            row=13, column=0, columnspan=3, sticky="w"
        )

        self.log_box = scrolledtext.ScrolledText(
            frame,
            width=72,
            height=10,
            state="disabled",
        )
        self.log_box.grid(row=14, column=0, columnspan=3, sticky="we", pady=(10, 0))

    def refresh_devices(self) -> None:
        self.refresh_input_devices()
        self.refresh_output_devices()

    def refresh_voices(self) -> None:
        voices = sorted(self.weights_root.glob("*.pth"))
        names = [voice.name for voice in voices]
        self.voice_combo["values"] = names
        if names and (not self.voice_var.get() or self.voice_var.get() not in names):
            self.voice_var.set(names[0])
        if not names:
            self.voice_var.set("")
            self.status_var.set("No .pth voices found in weights/")

    def selected_model_path(self) -> Optional[Path]:
        model_name = self.voice_var.get().strip()
        if not model_name:
            return None
        model_path = self.weights_root / model_name
        if not model_path.exists():
            return None
        return model_path

    def refresh_input_devices(self) -> None:
        self.input_device_map = {}
        labels = []
        hostapis = sd.query_hostapis()
        devices = sd.query_devices()
        hostapi_names = {idx: info.get("name", "Unknown API") for idx, info in enumerate(hostapis)}
        for idx, device in enumerate(devices):
            if int(device.get("max_input_channels", 0)) <= 0:
                continue
            host_name = hostapi_names.get(int(device.get("hostapi", -1)), "Unknown API")
            label = f"[{idx}] {device['name']} ({host_name})"
            self.input_device_map[label] = idx
            labels.append(label)

        self.input_device_combo["values"] = labels
        if not labels:
            self.input_device_var.set("")
            self.status_var.set("No microphone devices found.")
            return

        preferred = next(
            (label for label in labels if "microphone" in label.lower()),
            labels[0],
        )
        current = self.input_device_var.get()
        if current and current in self.input_device_map:
            return
        self.input_device_var.set(preferred)

    def refresh_output_devices(self) -> None:
        self.output_device_map = {}
        labels = []
        hostapis = sd.query_hostapis()
        devices = sd.query_devices()
        hostapi_names = {
            idx: info.get("name", "Unknown API") for idx, info in enumerate(hostapis)
        }
        for idx, device in enumerate(devices):
            if int(device.get("max_output_channels", 0)) <= 0:
                continue
            host_name = hostapi_names.get(int(device.get("hostapi", -1)), "Unknown API")
            label = f"[{idx}] {device['name']} ({host_name})"
            self.output_device_map[label] = idx
            labels.append(label)

        monitor_labels = ["(None)"] + labels
        self.output_device_combo["values"] = labels
        self.monitor_device_combo["values"] = monitor_labels

        if not labels:
            self.output_device_var.set("")
            self.monitor_device_var.set("(None)")
            self.status_var.set("No output devices found.")
            return

        current_output = self.output_device_var.get()
        if not current_output or current_output not in self.output_device_map:
            preferred_output = next(
                (label for label in labels if "cable input" in label.lower()),
                labels[0],
            )
            self.output_device_var.set(preferred_output)

        current_monitor = self.monitor_device_var.get()
        if current_monitor not in monitor_labels:
            preferred_monitor = next(
                (
                    label
                    for label in labels
                    if ("headphones" in label.lower() or "speakers" in label.lower())
                    and "cable input" not in label.lower()
                ),
                "(None)",
            )
            self.monitor_device_var.set(preferred_monitor)

    def selected_input_device_id(self) -> Optional[int]:
        label = self.input_device_var.get().strip()
        if not label:
            return None
        return self.input_device_map.get(label)

    def selected_output_device_id(self) -> Optional[int]:
        label = self.output_device_var.get().strip()
        if not label:
            return None
        return self.output_device_map.get(label)

    def selected_monitor_device_id(self) -> Optional[int]:
        label = self.monitor_device_var.get().strip()
        if not label or label == "(None)":
            return None
        return self.output_device_map.get(label)

    def on_voice_selected(self, _event=None) -> None:
        selected = self.voice_var.get().strip()
        if not selected:
            return
        if self.process and self.process.poll() is None:
            if self.active_model_name and selected != self.active_model_name:
                self.status_var.set(
                    f"Voice selected: {selected}. Press Start Mic to switch."
                )
        else:
            self.status_var.set(f"Voice selected: {selected}")

    def start_mic(self) -> None:
        selected_model_path = self.selected_model_path()
        selected_model_name = self.voice_var.get().strip()
        if not selected_model_name:
            messagebox.showerror("No voice", "Pick a voice model first.")
            return
        if selected_model_path is None:
            messagebox.showerror("Missing voice", "Selected .pth file was not found in weights/.")
            return
        if not self.python_exe.exists():
            messagebox.showerror("Missing Python", f"Not found: {self.python_exe}")
            return
        if not self.runner.exists():
            messagebox.showerror("Missing runner", f"Not found: {self.runner}")
            return
        input_device_id = self.selected_input_device_id()
        if input_device_id is None:
            messagebox.showerror("No mic", "Pick a microphone first.")
            return
        output_device_id = self.selected_output_device_id()
        if output_device_id is None:
            messagebox.showerror("No output", "Pick a call output device first.")
            return
        monitor_device_id = self.selected_monitor_device_id()
        if self.process and self.process.poll() is None:
            if (
                self.active_model_name == selected_model_name
                and self.active_input_device_id == input_device_id
                and self.active_output_device_id == output_device_id
                and self.active_monitor_device_id == monitor_device_id
            ):
                self.status_var.set(
                    f"Already running with {selected_model_name} on selected devices."
                )
                return
            self.pending_model_name = selected_model_name
            self.pending_input_device_id = input_device_id
            self.pending_output_device_id = output_device_id
            self.pending_monitor_device_id = monitor_device_id
            self.status_var.set(
                f"Switching to {selected_model_name} with selected audio routing..."
            )
            self.process.terminate()
            return

        pitch = self.pitch_var.get().strip() or "0"
        index_rate = self.index_rate_var.get().strip() or "0.08"

        command = [
            str(self.python_exe),
            "-u",
            str(self.runner),
            "--model",
            str(selected_model_path),
            "--pitch",
            pitch,
            "--f0-method",
            "rmvpe",
            "--index-rate",
            index_rate,
            "--input-device",
            str(input_device_id),
            "--output-device",
            str(output_device_id),
        ]
        if monitor_device_id is not None:
            command.extend(["--monitor-device", str(monitor_device_id)])
        try:
            self._append_log(
                "Starting realtime mic process:\n"
                + " ".join(command)
                + "\n"
            )
            self.process = subprocess.Popen(
                command,
                cwd=str(self.repo_root),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
        except Exception as exc:
            messagebox.showerror("Start failed", str(exc))
            return

        self.start_button.configure(state="disabled")
        self.stop_button.configure(state="normal")
        self.active_model_name = selected_model_name
        self.active_input_device_id = input_device_id
        self.active_output_device_id = output_device_id
        self.active_monitor_device_id = monitor_device_id
        self.status_var.set(
            f"Running: {selected_model_name} | mic='{self.input_device_var.get()}' | call out='{self.output_device_var.get()}' | hear='{self.monitor_device_var.get()}'"
        )
        threading.Thread(target=self._stream_logs, daemon=True).start()
        threading.Thread(target=self._watch_process, daemon=True).start()

    def _stream_logs(self) -> None:
        if not self.process or not self.process.stdout:
            return
        for line in self.process.stdout:
            self.root.after(0, lambda text=line: self._append_log(text))

    def _append_log(self, text: str) -> None:
        self.log_box.configure(state="normal")
        self.log_box.insert("end", text)
        self.log_box.see("end")
        self.log_box.configure(state="disabled")
        self.log_lines += text.count("\n")
        if self.log_lines > 1200:
            self.log_box.configure(state="normal")
            self.log_box.delete("1.0", "200.0")
            self.log_box.configure(state="disabled")
            self.log_lines = 1000

    def _watch_process(self) -> None:
        if not self.process:
            return
        self.process.wait()
        self.root.after(0, self._on_process_exit)

    def _on_process_exit(self) -> None:
        restart_model_name = self.pending_model_name
        restart_input_device_id = self.pending_input_device_id
        restart_output_device_id = self.pending_output_device_id
        restart_monitor_device_id = self.pending_monitor_device_id
        if (
            restart_model_name
            or restart_input_device_id is not None
            or restart_output_device_id is not None
            or self.pending_monitor_device_id is not None
        ):
            self.pending_model_name = None
            self.pending_input_device_id = None
            self.pending_output_device_id = None
            self.pending_monitor_device_id = None
            self.start_button.configure(state="normal")
            self.stop_button.configure(state="disabled")
            self.process = None
            if restart_model_name:
                self.voice_var.set(restart_model_name)
            if restart_input_device_id is not None:
                for label, value in self.input_device_map.items():
                    if value == restart_input_device_id:
                        self.input_device_var.set(label)
                        break
            if restart_output_device_id is not None:
                for label, value in self.output_device_map.items():
                    if value == restart_output_device_id:
                        self.output_device_var.set(label)
                        break
            if restart_monitor_device_id is None:
                self.monitor_device_var.set("(None)")
            else:
                for label, value in self.output_device_map.items():
                    if value == restart_monitor_device_id:
                        self.monitor_device_var.set(label)
                        break
            self.start_mic()
            return
        self.start_button.configure(state="normal")
        self.stop_button.configure(state="disabled")
        code = self.process.returncode if self.process else 0
        self.status_var.set(f"Stopped (exit code {code})")
        self.active_model_name = None
        self.active_input_device_id = None
        self.active_output_device_id = None
        self.active_monitor_device_id = None
        self.process = None

    def stop_mic(self) -> None:
        self.pending_model_name = None
        self.pending_input_device_id = None
        self.pending_output_device_id = None
        self.pending_monitor_device_id = None
        if self.process and self.process.poll() is None:
            self.process.terminate()
            self.status_var.set("Stopping...")

    def on_close(self) -> None:
        self.pending_model_name = None
        self.pending_input_device_id = None
        self.pending_output_device_id = None
        self.pending_monitor_device_id = None
        if self.process and self.process.poll() is None:
            self.process.terminate()
        self.root.destroy()


def main() -> None:
    root = tk.Tk()
    style = ttk.Style(root)
    if "clam" in style.theme_names():
        style.theme_use("clam")
    RealtimeVoiceGui(root)
    root.mainloop()


if __name__ == "__main__":
    main()
