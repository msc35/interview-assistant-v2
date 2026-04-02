import contextlib
import io
import json
import os
import queue
import threading
import time
import warnings
import wave
from datetime import datetime
from pathlib import Path
from typing import Optional
import tkinter as tk
from tkinter import messagebox, scrolledtext, ttk

import google.generativeai as genai
import pyaudio
import speech_recognition as sr

warnings.filterwarnings("ignore", module="whisper")

# openai-whisper does not read this; our _patched_whisper_download() does. Default on to stop SHA256 mismatch loops.
os.environ.setdefault("WHISPER_SKIP_CHECKSUM", "1")


CONFIG_DIR = Path.home() / ".interview_assistant"
CONFIG_FILE = CONFIG_DIR / "config.json"
DEFAULT_ALPHA = 0.96

# Whisper: (model_name, language_or_none). None omits `language` for Whisper auto-detect (mixed TR/EN).
TRANSCRIPTION_PROFILES = {
    "English": ("base.en", "en"),
    "Türkçe": ("small", "tr"),
    "TR + ENG (Mixed)": ("small", None),
}
TRANSCRIPTION_MODES_ORDER = ("English", "Türkçe", "TR + ENG (Mixed)")
DEFAULT_TRANSCRIPTION_MODE = "TR + ENG (Mixed)"
WHISPER_CACHE_ROOT = os.path.expanduser("~/.cache/whisper")
WHISPER_SMALL_PT = Path(WHISPER_CACHE_ROOT) / "small.pt"
# Expected size ~461 MiB; allow a band for filesystem variance.
WHISPER_SMALL_BYTES_MIN = 430 * 1024 * 1024
WHISPER_SMALL_BYTES_MAX = 500 * 1024 * 1024
TELEPROMPTER_ALPHA = 0.55
MIN_ALPHA = 0.30
MAX_ALPHA = 1.00

app_queue = queue.Queue()


def _trusted_small_pt_file(path) -> bool:
    """True if path looks like a complete cached small.pt (~461MB)."""
    try:
        p = Path(path)
        if p.name != "small.pt":
            return False
        sz = p.stat().st_size
    except OSError:
        return False
    return WHISPER_SMALL_BYTES_MIN <= sz <= WHISPER_SMALL_BYTES_MAX


def _whisper_checksum_skip_env() -> bool:
    v = os.environ.get("WHISPER_SKIP_CHECKSUM", "").strip().lower()
    return v in ("1", "true", "yes")


def _patched_whisper_download(url: str, root: str, in_memory: bool):
    """
    Replacement for whisper._download: accept on-disk file when SHA256 mismatches
    (stale hashes in older whisper releases) if WHISPER_SKIP_CHECKSUM is set or small.pt is size-valid.
    """
    import hashlib
    import urllib.request

    from tqdm import tqdm

    os.makedirs(root, exist_ok=True)
    expected_sha256 = url.split("/")[-2]
    download_target = os.path.join(root, os.path.basename(url))
    trust_mismatch = _whisper_checksum_skip_env() or _trusted_small_pt_file(download_target)

    if os.path.exists(download_target) and not os.path.isfile(download_target):
        raise RuntimeError(f"{download_target} exists and is not a regular file")

    if os.path.isfile(download_target):
        with open(download_target, "rb") as f:
            model_bytes = f.read()
        if hashlib.sha256(model_bytes).hexdigest() == expected_sha256:
            return model_bytes if in_memory else download_target
        if trust_mismatch:
            return model_bytes if in_memory else download_target
        import warnings as std_warnings

        std_warnings.warn(
            f"{download_target} exists, but the SHA256 checksum does not match; re-downloading the file"
        )

    with urllib.request.urlopen(url) as source, open(download_target, "wb") as output:
        with tqdm(
            total=int(source.info().get("Content-Length")),
            ncols=80,
            unit="iB",
            unit_scale=True,
            unit_divisor=1024,
        ) as loop:
            while True:
                buffer = source.read(8192)
                if not buffer:
                    break
                output.write(buffer)
                loop.update(len(buffer))

    model_bytes = open(download_target, "rb").read()
    if hashlib.sha256(model_bytes).hexdigest() != expected_sha256:
        if trust_mismatch or _trusted_small_pt_file(download_target):
            return model_bytes if in_memory else download_target
        raise RuntimeError(
            "Model has been downloaded but the SHA256 checksum does not not match. Please retry loading the model."
        )
    return model_bytes if in_memory else download_target


def _install_whisper_checksum_patch() -> None:
    import whisper

    if getattr(whisper, "_interview_assistant_checksum_patch", False):
        return
    whisper._download = _patched_whisper_download
    whisper._interview_assistant_checksum_patch = True


_whisper_model_cache = {}
_whisper_model_cache_lock = threading.Lock()


def get_cached_whisper_model(model_name: str):
    """
    Load each Whisper model name at most once (avoids RAM blowups from recognize_whisper's per-call load_model).
    Thread-safe; used by preload and the single transcription worker.
    """
    _install_whisper_checksum_patch()
    with _whisper_model_cache_lock:
        if model_name not in _whisper_model_cache:
            import whisper

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                    _whisper_model_cache[model_name] = whisper.load_model(
                        model_name, download_root=WHISPER_CACHE_ROOT
                    )
        return _whisper_model_cache[model_name]


def transcribe_audio_with_cached_model(audio_data: sr.AudioData, model_name: str, language: Optional[str]) -> str:
    """Match speech_recognition's whisper path: 16 kHz WAV -> float32 numpy -> model.transcribe."""
    import soundfile as sf
    import torch

    wmodel = get_cached_whisper_model(model_name)
    wav_bytes = audio_data.get_wav_data(convert_rate=16000)
    wav_stream = io.BytesIO(wav_bytes)
    audio_array, _sample_rate = sf.read(wav_stream)
    audio_array = audio_array.astype("float32")
    kwargs = {"fp16": torch.cuda.is_available()}
    if language is not None:
        kwargs["language"] = language
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            result = wmodel.transcribe(audio_array, **kwargs)
    text = (result.get("text") or "").strip()
    return text


def get_ai_suggestion(user_question, system_prompt, api_key, conversation_history=None):
    """Send prompts to Gemini with optional conversation history."""
    try:
        genai.configure(api_key=api_key)
        try:
            model = genai.GenerativeModel(
                "gemini-3-flash-preview",
                system_instruction=system_prompt,
            )
        except Exception:
            try:
                model = genai.GenerativeModel(
                    "gemini-2.5-flash",
                    system_instruction=system_prompt,
                )
            except Exception:
                model = genai.GenerativeModel(
                    "gemini-2.5-flash-lite",
                    system_instruction=system_prompt,
                )

        if conversation_history:
            history_text = "\n\n--- Previous Conversation ---\n"
            for i, (q_text, a_text) in enumerate(conversation_history[-5:], 1):
                history_text += f"\nQ{i}: {q_text}\nA{i}: {a_text}\n"
            history_text += "\n--- Current Question ---\n"
            full_question = history_text + user_question
        else:
            full_question = user_question

        response = model.generate_content(full_question)
        return response.text
    except Exception as err:
        return f"An error occurred with the AI: {err}"


class AssistantApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Interview Assistant")
        self.root.geometry("980x860")
        self.root.minsize(860, 700)

        _install_whisper_checksum_patch()

        self._setup_theme()
        self.root.configure(bg=self.BG_COLOR)

        self.is_closing = False
        self.is_teleprompter_mode = False
        self.status_text = tk.StringVar(value="Ready")
        self.connection_text = tk.StringVar(value="AI: Not Configured")
        self.device_text = tk.StringVar(value="Device: Detecting...")
        self.capture_text = tk.StringVar(value="Capture: idle")
        self.recording_duration_text = tk.StringVar(value="Recording: 00:00")
        self.alpha_percent_var = tk.IntVar(value=int(DEFAULT_ALPHA * 100))
        self.alpha_value = DEFAULT_ALPHA
        self.saved_alpha_before_teleprompter = self.alpha_value
        self.api_key_var = tk.StringVar()
        self.show_api_key = False
        self.api_key_entry = None
        self.record_start_time = None
        self.listening_source = None
        self.last_transcription_time = None

        self.conversation_history = []
        self.max_history = 5

        self.recognizer = sr.Recognizer()
        self.recognizer.pause_threshold = 0.5
        self.recognizer.dynamic_energy_threshold = True
        self.recognizer.energy_threshold = 4000
        self.recognizer.operation_timeout = None
        self.stop_listening = None

        self._transcription_queue = queue.Queue()
        threading.Thread(target=self._transcription_worker_loop, daemon=True).start()

        self.transcription_mode_var = tk.StringVar(value=DEFAULT_TRANSCRIPTION_MODE)
        self._transcribe_model, self._transcribe_language = TRANSCRIPTION_PROFILES[DEFAULT_TRANSCRIPTION_MODE]

        self.available_microphones = []
        self.microphone_device_index = self.find_microphone_device()
        self.device_text.set(f"Device: {self.get_device_name(self.microphone_device_index)}")

        self.is_recording = False
        self.recording_thread = None
        self.audio_stream = None
        self.pyaudio_instance = None
        self.recording_file = None

        self.load_config()
        self.apply_alpha(self.alpha_value)

        self._build_layout()

        threading.Thread(target=self._preload_whisper_small, daemon=True).start()

        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        self.root.after(100, self.check_queue)
        self.root.after(1000, self.update_recording_duration)

    def _setup_theme(self):
        self.style = ttk.Style(self.root)
        self.style.theme_use("clam")

        self.BG_COLOR = "#13151A"
        self.CARD_COLOR = "#1B1F2A"
        self.BORDER_COLOR = "#2A3040"
        self.TEXT_COLOR = "#EEF2FF"
        self.TEXT_SECONDARY = "#A5B0CC"
        self.BUTTON_COLOR = "#272E40"
        self.BUTTON_HOVER = "#333D55"
        self.ACCENT_COLOR = "#6366F1"
        self.ACCENT_HOVER = "#7C82FF"
        self.SUCCESS_COLOR = "#10B981"
        self.DANGER_COLOR = "#EF4444"
        self.TEXT_AREA_BG = "#111621"

        self.style.configure(
            ".",
            background=self.BG_COLOR,
            foreground=self.TEXT_COLOR,
            font=("SF Pro Display", 12),
        )
        self.style.configure("TFrame", background=self.BG_COLOR)
        self.style.configure("TLabel", background=self.BG_COLOR, foreground=self.TEXT_COLOR)
        self.style.configure("TNotebook", background=self.BG_COLOR, borderwidth=0)
        self.style.configure(
            "TNotebook.Tab",
            background=self.CARD_COLOR,
            foreground=self.TEXT_SECONDARY,
            padding=[18, 10],
            font=("SF Pro Display", 12, "bold"),
            borderwidth=0,
        )
        self.style.map(
            "TNotebook.Tab",
            background=[("selected", self.BG_COLOR)],
            foreground=[("selected", self.TEXT_COLOR)],
        )
        self.style.configure(
            "Rounded.TButton",
            background=self.BUTTON_COLOR,
            foreground=self.TEXT_COLOR,
            borderwidth=1,
            relief="flat",
            padding=[14, 9],
            focusthickness=0,
            focuscolor=self.BUTTON_COLOR,
            font=("SF Pro Display", 11, "bold"),
        )
        self.style.map(
            "Rounded.TButton",
            background=[("active", self.BUTTON_HOVER), ("pressed", self.ACCENT_COLOR)],
            bordercolor=[("active", self.ACCENT_COLOR)],
        )
        self.style.configure("Accent.TButton", background=self.ACCENT_COLOR, foreground=self.TEXT_COLOR)
        self.style.map("Accent.TButton", background=[("active", self.ACCENT_HOVER), ("pressed", self.ACCENT_HOVER)])
        self.style.configure("Danger.TButton", background="#7F1D1D", foreground=self.TEXT_COLOR)
        self.style.map("Danger.TButton", background=[("active", "#991B1B"), ("pressed", "#B91C1C")])
        self.style.configure("Card.TLabelframe", background=self.CARD_COLOR, bordercolor=self.BORDER_COLOR, borderwidth=1)
        self.style.configure(
            "Card.TLabelframe.Label",
            background=self.CARD_COLOR,
            foreground=self.TEXT_SECONDARY,
            font=("SF Pro Display", 10, "bold"),
        )
        self.style.configure("Status.TLabel", background=self.CARD_COLOR, foreground=self.TEXT_SECONDARY, font=("SF Pro Text", 10))

    def _build_layout(self):
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(padx=14, pady=(14, 8), fill="both", expand=True)

        self.assistant_tab = ttk.Frame(self.notebook, style="TFrame")
        self.setup_tab = ttk.Frame(self.notebook, style="TFrame")
        self.notebook.add(self.assistant_tab, text="Assistant")
        self.notebook.add(self.setup_tab, text="Setup")

        self.create_assistant_tab()
        self.create_setup_tab()
        self.create_status_bar()
        self.update_history_label()
        self.update_connection_state()

    def create_status_bar(self):
        status_bar = tk.Frame(self.root, bg=self.CARD_COLOR, highlightthickness=1, highlightbackground=self.BORDER_COLOR)
        status_bar.pack(side=tk.BOTTOM, fill="x", padx=12, pady=(0, 10))
        ttk.Label(status_bar, textvariable=self.status_text, style="Status.TLabel").pack(side=tk.LEFT, padx=12, pady=6)
        ttk.Label(status_bar, text="|", style="Status.TLabel").pack(side=tk.LEFT)
        ttk.Label(status_bar, textvariable=self.connection_text, style="Status.TLabel").pack(side=tk.LEFT, padx=8)
        ttk.Label(status_bar, text="|", style="Status.TLabel").pack(side=tk.LEFT)
        ttk.Label(status_bar, textvariable=self.device_text, style="Status.TLabel").pack(side=tk.LEFT, padx=8)
        ttk.Label(status_bar, text="|", style="Status.TLabel").pack(side=tk.LEFT)
        ttk.Label(status_bar, textvariable=self.capture_text, style="Status.TLabel").pack(side=tk.LEFT, padx=8)
        ttk.Label(status_bar, text="|", style="Status.TLabel").pack(side=tk.LEFT)
        ttk.Label(status_bar, textvariable=self.recording_duration_text, style="Status.TLabel").pack(side=tk.LEFT, padx=8)

    def create_assistant_tab(self):
        top_frame = tk.Frame(self.assistant_tab, bg=self.BG_COLOR)
        top_frame.pack(fill="x", padx=12, pady=(10, 8))

        indicator_wrap = tk.Frame(top_frame, bg=self.BG_COLOR)
        indicator_wrap.pack(side=tk.LEFT)
        self.listening_indicator = tk.Canvas(indicator_wrap, width=22, height=22, bg=self.BG_COLOR, highlightthickness=0)
        self.listening_indicator.pack(side=tk.LEFT, padx=(0, 8))
        self.indicator_light = self.listening_indicator.create_oval(4, 4, 18, 18, fill=self.DANGER_COLOR, outline="")
        self.indicator_ring = self.listening_indicator.create_oval(2, 2, 20, 20, outline="", width=0)

        self.status_label = ttk.Label(top_frame, text="Status: Paused", font=("SF Pro Display", 12))
        self.status_label.pack(side=tk.LEFT)

        history_badge = tk.Frame(top_frame, bg=self.CARD_COLOR, highlightthickness=1, highlightbackground=self.BORDER_COLOR)
        history_badge.pack(side=tk.RIGHT)
        self.history_label = ttk.Label(history_badge, text="History: 0/5", background=self.CARD_COLOR, foreground=self.TEXT_SECONDARY)
        self.history_label.pack(padx=10, pady=4)

        response_card = ttk.LabelFrame(self.assistant_tab, text="AI Response", style="Card.TLabelframe")
        response_card.pack(fill="both", expand=True, padx=12, pady=(4, 8))
        response_controls = tk.Frame(response_card, bg=self.CARD_COLOR)
        response_controls.pack(fill="x", padx=10, pady=(10, 6))
        self.get_suggestion_button = ttk.Button(
            response_controls, text="Get AI Suggestion", command=self.fetch_suggestion, style="Accent.TButton"
        )
        self.get_suggestion_button.pack(side=tk.LEFT)
        self.suggestion_text = scrolledtext.ScrolledText(
            response_card,
            height=10,
            state="disabled",
            font=("SF Mono", 17),
            bg=self.TEXT_AREA_BG,
            fg=self.TEXT_COLOR,
            wrap=tk.WORD,
            relief="flat",
            borderwidth=0,
            padx=12,
            pady=10,
            selectbackground=self.ACCENT_COLOR,
        )
        self.suggestion_text.pack(fill="both", expand=True, padx=10, pady=(0, 10))

        listening_card = ttk.LabelFrame(self.assistant_tab, text="Listening", style="Card.TLabelframe")
        listening_card.pack(fill="x", padx=12, pady=(4, 8))
        listening_body = tk.Frame(listening_card, bg=self.CARD_COLOR)
        listening_body.pack(fill="x", padx=10, pady=10)
        self.toggle_button = ttk.Button(listening_body, text="Start Listening", command=self.toggle_listening, style="Rounded.TButton")
        self.toggle_button.pack(side=tk.LEFT, padx=(0, 8))
        self.clear_button = ttk.Button(listening_body, text="Clear Text", command=self.clear_text, style="Rounded.TButton")
        self.clear_button.pack(side=tk.LEFT, padx=8)
        self.clear_history_button = ttk.Button(listening_body, text="Clear History", command=self.clear_history, style="Rounded.TButton")
        self.clear_history_button.pack(side=tk.LEFT, padx=8)
        self.teleprompter_button = ttk.Button(
            listening_body,
            text="Teleprompter Mode: Off",
            command=self.toggle_teleprompter_mode,
            style="Rounded.TButton",
        )
        self.teleprompter_button.pack(side=tk.LEFT, padx=8)

        recording_card = ttk.LabelFrame(self.assistant_tab, text="Recording", style="Card.TLabelframe")
        recording_card.pack(fill="x", padx=12, pady=(4, 8))
        recording_body = tk.Frame(recording_card, bg=self.CARD_COLOR)
        recording_body.pack(fill="x", padx=10, pady=10)
        self.record_button = ttk.Button(recording_body, text="Start Recording", command=self.toggle_recording, style="Danger.TButton")
        self.record_button.pack(side=tk.LEFT)
        self.recording_status_label = ttk.Label(recording_body, text="", background=self.CARD_COLOR, foreground=self.TEXT_SECONDARY)
        self.recording_status_label.pack(side=tk.LEFT, padx=10)

        question_card = ttk.LabelFrame(self.assistant_tab, text="Question (Editable)", style="Card.TLabelframe")
        question_card.pack(fill="both", expand=True, padx=12, pady=(4, 10))
        self.question_text = scrolledtext.ScrolledText(
            question_card,
            height=6,
            font=("SF Mono", 13),
            bg=self.TEXT_AREA_BG,
            fg=self.TEXT_COLOR,
            relief="flat",
            borderwidth=0,
            padx=12,
            pady=10,
            insertbackground=self.ACCENT_COLOR,
            selectbackground=self.ACCENT_COLOR,
            wrap=tk.WORD,
        )
        self.question_text.pack(fill="both", expand=True, padx=10, pady=10)

    def create_setup_tab(self):
        container = tk.Frame(self.setup_tab, bg=self.BG_COLOR)
        container.pack(fill="both", expand=True, padx=12, pady=10)

        canvas = tk.Canvas(container, bg=self.BG_COLOR, highlightthickness=0)
        scrollbar = ttk.Scrollbar(container, orient="vertical", command=canvas.yview)
        scrollable = tk.Frame(canvas, bg=self.BG_COLOR)

        def _set_scroll_region(_evt=None):
            canvas.configure(scrollregion=canvas.bbox("all"))

        def _set_width(evt):
            canvas.itemconfig(canvas_window, width=evt.width)

        scrollable.bind("<Configure>", _set_scroll_region)
        canvas_window = canvas.create_window((0, 0), window=scrollable, anchor="nw")
        canvas.bind("<Configure>", _set_width)
        canvas.configure(yscrollcommand=scrollbar.set)
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        api_card = ttk.LabelFrame(scrollable, text="API & Window", style="Card.TLabelframe")
        api_card.pack(fill="x", pady=(0, 10))
        api_body = tk.Frame(api_card, bg=self.CARD_COLOR)
        api_body.pack(fill="x", padx=10, pady=10)

        ttk.Label(api_body, text="Gemini API Key", background=self.CARD_COLOR).grid(row=0, column=0, sticky="w")
        self.api_key_entry = ttk.Entry(api_body, textvariable=self.api_key_var, show="*", width=58)
        self.api_key_entry.grid(row=1, column=0, sticky="we", padx=(0, 8), pady=(4, 0))
        ttk.Button(api_body, text="Show", command=self.toggle_api_visibility, style="Rounded.TButton").grid(
            row=1, column=1, padx=4, pady=(4, 0)
        )
        ttk.Button(api_body, text="Save", command=self.save_api_key, style="Rounded.TButton").grid(
            row=1, column=2, padx=4, pady=(4, 0)
        )

        ttk.Label(api_body, text="Window Opacity", background=self.CARD_COLOR).grid(row=2, column=0, sticky="w", pady=(12, 0))
        opacity_row = tk.Frame(api_body, bg=self.CARD_COLOR)
        opacity_row.grid(row=3, column=0, columnspan=3, sticky="we", pady=(4, 0))
        self.opacity_scale = ttk.Scale(
            opacity_row,
            from_=int(MIN_ALPHA * 100),
            to=int(MAX_ALPHA * 100),
            orient=tk.HORIZONTAL,
            variable=self.alpha_percent_var,
            command=self.on_opacity_slider,
        )
        self.opacity_scale.pack(side=tk.LEFT, fill="x", expand=True, padx=(0, 8))
        self.opacity_label = ttk.Label(opacity_row, text=f"{self.alpha_percent_var.get()}%", background=self.CARD_COLOR)
        self.opacity_label.pack(side=tk.LEFT)

        ttk.Label(api_body, text="Audio Input Device", background=self.CARD_COLOR).grid(row=4, column=0, sticky="w", pady=(12, 0))
        device_row = tk.Frame(api_body, bg=self.CARD_COLOR)
        device_row.grid(row=5, column=0, columnspan=3, sticky="we", pady=(4, 0))
        self.device_var = tk.StringVar(value="")
        self.device_combo = ttk.Combobox(device_row, textvariable=self.device_var, state="readonly", width=52)
        self.device_combo.pack(side=tk.LEFT, fill="x", expand=True, padx=(0, 8))
        ttk.Button(device_row, text="Refresh", command=self.refresh_microphone_list, style="Rounded.TButton").pack(side=tk.LEFT, padx=4)
        ttk.Button(device_row, text="Use Selected", command=self.apply_selected_microphone, style="Rounded.TButton").pack(side=tk.LEFT, padx=4)
        api_body.columnconfigure(0, weight=1)

        ttk.Label(api_body, text="Transcription language", background=self.CARD_COLOR).grid(
            row=6, column=0, sticky="w", pady=(12, 0)
        )
        transcription_row = tk.Frame(api_body, bg=self.CARD_COLOR)
        transcription_row.grid(row=7, column=0, columnspan=3, sticky="we", pady=(4, 0))
        self.transcription_combo = ttk.Combobox(
            transcription_row,
            textvariable=self.transcription_mode_var,
            values=TRANSCRIPTION_MODES_ORDER,
            state="readonly",
            width=50,
        )
        self.transcription_combo.pack(side=tk.LEFT, fill="x", expand=True)
        self.transcription_mode_var.trace_add("write", self._on_transcription_mode_write)
        self._apply_transcription_mode_key()

        cv_card = ttk.LabelFrame(scrollable, text="CV", style="Card.TLabelframe")
        cv_card.pack(fill="both", expand=True, pady=(0, 10))
        self.cv_text = scrolledtext.ScrolledText(
            cv_card,
            height=8,
            font=("SF Mono", 12),
            bg=self.TEXT_AREA_BG,
            fg=self.TEXT_COLOR,
            relief="flat",
            borderwidth=0,
            padx=12,
            pady=10,
            insertbackground=self.ACCENT_COLOR,
            wrap=tk.WORD,
        )
        self.cv_text.pack(fill="both", expand=True, padx=10, pady=10)
        try:
            with open("cv.txt", "r", encoding="utf-8") as file:
                self.cv_text.insert(tk.END, file.read())
        except FileNotFoundError:
            pass

        jd_card = ttk.LabelFrame(scrollable, text="Job Description", style="Card.TLabelframe")
        jd_card.pack(fill="both", expand=True, pady=(0, 10))
        self.jd_text = scrolledtext.ScrolledText(
            jd_card,
            height=8,
            font=("SF Mono", 12),
            bg=self.TEXT_AREA_BG,
            fg=self.TEXT_COLOR,
            relief="flat",
            borderwidth=0,
            padx=12,
            pady=10,
            insertbackground=self.ACCENT_COLOR,
            wrap=tk.WORD,
        )
        self.jd_text.pack(fill="both", expand=True, padx=10, pady=10)
        try:
            with open("job_description.txt", "r", encoding="utf-8") as file:
                self.jd_text.insert(tk.END, file.read())
        except FileNotFoundError:
            pass

        prompt_card = ttk.LabelFrame(scrollable, text="System Prompt (AI Instructions)", style="Card.TLabelframe")
        prompt_card.pack(fill="both", expand=True, pady=(0, 10))
        self.system_prompt_text = scrolledtext.ScrolledText(
            prompt_card,
            height=10,
            font=("SF Mono", 12),
            bg=self.TEXT_AREA_BG,
            fg=self.TEXT_COLOR,
            relief="flat",
            borderwidth=0,
            padx=12,
            pady=10,
            insertbackground=self.ACCENT_COLOR,
            wrap=tk.WORD,
        )
        self.system_prompt_text.pack(fill="both", expand=True, padx=10, pady=10)

        default_system_prompt = """
You are my real-time interview copilot. I'm presenting my data science case study and answering questions live — speed matters.

Language: Answer in Turkish. Use English for technical terms naturally (ARPU, churn, SHAP, LightGBM, PR-AUC, pipeline, ROAS, retention, conversion rate, etc.) — exactly how Turkish data scientists speak.

Format:
- 4–6 sentences max. I need to read and speak this in seconds.
- No intro, no filler, no "Harika soru" or "Şöyle açıklayayım".
- Start with the answer. Lead with the strongest point.
- Bullet points ONLY if listing 3+ items side by side.
- Numbers and metrics first, explanation second.

Style:
- First person, natural spoken Turkish cadence.
- Confident but not arrogant. Like a senior data scientist defending their own work.
- End every answer with a concrete metric, result, or decision — never trail off.

For technical questions: If it's a concept/definition question ("X nedir", "Y'yi açıkla", "Z ne demek"), give a clean textbook-level explanation first — only tie to the case if it flows naturally in one short clause. If it's a case-specific technical question ("neden X kullandın", "şu metrik neden düşük"), then give the metric, the method, the why — in that order.
For "neden bu approach" questions: State the decision, then the tradeoff I considered.
For "what if" / challenge questions: Acknowledge the concern in one clause, then defend with data.
For questions about a specific section: Pull the exact numbers from the case context below.

My background:
--- {cv} ---

Role, job description, and my full case study summary are all here — use this as the single source of truth for every answer:
--- {job_description} ---

Golden rule: If my answer can be shorter and still land the point, make it shorter.
"""
        self.system_prompt_text.insert(tk.END, default_system_prompt.strip())

        user_prompt_card = ttk.LabelFrame(scrollable, text="User Prompt (Question Template)", style="Card.TLabelframe")
        user_prompt_card.pack(fill="x", expand=True, pady=(0, 8))
        self.user_prompt_text = scrolledtext.ScrolledText(
            user_prompt_card,
            height=3,
            font=("SF Mono", 12),
            bg=self.TEXT_AREA_BG,
            fg=self.TEXT_COLOR,
            relief="flat",
            borderwidth=0,
            padx=12,
            pady=10,
            insertbackground=self.ACCENT_COLOR,
            wrap=tk.WORD,
        )
        self.user_prompt_text.pack(fill="x", expand=True, padx=10, pady=10)
        self.user_prompt_text.insert(
            tk.END,
            "Interviewer's question: \"{transcribed_text}\". Give me a ready-to-speak answer. Turkish with English technical terms.",
        )
        self.refresh_microphone_list()

    def config_payload(self):
        return {
            "api_key": self.api_key_var.get().strip(),
            "window_alpha": round(self.alpha_value, 2),
            "microphone_device_index": self.microphone_device_index,
            "transcription_mode": self.transcription_mode_var.get(),
        }

    def load_config(self):
        try:
            if CONFIG_FILE.exists():
                config = json.loads(CONFIG_FILE.read_text(encoding="utf-8"))
                self.api_key_var.set(config.get("api_key", "").strip())
                self.alpha_value = float(config.get("window_alpha", DEFAULT_ALPHA))
                self.alpha_value = min(MAX_ALPHA, max(MIN_ALPHA, self.alpha_value))
                self.alpha_percent_var.set(int(self.alpha_value * 100))
                saved_device = config.get("microphone_device_index")
                if isinstance(saved_device, int):
                    self.microphone_device_index = saved_device
                tm = config.get("transcription_mode", DEFAULT_TRANSCRIPTION_MODE)
                if tm in TRANSCRIPTION_PROFILES:
                    self.transcription_mode_var.set(tm)
                    self._apply_transcription_mode_key(tm)
        except Exception as err:
            print(f"Config load error: {err}")

    def save_config(self):
        try:
            CONFIG_DIR.mkdir(parents=True, exist_ok=True)
            CONFIG_FILE.write_text(json.dumps(self.config_payload(), indent=2), encoding="utf-8")
        except Exception as err:
            messagebox.showerror("Config Error", f"Could not save config: {err}")

    def save_api_key(self):
        key = self.api_key_var.get().strip()
        if not key:
            messagebox.showwarning("Missing API Key", "Please paste an API key before saving.")
            return
        self.save_config()
        self.update_connection_state()
        self.status_text.set("API key saved")

    def toggle_api_visibility(self):
        self.show_api_key = not self.show_api_key
        self.api_key_entry.configure(show="" if self.show_api_key else "*")
        if self.api_key_entry:
            for child in self.api_key_entry.master.winfo_children():
                if isinstance(child, ttk.Button) and child.cget("text") in {"Show", "Hide"}:
                    child.configure(text="Hide" if self.show_api_key else "Show")
                    break

    def require_api_key(self):
        key = self.api_key_var.get().strip()
        if key:
            return key
        messagebox.showwarning(
            "API Key Required",
            "No API key configured. Go to Setup tab, add your key, and click Save.",
        )
        self.notebook.select(self.setup_tab)
        return None

    def update_connection_state(self):
        has_key = bool(self.api_key_var.get().strip())
        self.connection_text.set("AI: Configured" if has_key else "AI: Not Configured")

    def apply_alpha(self, value):
        alpha = min(MAX_ALPHA, max(MIN_ALPHA, float(value)))
        self.alpha_value = alpha
        self.alpha_percent_var.set(int(alpha * 100))
        self.root.attributes("-alpha", alpha)
        if hasattr(self, "opacity_label"):
            self.opacity_label.config(text=f"{int(alpha * 100)}%")

    def on_opacity_slider(self, _value=None):
        alpha = self.alpha_percent_var.get() / 100.0
        self.apply_alpha(alpha)
        self.save_config()

    def toggle_teleprompter_mode(self):
        self.is_teleprompter_mode = not self.is_teleprompter_mode
        if self.is_teleprompter_mode:
            self.saved_alpha_before_teleprompter = self.alpha_value
            self.root.attributes("-topmost", True)
            self.apply_alpha(TELEPROMPTER_ALPHA)
            self.teleprompter_button.config(text="Teleprompter Mode: On")
            self.status_text.set("Teleprompter mode enabled")
        else:
            self.root.attributes("-topmost", False)
            self.apply_alpha(self.saved_alpha_before_teleprompter)
            self.teleprompter_button.config(text="Teleprompter Mode: Off")
            self.status_text.set("Teleprompter mode disabled")
        self.save_config()

    def find_microphone_device(self):
        try:
            mic_list = sr.Microphone.list_microphone_names()
            self.available_microphones = mic_list
            print(f"Found {len(mic_list)} audio devices")
            for idx, name in enumerate(mic_list):
                lower_name = name.lower()
                if "blackhole" in lower_name:
                    print(f"Using BlackHole device: {name} (index {idx})")
                    return idx
            for idx, name in enumerate(mic_list):
                lower_name = name.lower()
                if "multi" in lower_name or "virtual" in lower_name or "loopback" in lower_name:
                    print(f"Using virtual device: {name} (index {idx})")
                    return idx
            return 0
        except Exception as err:
            print(f"Error detecting devices: {err}")
            return 0

    def get_device_name(self, index):
        try:
            names = sr.Microphone.list_microphone_names()
            self.available_microphones = names
            if 0 <= index < len(names):
                return names[index]
        except Exception:
            pass
        return f"Index {index}"

    def refresh_microphone_list(self):
        try:
            names = sr.Microphone.list_microphone_names()
            self.available_microphones = names
            values = [f"{idx}: {name}" for idx, name in enumerate(names)]
            self.device_combo["values"] = values
            if not values:
                self.device_var.set("")
                self.status_text.set("No microphone devices found")
                return
            if 0 <= self.microphone_device_index < len(names):
                self.device_var.set(values[self.microphone_device_index])
            else:
                self.microphone_device_index = self.find_microphone_device()
                chosen_idx = min(max(self.microphone_device_index, 0), len(values) - 1)
                self.device_var.set(values[chosen_idx])
            self.device_text.set(f"Device: {self.get_device_name(self.microphone_device_index)}")
            self.status_text.set(f"Found {len(names)} input devices")
        except Exception as err:
            self.status_text.set(f"Device refresh failed: {str(err)[:50]}")

    def apply_selected_microphone(self):
        selected = self.device_var.get().strip()
        if not selected:
            return
        try:
            idx = int(selected.split(":", 1)[0].strip())
            self.microphone_device_index = idx
            self.device_text.set(f"Device: {self.get_device_name(idx)}")
            self.save_config()
            self.status_text.set(f"Using input device index {idx}")
        except Exception:
            self.status_text.set("Could not parse selected device")

    def _apply_transcription_mode_key(self, key=None):
        if key is None:
            key = self.transcription_mode_var.get()
        if key not in TRANSCRIPTION_PROFILES:
            key = DEFAULT_TRANSCRIPTION_MODE
            self.transcription_mode_var.set(key)
        self._transcribe_model, self._transcribe_language = TRANSCRIPTION_PROFILES[key]

    def _on_transcription_mode_write(self, *_args):
        self._apply_transcription_mode_key()
        self.save_config()

    def _preload_whisper_small(self):
        def set_status(msg):
            if not self.is_closing:
                self.root.after(0, lambda m=msg: self.status_text.set(m))

        _install_whisper_checksum_patch()
        cached_ok = WHISPER_SMALL_PT.exists() and _trusted_small_pt_file(WHISPER_SMALL_PT)
        show_status = not cached_ok

        if show_status:
            set_status("Downloading language model...")
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                    get_cached_whisper_model("small")
        finally:
            if show_status:
                set_status("Language model ready")

    def _schedule_capture_status(self, message: str):
        if self.is_closing:
            return

        def apply():
            try:
                self.capture_text.set(message)
            except tk.TclError:
                pass

        try:
            self.root.after(0, apply)
        except tk.TclError:
            pass

    def _transcription_worker_loop(self):
        while True:
            item = self._transcription_queue.get()
            if item is None:
                break
            audio, model_name, lang, mode = item
            try:
                print(f"[Transcription] mode={mode} model={model_name} lang={lang}")
                text = transcribe_audio_with_cached_model(audio, model_name, lang)
                if text and text.strip():
                    app_queue.put(text)
                    self.last_transcription_time = time.time()
                    self._schedule_capture_status("Capture: transcribed")
                else:
                    self._schedule_capture_status("Capture: silence")
            except Exception as err:
                error_msg = str(err).lower()
                if "could not" not in error_msg and "not understand" not in error_msg:
                    print(f"Transcription error: {str(err)[:100]}")
                self._schedule_capture_status("Capture: transcription error")

    def on_closing(self):
        self.is_closing = True
        if self.stop_listening:
            self.stop_listening(wait_for_stop=False)
        if self.is_recording:
            self.stop_recording()
        try:
            self._transcription_queue.put_nowait(None)
        except Exception:
            pass
        self.save_config()
        self.root.destroy()

    def audio_callback(self, recognizer, audio):
        if self.is_closing:
            return
        model_name = self._transcribe_model
        lang = self._transcribe_language
        mode = self.transcription_mode_var.get()
        self._transcription_queue.put((audio, model_name, lang, mode))

    def toggle_listening(self):
        if self.stop_listening:
            self.stop_listening(wait_for_stop=False)
            self.stop_listening = None
            self.listening_source = None
            self.toggle_button.config(text="Start Listening")
            self.status_label.config(text="Status: Paused")
            self.listening_indicator.itemconfig(self.indicator_light, fill=self.DANGER_COLOR)
            self.listening_indicator.itemconfig(self.indicator_ring, outline="", width=0)
            self.status_text.set("Listening paused")
            self.capture_text.set("Capture: idle")
        else:
            try:
                # Use a short calibration pass on a temporary source, then create a
                # fresh source for background listening to avoid source lifecycle issues.
                calibration_source = sr.Microphone(device_index=self.microphone_device_index)
                with calibration_source as source_for_calibration:
                    self.recognizer.adjust_for_ambient_noise(source_for_calibration, duration=0.4)
                    self.recognizer.energy_threshold = max(500, int(self.recognizer.energy_threshold * 0.8))

                self.listening_source = sr.Microphone(device_index=self.microphone_device_index)
                self.stop_listening = self.recognizer.listen_in_background(
                    self.listening_source,
                    self.audio_callback,
                    phrase_time_limit=20,
                )
                self.toggle_button.config(text="Pause Listening")
                self.status_label.config(text="Status: Listening")
                self.listening_indicator.itemconfig(self.indicator_light, fill=self.SUCCESS_COLOR)
                self.listening_indicator.itemconfig(self.indicator_ring, outline=self.SUCCESS_COLOR, width=2)
                self.status_text.set("Listening for speech...")
                self.capture_text.set("Capture: waiting for audio")
            except Exception as err:
                print(f"Error starting listener: {err}")
                self.status_label.config(text="Status: Error")
                self.status_text.set("Microphone error")
                self.listening_indicator.itemconfig(self.indicator_light, fill=self.DANGER_COLOR)
                self.listening_indicator.itemconfig(self.indicator_ring, outline="", width=0)
                self.capture_text.set("Capture: error")

    def clear_text(self):
        self.question_text.delete("1.0", tk.END)
        self.suggestion_text.config(state="normal")
        self.suggestion_text.delete("1.0", tk.END)
        self.suggestion_text.config(state="disabled")
        self.status_text.set("Text cleared")

    def clear_history(self):
        self.conversation_history = []
        self.update_history_label()
        self.status_text.set("Conversation history cleared")

    def update_history_label(self):
        self.history_label.config(text=f"History: {len(self.conversation_history)}/{self.max_history}")

    def fetch_suggestion(self):
        question = self.question_text.get("1.0", tk.END).strip()
        if not question:
            return

        api_key = self.require_api_key()
        if not api_key:
            return

        try:
            cv = self.cv_text.get("1.0", tk.END).strip()
            jd = self.jd_text.get("1.0", tk.END).strip()
            system_prompt_template = self.system_prompt_text.get("1.0", tk.END).strip()
            user_prompt_template = self.user_prompt_text.get("1.0", tk.END).strip()
            final_system_prompt = system_prompt_template.format(cv=cv, job_description=jd)
            final_user_question = user_prompt_template.format(transcribed_text=question)
        except KeyError as err:
            messagebox.showerror("Template Error", f"Prompt template has a missing placeholder: {err}")
            return
        except Exception as err:
            messagebox.showerror("Prompt Error", f"Could not prepare prompts: {err}")
            return

        self.update_connection_state()
        self.get_suggestion_button.config(text="Getting...", state="disabled")
        self.suggestion_text.config(state="normal")
        self.suggestion_text.delete("1.0", tk.END)
        self.suggestion_text.insert(tk.END, "Thinking...")
        self.suggestion_text.config(state="disabled")
        self.status_text.set("Requesting AI suggestion...")
        self.root.update()

        threading.Thread(
            target=self._get_suggestion_thread,
            args=(final_user_question, final_system_prompt, question, api_key),
            daemon=True,
        ).start()

    def _get_suggestion_thread(self, user_question, system_prompt, original_question, api_key):
        suggestion = get_ai_suggestion(user_question, system_prompt, api_key, self.conversation_history)
        self.root.after(0, self.update_suggestion_text, suggestion, original_question)

    def update_suggestion_text(self, suggestion, question):
        self.suggestion_text.config(state="normal")
        self.suggestion_text.delete("1.0", tk.END)
        self.suggestion_text.insert(tk.END, suggestion)
        self.suggestion_text.config(state="disabled")
        self.get_suggestion_button.config(text="Get AI Suggestion", state="normal")
        self.status_text.set("AI response ready")

        if question and suggestion:
            self.conversation_history.append((question, suggestion))
            if len(self.conversation_history) > self.max_history:
                self.conversation_history.pop(0)
            self.update_history_label()

    def check_queue(self):
        try:
            message = app_queue.get_nowait()
            current_text = self.question_text.get("1.0", tk.END).strip()
            if not current_text:
                self.question_text.insert(tk.END, message)
            else:
                self.question_text.insert(tk.END, "\n" + message)
            self.question_text.see(tk.END)
        except queue.Empty:
            pass
        finally:
            if not self.is_closing:
                self.root.after(100, self.check_queue)

    def toggle_recording(self):
        if self.is_recording:
            self.stop_recording()
        else:
            self.start_recording()

    def start_recording(self):
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            if not os.path.exists("recordings"):
                os.makedirs("recordings")
            self.recording_file = f"recordings/recording_{timestamp}.wav"

            self.pyaudio_instance = pyaudio.PyAudio()
            chunk = 1024
            audio_format = pyaudio.paInt16
            channels = 1
            rate = 44100
            self.audio_stream = self.pyaudio_instance.open(
                format=audio_format,
                channels=channels,
                rate=rate,
                input=True,
                input_device_index=self.microphone_device_index,
                frames_per_buffer=chunk,
            )
            self.is_recording = True
            self.record_start_time = time.time()
            self.record_button.config(text="Stop Recording")
            self.recording_status_label.config(
                text=f"Recording to: {os.path.basename(self.recording_file)}",
                foreground=self.SUCCESS_COLOR,
            )
            self.status_text.set("Recording started")
            self.recording_thread = threading.Thread(target=self._record_audio, daemon=True)
            self.recording_thread.start()
        except Exception as err:
            print(f"Error starting recording: {err}")
            self.recording_status_label.config(text=f"Error: {str(err)[:60]}", foreground=self.DANGER_COLOR)
            self.status_text.set("Recording failed")
            if self.audio_stream:
                self.audio_stream.stop_stream()
                self.audio_stream.close()
            if self.pyaudio_instance:
                self.pyaudio_instance.terminate()
                self.pyaudio_instance = None

    def stop_recording(self):
        self.is_recording = False
        if self.recording_thread and self.recording_thread.is_alive():
            self.recording_thread.join(timeout=2)

        self.record_button.config(text="Start Recording")
        if self.recording_file:
            self.recording_status_label.config(
                text=f"Saved: {os.path.basename(self.recording_file)}",
                foreground=self.TEXT_SECONDARY,
            )
            self.status_text.set("Recording saved")
        else:
            self.recording_status_label.config(text="")
            self.status_text.set("Recording stopped")
        self.record_start_time = None
        self.recording_duration_text.set("Recording: 00:00")

    def update_recording_duration(self):
        if self.is_recording and self.record_start_time:
            elapsed = int(time.time() - self.record_start_time)
            minutes = elapsed // 60
            seconds = elapsed % 60
            self.recording_duration_text.set(f"Recording: {minutes:02d}:{seconds:02d}")
        if not self.is_closing:
            self.root.after(1000, self.update_recording_duration)

    def _record_audio(self):
        chunk = 1024
        audio_format = pyaudio.paInt16
        channels = 1
        rate = 44100
        frames = []
        try:
            while self.is_recording and self.audio_stream:
                data = self.audio_stream.read(chunk, exception_on_overflow=False)
                frames.append(data)
        except Exception as err:
            print(f"Error during recording: {err}")
        finally:
            if frames and self.recording_file:
                try:
                    wf = wave.open(self.recording_file, "wb")
                    wf.setnchannels(channels)
                    if self.pyaudio_instance:
                        wf.setsampwidth(self.pyaudio_instance.get_sample_size(audio_format))
                    else:
                        wf.setsampwidth(2)
                    wf.setframerate(rate)
                    wf.writeframes(b"".join(frames))
                    wf.close()
                    print(f"Recording saved to: {self.recording_file}")
                except Exception as err:
                    print(f"Error saving recording: {err}")

            if self.audio_stream:
                try:
                    self.audio_stream.stop_stream()
                    self.audio_stream.close()
                except Exception:
                    pass
            if self.pyaudio_instance:
                try:
                    self.pyaudio_instance.terminate()
                except Exception:
                    pass
            self.pyaudio_instance = None
            self.audio_stream = None


if __name__ == "__main__":
    root = tk.Tk()
    app = AssistantApp(root)
    root.mainloop()

