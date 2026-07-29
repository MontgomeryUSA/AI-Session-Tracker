# SessionMemory.spec — build with:  pyinstaller packaging/SessionMemory.spec
#
# The point of this file is that everything the app will EVER need is inside the
# bundle at install time. After the user runs the installer they can unplug the
# machine from the network permanently and nothing degrades.
#
# Before building, on a CONNECTED machine, populate these:
#
#   models/hub/            Hugging Face cache: faster-whisper + pyannote weights.
#                          Easiest way to fill it correctly is to run the app
#                          once with SM_OFFLINE=0 and a real HF_TOKEN, and let
#                          it download; the cache layout it produces is the one
#                          HF_HUB_OFFLINE=1 knows how to read.
#
#   models/gguf/           llama.cpp weights, loaded in-process. Two files:
#                              bge-m3-Q8_0.gguf              (~610 MB, 1024-dim)
#                              gemma-3-4b-it-Q4_K_M.gguf     (~2.5 GB)
#                          Any GGUF works; set SM_EMBED_GGUF / SM_CHAT_GGUF to
#                          point at different filenames. If you swap the EMBED
#                          model, its dimension must match EMBED_DIM (1024) or
#                          the vault's vec_chunks table won't accept the vectors.
#
#   assets/fonts/*.ttf     Optional. Without them the app uses system fonts.
#
# The weights are the bulk of the installer: whisper-small + pyannote + bge-m3 +
# a 4B chat model lands around 6-8 GB. That's the honest cost of an app that has
# no network at all.

import sys
from PyInstaller.utils.hooks import collect_data_files, collect_dynamic_libs

block_cipher = None

datas = [
    ("../models", "models"),      # HF cache (whisper, pyannote) + gguf/
    ("../assets", "assets"),      # fonts, icons
]

# These ship data files that PyInstaller's analysis won't find by import alone.
for pkg in ("whisperx", "pyannote.audio", "lightning", "lightning_fabric",
            "speechbrain", "torchaudio", "sqlite_vec", "sounddevice", "soundfile",
            "llama_cpp"):
    try:
        datas += collect_data_files(pkg)
    except Exception:
        pass

binaries = []
# llama_cpp ships libllama.so/.dylib/.dll -- without this the app builds fine
# and then dies on first model load with "shared library not found".
for pkg in ("sqlcipher3", "soundfile", "sounddevice", "ctranslate2", "llama_cpp"):
    try:
        binaries += collect_dynamic_libs(pkg)
    except Exception:
        pass

hiddenimports = [
    "sqlcipher3", "sqlite_vec", "sounddevice", "soundfile",
    "scipy.signal", "scipy.special._cdflib",
    "whisperx", "faster_whisper", "pyannote.audio",
    "llama_cpp", "llama_cpp.llama_cpp",
    "PySide6.QtMultimedia",
    "session_store", "functions", "hf_cache_setup", "local_llm",
]

a = Analysis(
    ["../run_app.py"],
    pathex=[".."],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    excludes=["tkinter", "matplotlib", "pytest", "IPython"],
    cipher=block_cipher,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="SessionMemory",
    console=False,          # no terminal window; logs go to the UI panel
    icon="icon.icns" if sys.platform == "darwin" else "icon.ico",
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    name="SessionMemory",
)

if sys.platform == "darwin":
    app = BUNDLE(
        coll,
        name="SessionMemory.app",
        bundle_identifier="com.yourclinic.sessionmemory",
        info_plist={
            # macOS refuses microphone access without this string, and the
            # failure is silent -- the stream just returns zeros forever.
            "NSMicrophoneUsageDescription":
                "Session Memory records therapy sessions on this device. "
                "Audio never leaves this Mac.",
            "LSMinimumSystemVersion": "12.0",
        },
    )
