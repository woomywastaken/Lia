# Recipe for PyInstaller: packs lia_app.py, server.py, the page and Python itself into one Lia.exe.
# Build with build_exe.bat (or: python -m PyInstaller Lia.spec --noconfirm).
# Set LIA_BUNDLE_MODELS=1 before building to pack the models folder inside the .exe too (big and slow to start).
import os, importlib.util
from PyInstaller.utils.hooks import collect_all, collect_submodules

datas = [("lia.html", ".")]
if os.path.isdir("static"):
    datas.append(("static", "static"))
if os.environ.get("LIA_BUNDLE_MODELS") == "1" and os.path.isdir("models"):
    datas.append(("models", "models"))
binaries = []
hidden = ["server"] + collect_submodules("uvicorn")

# Optional parts (voice in, natural voice). Each is packed only if it is installed in this Python.
for pkg in ["faster_whisper", "ctranslate2", "tokenizers", "onnxruntime", "kokoro_onnx", "espeakng_loader",
            "phonemizer", "language_tags", "segments", "csvw", "misaki"]:
    try:
        if importlib.util.find_spec(pkg) is None:
            continue
        d, b, h = collect_all(pkg)
        datas += d; binaries += b; hidden += h
    except Exception as e:
        print("Lia.spec: skipped", pkg, e)
if importlib.util.find_spec("cv2") is not None:
    hidden.append("cv2")                                # face sign-in (PyInstaller knows how to pack OpenCV)

a = Analysis(["lia_app.py"], pathex=["."], binaries=binaries, datas=datas, hiddenimports=hidden,
             excludes=["torch", "transformers", "tensorflow", "tkinter", "matplotlib", "IPython", "pytest"])
pyz = PYZ(a.pure)
exe = EXE(pyz, a.scripts, a.binaries, a.datas, [], name="Lia", console=True, upx=False,
          icon="lia.ico" if os.path.exists("lia.ico") else None)
