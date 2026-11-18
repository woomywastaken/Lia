Lia

An offline AI health check assistant for astronauts. Built by Team Asclepius for NASA HUNCH 2026.

⚠️ Lia is a student research prototype, not a doctor and not a medical device. If someone is seriously ill, alert the crew commander immediately. Do not wait for Lia.

Overview

On a long mission, an astronaut who falls ill may be far from a doctor, and it can be hard to tell how urgent the symptoms are. Lia asks the questions a clinician would ask, one at a time, then rates how urgent the case is so the crew can choose between watching and alerting the commander.

Everything runs on one computer with no internet connection, using MedGemma 4B served locally by LM Studio.

Features
Guided questions: one at a time, then a short summary.
Results panel: urgency (low, medium or high) first, then ranked likely causes, what to do now, and when to call the commander.
Safety rules: code-based checks run no matter what the model answers. They can raise urgency but never lower it, so serious cases are not rated low.
Odd-reading flags: impossible or suspicious values (for example 140 °F or a pulse of 1800) show a "Recheck these readings" note.
Local AI, always an alert: the screen always tells the crew to alert the commander if someone is seriously ill.
Voice in and out (optional): offline speech recognition (Whisper) and natural voice (Kokoro).
Face sign-in (optional): stores only a list of numbers per person, not photos, on this computer, with a Remove button. It is a convenience, not security.
Watch and sensor input: pulse, oxygen and skin temperature from a watch, or cabin CO2 and radiation. Tested with simulated data; not yet tested on a Garmin Venu SQ.
Menu: previous checks and health over time, plus a °F/°C setting.
How it works
Watch or sensors  →  Local server (server.py + safety rules)  →  MedGemma 4B (LM Studio)
                              ↑
                    Lia web page (lia.html)
The student describes symptoms, or a watch sends readings.
Lia asks follow-up questions.
The model suggests causes and urgency.
Safety rules check the urgency and flag odd readings.
The results panel and commander alert appear.
Results

49 test cases in three sets:

Set	Cases	Top-3 cause	Urgency	Under-rated
Practice (tuned on)	24	22/24	20/24	2
Hold-out (first run, before safety rules)	10	9/10	8/10	2
Fresh (run once, independent)	15	14/15 (13/15 strict)	10/15	3

The hold-out set ran after the safety rules were written, so it is not a fair test. The fresh set is the most trustworthy.

Flawed-data cases: odd readings were flagged for recheck in 15/15 cases. The check that urgency is at or above a reviewer's minimum is still pending.

What we found: chest pain and fainting may be under-rated, so a medical reviewer should decide those rules. All safety rules in server.py are marked DRAFT and need clinician review.

Getting started (Windows)
1. Install the model server
Install LM Studio.
Download MedGemma 4B (you must accept Google's terms on Hugging Face first).
Load the model and start LM Studio's local server (default http://127.0.0.1:1234/v1).

LM Studio stays a separate program. Lia connects to it by itself and shows "Model offline" in the corner until it can.

2. Install Lia
bash
python -m venv venv
venv\Scripts\activate
pip install fastapi uvicorn pydantic

Optional extras:

bash
pip install faster-whisper     # voice input
pip install kokoro-onnx        # natural voice
pip install opencv-python numpy  # face sign-in

There is no requirements.txt yet. The package list above comes from the imports in the code, so adjust it if something is missing.

3. Download the optional model files (once, while online)
bat
get_models.bat

This puts about 900 MB into a models/ folder: Kokoro voice (~350 MB), face detection and recognition (~40 MB), and Whisper small.en (~480 MB). Files already there are skipped.

4. Run
bat
start_lia.bat

or python lia_app.py. Lia starts a server on this computer and opens in its own window (Edge or Chrome app window).

Option	What it does
--demo	Canned answers, no AI model needed (same as LIA_BACKEND=fake)
--browser	Open in your normal browser instead of an app window
--port N	Use another port (default 8000)
Build an .exe
bat
build_exe.bat          :: dist\Lia.exe + dist\models (starts fast, recommended)
build_exe.bat single   :: models packed inside one big .exe (slow to start)

Build on Windows with the same Python you run Lia with. Hand in the whole dist folder.

Configuration
Variable	Default	Meaning
LIA_BACKEND	lmstudio	lmstudio, fake (canned replies) or direct (load MedGemma with transformers, for Kaggle or a big GPU)
LIA_URL	http://127.0.0.1:1234/v1	LM Studio server address
LIA_MODEL	auto-detect	Model name to use
LIA_PORT	8000	Server port
LIA_MODELS	./models	Folder with voice, face and speech models
LIA_VOICE	af_heart	Natural voice name
Project layout
lia.html             The web page (chat, results, menu)
lia_app.py           Starts the server and opens the window
server.py            FastAPI server, prompts, safety rules, voice, face sign-in
static/              Front-end helpers
logs/                Local check logs (lia_log.jsonl)
models/              Optional voice, face and speech models (not in git)
start_lia.bat        Run from source
get_models.bat       Download optional models
build_exe.bat        Build Lia.exe with PyInstaller
Lia.spec             PyInstaller spec

Main server routes: /health, /session, /reply, /results, /photo, /analyze, /transcribe, /speak, and /face/*.

Limitations
A 4B model makes mistakes and can sound confident while doing it.
The test set is small (49 cases) and uses made-up cases.
Safety rules are drafts and need review by a medical professional.
Sensor input has only been tested with simulated data.
Not approved for real medical use.
Next steps
Real sensor feeds, including the Garmin Venu SQ.
Test larger medical models and compare against MedGemma.
Clinician-written safety rules.
More independent test cases.
Suggested .gitignore
venv/
models/
logs/
lia_window/
dist/
build/
__pycache__/

lia_window/ is the app window's browser profile (cache, saved permissions) and should not be committed.

Team

Team Asclepius: Olivia Schmookler, Caiden Smith, Nahidur Choudhury Passaic County Technical Vocational School STEM Institute Advisor: Anjali Wahi, awahi@pcti.tec.nj.us

Acknowledgements
MedGemma by Google
LM Studio
Kokoro, faster-whisper, and OpenCV's YuNet and SFace models
NASA HUNCH
License

[Add a license here, e.g. MIT]
