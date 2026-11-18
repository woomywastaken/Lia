@echo off
rem Downloads the model files Lia's extras need into the models folder. Run once, while online.
rem Files that are already there are skipped. About 900 MB in total.
cd /d "%~dp0"
if exist venv\Scripts\activate.bat call venv\Scripts\activate.bat
if not exist models mkdir models

echo.
echo [1/3] Natural voice (Kokoro, about 350 MB)
if not exist models\kokoro-v1.0.onnx curl.exe -L --fail -o models\kokoro-v1.0.onnx https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/kokoro-v1.0.onnx
if not exist models\voices-v1.0.bin curl.exe -L --fail -o models\voices-v1.0.bin https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/voices-v1.0.bin

echo.
echo [2/3] Face sign-in (OpenCV YuNet and SFace, about 40 MB)
if not exist models\face_detection_yunet_2023mar.onnx curl.exe -L --fail -o models\face_detection_yunet_2023mar.onnx https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx
if not exist models\face_recognition_sface_2021dec.onnx curl.exe -L --fail -o models\face_recognition_sface_2021dec.onnx https://github.com/opencv/opencv_zoo/raw/main/models/face_recognition_sface/face_recognition_sface_2021dec.onnx

echo.
echo [3/3] Speech recognition (Whisper small.en, about 480 MB)
if not exist models\faster-whisper-small.en\model.bin python -c "from faster_whisper.utils import download_model; download_model('small.en', output_dir=r'models\faster-whisper-small.en')"

echo.
echo Files now in the models folder:
dir /b models
echo.
echo If one is missing, check the internet connection and run this again.
pause
