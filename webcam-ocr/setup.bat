@echo off
echo === Installing Tesseract OCR ===
winget install --id UB-Mannheim.TesseractOCR -e --accept-source-agreements --accept-package-agreements
echo.
echo === Installing Python libraries ===
py -m pip install opencv-python pytesseract
echo.
echo === Looking for cameras ===
py "%~dp0webcam_ocr.py" --list
echo.
echo Setup done. Now run:  py webcam_ocr.py --camera 1   (use the number of your webcam)
