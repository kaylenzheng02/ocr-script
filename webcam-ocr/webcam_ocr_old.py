"""
webcam_ocr.py - read text from a webcam and print it to the terminal.

Usage:
    python webcam_ocr.py --list          # find which camera index is your external webcam
    python webcam_ocr.py --camera 1      # open camera 1 (external webcams are often 1 on laptops)

In the preview window:
    SPACE  -> capture the current frame, run OCR, print the text
    a      -> toggle auto mode (OCR roughly every 2 seconds)
    q      -> quit
"""
import argparse
import time

import cv2
import pytesseract

# Windows only: if Tesseract isn't on your PATH, point to it directly, e.g.
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"


def list_cameras(max_index=5):
    print("Probing camera indexes...")
    for i in range(max_index):
        cap = cv2.VideoCapture(i)
        ok, _ = cap.read()
        print(f"  camera {i}: {'available' if ok else 'not found'}")
        cap.release()


def preprocess(frame):
    """Grayscale + threshold: makes printed text much easier for Tesseract."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_CUBIC)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return thresh


def run_ocr(frame):
    text = pytesseract.image_to_string(preprocess(frame), config="--psm 6")
    text = text.strip()
    print("\n----- OCR result -----")
    print(text if text else "(no text found)")
    print("----------------------")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--camera", type=int, default=0, help="camera index")
    parser.add_argument("--list", action="store_true", help="list available cameras")
    args = parser.parse_args()

    if args.list:
        list_cameras()
        return

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise SystemExit(f"Could not open camera {args.camera}. Try --list.")

    # Ask for a higher resolution; better resolution = better OCR
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    print("SPACE = read text | a = toggle auto mode | q = quit")
    auto, last = False, 0.0

    while True:
        ok, frame = cap.read()
        if not ok:
            print("Failed to read frame.")
            break

        cv2.imshow("Webcam OCR (SPACE to read, q to quit)", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break
        elif key == ord(" "):
            run_ocr(frame)
        elif key == ord("a"):
            auto = not auto
            print(f"Auto mode {'ON' if auto else 'OFF'}")

        if auto and time.time() - last > 2:
            run_ocr(frame)
            last = time.time()

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
