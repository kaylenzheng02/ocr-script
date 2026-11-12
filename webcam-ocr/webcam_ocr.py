"""Read text from a webcam and print it to the terminal.

Usage:
    py webcam_ocr.py --list          find which camera number is your webcam
    py webcam_ocr.py --camera 1      start reading

Windows:
    Camera          live feed with a guide box (green = sharp, red = blurry)
    What OCR sees   the cleaned-up image Tesseract actually reads

Keys (click the Camera window first):
    SPACE   read the text in the box now
    a       auto mode on/off (prints only when the text changes)
    p       cycle layout mode
    t       cycle cleanup mode
    [ ]     shrink / grow the guide box
    q       quit
"""

from __future__ import annotations

import argparse
import os
import time
from dataclasses import dataclass

import cv2
import numpy as np
import pytesseract

# --------------------------------------------------------------------------- #
# Settings
# --------------------------------------------------------------------------- #

WINDOWS_TESSERACT = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

LAYOUT_MODES = [  # (Tesseract page segmentation mode, label)
    (3, "auto layout"),
    (6, "block of text"),
    (4, "single column"),
    (11, "scattered words"),
]
CLEANUP_MODES = ["adaptive", "otsu", "gray"]
BULLET_LOOKALIKES = {"o", "O", "0", "e", "*", "°", "•", "»", "-", "—", "+"}

MIN_CONFIDENCE = 55  # drop words Tesseract is less sure about (0-100)
MIN_SHARPNESS = 60  # below this a frame counts as blurry
MIN_SPECK_AREA = 40  # dark blobs smaller than this (px) are treated as noise
AUTO_INTERVAL = 1.5  # seconds between reads in auto mode

BOX_SCALE_DEFAULT, BOX_SCALE_MIN, BOX_SCALE_MAX, BOX_SCALE_STEP = 0.6, 0.2, 0.95, 0.05

GREEN, RED, WHITE, BLACK = (0, 200, 0), (0, 0, 255), (255, 255, 255), (0, 0, 0)

if os.path.exists(WINDOWS_TESSERACT):
    pytesseract.pytesseract.tesseract_cmd = WINDOWS_TESSERACT


# --------------------------------------------------------------------------- #
# Camera
# --------------------------------------------------------------------------- #


def open_camera(index: int) -> cv2.VideoCapture:
    """Open a camera; DirectShow starts much faster on Windows."""
    if os.name == "nt":
        return cv2.VideoCapture(index, cv2.CAP_DSHOW)
    return cv2.VideoCapture(index)


def list_cameras(max_index: int = 5) -> None:
    print("Looking for cameras...")
    for i in range(max_index):
        cap = open_camera(i)
        ok, _ = cap.read()
        cap.release()
        print(f"  camera {i}: {'available' if ok else 'not found'}")


# --------------------------------------------------------------------------- #
# Image processing
# --------------------------------------------------------------------------- #


def guide_box(frame: np.ndarray, scale: float) -> tuple[int, int, int, int]:
    """Return (x, y, width, height) of a centered box covering `scale` of the frame."""
    h, w = frame.shape[:2]
    bw, bh = int(w * scale), int(h * scale * 0.75)
    return (w - bw) // 2, (h - bh) // 2, bw, bh


def sharpness(gray: np.ndarray) -> float:
    """Variance of the Laplacian: higher means sharper edges."""
    return cv2.Laplacian(gray, cv2.CV_64F).var()


def remove_specks(binary: np.ndarray) -> np.ndarray:
    """Erase tiny dark dots that Tesseract would read as punctuation."""
    _, labels, stats, _ = cv2.connectedComponentsWithStats(cv2.bitwise_not(binary))
    is_speck = stats[:, cv2.CC_STAT_AREA] < MIN_SPECK_AREA
    is_speck[0] = False  # label 0 is the background
    cleaned = binary.copy()
    cleaned[is_speck[labels]] = 255
    return cleaned


def preprocess(crop: np.ndarray, mode: str) -> np.ndarray:
    """Turn a color crop into a clean, high-contrast image for Tesseract."""
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    gray = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)

    if mode == "gray":
        out = gray
    else:
        if mode == "otsu":
            _, out = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        else:  # adaptive: thresholds each region separately, so shadows don't matter
            out = cv2.adaptiveThreshold(
                gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 41, 20
            )
        if out.mean() < 127:  # light text on dark background -> flip it
            out = cv2.bitwise_not(out)
        out = remove_specks(out)

    # White margin so letters touching the edge aren't clipped
    return cv2.copyMakeBorder(out, 20, 20, 20, 20, cv2.BORDER_CONSTANT, value=255)


# --------------------------------------------------------------------------- #
# OCR
# --------------------------------------------------------------------------- #


def read_text(image: np.ndarray, psm: int) -> str:
    """Run Tesseract, keep confident real words, and rebuild the lines."""
    data = pytesseract.image_to_data(
        image, config=f"--oem 1 --psm {psm}", output_type=pytesseract.Output.DICT
    )

    lines: dict[tuple[int, int, int], list[str]] = {}
    for word, conf, block, par, line in zip(
        data["text"],
        data["conf"],
        data["block_num"],
        data["par_num"],
        data["line_num"],
        strict=True,
    ):
        word = word.strip()
        if not word or float(conf) < MIN_CONFIDENCE:
            continue
        if not any(ch.isalnum() for ch in word):  # lone punctuation like | ~ ;
            continue
        lines.setdefault((block, par, line), []).append(word)

    result = []
    for _, words in sorted(lines.items()):
        if len(words) > 1 and words[0] in BULLET_LOOKALIKES:
            words = words[1:]
        result.append(" ".join(words))
    return "\n".join(result)


def print_result(text: str) -> None:
    print("\n----- OCR result -----")
    print(text or "(no confident text found - move closer, hold still, add light)")
    print("----------------------")


# --------------------------------------------------------------------------- #
# App
# --------------------------------------------------------------------------- #


@dataclass
class State:
    layout: int = 0
    cleanup: int = 0
    box_scale: float = BOX_SCALE_DEFAULT
    auto: bool = False
    last_auto_read: float = 0.0
    last_text: str | None = None

    @property
    def psm(self) -> int:
        return LAYOUT_MODES[self.layout][0]

    @property
    def cleanup_mode(self) -> str:
        return CLEANUP_MODES[self.cleanup]


def draw_overlay(frame, box, is_sharp: bool, state: State) -> np.ndarray:
    x, y, bw, bh = box
    view = frame.copy()
    cv2.rectangle(view, (x, y), (x + bw, y + bh), GREEN if is_sharp else RED, 3)

    status = " | ".join(
        [
            f"layout: {LAYOUT_MODES[state.layout][1]}",
            f"cleanup: {state.cleanup_mode}",
            f"auto: {'ON' if state.auto else 'off'}",
            f"focus: {'OK' if is_sharp else 'BLURRY'}",
        ]
    )
    for color, thickness in ((BLACK, 4), (WHITE, 2)):  # outlined text
        cv2.putText(view, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, thickness)
    return view


def handle_key(key: int, state: State, processed: np.ndarray, is_sharp: bool) -> bool:
    """Apply a keypress. Returns False when the user wants to quit."""
    if key == ord("q"):
        return False
    if key == ord(" "):
        if not is_sharp:
            print("(image looks blurry - reading anyway)")
        print_result(read_text(processed, state.psm))
    elif key == ord("a"):
        state.auto, state.last_text = not state.auto, None
        print(f"Auto mode {'ON' if state.auto else 'OFF'}")
    elif key == ord("p"):
        state.layout = (state.layout + 1) % len(LAYOUT_MODES)
        print(f"Layout mode: {LAYOUT_MODES[state.layout][1]}")
    elif key == ord("t"):
        state.cleanup = (state.cleanup + 1) % len(CLEANUP_MODES)
        print(f"Cleanup mode: {state.cleanup_mode}")
    elif key == ord("["):
        state.box_scale = max(BOX_SCALE_MIN, state.box_scale - BOX_SCALE_STEP)
    elif key == ord("]"):
        state.box_scale = min(BOX_SCALE_MAX, state.box_scale + BOX_SCALE_STEP)
    return True


def auto_read(state: State, processed: np.ndarray, is_sharp: bool) -> None:
    """In auto mode, read sharp frames periodically and print only new text."""
    now = time.time()
    if not (state.auto and is_sharp and now - state.last_auto_read > AUTO_INTERVAL):
        return
    state.last_auto_read = now
    text = read_text(processed, state.psm)
    if text and text != state.last_text:
        print_result(text)
        state.last_text = text


def run(camera_index: int) -> None:
    cap = open_camera(camera_index)
    if not cap.isOpened():
        raise SystemExit(f"Could not open camera {camera_index}. Try --list.")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    cap.set(cv2.CAP_PROP_AUTOFOCUS, 1)

    print("Keys" + __doc__.split("Keys", 1)[1])
    state = State()

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("Failed to read from camera.")
                break

            box = guide_box(frame, state.box_scale)
            x, y, bw, bh = box
            crop = frame[y : y + bh, x : x + bw]
            is_sharp = sharpness(cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)) >= MIN_SHARPNESS
            processed = preprocess(crop, state.cleanup_mode)

            cv2.imshow("Camera", draw_overlay(frame, box, is_sharp, state))
            cv2.imshow("What OCR sees", processed)

            key = cv2.waitKey(1) & 0xFF
            if not handle_key(key, state, processed, is_sharp):
                break
            auto_read(state, processed, is_sharp)
    finally:
        cap.release()
        cv2.destroyAllWindows()


def main() -> None:
    parser = argparse.ArgumentParser(description="Read text from a webcam.")
    parser.add_argument("--camera", type=int, default=0, help="camera number (see --list)")
    parser.add_argument("--list", action="store_true", help="list available cameras")
    args = parser.parse_args()

    if args.list:
        list_cameras()
    else:
        run(args.camera)


if __name__ == "__main__":
    main()
