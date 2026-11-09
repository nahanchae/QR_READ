"""
Real-time Data Matrix scanner: captures the screen (or a chosen region) and
decodes any Data Matrix barcode(s) it finds, live.

Usage:
    python datamatrix_scanner.py            # select monitor + region interactively
    python datamatrix_scanner.py --full     # scan the whole primary monitor, no region select
    python datamatrix_scanner.py --monitor 2  # pick a specific monitor (1 = primary)

Controls while running:
    q       quit
    r       re-select the capture region
    s       save the current preview frame as a PNG

Decoded values are printed to the console as soon as they change, and shown
as an overlay on the preview window.
"""

import argparse
import time
from datetime import datetime

import cv2
import numpy as np
import mss
import zxingcpp


def grab_monitor_list(sct: mss.mss):
    # monitors[0] is the "all monitors combined" pseudo-monitor; skip it for selection.
    return sct.monitors[1:]


def select_monitor(sct: mss.mss, requested: int | None):
    monitors = grab_monitor_list(sct)
    if requested is not None:
        idx = requested - 1
        if idx < 0 or idx >= len(monitors):
            raise SystemExit(f"--monitor {requested} is out of range (1..{len(monitors)})")
        return monitors[idx]
    if len(monitors) == 1:
        return monitors[0]
    print("Detected monitors:")
    for i, m in enumerate(monitors, start=1):
        print(f"  {i}: {m['width']}x{m['height']} at ({m['left']},{m['top']})")
    while True:
        raw = input(f"Select monitor [1-{len(monitors)}] (default 1): ").strip()
        if raw == "":
            return monitors[0]
        if raw.isdigit() and 1 <= int(raw) <= len(monitors):
            return monitors[int(raw) - 1]
        print("Invalid choice, try again.")


def select_region(sct: mss.mss, monitor: dict) -> dict:
    """Show a snapshot of the monitor and let the user drag a box to scan."""
    raw = sct.grab(monitor)
    frame = np.ascontiguousarray(np.array(raw)[:, :, :3])  # BGRA -> BGR

    # Downscale huge screens so the selection window fits on screen.
    h, w = frame.shape[:2]
    max_dim = 1400
    scale = min(1.0, max_dim / max(h, w))
    preview = cv2.resize(frame, (int(w * scale), int(h * scale))) if scale < 1.0 else frame

    print("Drag a box around the area to scan, then press ENTER or SPACE.")
    print("Press 'c' to cancel and scan the full monitor instead.")
    box = cv2.selectROI("Select scan region - ENTER to confirm", preview, showCrosshair=True)
    cv2.destroyWindow("Select scan region - ENTER to confirm")
    cv2.waitKey(1)  # Windows: pump events so the window actually closes

    x, y, bw, bh = box
    if bw == 0 or bh == 0:
        return monitor  # user cancelled -> full monitor

    x, y, bw, bh = (int(v / scale) for v in (x, y, bw, bh))
    return {
        "left": monitor["left"] + x,
        "top": monitor["top"] + y,
        "width": bw,
        "height": bh,
    }


def format_position(pos) -> str:
    pts = [pos.top_left, pos.top_right, pos.bottom_right, pos.bottom_left]
    return " ".join(f"({p.x},{p.y})" for p in pts)


def main():
    parser = argparse.ArgumentParser(description="Real-time Data Matrix screen scanner")
    parser.add_argument("--full", action="store_true", help="scan the whole monitor, skip region selection")
    parser.add_argument("--monitor", type=int, default=None, help="monitor number to use (1 = primary)")
    parser.add_argument("--fps", type=float, default=10.0, help="capture rate, frames per second (default 10)")
    args = parser.parse_args()

    interval = 1.0 / max(args.fps, 0.1)

    with mss.mss() as sct:
        monitor = select_monitor(sct, args.monitor)
        region = monitor if args.full else select_region(sct, monitor)

        print(f"Scanning region: {region}")
        print("Press 'q' in the preview window to quit, 'r' to reselect region, 's' to save a frame.")

        last_values: set[str] = set()
        window = "Data Matrix Scanner (q=quit, r=reselect, s=save)"
        cv2.namedWindow(window, cv2.WINDOW_NORMAL)

        try:
            while True:
                if cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1:
                    break

                t0 = time.time()
                raw = sct.grab(region)
                frame = np.ascontiguousarray(np.array(raw)[:, :, :3])  # BGRA -> BGR

                results = zxingcpp.read_barcodes(frame, formats=zxingcpp.BarcodeFormat.DataMatrix)

                current_values = set()
                for r in results:
                    if not r.valid:
                        continue
                    current_values.add(r.text)
                    pos = r.position
                    pts = np.array(
                        [[p.x, p.y] for p in (pos.top_left, pos.top_right, pos.bottom_right, pos.bottom_left)],
                        dtype=np.int32,
                    )
                    cv2.polylines(frame, [pts], isClosed=True, color=(0, 255, 0), thickness=2)
                    label_pos = (pos.top_left.x, max(pos.top_left.y - 10, 15))
                    cv2.putText(
                        frame, r.text, label_pos,
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1, cv2.LINE_AA,
                    )

                new_values = current_values - last_values
                for v in new_values:
                    ts = datetime.now().strftime("%H:%M:%S")
                    print(f"[{ts}] Data Matrix decoded: {v}")
                last_values = current_values

                cv2.imshow(window, frame)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):  # q or ESC
                    break
                if cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1:
                    break  # closed with the X button
                elif key == ord("r"):
                    cv2.destroyWindow(window)
                    cv2.waitKey(1)
                    region = select_region(sct, monitor)
                    print(f"New scan region: {region}")
                    last_values = set()
                    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
                elif key == ord("s"):
                    fname = f"scan_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
                    cv2.imwrite(fname, frame)
                    print(f"Saved {fname}")

                elapsed = time.time() - t0
                if elapsed < interval:
                    time.sleep(interval - elapsed)
        except KeyboardInterrupt:
            pass  # Ctrl+C in the console
        finally:
            cv2.destroyAllWindows()
            for _ in range(5):  # Windows: let the close events be processed
                cv2.waitKey(1)


if __name__ == "__main__":
    main()
