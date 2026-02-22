import json
import sys

import av
import cv2
import numpy as np

MKV_PATH = "a754c773-f455-47c3-8e23-a0d733c4ccdd.mkv"

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
    (5, 9), (9, 13), (13, 17),
]

HANDEDNESS_COLORS = {
    "Left": (0, 255, 0),
    "Right": (0, 128, 255),
}


def extract_hand_landmarks(mkv_path: str) -> dict[int, list]:
    """Map subtitle packets to video frame indices using MKV timestamps."""
    def _extract_hands(entry: dict) -> list:
        if isinstance(entry.get("h"), list):
            return entry["h"]
        if isinstance(entry.get("hands"), list):
            return entry["hands"]
        payload = entry.get("payload")
        if isinstance(payload, dict):
            if isinstance(payload.get("h"), list):
                return payload["h"]
            if isinstance(payload.get("hands"), list):
                return payload["hands"]
        return []

    def _select_hand_stream_index(path: str) -> int | None:
        probe = av.open(path)
        subtitle_indices = [i for i, s in enumerate(probe.streams) if s.type == "subtitle"]
        probe.close()
        for idx in subtitle_indices:
            c = av.open(path)
            stream = c.streams[idx]
            found = False
            checked = 0
            for pkt in c.demux(stream):
                if pkt.size == 0:
                    continue
                raw = bytes(pkt).decode("utf-8", errors="replace").strip()
                if not raw:
                    continue
                try:
                    entry = json.loads(raw)
                except json.JSONDecodeError:
                    continue
                checked += 1
                if _extract_hands(entry):
                    found = True
                    break
                if checked >= 50:
                    break
            c.close()
            if found:
                return idx
        return None

    stream_idx = _select_hand_stream_index(mkv_path)
    if stream_idx is None:
        return {}

    container = av.open(mkv_path)
    video_stream = container.streams.video[0]
    subtitle_stream = container.streams[stream_idx]
    fps = float(video_stream.average_rate) if video_stream.average_rate else 25.0
    video_start_time = (
        float(video_stream.start_time * video_stream.time_base)
        if video_stream.start_time is not None
        else 0.0
    )
    hands_by_frame: dict[int, list] = {}

    for pkt in container.demux(subtitle_stream):
        if pkt.size == 0 or pkt.pts is None:
            continue
        raw = bytes(pkt).decode("utf-8").strip()
        if not raw:
            continue
        entry = json.loads(raw)
        pkt_time = float(pkt.pts * subtitle_stream.time_base)
        frame_idx = int(round((pkt_time - video_start_time) * fps))
        if frame_idx < 0:
            continue
        hands_by_frame[frame_idx] = _extract_hands(entry)
    container.close()
    return hands_by_frame


def draw_hand(frame: np.ndarray, hand: dict, w: int, h: int) -> None:
    landmarks = hand["landmarks"]
    color = HANDEDNESS_COLORS.get(hand.get("handedness", ""), (255, 255, 255))

    pts = [(int(lm[0] * w), int(lm[1] * h)) for lm in landmarks]

    for a, b in HAND_CONNECTIONS:
        cv2.line(frame, pts[a], pts[b], color, 2, cv2.LINE_AA)

    for pt in pts:
        cv2.circle(frame, pt, 4, color, -1, cv2.LINE_AA)

    label = hand.get("handedness", "")
    if label:
        score = hand.get("score", 0)
        cv2.putText(
            frame,
            f"{label} {score:.2f}",
            (pts[0][0] + 8, pts[0][1] - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            1,
            cv2.LINE_AA,
        )


def main() -> None:
    hands_by_frame = extract_hand_landmarks(MKV_PATH)

    cap = cv2.VideoCapture(MKV_PATH)
    if not cap.isOpened():
        print(f"Cannot open {MKV_PATH}")
        sys.exit(1)

    paused = False
    frame_idx = 0
    latest_hands: list = []

    while True:
        if not paused:
            ret, frame = cap.read()
            if not ret:
                break

            h, w = frame.shape[:2]
            if frame_idx in hands_by_frame:
                latest_hands = hands_by_frame[frame_idx]
            hands = latest_hands
            for hand in hands:
                draw_hand(frame, hand, w, h)

            cv2.putText(
                frame,
                f"Frame {frame_idx}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

            display = frame
            frame_idx += 1

        cv2.imshow("RGB + Hand Landmarks", display)

        key = cv2.waitKey(40) & 0xFF  # ~25 fps
        if key == ord("q") or key == 27:
            break
        elif key == ord(" "):
            paused = not paused

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
