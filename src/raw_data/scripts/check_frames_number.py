import json
import sys

import cv2
import numpy as np

MKV_PATH = "a754c773-f455-47c3-8e23-a0d733c4ccdd.mkv"
HANDS_PATH = "a754c773-f455-47c3-8e23-a0d733c4ccdd_hands.ndjson"

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


def load_hand_data(path: str) -> dict[int, list]:
    hands_by_frame: dict[int, list] = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            hands_by_frame[entry["f"]] = entry["h"]
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
    hands_by_frame = load_hand_data(HANDS_PATH)

    cap = cv2.VideoCapture(MKV_PATH)
    if not cap.isOpened():
        print(f"Cannot open {MKV_PATH}")
        sys.exit(1)

    paused = False
    frame_idx = 0

    while True:
        if not paused:
            ret, frame = cap.read()
            if not ret:
                break

            h, w = frame.shape[:2]
            hands = hands_by_frame.get(frame_idx, [])
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
