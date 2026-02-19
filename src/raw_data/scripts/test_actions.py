import json
import sys
from dataclasses import dataclass, field

import cv2
import numpy as np

MKV_PATH = "421c63b0-1036-4300-b0a3-0b0d81f8c68b.mkv"
HANDS_PATH = "421c63b0-1036-4300-b0a3-0b0d81f8c68b_hands.ndjson"
OUT_PATH = "421c63b0-1036-4300-b0a3-0b0d81f8c68b_actions.ndjson"

# CS:GO action vector layout (51 dims total)
# [0:11]  keys: w=0, a=1, s=2, d=3, space=4, ctrl=5, shift=6, 1=7, 2=8, 3=9, r=10
# [11]    left click
# [12]    right click
# [13:35] mouse_x one-hot (22 bins)
# [35:51] mouse_y one-hot (15 bins)
N_KEYS = 11
N_CLICKS = 2
N_MOUSE_X = 23
N_MOUSE_Y = 15
ACTION_DIM = N_KEYS + N_CLICKS + N_MOUSE_X + N_MOUSE_Y  # 51
MOUSE_X_ZERO_IDX = 11  # index of 0 in MOUSE_X_POSSIBLES
MOUSE_Y_ZERO_IDX = 7   # index of 0 in MOUSE_Y_POSSIBLES

# MediaPipe hand landmark indices
WRIST = 0
INDEX_MCP = 5
MIDDLE_MCP = 9
RING_MCP = 13
PINKY_MCP = 17
INDEX_TIP = 8
MIDDLE_TIP = 12
RING_TIP = 16
PINKY_TIP = 20

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
    (5, 9), (9, 13), (13, 17),
]

DIRECTION_THRESHOLD = 0.008
GRIP_THRESHOLD = 1.1
SMOOTHING = 0.4


@dataclass
class ActionState:
    prev_x: float | None = None
    prev_y: float | None = None
    smooth_dx: float = 0.0
    smooth_dy: float = 0.0
    pending_dx: float = 0.0
    pending_dy: float = 0.0
    left: bool = False
    right: bool = False
    up: bool = False
    down: bool = False
    grip: bool = False
    history: list = field(default_factory=list)


def load_hand_data(path: str) -> dict[int, list]:
    data: dict[int, list] = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            data[entry["f"]] = entry["h"]
    return data


def get_right_hand(hands: list) -> dict | None:
    for h in hands:
        if h.get("handedness") == "Right":
            return h
    return None


def detect_grip(landmarks: list) -> bool:
    """Closed hand when fingertips are close to wrist relative to hand size."""
    wrist = np.array(landmarks[WRIST][:2])
    palm_ref = np.array(landmarks[MIDDLE_MCP][:2])
    hand_size = np.linalg.norm(palm_ref - wrist)
    if hand_size < 1e-6:
        return False

    tips = [INDEX_TIP, MIDDLE_TIP, RING_TIP, PINKY_TIP]
    mcps = [INDEX_MCP, MIDDLE_MCP, RING_MCP, PINKY_MCP]

    tip_dists = [np.linalg.norm(np.array(landmarks[t][:2]) - wrist) for t in tips]
    mcp_dists = [np.linalg.norm(np.array(landmarks[m][:2]) - wrist) for m in mcps]

    ratios = [td / md if md > 1e-6 else 2.0 for td, md in zip(tip_dists, mcp_dists)]
    return np.mean(ratios) < GRIP_THRESHOLD


def update_actions(state: ActionState, landmarks: list | None) -> None:
    # Apply the *previous* frame's pending deltas as this frame's actions.
    # The delta pos[N] - pos[N-1] describes movement during frame N-1.
    state.left = state.pending_dx < -DIRECTION_THRESHOLD
    state.right = state.pending_dx > DIRECTION_THRESHOLD
    state.up = state.pending_dy < -DIRECTION_THRESHOLD
    state.down = state.pending_dy > DIRECTION_THRESHOLD

    if landmarks is None:
        state.prev_x = state.prev_y = None
        state.smooth_dx = state.smooth_dy = 0.0
        state.pending_dx = state.pending_dy = 0.0
        state.grip = False
        return

    state.grip = detect_grip(landmarks)

    x, y = landmarks[INDEX_MCP][0], landmarks[INDEX_MCP][1]
    if state.prev_x is not None:
        raw_dx = x - state.prev_x
        raw_dy = y - state.prev_y
        state.smooth_dx = SMOOTHING * raw_dx + (1 - SMOOTHING) * state.smooth_dx
        state.smooth_dy = SMOOTHING * raw_dy + (1 - SMOOTHING) * state.smooth_dy

    state.pending_dx = state.smooth_dx
    state.pending_dy = state.smooth_dy
    state.prev_x, state.prev_y = x, y


def encode_action(state: ActionState) -> list[float]:
    vec = [0.0] * ACTION_DIM
    if state.up:
        vec[0] = 1.0     # w
    if state.left:
        vec[1] = 1.0     # a
    if state.down:
        vec[2] = 1.0     # s
    if state.right:
        vec[3] = 1.0     # d
    if state.grip:
        vec[4] = 1.0     # space
    # mouse one-hots default to the zero bin
    vec[N_KEYS + N_CLICKS + MOUSE_X_ZERO_IDX] = 1.0
    vec[N_KEYS + N_CLICKS + N_MOUSE_X + MOUSE_Y_ZERO_IDX] = 1.0
    return vec


def draw_landmarks(frame: np.ndarray, hand: dict, w: int, h: int) -> None:
    lm = hand["landmarks"]
    pts = [(int(l[0] * w), int(l[1] * h)) for l in lm]
    color = (0, 128, 255)

    for a, b in HAND_CONNECTIONS:
        cv2.line(frame, pts[a], pts[b], color, 2, cv2.LINE_AA)
    for pt in pts:
        cv2.circle(frame, pt, 4, color, -1, cv2.LINE_AA)

    # Highlight tracking point
    cv2.circle(frame, pts[INDEX_MCP], 8, (0, 255, 255), 2, cv2.LINE_AA)


def draw_key(
    frame: np.ndarray, cx: int, cy: int, w: int, h: int, active: bool, label: str
) -> None:
    color_bg = (0, 200, 80) if active else (60, 60, 60)
    color_border = (100, 255, 130) if active else (140, 140, 140)
    color_text = (255, 255, 255) if active else (180, 180, 180)

    x1, y1 = cx - w // 2, cy - h // 2
    x2, y2 = cx + w // 2, cy + h // 2
    cv2.rectangle(frame, (x1, y1), (x2, y2), color_bg, -1)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color_border, 2, cv2.LINE_AA)

    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.5 if len(label) <= 2 else 0.4
    (tw, th), _ = cv2.getTextSize(label, font, scale, 1)
    cv2.putText(
        frame, label,
        (cx - tw // 2, cy + th // 2),
        font, scale, color_text, 1, cv2.LINE_AA,
    )


def draw_hud(frame: np.ndarray, state: ActionState) -> None:
    fh, fw = frame.shape[:2]
    ks = 44
    pad = 6
    margin = 20

    # Center of the arrow key cluster
    cluster_cx = fw - margin - ks - pad - ks // 2
    cluster_cy = fh - margin - ks - pad - ks // 2

    # Arrow keys:  [up] on top, [left][down][right] on bottom row
    draw_key(frame, cluster_cx, cluster_cy - ks - pad, ks, ks, state.up, "^")
    draw_key(frame, cluster_cx - ks - pad, cluster_cy, ks, ks, state.left, "<")
    draw_key(frame, cluster_cx, cluster_cy, ks, ks, state.down, "v")
    draw_key(frame, cluster_cx + ks + pad, cluster_cy, ks, ks, state.right, ">")

    # Space bar below arrows
    space_w = 3 * ks + 2 * pad
    space_cy = cluster_cy + ks + pad
    draw_key(frame, cluster_cx, space_cy, space_w, ks - 8, state.grip, "SPACE")

    # Tracking dot indicator
    if state.prev_x is not None:
        ix = int(state.prev_x * fw)
        iy = int(state.prev_y * fh)
        size = 10
        cv2.line(frame, (ix - size, iy), (ix + size, iy), (0, 255, 255), 2, cv2.LINE_AA)
        cv2.line(frame, (ix, iy - size), (ix, iy + size), (0, 255, 255), 2, cv2.LINE_AA)


def main() -> None:
    hands_by_frame = load_hand_data(HANDS_PATH)
    cap = cv2.VideoCapture(MKV_PATH)
    if not cap.isOpened():
        print(f"Cannot open {MKV_PATH}")
        sys.exit(1)

    state = ActionState()
    action_records: list[dict] = []
    paused = False
    frame_idx = 0

    while True:
        if not paused:
            ret, frame = cap.read()
            if not ret:
                break

            h, w = frame.shape[:2]
            hands = hands_by_frame.get(frame_idx, [])
            right = get_right_hand(hands)

            if right:
                draw_landmarks(frame, right, w, h)
                update_actions(state, right["landmarks"])
            else:
                update_actions(state, None)

            action_records.append({
                "f": frame_idx,
                "action": encode_action(state),
            })

            draw_hud(frame, state)

            cv2.putText(
                frame, f"Frame {frame_idx}",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                (255, 255, 255), 2, cv2.LINE_AA,
            )

            display = frame
            frame_idx += 1

        cv2.imshow("Actions from Hand Tracking", display)

        key = cv2.waitKey(40) & 0xFF
        if key == ord("q") or key == 27:
            break
        elif key == ord(" "):
            paused = not paused

    cap.release()
    cv2.destroyAllWindows()

    with open(OUT_PATH, "w") as f:
        for rec in action_records:
            f.write(json.dumps(rec) + "\n")
    print(f"Wrote {len(action_records)} frames to {OUT_PATH}")


if __name__ == "__main__":
    main()
