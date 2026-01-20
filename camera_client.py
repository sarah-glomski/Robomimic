import cv2
import socket
import json
import time
import mediapipe as mp

# -----------------------------
# Networking
# -----------------------------
SERVER_IP = "127.0.0.1"  # change to Linux IP later
PORT = 5005

sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
sock.connect((SERVER_IP, PORT))
print("Connected to receiver")

# -----------------------------
# MediaPipe setup
# -----------------------------
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

INDEX_TIP_ID = mp_hands.HandLandmark.INDEX_FINGER_TIP

# -----------------------------
# Webcam
# -----------------------------
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("Could not open webcam")

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(frame_rgb)

        h, w, _ = frame.shape

        if result.multi_hand_landmarks and result.multi_handedness:
            for hand_landmarks, handedness in zip(
                result.multi_hand_landmarks,
                result.multi_handedness
            ):
                label = handedness.classification[0].label

                # Only track RIGHT hand (flipped on webcam)
                if label != "Left":
                    continue

                tip = hand_landmarks.landmark[INDEX_TIP_ID]

                x_norm = tip.x
                y_norm = tip.y

                x_px = x_norm * w
                y_px = y_norm * h

                payload = {
                    "timestamp": time.time(),
                    "x_px": x_px,
                    "y_px": y_px,
                    "x_norm": x_norm,
                    "y_norm": y_norm
                }

                sock.sendall((json.dumps(payload) + "\n").encode("utf-8"))

                # Visualization
                cv2.circle(
                    frame,
                    (int(x_px), int(y_px)),
                    10,
                    (0, 255, 0),
                    -1
                )

                mp_draw.draw_landmarks(
                    frame,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS
                )

                break  # only send one hand per frame

        cv2.imshow("Right Index Finger Tracking", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

except KeyboardInterrupt:
    pass

finally:
    cap.release()
    hands.close()
    sock.close()
    cv2.destroyAllWindows()
