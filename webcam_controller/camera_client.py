import cv2
import socket
import json
import time
import mediapipe as mp
import numpy as np

# ===============================
# Networking
# ===============================
SERVER_IP = "127.0.0.1"   # Linux later
PORT = 5005

sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
sock.connect((SERVER_IP, PORT))
print("Connected to robot receiver")

# ===============================
# MediaPipe
# ===============================
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7,
)
INDEX_TIP = mp_hands.HandLandmark.INDEX_FINGER_TIP

# ===============================
# Parameters
# ===============================
PIXEL_TO_MM = 0.3        # tune this
DEADZONE_PX = 3

prev_x = None
prev_y = None

cap = cv2.VideoCapture(0)

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        h, w, _ = frame.shape
        result = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        if result.multi_hand_landmarks and result.multi_handedness:
            for lm, handedness in zip(
                result.multi_hand_landmarks,
                result.multi_handedness
            ):
                if handedness.classification[0].label != "Right":
                    continue

                tip = lm.landmark[INDEX_TIP]
                x_px = tip.x * w
                y_px = tip.y * h

                if prev_x is not None:
                    dx_px = x_px - prev_x
                    dy_px = y_px - prev_y

                    if abs(dx_px) > DEADZONE_PX or abs(dy_px) > DEADZONE_PX:
                        # image → robot mapping
                        dy_mm =  PIXEL_TO_MM * dx_px
                        dz_mm = -PIXEL_TO_MM * dy_px

                        payload = {
                            "dy_mm": dy_mm,
                            "dz_mm": dz_mm
                        }
                        sock.sendall((json.dumps(payload) + "\n").encode())

                prev_x = x_px
                prev_y = y_px

                # visualization
                cv2.circle(frame, (int(x_px), int(y_px)), 8, (0,255,0), -1)
                break

        else:
            prev_x = None
            prev_y = None

        cv2.imshow("Finger Teleop", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    cap.release()
    hands.close()
    sock.close()
    cv2.destroyAllWindows()
