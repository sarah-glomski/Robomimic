#!/usr/bin/env python3

import os
import sys
import time
import socket
import json

sys.path.append(os.path.join(os.path.dirname(__file__), '../../..'))
from xarm.wrapper import XArmAPI

# ===============================
# Networking
# ===============================
HOST = "0.0.0.0"
PORT = 5005

sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
sock.bind((HOST, PORT))
sock.listen(1)
print("Waiting for camera sender...")
conn, _ = sock.accept()

buffer = ""

# ===============================
# Robot setup
# ===============================
# ip = sys.argv[1] if len(sys.argv) > 1 else input("xArm IP: ")
ip = '192.168.1.153'

arm = XArmAPI(ip)
arm.motion_enable(True)
arm.set_mode(0)
arm.set_state(0)
arm.move_gohome(wait=True)

x = 200
y = 0
z = 150

arm.set_position(
    x=x, y=y, z=z,
    roll=-180, pitch=0, yaw=0,
    speed=100, wait=True
)

arm.set_mode(7)
arm.set_state(0)
time.sleep(0.5)

# ===============================
# Limits
# ===============================
Y_MIN, Y_MAX = -150, 150
Z_MIN, Z_MAX = 150, 450

print("Delta teleop running")

try:
    while True:
        data = conn.recv(1024).decode()
        if not data:
            break

        buffer += data
        while "\n" in buffer:
            msg, buffer = buffer.split("\n", 1)
            payload = json.loads(msg)

            dy = payload["dy_mm"]
            dz = payload["dz_mm"]

            y += dy
            z += dz

            y = max(Y_MIN, min(Y_MAX, y))
            z = max(Z_MIN, min(Z_MAX, z))

            arm.set_position(
                x=x,
                y=y,
                z=z,
                roll=-180,
                pitch=0,
                yaw=0,
                speed=200,
                wait=False
            )

except KeyboardInterrupt:
    pass

finally:
    arm.set_mode(0)
    arm.set_state(0)
    arm.move_gohome(wait=True)
    arm.disconnect()
    conn.close()
    sock.close()
