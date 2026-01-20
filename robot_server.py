import socket
import json

HOST = "127.0.0.1"   # use "0.0.0.0" on Linux later
PORT = 5005

sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
sock.bind((HOST, PORT))
sock.listen(1)

print(f"Listening on {HOST}:{PORT}...")
conn, addr = sock.accept()
print(f"Connected by {addr}")

buffer = ""

try:
    while True:
        data = conn.recv(1024).decode("utf-8")
        if not data:
            break

        buffer += data

        while "\n" in buffer:
            msg, buffer = buffer.split("\n", 1)
            payload = json.loads(msg)

            print(
                f"t={payload['timestamp']:.3f} | "
                f"x_px={payload['x_px']:.1f}, "
                f"y_px={payload['y_px']:.1f} | "
                f"x_norm={payload['x_norm']:.3f}, "
                f"y_norm={payload['y_norm']:.3f}"
            )

except KeyboardInterrupt:
    pass
finally:
    conn.close()
    sock.close()
