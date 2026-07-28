import json
import socket
import time

PROTOCOL_VERSION = 1


class Connection:
    """Newline delimited JSON over TCP. One request, one response."""

    def __init__(self, host="127.0.0.1", port=5005, timeout=30.0):
        deadline = time.monotonic() + timeout
        while True:
            try:
                self.sock = socket.create_connection((host, port), 1.0)
                break
            except OSError:
                if time.monotonic() > deadline:
                    raise
                time.sleep(0.1)

        self.sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        self.buffer = b""

    def request(self, message):
        self.sock.sendall(json.dumps(message).encode() + b"\n")

        while b"\n" not in self.buffer:
            chunk = self.sock.recv(1 << 16)
            if not chunk:
                raise ConnectionError("unity closed the connection")
            self.buffer += chunk

        line, self.buffer = self.buffer.split(b"\n", 1)
        return json.loads(line.decode())

    def send(self, message):
        self.sock.sendall(json.dumps(message).encode() + b"\n")

    def close(self):
        self.sock.close()
