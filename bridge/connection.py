import json
import socket
import time

PROTOCOL_VERSION = 2


class Connection:
    """Newline delimited JSON over TCP. One request, one response."""

    def __init__(self, host="127.0.0.1", port=5005, timeout=30.0, response_timeout=30.0):
        print("waiting for unity on %s:%d" % (host, port))
        start = time.monotonic()
        deadline = start + timeout
        while True:
            try:
                self.sock = socket.create_connection((host, port), 1.0)
                break
            except OSError:
                if time.monotonic() > deadline:
                    raise
                time.sleep(0.1)

        # create_connection's timeout applies to the attempt and then stays on the
        # socket: without this every recv would inherit the one second retry budget.
        self.sock.settimeout(response_timeout)
        self.sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        self.buffer = b""
        print("connected after %.1fs" % (time.monotonic() - start))

    def request(self, message):
        self.send(message)
        return self.recv()

    def send(self, message):
        self.sock.sendall(json.dumps(message).encode() + b"\n")

    def recv(self):
        while b"\n" not in self.buffer:
            chunk = self.sock.recv(1 << 16)
            if not chunk:
                raise ConnectionError("unity closed the connection")
            self.buffer += chunk

        line, self.buffer = self.buffer.split(b"\n", 1)
        return json.loads(line.decode())

    def close(self):
        self.sock.close()
        print("connection closed")
