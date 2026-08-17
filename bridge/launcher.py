"""Unity player processes started from python, one per bridge port."""

import atexit
import os
import subprocess


def executable(path):
    """A macOS bundle is a directory; the player inside it is what runs."""
    if not path.endswith(".app"):
        return path
    macos = os.path.join(path, "Contents", "MacOS")
    return os.path.join(macos, os.listdir(macos)[0])


class Unity:
    def __init__(self, path, port, arenas, graphics=False, log=None, extra=()):
        command = [executable(path), "--port", str(port), "--arenas", str(arenas)]
        if not graphics:
            command += ["-batchmode", "-nographics"]
        if log is not None:
            os.makedirs(os.path.dirname(log), exist_ok=True)
            command += ["-logFile", log]
        command += list(extra)

        print("starting %s" % " ".join(command))
        # Own session: ctrl-c goes to the terminal's whole process group, and a
        # player killed mid-request leaves python reading a socket nobody answers.
        self.process = subprocess.Popen(command, start_new_session=True)
        atexit.register(self.kill)

    def stop(self, timeout=10.0):
        """Deadline for the quit command that was already sent over the socket."""
        try:
            self.process.wait(timeout)
        except subprocess.TimeoutExpired:
            print("unity did not quit in %.0fs, terminating" % timeout)
            self.process.terminate()
            try:
                self.process.wait(5.0)
            except subprocess.TimeoutExpired:
                self.kill()
        atexit.unregister(self.kill)

    def kill(self):
        if self.process.poll() is None:
            self.process.kill()
