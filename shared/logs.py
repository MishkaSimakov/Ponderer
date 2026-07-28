import os
import time

ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "logs")

SOURCES = ("brick", "sim")


def run_prefix(source, name):
    """Path prefix for one run's files: logs/<source>/<name>-<timestamp>.

    Independent of the working directory, and utc because the brick runs in utc
    while the host does not: names from both must stay comparable.
    """
    if source not in SOURCES:
        raise ValueError("unknown log source %r" % source)
    directory = os.path.join(ROOT, source)
    os.makedirs(directory, exist_ok=True)
    return os.path.join(directory, name + "-" + time.strftime("%Y%m%d-%H%M%S", time.gmtime()))


def latest(source, name):
    """Newest run of a given name, by timestamp in the file name."""
    directory = os.path.join(ROOT, source)
    runs = sorted(f for f in os.listdir(directory) if f.startswith(name + "-"))
    if not runs:
        raise IOError("no %s runs in %s" % (name, directory))
    return os.path.join(directory, runs[-1])
