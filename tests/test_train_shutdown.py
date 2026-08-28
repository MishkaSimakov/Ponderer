"""The weights are the point of a run, so every way one ends must save and export them.

cloud_train.sh ends a run with kill, and SIGTERM raises SystemExit, which is a
BaseException: it once unwound straight past the save. Each mode here is one branch
of the shutdown in train.py, driven in a subprocess because a signal needs one.
"""

import os
import signal
import subprocess
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DRIVER = os.path.join(ROOT, "tests", "shutdown_driver.py")
NAME = "run"
TIMEOUT = 300


def spawn(mode, tmp_path):
    """train.py in a subprocess, writing its run and its npz under tmp_path."""
    return subprocess.Popen(
        [sys.executable, DRIVER, mode, str(tmp_path / "tb"), str(tmp_path / "policy"), NAME],
        cwd=ROOT, env=dict(os.environ, PYTHONPATH=ROOT, PYTHONUNBUFFERED="1"),
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)


def weights(process, tmp_path):
    out = process.communicate(timeout=TIMEOUT)[0]
    assert process.returncode == 0, out
    assert (tmp_path / "policy" / (NAME + ".npz")).exists(), out
    assert (tmp_path / "tb" / (NAME + "_1") / "model.zip").exists(), out


@pytest.mark.parametrize("mode", ("steps", "keyboard", "oserror"))
def test_a_run_that_ends_on_its_own_saves_the_weights(mode, tmp_path):
    weights(spawn(mode, tmp_path), tmp_path)


def test_a_run_killed_with_sigterm_saves_the_weights(tmp_path):
    process = spawn("sigterm", tmp_path)

    for line in process.stdout:
        if line.strip() == "READY":
            break
    else:
        pytest.fail("the run never reached the step that asks to be killed")

    process.send_signal(signal.SIGTERM)
    weights(process, tmp_path)
