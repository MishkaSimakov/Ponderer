"""Constants that live in more than one language and must agree.

Nothing here imports unity or the brick: the c# and the brick sources are read as
text, so the whole file runs on the host with no build and no hardware.
"""

import ast
import os
import re

import pytest

from conftest import ROOT, SCRIPTS, read
from bridge.connection import PROTOCOL_VERSION
from shared.observation import COLUMNS, DIM

BRIDGE = read("simulator", "Assets", "Scripts", "Bridge.cs")
CONTROLLER = read("simulator", "Assets", "Scripts", "Robot", "RobotController.cs")
SCENE = read("simulator", "Assets", "Scenes", "Simulation.unity")

# c# expression in RobotController.Observe -> column it writes.
UNITY_OBSERVATION = {
    "ElapsedSeconds": "t",
    "ultrasonic.DistanceCm": "distance",
    "leftColor.Reflected": "left_color",
    "rightColor.Reflected": "right_color",
    "leftMotor.GetDegrees()": "left_position",
    "rightMotor.GetDegrees()": "right_position",
}

# python expression in BrickRobot._observe -> column it returns.
BRICK_OBSERVATION = {
    "time.monotonic() - self.start": "t",
    "self.distance.distance_centimeters_continuous": "distance",
    "self.left_color.reflected_light_intensity": "left_color",
    "self.right_color.reflected_light_intensity": "right_color",
    "self.left_motor.position": "left_position",
    "self.right_motor.position": "right_position",
}

# The brick runs python 3.9 with numpy and ev3dev2, nothing else.
BRICK_PYTHON = (3, 9)
HOST_ONLY = {"torch", "stable_baselines3", "sb3_contrib", "gymnasium", "tensorboard",
             "rl", "bridge"}


def constant(source, name):
    """A numeric literal assigned to name, in c# or in python."""
    return float(re.search(r"\b%s\s*=\s*(-?[\d.]+)f?" % name, source).group(1))


def serialized_fields(source):
    return set(re.findall(r"\[SerializeField\]\s+\w+\s+(\w+)", source))


def scene_block(source, component):
    """The lines unity wrote for one MonoBehaviour."""
    block = source.split("m_EditorClassIdentifier: Assembly-CSharp::" + component)[1]
    return block.split("--- !u!")[0]


def scene_fields(source, component):
    """Field names in that block; m_ prefixed keys are unity's own."""
    keys = re.findall(r"^  (\w+):", scene_block(source, component), re.MULTILINE)
    return set(k for k in keys if not k.startswith("m_"))


def scene_value(source, component, field):
    """None when the scene omits the field, which leaves the script initializer."""
    found = re.search(r"^  %s: (-?[\d.]+)" % field, scene_block(source, component),
                      re.MULTILINE)
    return None if found is None else float(found.group(1))


def sources(*directories):
    for directory in directories:
        for base, _, files in os.walk(os.path.join(ROOT, directory)):
            for f in files:
                if f.endswith(".py"):
                    yield os.path.join(base, f)


def test_protocol_version_matches_unity():
    assert PROTOCOL_VERSION == int(constant(BRIDGE, "Version"))


def test_observation_dim_matches_unity():
    assert int(constant(CONTROLLER, "ObsDim")) == DIM


def test_action_dim_is_two_volts():
    from shared.runner import LOG_COLUMNS

    assert int(constant(CONTROLLER, "ActionDim")) == 2
    assert LOG_COLUMNS[:2] == ["volts_left", "volts_right"]
    assert LOG_COLUMNS[2:] == COLUMNS


def test_unity_writes_the_columns_in_order():
    body = CONTROLLER.split("public void Observe")[1]
    written = re.findall(r"destination\[offset \+ (\d+)\] = ([^;]+);", body)

    assert [int(i) for i, _ in written] == list(range(DIM))
    assert [UNITY_OBSERVATION[e.strip()] for _, e in written] == COLUMNS


def test_brick_returns_the_columns_in_order():
    tree = ast.parse(read("brick", "brick_robot.py"))
    observe = next(n for n in ast.walk(tree)
                   if isinstance(n, ast.FunctionDef) and n.name == "_observe")
    returned = next(n for n in observe.body if isinstance(n, ast.Return)).value

    assert [BRICK_OBSERVATION[ast.unparse(e)] for e in returned.elts] == COLUMNS


def test_control_period_matches_the_brick():
    """The value unity runs: the scene's, or the field initializer it leaves alone."""
    period = scene_value(SCENE, "Bridge", "controlPeriod")
    if period is None:
        period = constant(BRIDGE, "controlPeriod")

    frequency = constant(read("brick", "run.py"), "FREQUENCY")
    assert abs(period - 1.0 / frequency) < 1e-4


def test_scene_has_no_fields_the_bridge_script_dropped():
    assert scene_fields(SCENE, "Bridge") <= serialized_fields(BRIDGE)


@pytest.mark.parametrize("path", sorted(sources("shared", "brick")))
def test_brick_sources_parse_on_python_39(path):
    """feature_version only checks syntax; a 3.10 stdlib call still slips through."""
    with open(path) as f:
        ast.parse(f.read(), path, feature_version=BRICK_PYTHON)


@pytest.mark.parametrize("path", sorted(sources("shared", "brick")))
def test_brick_sources_import_nothing_the_brick_lacks(path):
    with open(path) as f:
        tree = ast.parse(f.read(), path)

    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(a.name.split(".")[0] for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0:
            imported.add(node.module.split(".")[0])

    assert not imported & HOST_ONLY
