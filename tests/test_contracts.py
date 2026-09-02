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
PROTOCOL = read("simulator", "Assets", "Scripts", "Protocol.cs")
SCENE = read("simulator", "Assets", "Scenes", "Simulation.unity")
ARENA_CS = read("simulator", "Assets", "Scripts", "Arena.cs")
ARENA_PREFAB = read("simulator", "Assets", "Prefabs", "Arena.prefab")
REWARDS = ("simulator", "Assets", "Scripts", "Rewards")

# c# expression in RobotController.Observe -> column it writes.
UNITY_OBSERVATION = {
    "ElapsedSeconds": "t",
    "leftColor.Reflected": "left_color",
    "rightColor.Reflected": "right_color",
    "leftMotor.GetDegrees()": "left_position",
    "rightMotor.GetDegrees()": "right_position",
}

# python expression in BrickRobot._observe -> column it returns. The reads are sysfs
# by hand, so the names here are the descriptors the robot opens, not ev3dev2's.
BRICK_OBSERVATION = {
    "time.monotonic() - self.start": "t",
    "int(os.pread(self.left_light, READ_SIZE, 0))": "left_color",
    "int(os.pread(self.right_light, READ_SIZE, 0))": "right_color",
    "int(os.pread(self.left_tacho, READ_SIZE, 0))": "left_position",
    "int(os.pread(self.right_tacho, READ_SIZE, 0))": "right_position",
}

# ev3dev ships python 3.5.3 and numpy 1.12; there is no other interpreter on the
# brick. sync.sh copies experiments/ there too, so those scripts obey the same rule.
BRICK_PYTHON = (3, 5)
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


def test_the_brick_is_not_pinned_to_a_control_rate():
    """How long a step takes is not a number the brick is set to: it is whatever one
    pass of the loop costs, reported by the observation clock. The constant that used
    to have to agree with unity must not come back."""
    assert re.search(r"\bFREQUENCY\b", read("brick", "run.py")) is None
    assert "sleep" not in read("brick", "brick_robot.py")
    assert "sleep" not in read("shared", "runner.py")


def test_unity_is_not_pinned_to_a_control_rate():
    """The step length arrives with every step request; there is no period to keep."""
    assert "controlPeriod" not in BRIDGE


def request_fields(source):
    """Field names unity will deserialize a request into."""
    block = source.split("public class Request")[1].split("}")[0]
    return set(re.findall(r"^\s+public [\w\[\]]+ (\w+);", block, re.MULTILINE))


def sent_keys(source):
    """String keys of every dict literal python hands to the connection."""
    keys = set()
    for node in ast.walk(ast.parse(source)):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
            continue
        if node.func.attr not in ("request", "send"):
            continue
        for argument in node.args:
            if isinstance(argument, ast.Dict):
                keys.update(k.value for k in argument.keys if isinstance(k, ast.Constant))
    return keys


def test_every_key_python_sends_is_a_request_field():
    """A key unity has no field for is dropped in silence, and reads there as zero."""
    assert sent_keys(read("bridge", "sim_robot.py")) <= request_fields(PROTOCOL)


def test_the_step_request_carries_the_step_length():
    assert "dt" in request_fields(PROTOCOL)
    assert "dt" in sent_keys(read("bridge", "sim_robot.py"))


def test_the_arena_prefab_has_no_fields_the_scripts_dropped():
    """A renamed field leaves its old value in the prefab, where it is read by nothing
    and looks like it is still in force. Open the prefab in the editor and save it."""
    for component, source in (("Arena", ARENA_CS),
                              ("StepPenalty", read(*REWARDS, "StepPenalty.cs")),
                              ("OffTrackPenalty", read(*REWARDS, "OffTrackPenalty.cs")),
                              ("JerkPenalty", read(*REWARDS, "JerkPenalty.cs"))):
        assert scene_fields(ARENA_PREFAB, component) <= serialized_fields(source), component


def test_scene_has_no_fields_the_bridge_script_dropped():
    assert scene_fields(SCENE, "Bridge") <= serialized_fields(BRIDGE)


@pytest.mark.parametrize("path", sorted(sources("shared", "brick", "experiments")))
def test_brick_sources_parse_on_the_brick_python(path):
    """feature_version only checks syntax; a newer stdlib call still slips through."""
    with open(path) as f:
        ast.parse(f.read(), path, feature_version=BRICK_PYTHON)


@pytest.mark.parametrize("path", sorted(sources("shared", "brick", "experiments")))
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


def test_the_python_check_is_set_to_the_version_the_brick_runs():
    """Set too high, the check above accepts syntax the brick would reject. Variable
    annotations are 3.6, so this pins it to the 3.5 ev3dev ships. It is a floor and
    not a guarantee: feature_version does not gate f-strings, which are 3.6 too."""
    with pytest.raises(SyntaxError):
        ast.parse("x: int = 1", feature_version=BRICK_PYTHON)
