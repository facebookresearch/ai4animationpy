# Copyright (c) Meta Platforms, Inc. and affiliates.
import glob
import os
import struct
import sys
from pathlib import Path

import pyray as pr
import raylib as rl
from ai4animation import AI4Animation, Utility
from ai4animation.Math import Rotation, Vector3

THIS_DIR = Path(__file__).resolve().parent

CONTROLLER_TEXTURE = pr.load_texture(str(THIS_DIR / "resources/xbox.png"))
STICK_DEADZONE = 0.1
TRIGGER_DEADZONE = -0.9
TRIGGER_PRESSED_THRESHOLD = -0.5
CONTROLLER_ID = 0

# Linux joydev axis indices (Xbox layout)
_JS_LEFT_X = 0
_JS_LEFT_Y = 1
_JS_RIGHT_X = 3
_JS_RIGHT_Y = 4
_JS_LEFT_TRIGGER = 2
_JS_RIGHT_TRIGGER = 5

# Linux joydev button indices (Xbox layout)
_JS_BTN_LEFT_THUMB = 9
_JS_BTN_RIGHT_THUMB = 10

_JS_EVENT_SIZE = 8  # struct js_event: u32 time, s16 value, u8 type, u8 number
_GAMEPAD_ALIASES = ("xbox", "x-box", "playstation")


class _JoydevReader:
    """Reads gamepad input directly from Linux joydev (/dev/input/js*).

    GLFW (used by raylib) fails to read Xbox Series controllers on some
    Linux systems despite detecting them. This bypasses GLFW entirely.
    """

    def __init__(self):
        self.fd = None
        self.axes = {}
        self.buttons = {}
        self._open()

    def _open(self):
        for jsdir in sorted(glob.glob("/sys/class/input/js*")):
            name_file = os.path.join(jsdir, "device", "name")
            try:
                with open(name_file) as f:
                    name = f.read().strip()
                if any(a in name.lower() for a in _GAMEPAD_ALIASES):
                    jsnum = os.path.basename(jsdir)
                    path = f"/dev/input/{jsnum}"
                    self.fd = os.open(path, os.O_RDONLY | os.O_NONBLOCK)
                    print(f"[InputSystem] Opened joydev: {path} ({name})")
                    return
            except (FileNotFoundError, PermissionError):
                continue
        print("[InputSystem] No gamepad found via joydev, using keyboard fallback")

    @property
    def available(self):
        return self.fd is not None

    def poll(self):
        """Drain all pending joydev events. Call once per frame."""
        if self.fd is None:
            return
        while True:
            try:
                data = os.read(self.fd, _JS_EVENT_SIZE)
                _ts, value, etype, number = struct.unpack("IhBB", data)
                if etype & 0x02:  # axis (including INIT+AXIS)
                    self.axes[number] = value / 32767.0
                elif etype & 0x01:  # button (including INIT+BUTTON)
                    self.buttons[number] = bool(value)
            except BlockingIOError:
                break

    def get_axis(self, index):
        return self.axes.get(index, 0.0)

    def get_button(self, index):
        return self.buttons.get(index, False)

    def close(self):
        if self.fd is not None:
            os.close(self.fd)
            self.fd = None


_joydev = _JoydevReader() if sys.platform == "linux" else None


def GamepadAvailable() -> bool:
    if _joydev is not None:
        return _joydev.available
    return rl.IsGamepadAvailable(0)


def _poll():
    if _joydev is not None:
        _joydev.poll()


def _apply_deadzone(value, deadzone=STICK_DEADZONE):
    return 0.0 if -deadzone < value < deadzone else value


def GetLeftStick():
    _poll()
    if not GamepadAvailable():
        wasd = GetWASDQE()
        return (wasd[0], wasd[2])
    if _joydev is not None and _joydev.available:
        x = _apply_deadzone(_joydev.get_axis(_JS_LEFT_X))
        y = _apply_deadzone(_joydev.get_axis(_JS_LEFT_Y))
        return (x, -y)
    x = pr.get_gamepad_axis_movement(0, pr.GamepadAxis.GAMEPAD_AXIS_LEFT_X)
    y = pr.get_gamepad_axis_movement(0, pr.GamepadAxis.GAMEPAD_AXIS_LEFT_Y)
    x = _apply_deadzone(x)
    y = _apply_deadzone(y)
    return (x, -y)


def GetRightStick():
    _poll()
    if not GamepadAvailable():
        x = float(rl.IsKeyDown(rl.KEY_RIGHT)) - float(rl.IsKeyDown(rl.KEY_LEFT))
        y = float(rl.IsKeyDown(rl.KEY_UP)) - float(rl.IsKeyDown(rl.KEY_DOWN))
        return (x, y)
    if _joydev is not None and _joydev.available:
        x = _apply_deadzone(_joydev.get_axis(_JS_RIGHT_X))
        y = _apply_deadzone(_joydev.get_axis(_JS_RIGHT_Y))
        return (x, -y)
    x = pr.get_gamepad_axis_movement(0, pr.GamepadAxis.GAMEPAD_AXIS_RIGHT_X)
    y = pr.get_gamepad_axis_movement(0, pr.GamepadAxis.GAMEPAD_AXIS_RIGHT_Y)
    x = _apply_deadzone(x)
    y = _apply_deadzone(y)
    return (x, -y)


def GetLeftTrigger():
    _poll()
    if _joydev is not None and _joydev.available:
        value = _joydev.get_axis(_JS_LEFT_TRIGGER)
        if value < TRIGGER_DEADZONE:
            value = -1.0
        return value
    LogErrorIfGamepadNotAvailable()
    value = pr.get_gamepad_axis_movement(0, pr.GamepadAxis.GAMEPAD_AXIS_LEFT_TRIGGER)
    if value < TRIGGER_DEADZONE:
        value = -1.0
    return value


def GetRightTrigger():
    _poll()
    if _joydev is not None and _joydev.available:
        value = _joydev.get_axis(_JS_RIGHT_TRIGGER)
        if value < TRIGGER_DEADZONE:
            value = -1.0
        return value
    LogErrorIfGamepadNotAvailable()
    value = pr.get_gamepad_axis_movement(0, pr.GamepadAxis.GAMEPAD_AXIS_RIGHT_TRIGGER)
    if value < TRIGGER_DEADZONE:
        value = -1.0
    return value


def IsLeftStickPressed():
    _poll()
    if not GamepadAvailable():
        return rl.IsKeyDown(rl.KEY_LEFT_SHIFT)
    if _joydev is not None and _joydev.available:
        return _joydev.get_button(_JS_BTN_LEFT_THUMB)
    return pr.is_gamepad_button_down(0, pr.GamepadButton.GAMEPAD_BUTTON_LEFT_THUMB)


def IsRightStickPressed():
    _poll()
    if not GamepadAvailable():
        return False
    if _joydev is not None and _joydev.available:
        return _joydev.get_button(_JS_BTN_RIGHT_THUMB)
    return pr.is_gamepad_button_down(0, pr.GamepadButton.GAMEPAD_BUTTON_RIGHT_THUMB)


def LogErrorIfGamepadNotAvailable():
    if not GamepadAvailable():
        print("Error: Gamepad not available")


def IsL1Pressed():
    LogErrorIfGamepadNotAvailable()
    return pr.is_gamepad_button_pressed(
        CONTROLLER_ID, pr.GamepadButton.GAMEPAD_BUTTON_LEFT_TRIGGER_1
    )


def IsL1Down():
    LogErrorIfGamepadNotAvailable()
    return pr.is_gamepad_button_down(
        CONTROLLER_ID, pr.GamepadButton.GAMEPAD_BUTTON_LEFT_TRIGGER_1
    )


def IsR1Pressed():
    LogErrorIfGamepadNotAvailable()
    return pr.is_gamepad_button_pressed(
        CONTROLLER_ID, pr.GamepadButton.GAMEPAD_BUTTON_RIGHT_TRIGGER_1
    )


def IsR1Down():
    LogErrorIfGamepadNotAvailable()
    return pr.is_gamepad_button_down(
        CONTROLLER_ID, pr.GamepadButton.GAMEPAD_BUTTON_RIGHT_TRIGGER_1
    )


def IsL2Pressed():
    LogErrorIfGamepadNotAvailable()
    return pr.is_gamepad_button_pressed(
        CONTROLLER_ID, pr.GamepadButton.GAMEPAD_BUTTON_LEFT_TRIGGER_2
    )


def IsL2Down():
    LogErrorIfGamepadNotAvailable()
    return (
        pr.is_gamepad_button_down(
            CONTROLLER_ID, pr.GamepadButton.GAMEPAD_BUTTON_LEFT_TRIGGER_2
        )
        or GetLeftTrigger() > TRIGGER_PRESSED_THRESHOLD
    )


def IsR2Pressed():
    LogErrorIfGamepadNotAvailable()
    return pr.is_gamepad_button_pressed(
        CONTROLLER_ID, pr.GamepadButton.GAMEPAD_BUTTON_RIGHT_TRIGGER_2
    )


def IsR2Down():
    LogErrorIfGamepadNotAvailable()
    return (
        pr.is_gamepad_button_down(
            CONTROLLER_ID, pr.GamepadButton.GAMEPAD_BUTTON_RIGHT_TRIGGER_2
        )
        or GetRightTrigger() > TRIGGER_PRESSED_THRESHOLD
    )


def IsRightFaceRightPressed():
    LogErrorIfGamepadNotAvailable()
    return pr.is_gamepad_button_pressed(
        CONTROLLER_ID, pr.GamepadButton.GAMEPAD_BUTTON_RIGHT_FACE_RIGHT
    )


def IsRightFaceDownPressed():
    LogErrorIfGamepadNotAvailable()
    return pr.is_gamepad_button_pressed(
        CONTROLLER_ID, pr.GamepadButton.GAMEPAD_BUTTON_RIGHT_FACE_DOWN
    )


def IsRightFaceLeftPressed():
    LogErrorIfGamepadNotAvailable()
    return pr.is_gamepad_button_pressed(
        CONTROLLER_ID, pr.GamepadButton.GAMEPAD_BUTTON_RIGHT_FACE_LEFT
    )


def GetButtonA():
    return rl.IsGamepadButtonDown(CONTROLLER_ID, rl.GAMEPAD_BUTTON_RIGHT_FACE_DOWN)


def GetCurrentKey():
    key = rl.GetCharPressed()
    return chr(key) if (key >= 32) and (key <= 125) else None


def GetKey(id):
    return rl.IsKeyPressed(id)

def GetWASDQE():
    input = [0, 0, 0]
    if rl.IsKeyDown(rl.KEY_S):
        input[2] -= 1
    if rl.IsKeyDown(rl.KEY_W):
        input[2] += 1
    if rl.IsKeyDown(rl.KEY_A):
        input[0] -= 1
    if rl.IsKeyDown(rl.KEY_D):
        input[0] += 1
    if rl.IsKeyDown(rl.KEY_Q):
        input[1] -= 1
    if rl.IsKeyDown(rl.KEY_E):
        input[1] += 1
    return Vector3.Create(input)


# A secondary mapping to support keyboard approximation of two joysticks
def GetIJKL():
    x = 0
    y = 0
    if rl.IsKeyDown(rl.KEY_K):
        y -= 1
    if rl.IsKeyDown(rl.KEY_I):
        y += 1
    if rl.IsKeyDown(rl.KEY_J):
        x -= 1
    if rl.IsKeyDown(rl.KEY_L):
        x += 1
    return [x, y]


def GetMousePositionOnScreen():  # Get mouse position XY in screen space
    position = rl.GetMousePosition()
    return (position.x, position.y)


def GetMouseDeltaOnScreen():  # Get mouse delta XY in screen space
    position = rl.GetMouseDelta()
    return (position.x, position.y)


def GetWorldPositionOnScreen(position, camera):
    return Vector3.FromRayLib(rl.GetWorldToScreen(position.tolist(), camera))


def GetMousePositionInWorld(camera):
    ray = rl.GetScreenToWorldRay(rl.GetMousePosition(), camera)
    size = 25
    a = (-size, 0, -size)
    b = (-size, 0, size)
    c = (size, 0, size)
    d = (size, 0, -size)
    info = rl.GetRayCollisionQuad(ray, a, b, c, d)
    return Vector3.FromRayLib(info.point)


def GetMousePositionInSpace(camera, space):
    ray = rl.GetScreenToWorldRay(rl.GetMousePosition(), camera.Camera)
    size = 25
    r = camera.Entity.GetRotation()
    p = space.GetPosition()
    a = Vector3.ToRayLib(p + Rotation.Multiply(r, Vector3.Create(-size, -size, 0)))
    b = Vector3.ToRayLib(p + Rotation.Multiply(r, Vector3.Create(-size, size, 0)))
    c = Vector3.ToRayLib(p + Rotation.Multiply(r, Vector3.Create(size, size, 0)))
    d = Vector3.ToRayLib(p + Rotation.Multiply(r, Vector3.Create(size, -size, 0)))
    info = rl.GetRayCollisionQuad(ray, a, b, c, d)
    return Vector3.FromRayLib(info.point)


def DrawController(x, y, scale):
    ratio = AI4Animation.Standalone.ScaleRatio()
    x, y = AI4Animation.Standalone.ToScreen((x, y))
    w = scale * CONTROLLER_TEXTURE.width * ratio
    h = scale * CONTROLLER_TEXTURE.height * ratio
    pr.draw_texture_ex(
        CONTROLLER_TEXTURE,
        pr.Vector2(int(x - w / 2), int(y - h / 2)),
        0.0,
        scale * ratio,
        rl.DARKGRAY,
    )

    left_stick_color = pr.RED if IsLeftStickPressed() else pr.BLACK
    right_stick_color = pr.RED if IsRightStickPressed() else pr.BLACK

    # Left Stick
    pos_x = -140 * ratio
    pos_y = -73 * ratio
    outer = 40 * ratio
    middle = 30 * ratio
    inner = 20 * ratio
    stick_x, stick_y = GetLeftStick()
    pr.draw_circle(
        int(x + pos_x * scale), int(y + pos_y * scale), int(outer * scale), pr.BLACK
    )
    pr.draw_circle(
        int(x + pos_x * scale),
        int(y + pos_y * scale),
        int(middle * scale),
        pr.LIGHTGRAY,
    )
    pr.draw_circle(
        int(x + scale * (pos_x + stick_x * inner)),
        int(y + scale * (pos_y - stick_y * inner)),
        int(inner * scale),
        left_stick_color,
    )

    # Right Stick
    pos_x = 61 * ratio
    pos_y = 12 * ratio
    outer = 40 * ratio
    middle = 30 * ratio
    inner = 20 * ratio
    stick_x, stick_y = GetRightStick()
    pr.draw_circle(
        int(x + pos_x * scale), int(y + pos_y * scale), int(outer * scale), pr.BLACK
    )
    pr.draw_circle(
        int(x + pos_x * scale),
        int(y + pos_y * scale),
        int(middle * scale),
        pr.LIGHTGRAY,
    )
    pr.draw_circle(
        int(x + scale * (pos_x + stick_x * inner)),
        int(y + scale * (pos_y - stick_y * inner)),
        int(inner * scale),
        right_stick_color,
    )


def DrawWASDQE(x, y, scale):
    DrawKeySet(
        x, y, scale, [[rl.KEY_Q, rl.KEY_W, rl.KEY_E], [rl.KEY_A, rl.KEY_S, rl.KEY_D]]
    )


def DrawIJKL(x, y, scale):
    DrawKeySet(x, y, scale, [[None, rl.KEY_I, None], [rl.KEY_J, rl.KEY_K, rl.KEY_L]])


# Given a list of list of keys, draw a graphical representation of the key state
def DrawKeySet(x, y, scale, keySet):
    ratio = AI4Animation.Standalone.ScaleRatio()
    x, y = AI4Animation.Standalone.ToScreen((x, y))
    outer = 120 * ratio * scale
    spacing = 20 * ratio * scale
    border = 2

    if isinstance(keySet, int):
        keySet = [[keySet]]
    elif isinstance(keySet[0], int):
        keySet = [keySet]
    for j, row in enumerate(keySet):
        for i, key in enumerate(row):
            if key is None:
                continue
            value = rl.IsKeyDown(key)
            pos_x = x + (outer + spacing) * i
            pos_y = y + (outer + spacing) * j
            size = outer
            pr.draw_rectangle_rounded([pos_x, pos_y, size, size], 0.2, 10, pr.DARKGRAY)
            pos_x = x + (outer + spacing) * i + border
            pos_y = y + (outer + spacing) * j + border
            size = outer - 2 * border
            color = pr.LIGHTGRAY if value else pr.GRAY
            pr.draw_rectangle_rounded([pos_x, pos_y, size, size], 0.2, 10, color)
            text = Utility.ToBytes(chr(key))
            size = int(outer / 2)
            w = rl.MeasureText(text, size)
            h = size
            pos_x = x + (outer + spacing) * i + outer / 2 - w / 2
            pos_y = y + (outer + spacing) * j + outer / 2 - h / 2
            rl.DrawText(text, int(pos_x), int(pos_y), size, rl.BLACK)
