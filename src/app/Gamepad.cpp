// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Rajesh D'Monte
#include "Gamepad.h"
#include "../core/Log.h"

#define GLFW_INCLUDE_NONE
#include <GLFW/glfw3.h>

#include <cmath>

namespace pt::app {

namespace {

// Smooth radial deadzone on a 2D stick. Inside the deadzone the stick
// reads as 0; outside, the remaining magnitude is rescaled to span
// [0, 1] so there's no sudden snap at the boundary. The direction
// (atan2) is preserved.
void ApplyStickDeadzone(float& x, float& y, float deadzone) {
    if (deadzone <= 0.0f) return;
    if (deadzone >= 1.0f) { x = 0.0f; y = 0.0f; return; }
    float mag = std::sqrt(x * x + y * y);
    if (mag <= deadzone) { x = 0.0f; y = 0.0f; return; }
    // Scale the surviving magnitude to span [0, 1]
    float scaled  = (mag - deadzone) / (1.0f - deadzone);
    // Clamp to 1 -- some pads briefly report >1 due to mechanical slop.
    if (scaled > 1.0f) scaled = 1.0f;
    float k       = scaled / mag;
    x *= k;
    y *= k;
}

// One-shot trigger remap from GLFW's [-1, 1] (rest = -1) to [0, 1]
// (rest = 0). A small zero-floor swallows the few-percent rattle most
// triggers emit at idle so cam_sprint doesn't blip on.
float TriggerToUnit(float v) {
    float t = (v + 1.0f) * 0.5f;
    if (t < 0.02f) t = 0.0f;
    if (t > 1.0f)  t = 1.0f;
    return t;
}

}  // namespace

GamepadState PollGamepad(float deadzone) {
    GamepadState s;

    // GLFW_JOYSTICK_1 is slot 0; we only poll the first present pad.
    // glfwGetGamepadState only succeeds if (a) a joystick is connected
    // AND (b) GLFW has an SDL2-style mapping for it. That filters out
    // generic HID devices that don't follow the gamepad layout.
    if (glfwJoystickPresent(GLFW_JOYSTICK_1) != GLFW_TRUE) return s;

    GLFWgamepadstate gs;
    if (glfwGetGamepadState(GLFW_JOYSTICK_1, &gs) != GLFW_TRUE) return s;

    s.connected = true;
    const char* nm = glfwGetGamepadName(GLFW_JOYSTICK_1);
    if (nm != nullptr) s.name = nm;

    // Raw axes from GLFW. The struct layout per GLFW docs:
    //   axes[GLFW_GAMEPAD_AXIS_LEFT_X]        [-1, 1] left/right
    //   axes[GLFW_GAMEPAD_AXIS_LEFT_Y]        [-1, 1] up/down (-1 = up)
    //   axes[GLFW_GAMEPAD_AXIS_RIGHT_X]       [-1, 1]
    //   axes[GLFW_GAMEPAD_AXIS_RIGHT_Y]       [-1, 1] (-1 = up)
    //   axes[GLFW_GAMEPAD_AXIS_LEFT_TRIGGER]  [-1, 1] (-1 = rest)
    //   axes[GLFW_GAMEPAD_AXIS_RIGHT_TRIGGER] [-1, 1] (-1 = rest)
    s.left_x  =  gs.axes[GLFW_GAMEPAD_AXIS_LEFT_X];
    s.left_y  = -gs.axes[GLFW_GAMEPAD_AXIS_LEFT_Y];   // flip so +Y = forward
    s.right_x =  gs.axes[GLFW_GAMEPAD_AXIS_RIGHT_X];
    s.right_y = -gs.axes[GLFW_GAMEPAD_AXIS_RIGHT_Y];  // flip so +Y = look up

    ApplyStickDeadzone(s.left_x,  s.left_y,  deadzone);
    ApplyStickDeadzone(s.right_x, s.right_y, deadzone);

    s.left_trigger  = TriggerToUnit(gs.axes[GLFW_GAMEPAD_AXIS_LEFT_TRIGGER]);
    s.right_trigger = TriggerToUnit(gs.axes[GLFW_GAMEPAD_AXIS_RIGHT_TRIGGER]);

    return s;
}

bool LogConnectionTransitions(const GamepadState& s) {
    // Static so the engine can poll this every frame and only LOG_INFO
    // on the transition edge. Tracking name too: hot-swapping pads
    // (Xbox -> DualSense) still emits one transition pair.
    static bool        was_connected = false;
    static std::string last_name;

    if (s.connected && !was_connected) {
        LOG_INFO("gamepad: connected ({})", s.name.empty() ? "unknown" : s.name);
        was_connected = true;
        last_name     = s.name;
    } else if (!s.connected && was_connected) {
        LOG_INFO("gamepad: disconnected ({})",
                 last_name.empty() ? "unknown" : last_name);
        was_connected = false;
        last_name.clear();
    } else if (s.connected && s.name != last_name) {
        LOG_INFO("gamepad: changed ({} -> {})",
                 last_name.empty() ? "unknown" : last_name,
                 s.name.empty()    ? "unknown" : s.name);
        last_name = s.name;
    }
    return was_connected;
}

}  // namespace pt::app
