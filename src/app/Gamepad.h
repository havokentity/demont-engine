// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Rajesh D'Monte
#pragma once

// GLFW-backed gamepad polling for first-person camera control.
//
// Wraps glfwGetGamepadState / glfwJoystickPresent. GLFW maps known
// controllers (Xbox 360/One/Series, DualShock 4, DualSense, Switch Pro,
// generic XInput) onto a fixed SDL2-style layout (LX/LY/RX/RY/LT/RT
// axes + A/B/X/Y/bumpers/start/back/sticks/dpad buttons), so most pads
// work out of the box without per-device mapping.
//
// The engine polls this every frame from UpdateCamera. Cost is a single
// glfwGetGamepadState call on slot 0 -- effectively free relative to
// the rest of the frame work. Connect/disconnect is detected by per-
// frame edge detection on slot 0 (see LogConnectionTransitions); we do
// NOT use GLFW's joystick callback / OS-event delivery.
//
// MVP scope (issue #83):
//   - Left stick  -> forward/back + strafe.
//   - Right stick -> yaw + pitch (look).
//   - Triggers    -> sprint multiplier (analog blend: 0..1 LT|RT).
//
// Out of scope for MVP (deferred to follow-up):
//   - Action-binding layer (the issue's full plan).
//   - OS callback-style hot-plug events (vs. per-frame polling above).
//   - Rumble / haptics (would need SDL2 or HIDAPI).
//   - Button-to-key remapping (jump, fire, etc.).

#include <string>

namespace pt::app {

// One polled frame of gamepad state. All sticks/triggers are already
// deadzoned and scaled to [-1, 1] (sticks) or [0, 1] (triggers).
struct GamepadState {
    bool  connected   = false;   // a recognised pad is present in slot 0
    std::string name;            // controller name from GLFW (empty if !connected)

    // Sticks: standard XY where +X = right, +Y = "stick pushed away from
    // user" on the left stick, and +Y = "stick pushed away" on the right
    // stick. GLFW reports +Y = down (toward user) for both sticks; we
    // flip the sign here so callers get the more intuitive convention.
    float left_x      = 0.0f;
    float left_y      = 0.0f;    // +Y = forward
    float right_x     = 0.0f;
    float right_y     = 0.0f;    // +Y = look up

    // Triggers in [0, 1]. GLFW reports them in [-1, 1] (rest = -1); we
    // remap to [0, 1] (rest = 0) so caller code reads as a fraction.
    float left_trigger  = 0.0f;
    float right_trigger = 0.0f;
};

// Polls slot GLFW_JOYSTICK_1 (slot 0). Returns a fully-populated
// GamepadState; .connected is false if no gamepad is present or GLFW
// doesn't recognise it (no SDL2 mapping). Deadzone is applied to the
// sticks with a smooth radial scale so the usable range remains
// [0, 1] above the threshold.
//
// deadzone is the inner-radius cutoff in the stick's native [-1, 1]
// space (typical: 0.15). Sticks return 0 inside that radius and
// linearly interpolate (mag - deadzone) / (1 - deadzone) outside.
GamepadState PollGamepad(float deadzone);

// Logs the connection state when it transitions (connected ->
// disconnected or vice versa). Idempotent; safe to call every frame.
// Returns the current connected state for convenience.
bool LogConnectionTransitions(const GamepadState& s);

}  // namespace pt::app
