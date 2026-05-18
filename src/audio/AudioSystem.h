// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Rajesh D'Monte
//
// pt::audio::AudioSystem -- miniaudio-backed audio subsystem (MVP, issue #80).
//
// This is the Phase A skeleton from issue #80. It owns the platform audio
// device (CoreAudio on macOS, WASAPI on Windows, ALSA/PulseAudio on Linux
// via miniaudio's runtime backend selection) and a small voice pool for
// 3D-positioned WAV one-shots.
//
// Scope explicitly DEFERRED to follow-up PRs (per issue #80 Phase B):
//   - ray-traced occlusion against the renderer's TLAS (the headline)
//   - ray-traced reverb (cast diffuse rays from listener, integrate IR)
//   - HRTF (basic stereo gain panning only for MVP)
//   - music streaming (one-shots only for MVP)
//   - OGG / FLAC decode (WAV only for MVP; miniaudio supports the rest
//     natively, just not wired up here)
//   - doppler shift
//
// What IS wired:
//   - default-device open at 48 kHz stereo
//   - PCM-frames sound load (`LoadSound`) via miniaudio's WAV decoder
//   - 3D positional playback with 1/r distance attenuation (clamped at
//     minimum-distance 1m to avoid singularities) and basic equal-power
//     stereo pan from the listener's forward/right basis
//   - lock-free voice pool sized for kMaxVoices simultaneous voices;
//     overflow drops the new voice (returns kInvalidVoice) rather than
//     stealing
//   - Tick(camera_pos, camera_fwd) updates the listener state used by
//     the audio callback (atomic write of a small POD; no locks)
//
// Threading model:
//   - LoadSound / PlaySound / Stop* / Tick run on the main (engine) thread
//   - The miniaudio device callback runs on a high-priority audio thread
//     and reads voice + listener state via atomics. No allocation, no
//     locks, no syscalls in the callback.
//
// Spatial-audio scope (a clarification to the "what IS wired" block
// above): moving SOURCES are not supported in MVP. Source pos is fixed
// at PlaySound time and never re-read by the engine. The LISTENER
// (camera) can move freely -- Tick() publishes the latest listener
// pos / fwd each frame and recomputes per-voice stereo gains against
// the moving camera, so a one-shot fired from a fixed point will
// attenuate and pan correctly as the camera flies around it. The
// fixed-source limitation matters only for sounds attached to moving
// game objects, which is a follow-up.

#pragma once

#include <glm/glm.hpp>

#include <atomic>
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <string_view>
#include <vector>

namespace pt::audio {

// Opaque handle to a loaded sound asset. 0 is the invalid sentinel.
// Issued sequentially by LoadSound; never recycled (the sound bank
// is grow-only for the engine's lifetime).
using SoundHandle = std::uint32_t;
inline constexpr SoundHandle kInvalidSound = 0;

// Opaque handle to a currently-playing voice. 0 is the invalid sentinel.
// Encodes both the pool slot (low bits) and a generation counter (high
// bits) so a Stop() against a stale handle is a safe no-op when the
// slot has been recycled into a new voice.
using VoiceHandle = std::uint64_t;
inline constexpr VoiceHandle kInvalidVoice = 0;

class AudioSystem {
public:
    // Hard limit on concurrent voices. Sized for first-cut testing; if
    // the engine grows ambient-soundscape needs this can move higher.
    // Power of two so slot-index math (handle & (kMaxVoices - 1))
    // works cleanly in the callback.
    static constexpr std::uint32_t kMaxVoices = 64;

    AudioSystem();
    ~AudioSystem();

    AudioSystem(const AudioSystem&)            = delete;
    AudioSystem& operator=(const AudioSystem&) = delete;
    AudioSystem(AudioSystem&&)                 = delete;
    AudioSystem& operator=(AudioSystem&&)      = delete;

    // Open the default playback device at 48 kHz stereo F32. Returns
    // false on any miniaudio init failure -- in that case the rest of
    // the API silently no-ops (LoadSound returns kInvalidSound,
    // PlaySound returns kInvalidVoice) so the engine can continue
    // running headless without audio.
    bool Init();

    // Release the device + free all sound banks. Idempotent.
    void Shutdown();

    // Push listener state into the audio thread (atomic store of a
    // small POD; no locks). Camera forward is used for stereo panning;
    // camera position is used for per-voice distance attenuation.
    // Called once per engine tick.
    void Tick(const glm::vec3& camera_pos, const glm::vec3& camera_fwd);

    // Load a sound from disk. WAV only in MVP -- miniaudio's built-in
    // decoder handles 16-bit PCM, 24-bit PCM, 32-bit PCM, and float
    // formats; non-WAV files return kInvalidSound with an error
    // logged. Idempotent for the same path (returns the previously
    // loaded handle).
    SoundHandle LoadSound(std::string_view path);

    // Fire a 3D-positioned one-shot at world position `pos` with linear
    // gain `gain` (1.0 = unity, 0.0 = silent). Returns kInvalidVoice
    // if the voice pool is full or the sound handle is invalid.
    VoiceHandle PlaySound(SoundHandle sound, const glm::vec3& pos, float gain = 1.0f);

    // Stop a specific voice. Safe no-op for stale / already-finished
    // / kInvalidVoice handles -- the slot's generation counter is
    // checked against the high bits of the handle.
    void Stop(VoiceHandle voice);

    // Stop every currently-playing voice. Returns the number of voices
    // that were active before the call.
    std::uint32_t StopAll();

    // True iff Init() succeeded and the device is currently open.
    bool IsRunning() const noexcept { return running_.load(std::memory_order_acquire); }

    // Implementation detail exposed only so the miniaudio static
    // callback can reach into the pool from a C-style function
    // pointer. Definition lives entirely inside AudioSystem.cpp.
    struct Impl;

private:
    std::unique_ptr<Impl> impl_;

    std::atomic<bool> running_ {false};
};

}  // namespace pt::audio
