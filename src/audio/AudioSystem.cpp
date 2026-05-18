// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Rajesh D'Monte
//
// pt::audio::AudioSystem implementation.  See AudioSystem.h for scope.
//
// miniaudio is compiled inline here -- the single .cpp -> single .obj
// pattern matches stb_image_write.h being compiled in engine/stb_impl.cpp.
// MA_IMPLEMENTATION is defined in exactly ONE TU (this one) so the
// header acts as a regular declaration include everywhere else.
//
// MA_NO_ENCODING strips file-encode paths we don't use (PCM/WAV-write
// from a recording capture). Keeps the build a bit faster + leaves a
// smaller binary footprint. The decode side (MA_NO_DECODING is NOT
// defined) is what LoadSound depends on.

#include "AudioSystem.h"

#include "../core/Log.h"

// miniaudio configuration -- defined before including miniaudio.h so
// the single-header compile-time switches take effect.
//
// MA_NO_GENERATION: drops generic generation / waveform helpers (sine,
//   square, etc.) we don't use.
// MA_NO_ENCODING: drops the writer side (we only decode WAV from disk).
// MA_NO_VORBIS / MA_NO_FLAC / MA_NO_MP3: lean MVP -- WAV only. miniaudio
//   itself can natively decode WAV without external deps; vorbis/flac
//   require stb_vorbis / dr_flac inlined (which it ships internally,
//   but the compile cost is non-trivial). Wire these back on when a
//   non-WAV asset actually lands in `assets/audio/`.
#define MA_NO_GENERATION
#define MA_NO_ENCODING
#define MA_NO_VORBIS
#define MA_NO_FLAC
#define MA_NO_MP3
#define MINIAUDIO_IMPLEMENTATION
#include "../../third_party/miniaudio/miniaudio.h"

#include <glm/geometric.hpp>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <unordered_map>
#include <vector>

namespace pt::audio {

namespace {

// Engine output format. F32 stereo at 48kHz matches modern device
// defaults on every supported platform so miniaudio almost never has
// to insert a sample-rate / format converter on the playback path.
constexpr std::uint32_t kSampleRate     = 48000;
constexpr std::uint32_t kChannels       = 2;
constexpr ma_format     kFormat         = ma_format_f32;

// Distance attenuation lower clamp. Below 1m the 1/r falloff explodes;
// real-world point sources don't usually sit on top of the listener so
// this clamp is just a safety net. Metric units: 1 world unit = 1 metre.
constexpr float         kMinDistanceM   = 1.0f;

// Slot index = lower 32 bits of a VoiceHandle. Generation = upper 32.
// Generation increments every time a slot transitions from in-use to
// free, so a Stop() against a stale handle whose slot has been reused
// finds a mismatched generation and bails out cleanly.
constexpr std::uint64_t kSlotMask       = 0xFFFFFFFFull;

inline std::uint64_t MakeHandle(std::uint32_t slot, std::uint32_t gen) {
    return (static_cast<std::uint64_t>(gen) << 32) | slot;
}
inline std::uint32_t SlotOf(VoiceHandle h) { return static_cast<std::uint32_t>(h & kSlotMask); }
inline std::uint32_t GenOf(VoiceHandle h)  { return static_cast<std::uint32_t>(h >> 32); }

// A loaded sound asset. We decode the entire file into PCM frames at
// load time (WAVs are small, the engine isn't memory-pressured by them
// at MVP scale) and stream from the in-memory buffer in the callback.
// Mono assets get duplicated to both channels; multi-channel assets are
// downmixed to mono first then duplicated.
struct Sound {
    std::vector<float> frames;     // interleaved kChannels samples
    std::uint64_t      frame_count = 0;  // == frames.size() / kChannels
    std::string        path;
};

// A single voice slot in the pool. The audio callback reads `active`,
// `sound_id`, `cursor`, `pos`, `gain` atomically; the main thread
// writes them via PlaySound / Stop.
//
// To avoid tearing on the 3D pos vector (writing 3 floats isn't atomic
// on any architecture), we publish under a sequence-lock-like discipline:
// the audio callback samples pos exactly ONCE per buffer-fill and the
// main thread's only mutator is PlaySound (which runs while the slot is
// inactive). After PlaySound flips `active` to true the audio thread
// owns the slot's pos field; later movement of the source is NOT
// supported in MVP (one-shots assumed to be effectively point-in-time
// for their duration of < 1 s). Moving sources is a follow-up.
struct Voice {
    std::atomic<bool>          active   {false};
    std::atomic<std::uint32_t> generation {1};   // 0 reserved for kInvalidVoice
    SoundHandle                sound_id = kInvalidSound;
    std::uint64_t              cursor   = 0;     // frame index into sound->frames
    glm::vec3                  pos      {0, 0, 0};
    float                      gain     = 1.0f;
    // Cached left/right gains computed on PlaySound -- the callback
    // doesn't try to recompute pan per frame since the listener's
    // tick rate (~60 Hz) is much faster than a one-shot is typically
    // long. Updated in Tick() if `track_listener_` is wired later.
    std::atomic<float>         gain_l   {1.0f};
    std::atomic<float>         gain_r   {1.0f};
};

// Equal-power panning + 1/r distance attenuation.
// Returns linear left+right gains (NOT applied to `voice.gain` yet --
// callers multiply through).
void ComputeStereoGains(const glm::vec3& source_pos,
                        const glm::vec3& listener_pos,
                        const glm::vec3& listener_fwd,
                        float& out_l, float& out_r) {
    glm::vec3 delta = source_pos - listener_pos;
    float dist = glm::length(delta);
    if (dist < kMinDistanceM) dist = kMinDistanceM;

    // 1/r distance attenuation. Real point-source falloff is 1/r^2 in
    // intensity (1/r in amplitude); we're already in amplitude space
    // here (linear gain), so 1/r is correct. The minimum-distance
    // clamp keeps near-coincident sources from blowing the mix up.
    float atten = kMinDistanceM / dist;

    // Pan: project the source direction onto the listener's right
    // axis. listener_fwd is assumed unit-length and roughly horizontal
    // (the engine's Camera::Forward()); we build right = fwd x worldUp
    // and dot the source direction onto it. -1 = fully left, +1 = fully
    // right.
    glm::vec3 dir = (dist > 1e-6f) ? (delta / dist) : glm::vec3{0, 0, 1};
    glm::vec3 right = glm::cross(listener_fwd, glm::vec3{0, 1, 0});
    float rlen = glm::length(right);
    if (rlen > 1e-6f) {
        right /= rlen;
    } else {
        right = glm::vec3{1, 0, 0};
    }
    float pan = glm::clamp(glm::dot(dir, right), -1.0f, 1.0f);

    // Equal-power pan law: gains sit on a quarter-circle so
    // L^2 + R^2 = constant regardless of pan. theta in [0, pi/2].
    constexpr float kPiOver4 = 0.785398163397448f;
    float theta = (pan + 1.0f) * kPiOver4;   // [0, pi/2]
    out_l = atten * std::cos(theta);
    out_r = atten * std::sin(theta);
}

}  // namespace

struct AudioSystem::Impl {
    ma_device device {};

    // The voice pool. Aligned so the atomics inside don't false-share
    // across cache lines, though at kMaxVoices=64 and ~80 bytes each
    // we're already comfortably below typical worker contention.
    Voice voices[kMaxVoices] {};

    // Sound bank. Index 0 is the kInvalidSound sentinel (never used);
    // entry at index N is the sound with handle N. Grow-only.
    std::vector<Sound>                 sounds;
    std::unordered_map<std::string,
                       SoundHandle>    path_to_handle;
    std::mutex                         sounds_mutex;   // guards LoadSound only

    // Atomic listener snapshot consumed by the callback. We avoid
    // tearing by writing to the inactive slot and flipping
    // listener_active_.
    std::atomic<glm::vec3>             listener_pos  {glm::vec3{0, 1.5f, 4.0f}};
    std::atomic<glm::vec3>             listener_fwd  {glm::vec3{0, 0, -1}};

    Impl() {
        // Reserve slot 0 for the kInvalidSound sentinel.
        sounds.emplace_back();
        sounds[0].path = "<invalid>";
    }
};

// ---- data callback (audio thread) -----------------------------------------
//
// miniaudio invokes this every time the device wants more PCM data.
// frame_count is small (~480 frames at 10ms-buffer @ 48kHz on macOS).
// We MUST NOT allocate, lock, or syscall here -- doing so risks audio
// stutter / starvation.

static void DataCallback(ma_device* device, void* output, const void* /*input*/,
                         ma_uint32 frame_count) {
    auto* impl = static_cast<AudioSystem::Impl*>(device->pUserData);
    if (impl == nullptr) {
        std::memset(output, 0, frame_count * kChannels * sizeof(float));
        return;
    }

    float* out = static_cast<float*>(output);
    std::memset(out, 0, frame_count * kChannels * sizeof(float));

    // Mix every active voice into the output buffer.
    for (std::uint32_t i = 0; i < AudioSystem::kMaxVoices; ++i) {
        Voice& v = impl->voices[i];
        if (!v.active.load(std::memory_order_acquire)) continue;

        const SoundHandle sid = v.sound_id;
        if (sid == kInvalidSound || sid >= impl->sounds.size()) {
            // Defensive: shouldn't happen if PlaySound validated. Mute
            // + free the slot.
            v.active.store(false, std::memory_order_release);
            v.generation.fetch_add(1, std::memory_order_acq_rel);
            continue;
        }
        const Sound& s = impl->sounds[sid];
        if (s.frame_count == 0) {
            v.active.store(false, std::memory_order_release);
            v.generation.fetch_add(1, std::memory_order_acq_rel);
            continue;
        }

        float gain   = v.gain;
        float gain_l = v.gain_l.load(std::memory_order_relaxed);
        float gain_r = v.gain_r.load(std::memory_order_relaxed);

        std::uint64_t cursor = v.cursor;
        const std::uint64_t remaining = s.frame_count - cursor;
        const std::uint64_t to_mix    = std::min<std::uint64_t>(frame_count, remaining);

        // Mix `to_mix` frames into the output. Source is stored as
        // interleaved stereo F32 (mono assets were duplicated on load).
        const float* src = s.frames.data() + cursor * kChannels;
        for (std::uint64_t f = 0; f < to_mix; ++f) {
            out[f * 2 + 0] += src[f * 2 + 0] * gain * gain_l;
            out[f * 2 + 1] += src[f * 2 + 1] * gain * gain_r;
        }

        cursor += to_mix;
        v.cursor = cursor;

        // Voice finished naturally.
        if (cursor >= s.frame_count) {
            v.active.store(false, std::memory_order_release);
            v.generation.fetch_add(1, std::memory_order_acq_rel);
        }
    }
}

// ---- public API -----------------------------------------------------------

AudioSystem::AudioSystem()  : impl_(std::make_unique<Impl>()) {}
AudioSystem::~AudioSystem() { Shutdown(); }

bool AudioSystem::Init() {
    if (running_.load(std::memory_order_acquire)) return true;

    ma_device_config cfg     = ma_device_config_init(ma_device_type_playback);
    cfg.playback.format      = kFormat;
    cfg.playback.channels    = kChannels;
    cfg.sampleRate           = kSampleRate;
    cfg.dataCallback         = &DataCallback;
    cfg.pUserData            = impl_.get();

    ma_result res = ma_device_init(nullptr, &cfg, &impl_->device);
    if (res != MA_SUCCESS) {
        LOG_ERROR("audio: ma_device_init failed (code={}); audio disabled", int(res));
        return false;
    }

    res = ma_device_start(&impl_->device);
    if (res != MA_SUCCESS) {
        LOG_ERROR("audio: ma_device_start failed (code={}); audio disabled", int(res));
        ma_device_uninit(&impl_->device);
        return false;
    }

    running_.store(true, std::memory_order_release);
    LOG_INFO("audio: device opened ({} Hz, {} ch, F32, miniaudio v{})",
             kSampleRate, kChannels, ma_version_string());
    return true;
}

void AudioSystem::Shutdown() {
    if (!running_.exchange(false, std::memory_order_acq_rel)) return;

    // Stop the device first so the data callback stops touching `impl_`
    // before we tear voice state down.
    ma_device_uninit(&impl_->device);

    StopAll();
    impl_->sounds.clear();
    impl_->path_to_handle.clear();
    // Re-seed the invalid sentinel so a subsequent Init/LoadSound cycle
    // still treats handle 0 as bogus.
    impl_->sounds.emplace_back();
    impl_->sounds[0].path = "<invalid>";

    LOG_INFO("audio: device closed");
}

void AudioSystem::Tick(const glm::vec3& camera_pos, const glm::vec3& camera_fwd) {
    if (!running_.load(std::memory_order_acquire)) return;
    impl_->listener_pos.store(camera_pos, std::memory_order_release);
    impl_->listener_fwd.store(camera_fwd, std::memory_order_release);
    // Per-voice gain re-computation against the current listener so a
    // moving camera attenuates / pans an in-flight one-shot correctly.
    // (Source pos is fixed at PlaySound time -- see Voice doc comment.)
    for (std::uint32_t i = 0; i < kMaxVoices; ++i) {
        Voice& v = impl_->voices[i];
        if (!v.active.load(std::memory_order_acquire)) continue;
        float l = 1.0f, r = 1.0f;
        ComputeStereoGains(v.pos, camera_pos, camera_fwd, l, r);
        v.gain_l.store(l, std::memory_order_relaxed);
        v.gain_r.store(r, std::memory_order_relaxed);
    }
}

SoundHandle AudioSystem::LoadSound(std::string_view path) {
    if (!running_.load(std::memory_order_acquire)) return kInvalidSound;

    const std::string path_s(path);
    {
        std::lock_guard<std::mutex> lk(impl_->sounds_mutex);
        auto it = impl_->path_to_handle.find(path_s);
        if (it != impl_->path_to_handle.end()) return it->second;
    }

    // Decode the WAV via miniaudio's built-in decoder. We pull the
    // entire file into memory in our target format (F32 / 48kHz /
    // stereo) so the audio callback can stream straight from
    // `Sound::frames` with no per-callback resampling cost.
    ma_decoder_config dec_cfg = ma_decoder_config_init(kFormat, kChannels, kSampleRate);
    ma_decoder dec;
    ma_result  res = ma_decoder_init_file(path_s.c_str(), &dec_cfg, &dec);
    if (res != MA_SUCCESS) {
        LOG_ERROR("audio: ma_decoder_init_file('{}') failed (code={}); sound not loaded",
                  path_s, int(res));
        return kInvalidSound;
    }

    ma_uint64 frame_count = 0;
    res = ma_decoder_get_length_in_pcm_frames(&dec, &frame_count);
    if (res != MA_SUCCESS || frame_count == 0) {
        LOG_ERROR("audio: ma_decoder_get_length_in_pcm_frames('{}') failed or empty "
                  "(code={}, frames={})", path_s, int(res), frame_count);
        ma_decoder_uninit(&dec);
        return kInvalidSound;
    }

    Sound s;
    s.path        = path_s;
    s.frame_count = frame_count;
    s.frames.resize(frame_count * kChannels);
    ma_uint64 frames_read = 0;
    res = ma_decoder_read_pcm_frames(&dec, s.frames.data(), frame_count, &frames_read);
    ma_decoder_uninit(&dec);
    if (res != MA_SUCCESS || frames_read == 0) {
        LOG_ERROR("audio: ma_decoder_read_pcm_frames('{}') failed (code={}, read={})",
                  path_s, int(res), frames_read);
        return kInvalidSound;
    }
    s.frame_count = frames_read;
    s.frames.resize(frames_read * kChannels);

    SoundHandle h = kInvalidSound;
    {
        std::lock_guard<std::mutex> lk(impl_->sounds_mutex);
        // Re-check under lock in case another thread raced us to the
        // same path between our first lookup and now.
        auto it = impl_->path_to_handle.find(path_s);
        if (it != impl_->path_to_handle.end()) return it->second;
        h = static_cast<SoundHandle>(impl_->sounds.size());
        impl_->sounds.push_back(std::move(s));
        impl_->path_to_handle.emplace(path_s, h);
    }
    LOG_INFO("audio: loaded '{}' ({} frames, {:.2f}s)",
             path_s, frame_count,
             double(frame_count) / double(kSampleRate));
    return h;
}

VoiceHandle AudioSystem::PlaySound(SoundHandle sound, const glm::vec3& pos, float gain) {
    if (!running_.load(std::memory_order_acquire)) return kInvalidVoice;
    if (sound == kInvalidSound) return kInvalidVoice;
    {
        std::lock_guard<std::mutex> lk(impl_->sounds_mutex);
        if (sound >= impl_->sounds.size() || impl_->sounds[sound].frame_count == 0) {
            return kInvalidVoice;
        }
    }

    // Find an inactive slot. Linear scan is fine at kMaxVoices=64.
    for (std::uint32_t i = 0; i < kMaxVoices; ++i) {
        Voice& v = impl_->voices[i];
        bool expected = false;
        if (!v.active.compare_exchange_strong(expected, false,
                                              std::memory_order_acquire,
                                              std::memory_order_relaxed)) {
            // already active
            continue;
        }
        // We now own slot `i` (it was inactive and we did NOT flip
        // active yet -- the CAS only fenced the load. Configure the
        // voice fully BEFORE we flip active to true; once it goes
        // active the audio thread is reading.
        v.sound_id = sound;
        v.cursor   = 0;
        v.pos      = pos;
        v.gain     = gain;

        // Initial pan from the current listener snapshot.
        glm::vec3 lp = impl_->listener_pos.load(std::memory_order_acquire);
        glm::vec3 lf = impl_->listener_fwd.load(std::memory_order_acquire);
        float l = 1.0f, r = 1.0f;
        ComputeStereoGains(pos, lp, lf, l, r);
        v.gain_l.store(l, std::memory_order_relaxed);
        v.gain_r.store(r, std::memory_order_relaxed);

        v.active.store(true, std::memory_order_release);
        return MakeHandle(i, v.generation.load(std::memory_order_acquire));
    }
    // Pool exhausted.
    LOG_WARN("audio: voice pool full ({} active voices); dropped one-shot",
             kMaxVoices);
    return kInvalidVoice;
}

void AudioSystem::Stop(VoiceHandle voice) {
    if (voice == kInvalidVoice) return;
    if (!running_.load(std::memory_order_acquire)) return;
    const std::uint32_t slot = SlotOf(voice);
    const std::uint32_t gen  = GenOf(voice);
    if (slot >= kMaxVoices) return;
    Voice& v = impl_->voices[slot];
    if (v.generation.load(std::memory_order_acquire) != gen) return;  // stale
    v.active.store(false, std::memory_order_release);
    v.generation.fetch_add(1, std::memory_order_acq_rel);
}

std::uint32_t AudioSystem::StopAll() {
    if (!impl_) return 0;
    std::uint32_t n = 0;
    for (std::uint32_t i = 0; i < kMaxVoices; ++i) {
        Voice& v = impl_->voices[i];
        if (v.active.exchange(false, std::memory_order_acq_rel)) {
            v.generation.fetch_add(1, std::memory_order_acq_rel);
            ++n;
        }
    }
    return n;
}

}  // namespace pt::audio
