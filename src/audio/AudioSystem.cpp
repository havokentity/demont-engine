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
//   square, etc.) we don't use. Documented in miniaudio.h's header
//   compile-time-options table -- not a no-op.
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
#include <array>
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
//
// Stored in a vector<unique_ptr<Sound>>: the unique_ptr layer is what
// gives us address-stability across LoadSound calls. push_back on the
// outer vector may relocate the unique_ptr handles, but the audio
// callback dereferences via the pointer's value (which is stable across
// the relocation, since unique_ptr stores a raw heap pointer). Combined
// with Impl::sound_count -- an atomic snapshot of the valid index range
// the callback is allowed to see -- this gives the callback a safe
// read-side without a lock on the hot path. See the LoadSound +
// DataCallback comments for the publication discipline.
struct Sound {
    std::vector<float> frames;     // interleaved kChannels samples
    std::uint64_t      frame_count = 0;  // == frames.size() / kChannels
    std::string        path;
};

// A single voice slot in the pool. The audio callback reads `active`,
// `sound_id`, `cursor`, `pos`, `gain` atomically; the main thread
// writes them via PlaySound / Stop.
//
// Threading model:
//   - The audio callback is the SOLE writer of `cursor` -- the main
//     thread never touches it after PlaySound's initial zero-init
//     (which happens while `active==false`, so the callback is not
//     reading the slot at that point).
//   - The audio callback is the SOLE entity that flips `active` from
//     true -> false at voice-natural-end (cursor reached frame_count)
//     and the SOLE entity that bumps `generation` at that transition.
//     Stop() does NOT mutate `active` directly; instead it bumps
//     `stop_request` and the callback observes it on its next pass and
//     transitions the slot to free (along with the same generation
//     bump it does on natural completion). This eliminates the
//     "main thread flips active=false while callback is mid-read"
//     race that the simpler design had.
//   - PlaySound is the SOLE writer of `sound_id`, `pos`, `gain`, and
//     publishes them BEFORE flipping `active` from false -> true via a
//     release store. The audio callback's first load of `active` on
//     each pass is an acquire load, so observing `active==true` also
//     observes the published configuration. After publication the
//     callback alone owns the slot until it transitions it back to
//     free, so no further main-thread mutation is permitted.
struct Voice {
    std::atomic<bool>          active   {false};
    std::atomic<std::uint32_t> generation {1};   // 0 reserved for kInvalidVoice
    // Mailbox used by Stop() to ask the callback to free the slot at
    // its next pass. The callback consumes (reset to 0) when it acts.
    // Avoids the race where Stop()-on-main and callback-end-of-voice
    // both try to flip `active` and bump `generation`.
    std::atomic<std::uint32_t> stop_request {0};
    // sound_id / pos / gain are published BY PlaySound BEFORE the
    // active.store(true, release) handshake. The callback reads them
    // AFTER its active.load(acquire) sees true. Both ends are stable
    // for the duration of the voice (we never mutate them while
    // active==true).
    SoundHandle                sound_id = kInvalidSound;
    glm::vec3                  pos      {0, 0, 0};
    float                      gain     = 1.0f;
    // cursor is callback-only after PlaySound's initial zero. The main
    // thread does not read it.
    std::uint64_t              cursor   = 0;     // frame index into sound->frames
    // Cached left/right gains computed on PlaySound. Tick() may also
    // update these while the voice is active; both ends use atomic
    // load/store so the callback never observes a torn value.
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

    // Sound bank. Held by unique_ptr so push_back-induced reallocation
    // of the outer vector doesn't invalidate the Sound* the audio
    // callback dereferences. Index 0 is the kInvalidSound sentinel
    // (never used); entry at index N is the sound with handle N.
    // Grow-only.
    //
    // Publication discipline for cross-thread reads:
    //   - LoadSound builds a fully-decoded Sound under sounds_mutex,
    //     stores the unique_ptr into the bank, then atomically
    //     publishes `sound_count` (release) to make the new entry
    //     visible to the audio callback.
    //   - The audio callback loads `sound_count` (acquire) once per
    //     pass and only reads sounds[0..sound_count-1]. That gives
    //     it a stable view of the bank for the entire callback.
    //
    // FIXED array, not a vector: the callback dereferences
    // `sounds[sid]`, i.e. it reads the POINTER ARRAY itself. A
    // vector's push_back past its reserve() headroom reallocates that
    // array and frees the old one -- a use-after-free on the real-time
    // audio thread the moment the bank outgrew 256 distinct sounds.
    // unique_ptr boxing kept the *pointees* stable but never the
    // backing store. Loads past capacity are refused with a warning.
    static constexpr std::uint32_t kMaxSounds = 256;
    std::array<std::unique_ptr<Sound>, kMaxSounds> sounds {};
    std::uint32_t                       sound_size = 0;  // guarded by sounds_mutex
    std::atomic<std::uint32_t>          sound_count {0};
    std::unordered_map<std::string,
                       SoundHandle>     path_to_handle;
    std::mutex                          sounds_mutex;   // guards LoadSound bank-side only

    // Atomic listener snapshot consumed by callers of PlaySound (the
    // callback itself does NOT load these -- it uses the precomputed
    // gain_l/gain_r atomics on each voice). We store xyz as three
    // independent atomic<float>s rather than std::atomic<glm::vec3>
    // because 12-byte atomics are not lock-free on most platforms,
    // which would violate the header's "no locks" promise.
    std::atomic<float> listener_pos_x {0.0f};
    std::atomic<float> listener_pos_y {1.5f};
    std::atomic<float> listener_pos_z {4.0f};
    std::atomic<float> listener_fwd_x {0.0f};
    std::atomic<float> listener_fwd_y {0.0f};
    std::atomic<float> listener_fwd_z {-1.0f};
    static_assert(std::atomic<float>::is_always_lock_free,
                  "audio: atomic<float> must be lock-free for listener publication");
    static_assert(std::atomic<bool>::is_always_lock_free,
                  "audio: atomic<bool> must be lock-free for voice publication");
    static_assert(std::atomic<std::uint32_t>::is_always_lock_free,
                  "audio: atomic<uint32_t> must be lock-free for generation counter");

    Impl() {
        // Slot 0 is the kInvalidSound sentinel.
        sounds[0] = std::make_unique<Sound>();
        sounds[0]->path = "<invalid>";
        sound_size = 1;
        sound_count.store(sound_size, std::memory_order_release);
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

    // Snapshot the published sound-bank size ONCE per callback. The
    // pairing release store happens in LoadSound after the new entry
    // is fully constructed; that gives us a safe upper bound on which
    // indices into impl->sounds we may dereference for the rest of
    // this callback.
    const std::uint32_t sound_count =
        impl->sound_count.load(std::memory_order_acquire);

    // Mix every active voice into the output buffer.
    for (std::uint32_t i = 0; i < AudioSystem::kMaxVoices; ++i) {
        Voice& v = impl->voices[i];
        if (!v.active.load(std::memory_order_acquire)) continue;

        // Honour an asynchronous Stop() request before doing any work
        // on this voice. The main thread cannot flip `active` itself
        // (it would race with the callback's writes to cursor / the
        // free transition below); instead it bumps stop_request and we
        // drain it here. Doing the drain at the TOP of the per-voice
        // pass keeps the slot's transition-to-free atomic with respect
        // to PlaySound's slot-reclaim CAS (a slot we just freed here
        // is safe for PlaySound to immediately reuse).
        if (v.stop_request.exchange(0, std::memory_order_acq_rel) != 0u) {
            v.active.store(false, std::memory_order_release);
            v.generation.fetch_add(1, std::memory_order_acq_rel);
            continue;
        }

        // sid / pos / gain were published BY PlaySound BEFORE its
        // active.store(true, release). Our active.load(acquire) above
        // is the synchronizing read, so these are stable for the
        // lifetime of the voice.
        const SoundHandle sid = v.sound_id;
        if (sid == kInvalidSound || sid >= sound_count) {
            // Defensive: shouldn't happen if PlaySound validated.
            v.active.store(false, std::memory_order_release);
            v.generation.fetch_add(1, std::memory_order_acq_rel);
            continue;
        }
        const Sound& s = *impl->sounds[sid];
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
    // before we tear voice state down. After ma_device_uninit returns,
    // the audio thread is guaranteed not to be inside DataCallback any
    // longer, so we can directly clear voice state without going
    // through the stop_request mailbox.
    ma_device_uninit(&impl_->device);

    for (std::uint32_t i = 0; i < kMaxVoices; ++i) {
        Voice& v = impl_->voices[i];
        if (v.active.exchange(false, std::memory_order_acq_rel)) {
            v.generation.fetch_add(1, std::memory_order_acq_rel);
        }
        v.stop_request.store(0, std::memory_order_relaxed);
    }
    impl_->sound_count.store(0, std::memory_order_release);
    for (auto& s : impl_->sounds) s.reset();
    impl_->path_to_handle.clear();
    // Re-seed the invalid sentinel so a subsequent Init/LoadSound cycle
    // still treats handle 0 as bogus.
    impl_->sounds[0] = std::make_unique<Sound>();
    impl_->sounds[0]->path = "<invalid>";
    impl_->sound_size = 1;
    impl_->sound_count.store(impl_->sound_size, std::memory_order_release);

    LOG_INFO("audio: device closed");
}

void AudioSystem::Tick(const glm::vec3& camera_pos, const glm::vec3& camera_fwd) {
    if (!running_.load(std::memory_order_acquire)) return;
    impl_->listener_pos_x.store(camera_pos.x, std::memory_order_relaxed);
    impl_->listener_pos_y.store(camera_pos.y, std::memory_order_relaxed);
    impl_->listener_pos_z.store(camera_pos.z, std::memory_order_release);
    impl_->listener_fwd_x.store(camera_fwd.x, std::memory_order_relaxed);
    impl_->listener_fwd_y.store(camera_fwd.y, std::memory_order_relaxed);
    impl_->listener_fwd_z.store(camera_fwd.z, std::memory_order_release);
    // Per-voice gain re-computation against the current listener so a
    // moving camera attenuates / pans an in-flight one-shot correctly.
    // v.pos is published by PlaySound BEFORE active=true, and we only
    // observe slots that are currently active. The callback never
    // mutates v.pos. (Source pos is fixed at PlaySound time -- see
    // Voice doc comment.)
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
    // LoadSound is intentionally callable even when the device is NOT
    // running -- it only touches the in-memory sound bank and
    // miniaudio's file-decode path, neither of which depend on an open
    // playback device. This lets headless / CI builds and tools
    // pre-load assets before Init(), and PlaySound's `running_` guard
    // still correctly drops playback attempts on a closed device.

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
        if (impl_->sound_size >= Impl::kMaxSounds) {
            LOG_WARN("audio: sound bank full ({} distinct sounds); "
                     "'{}' not loaded", Impl::kMaxSounds, path_s);
            return kInvalidSound;
        }
        h = static_cast<SoundHandle>(impl_->sound_size);
        impl_->sounds[h] = std::make_unique<Sound>(std::move(s));
        impl_->path_to_handle.emplace(path_s, h);
        // Publish the new size to the audio callback with release
        // semantics. The unique_ptr pointee was fully constructed and
        // the slot written before this store, so a callback observing
        // sound_count > h also sees the published Sound contents.
        ++impl_->sound_size;
        impl_->sound_count.store(impl_->sound_size, std::memory_order_release);
    }
    LOG_INFO("audio: loaded '{}' ({} frames, {:.2f}s)",
             path_s, frame_count,
             double(frame_count) / double(kSampleRate));
    return h;
}

VoiceHandle AudioSystem::PlaySound(SoundHandle sound, const glm::vec3& pos, float gain) {
    // PlaySound is documented as called from a single thread (the main
    // engine thread). Validating the sound handle requires a quick
    // peek into the bank; we use the atomic sound_count snapshot the
    // callback also reads -- no mutex needed, and the vector entry at
    // `sound < sound_count` is guaranteed published because LoadSound
    // released sound_count AFTER its push_back.
    if (!running_.load(std::memory_order_acquire)) return kInvalidVoice;
    if (sound == kInvalidSound) return kInvalidVoice;
    const std::uint32_t known = impl_->sound_count.load(std::memory_order_acquire);
    if (sound >= known) return kInvalidVoice;
    if (impl_->sounds[sound]->frame_count == 0) return kInvalidVoice;

    // Find an inactive slot. Linear scan is fine at kMaxVoices=64.
    // We do a simple acquire load to check `active`; PlaySound is the
    // SOLE caller that flips active from false -> true and is
    // documented as single-threaded, so no CAS is required to claim
    // the slot. (If a future commit wants multi-threaded PlaySound,
    // restore the CAS to true-with-a-configuring-sentinel pattern.)
    for (std::uint32_t i = 0; i < kMaxVoices; ++i) {
        Voice& v = impl_->voices[i];
        if (v.active.load(std::memory_order_acquire)) continue;
        // Snapshot the current generation BEFORE we publish active=true.
        // After active=true the audio thread may complete the voice and
        // bump generation in the same callback pass; reading generation
        // here gives us the value the handle should encode (the slot's
        // generation AT the moment we claimed it). Stop(handle) later
        // does the same compare-against-original gen check.
        const std::uint32_t handle_gen =
            v.generation.load(std::memory_order_acquire);

        // Configure the voice fully BEFORE we flip active to true; once
        // it goes active the audio thread is reading.
        v.sound_id     = sound;
        v.cursor       = 0;
        v.pos          = pos;
        v.gain         = gain;
        v.stop_request.store(0, std::memory_order_relaxed);

        // Initial pan from the current listener snapshot. Atomic loads
        // of the three float components -- not torn (each load is
        // atomic). A brief inconsistency between x/y/z components if
        // Tick() is mid-update is tolerable: the next Tick() will
        // recompute gain_l/gain_r anyway and a single-callback bad pan
        // is inaudible.
        glm::vec3 lp{ impl_->listener_pos_x.load(std::memory_order_acquire),
                      impl_->listener_pos_y.load(std::memory_order_acquire),
                      impl_->listener_pos_z.load(std::memory_order_acquire) };
        glm::vec3 lf{ impl_->listener_fwd_x.load(std::memory_order_acquire),
                      impl_->listener_fwd_y.load(std::memory_order_acquire),
                      impl_->listener_fwd_z.load(std::memory_order_acquire) };
        float l = 1.0f, r = 1.0f;
        ComputeStereoGains(pos, lp, lf, l, r);
        v.gain_l.store(l, std::memory_order_relaxed);
        v.gain_r.store(r, std::memory_order_relaxed);

        v.active.store(true, std::memory_order_release);
        return MakeHandle(i, handle_gen);
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
    // Ask the callback to free the slot on its next pass rather than
    // flipping `active` ourselves. Direct flipping would race with the
    // callback's in-flight read of v.cursor (and would race with
    // PlaySound's reclaim of the same slot in the same callback pass,
    // since once `active==false` PlaySound treats it as free).
    v.stop_request.store(1, std::memory_order_release);
}

std::uint32_t AudioSystem::StopAll() {
    if (!impl_) return 0;
    std::uint32_t n = 0;
    for (std::uint32_t i = 0; i < kMaxVoices; ++i) {
        Voice& v = impl_->voices[i];
        if (v.active.load(std::memory_order_acquire)) {
            v.stop_request.store(1, std::memory_order_release);
            ++n;
        }
    }
    return n;
}

}  // namespace pt::audio
