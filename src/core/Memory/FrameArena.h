// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Rajesh D'Monte
#pragma once

#include "MemTag.h"

#include <atomic>
#include <cstddef>
#include <cstdint>

namespace pt::mem {

// Simple bump allocator backed by one mimalloc block. Not thread safe by
// default; the FrameArenaSet below gives each frame index its own arena so
// callers don't need a lock as long as they only touch the current arena.
class FrameArena {
public:
    FrameArena();
    ~FrameArena();
    FrameArena(const FrameArena&) = delete;
    FrameArena& operator=(const FrameArena&) = delete;

    void Init(std::size_t capacity_bytes, MemTag tag = MemTag::Render);
    void Shutdown();
    void Reset();

    // Returns nullptr if the arena is full.
    void* Alloc(std::size_t size, std::size_t align = 16);

    template <typename T, typename... Args>
    T* New(Args&&... args) {
        void* p = Alloc(sizeof(T), alignof(T));
        if (!p) return nullptr;
        return ::new (p) T(static_cast<Args&&>(args)...);
    }

    std::size_t Used()     const noexcept { return offset_; }
    std::size_t Capacity() const noexcept { return capacity_; }
    std::size_t Peak()     const noexcept { return peak_; }

private:
    std::uint8_t* base_     = nullptr;
    std::size_t   capacity_ = 0;
    std::size_t   offset_   = 0;
    std::size_t   peak_     = 0;
    MemTag        tag_      = MemTag::Render;
};

// Triple-buffered set so consecutive frames can each have their own arena
// without stomping on the previous frame while it's still in flight on the
// GPU. Currently only the active arena is exposed; rotation is manual.
class FrameArenaSet {
public:
    static constexpr int kBufferCount = 3;

    void Init(std::size_t per_arena_bytes, MemTag tag = MemTag::Render);
    void Shutdown();

    void Advance();  // call once per frame, just before producing draw work
    FrameArena& Current() noexcept { return arenas_[current_]; }

private:
    FrameArena arenas_[kBufferCount];
    int        current_ = 0;
};

}  // namespace pt::mem
