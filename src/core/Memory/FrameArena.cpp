// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Rajesh D'Monte
#include "FrameArena.h"
#include "Memory.h"

#include <cstdint>

namespace pt::mem {

FrameArena::FrameArena() = default;

FrameArena::~FrameArena() { Shutdown(); }

void FrameArena::Init(std::size_t capacity_bytes, MemTag tag) {
    Shutdown();
    tag_ = tag;
    capacity_ = capacity_bytes;
    base_ = static_cast<std::uint8_t*>(::pt::mem::Alloc(capacity_bytes, tag));
    offset_ = 0;
    peak_   = 0;
}

void FrameArena::Shutdown() {
    if (base_ != nullptr) {
        ::pt::mem::Free(base_);
        base_ = nullptr;
    }
    capacity_ = offset_ = peak_ = 0;
}

void FrameArena::Reset() {
    if (offset_ > peak_) peak_ = offset_;
    offset_ = 0;
}

void* FrameArena::Alloc(std::size_t size, std::size_t align) {
    if (base_ == nullptr) return nullptr;
    auto cur = reinterpret_cast<std::uintptr_t>(base_) + offset_;
    auto pad = (align - (cur & (align - 1))) & (align - 1);
    // Subtraction form so a huge `size` (e.g. a count*stride multiply
    // that wrapped) can't overflow the sum, pass the check, and hand
    // back an out-of-bounds pointer.
    if (pad > capacity_ - offset_ || size > capacity_ - offset_ - pad) {
        return nullptr;
    }
    offset_ += pad;
    void* result = base_ + offset_;
    offset_ += size;
    if (offset_ > peak_) peak_ = offset_;
    return result;
}

void FrameArenaSet::Init(std::size_t per_arena_bytes, MemTag tag) {
    for (auto& a : arenas_) a.Init(per_arena_bytes, tag);
    current_ = 0;
}

void FrameArenaSet::Shutdown() {
    for (auto& a : arenas_) a.Shutdown();
}

void FrameArenaSet::Advance() {
    current_ = (current_ + 1) % kBufferCount;
    arenas_[current_].Reset();
}

}  // namespace pt::mem
