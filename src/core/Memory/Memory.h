#pragma once

#include "MemTag.h"

#include <cstddef>

namespace pt::mem {

// Initialize the memory subsystem. Idempotent; safe to call after static
// initialization has already pulled allocations through `operator new`.
void Init();
void Shutdown();

// Tagged alloc/free. All of our memory routes through here -- including the
// global `operator new` overrides defined in Memory.cpp (those use the
// thread-local current tag, default `MemTag::Misc`).
void* Alloc(std::size_t size, MemTag tag);
void  Free(void* ptr);

// Returns the top of the thread-local tag stack. Defaults to MemTag::Misc.
MemTag CurrentTag();

// RAII helper: any allocation made inside the scope inherits `t`.
class TagScope {
public:
    explicit TagScope(MemTag t);
    ~TagScope();
    TagScope(const TagScope&) = delete;
    TagScope& operator=(const TagScope&) = delete;
    TagScope(TagScope&&) = delete;
    TagScope& operator=(TagScope&&) = delete;
private:
    MemTag prev_;
};

struct TagStats {
    std::size_t live_bytes  = 0;
    std::size_t peak_bytes  = 0;
    std::size_t alloc_count = 0;
    std::size_t free_count  = 0;
};

void GetReport(TagStats out[kMemTagCount]);
void PrintReport();

}  // namespace pt::mem
