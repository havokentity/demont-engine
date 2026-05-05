#include "Memory.h"
#include "../Log.h"

#include <mimalloc.h>

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <new>

#if defined(TRACY_ENABLE)
#  include <tracy/Tracy.hpp>
#  define PT_TRACY_ALLOC(p, s) TracyAlloc(p, s)
#  define PT_TRACY_FREE(p)     TracyFree(p)
#else
#  define PT_TRACY_ALLOC(p, s) ((void)0)
#  define PT_TRACY_FREE(p)     ((void)0)
#endif

namespace pt::mem {

namespace {

// Each allocation is laid out as: [AllocHeader][user payload].
// Header is exactly 16 bytes so the user pointer stays 16-byte aligned --
// matches __STDCPP_DEFAULT_NEW_ALIGNMENT__ on Apple Silicon.
struct alignas(16) AllocHeader {
    std::uint32_t tag;
    std::uint32_t magic;
    std::uint64_t size;
};
static_assert(sizeof(AllocHeader) == 16, "AllocHeader must be exactly 16 bytes");
static_assert(alignof(AllocHeader) == 16, "AllocHeader must be 16-byte aligned");

constexpr std::uint32_t kHeaderMagic = 0xC0FFEE42u;

struct AtomicStats {
    std::atomic<std::size_t> live{0};
    std::atomic<std::size_t> peak{0};
    std::atomic<std::size_t> allocs{0};
    std::atomic<std::size_t> frees{0};
};

// Zero-initialized at static-storage time -- safe to touch from allocations
// that occur during early static init (before main).
AtomicStats g_stats[kMemTagCount];

thread_local MemTag t_current_tag = MemTag::Misc;

inline void RecordAlloc(MemTag tag, std::size_t bytes) {
    auto& s = g_stats[static_cast<int>(tag)];
    auto live = s.live.fetch_add(bytes, std::memory_order_relaxed) + bytes;
    s.allocs.fetch_add(1, std::memory_order_relaxed);
    auto cur = s.peak.load(std::memory_order_relaxed);
    while (live > cur &&
           !s.peak.compare_exchange_weak(cur, live, std::memory_order_relaxed)) { }
}

inline void RecordFree(MemTag tag, std::size_t bytes) {
    auto& s = g_stats[static_cast<int>(tag)];
    s.live.fetch_sub(bytes, std::memory_order_relaxed);
    s.frees.fetch_add(1, std::memory_order_relaxed);
}

}  // namespace

void Init() {
    // mimalloc self-initializes lazily on first allocation.
}

void Shutdown() {
    // Optional: a final report would belong here; left to caller for now.
}

void* Alloc(std::size_t size, MemTag tag) {
    const std::size_t total = size + sizeof(AllocHeader);
    void* raw = mi_malloc_aligned(total, alignof(AllocHeader));
    if (raw == nullptr) {
        return nullptr;
    }
    auto* hdr  = static_cast<AllocHeader*>(raw);
    hdr->tag   = static_cast<std::uint32_t>(tag);
    hdr->magic = kHeaderMagic;
    hdr->size  = static_cast<std::uint64_t>(size);
    void* user = static_cast<char*>(raw) + sizeof(AllocHeader);
    RecordAlloc(tag, size);
    PT_TRACY_ALLOC(user, size);
    return user;
}

void Free(void* ptr) {
    if (ptr == nullptr) return;
    auto* hdr = reinterpret_cast<AllocHeader*>(
        static_cast<char*>(ptr) - sizeof(AllocHeader));
    if (hdr->magic != kHeaderMagic) {
        // Foreign pointer (allocated outside our path) -- safest is to leak it
        // rather than corrupt mimalloc's bookkeeping. In practice this should
        // never fire because all `operator new` calls route through Alloc.
        return;
    }
    const auto tag  = static_cast<MemTag>(hdr->tag);
    const auto size = static_cast<std::size_t>(hdr->size);
    hdr->magic = 0;  // poison so a double-free is detectable
    PT_TRACY_FREE(ptr);
    RecordFree(tag, size);
    mi_free(hdr);
}

MemTag CurrentTag() { return t_current_tag; }

TagScope::TagScope(MemTag t) : prev_(t_current_tag) { t_current_tag = t; }
TagScope::~TagScope() { t_current_tag = prev_; }

void GetReport(TagStats out[kMemTagCount]) {
    for (int i = 0; i < kMemTagCount; ++i) {
        out[i].live_bytes  = g_stats[i].live.load(std::memory_order_relaxed);
        out[i].peak_bytes  = g_stats[i].peak.load(std::memory_order_relaxed);
        out[i].alloc_count = g_stats[i].allocs.load(std::memory_order_relaxed);
        out[i].free_count  = g_stats[i].frees.load(std::memory_order_relaxed);
    }
}

void PrintReport() {
    TagStats s[kMemTagCount];
    GetReport(s);
    LOG_INFO("---- mem_report ----");
    LOG_INFO("{:<12} {:>14} {:>14} {:>10} {:>10}",
             "tag", "live(B)", "peak(B)", "allocs", "frees");
    std::size_t total_live = 0;
    std::size_t total_peak = 0;
    for (int i = 0; i < kMemTagCount; ++i) {
        LOG_INFO("{:<12} {:>14} {:>14} {:>10} {:>10}",
                 MemTagName(static_cast<MemTag>(i)),
                 s[i].live_bytes, s[i].peak_bytes,
                 s[i].alloc_count, s[i].free_count);
        total_live += s[i].live_bytes;
        total_peak += s[i].peak_bytes;
    }
    LOG_INFO("{:<12} {:>14} {:>14}", "TOTAL", total_live, total_peak);
}

}  // namespace pt::mem

// --- Global new/delete overrides ------------------------------------------
// Untagged allocations land here. They take whatever the thread-local tag
// currently is (default MemTag::Misc), so a TagScope wrapping a chunk of
// code attributes any STL allocations inside it to the right bucket.
//
// We intentionally do NOT implement the over-aligned variants (taking
// std::align_val_t) -- nothing in the project currently uses them. If a type
// with alignof > 16 appears later, add the matching overrides then.

void* operator new(std::size_t size) {
    void* p = pt::mem::Alloc(size, pt::mem::CurrentTag());
    if (p == nullptr) {
        // -fno-exceptions: cannot throw bad_alloc; abort instead.
        std::abort();
    }
    return p;
}

void* operator new[](std::size_t size) {
    return ::operator new(size);
}

void* operator new(std::size_t size, const std::nothrow_t&) noexcept {
    return pt::mem::Alloc(size, pt::mem::CurrentTag());
}

void* operator new[](std::size_t size, const std::nothrow_t&) noexcept {
    return pt::mem::Alloc(size, pt::mem::CurrentTag());
}

void operator delete(void* p) noexcept                  { pt::mem::Free(p); }
void operator delete[](void* p) noexcept                { pt::mem::Free(p); }
void operator delete(void* p, std::size_t) noexcept     { pt::mem::Free(p); }
void operator delete[](void* p, std::size_t) noexcept   { pt::mem::Free(p); }
void operator delete(void* p, const std::nothrow_t&) noexcept   { pt::mem::Free(p); }
void operator delete[](void* p, const std::nothrow_t&) noexcept { pt::mem::Free(p); }
