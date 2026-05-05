#pragma once

#include "Memory.h"
#include "MemTag.h"

#include <cstddef>
#include <cstdint>
#include <new>
#include <utility>

namespace pt::mem {

// O(1) free-list pool of fixed-size T objects.  Backed by one tagged heap
// block per slab; new slabs are allocated on demand when the free list is
// empty.  Objects are constructed in place via `Acquire()` and torn down via
// `Release()`.
template <typename T>
class PoolAllocator {
public:
    explicit PoolAllocator(std::size_t per_slab = 256, MemTag tag = MemTag::Misc)
        : per_slab_(per_slab), tag_(tag) {}

    PoolAllocator(const PoolAllocator&) = delete;
    PoolAllocator& operator=(const PoolAllocator&) = delete;

    ~PoolAllocator() { ReleaseAll(); }

    template <typename... Args>
    T* Acquire(Args&&... args) {
        if (free_list_ == nullptr) Grow();
        Slot* slot = free_list_;
        free_list_ = slot->next;
        ++live_;
        return ::new (slot->storage) T(std::forward<Args>(args)...);
    }

    void Release(T* ptr) {
        if (ptr == nullptr) return;
        ptr->~T();
        auto* slot = reinterpret_cast<Slot*>(ptr);
        slot->next = free_list_;
        free_list_ = slot;
        --live_;
    }

    std::size_t Live()    const noexcept { return live_; }
    std::size_t Slabs()   const noexcept { return slab_count_; }

    void ReleaseAll() {
        Slab* s = slabs_;
        while (s != nullptr) {
            Slab* next = s->next;
            Free(s);
            s = next;
        }
        slabs_       = nullptr;
        free_list_   = nullptr;
        live_        = 0;
        slab_count_  = 0;
    }

private:
    union Slot {
        Slot* next;
        alignas(T) unsigned char storage[sizeof(T)];
    };

    struct Slab {
        Slab* next;
        Slot  slots[1];  // trailing array, sized at allocation
    };

    void Grow() {
        std::size_t bytes = sizeof(Slab) + sizeof(Slot) * (per_slab_ - 1);
        auto* slab = static_cast<Slab*>(Alloc(bytes, tag_));
        if (slab == nullptr) return;
        slab->next = slabs_;
        slabs_ = slab;
        ++slab_count_;
        for (std::size_t i = 0; i < per_slab_; ++i) {
            slab->slots[i].next = free_list_;
            free_list_ = &slab->slots[i];
        }
    }

    Slab*       slabs_       = nullptr;
    Slot*       free_list_   = nullptr;
    std::size_t live_        = 0;
    std::size_t slab_count_  = 0;
    std::size_t per_slab_    = 256;
    MemTag      tag_         = MemTag::Misc;
};

}  // namespace pt::mem
