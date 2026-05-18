// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Rajesh D'Monte
//
// Tracy include shim.
//
// Why this header exists
// ----------------------
// Tracy's official entry point is `<tracy/Tracy.hpp>`, which provides
// macros like `ZoneScoped`, `ZoneScopedN("name")`, `FrameMark`, etc.
// Those macros expand to real RAII zone objects when TRACY_ENABLE is
// defined (set publicly on pt_core when PT_ENABLE_TRACY=ON), and to
// `(void)0` otherwise.
//
// The Tracy header is heavy (it pulls in <atomic>, <thread>, the entire
// profiler client lobby), so we don't want to drag it into every TU
// just to land a couple of `ZoneScopedN` calls. This shim lets call
// sites do:
//
//     #include "../core/Tracy.h"
//     ...
//     void HotFunc() {
//         PT_ZONE_SCOPED_N("Engine::HotFunc");
//         ...
//     }
//
// and get zero overhead -- both compile time and run time -- when
// PT_ENABLE_TRACY is OFF.
//
// Naming
// ------
// Tracy's own macros are unprefixed (`ZoneScopedN`) and live in the
// global namespace. We expose `PT_ZONE_SCOPED_N(name)` / `PT_FRAME_MARK`
// instead so we (a) don't fight grep for "ZoneScoped" with the upstream
// tutorial code, and (b) get a single point of control if we ever need
// to wrap the macros with extra metadata (memory tags, GPU contexts,
// etc.) in the future.
//
// String literals only
// --------------------
// `PT_ZONE_SCOPED_N` takes a string literal. Tracy stores the pointer,
// not a copy, so dynamic strings would dangle. For dynamic names use
// `ZoneScoped` + `ZoneName(buf, len)` directly via <tracy/Tracy.hpp>.

#pragma once

#if defined(TRACY_ENABLE)
#  include <tracy/Tracy.hpp>
#  define PT_ZONE_SCOPED            ZoneScoped
#  define PT_ZONE_SCOPED_N(name)    ZoneScopedN(name)
#  define PT_FRAME_MARK             FrameMark
#  define PT_FRAME_MARK_NAMED(name) FrameMarkNamed(name)
#else
#  define PT_ZONE_SCOPED            ((void)0)
#  define PT_ZONE_SCOPED_N(name)    ((void)0)
#  define PT_FRAME_MARK             ((void)0)
#  define PT_FRAME_MARK_NAMED(name) ((void)0)
#endif
