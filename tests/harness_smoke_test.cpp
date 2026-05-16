// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Rajesh D'Monte
//
// First test in the doctest harness. Trivial assertions only -- this
// exists to verify the build system produces a runnable ctest binary,
// the doctest dep is wired correctly, and `ctest` from the build root
// discovers + executes it.
//
// Real tests with actual project code-paths get added module-by-module
// in Phase 3 of the testing rollout (#47). This file stays as a
// canary: if doctest's main entry point or the CMake pt_add_test
// helper regresses, this test fails first and the diagnostic is clean
// (no project code involved).

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

TEST_CASE("harness: doctest framework is callable") {
    CHECK(1 + 1 == 2);
    CHECK_FALSE(false);
}

TEST_CASE("harness: SUBCASE isolation works") {
    int x = 0;

    SUBCASE("increment") {
        x++;
        CHECK(x == 1);
    }

    SUBCASE("decrement") {
        // Each SUBCASE runs against a fresh TEST_CASE setup, so x is 0
        // here (not 1 from the increment SUBCASE). This is the doctest
        // execution model that makes test isolation cheap.
        x--;
        CHECK(x == -1);
    }
}

TEST_CASE("harness: REQUIRE halts on first failure") {
    // CHECK records the failure and continues; REQUIRE halts the test
    // case. Used for preconditions where subsequent assertions would
    // be meaningless if the precondition fails.
    int* p = nullptr;
    int  v = 42;
    p = &v;
    REQUIRE(p != nullptr);
    CHECK(*p == 42);
}
