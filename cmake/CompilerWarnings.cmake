# Apply orthodox-C++23 compile/link flags to a single target.
# Per principles doc §08: -fno-exceptions, -fno-rtti, hard warnings.
# ASan is opt-in via PT_ENABLE_SANITIZERS (mimalloc + ASan needs MI_TRACK_ASAN).

function(pt_set_compiler_options target)
    target_compile_features(${target} PUBLIC cxx_std_23)

    target_compile_options(${target} PRIVATE
        -fno-exceptions
        -fno-rtti
        -fvisibility=hidden
        -fvisibility-inlines-hidden
        -Wall -Wextra -Wpedantic
        -Werror=return-type
        -Wshadow
        -Wno-unknown-pragmas
        -Wno-unused-parameter
    )

    target_compile_options(${target} PRIVATE
        $<$<CONFIG:Debug>:-O0>
        $<$<CONFIG:Debug>:-g>
        $<$<CONFIG:Release>:-O3>
        $<$<CONFIG:Release>:-ffast-math>
        $<$<CONFIG:Release>:-DNDEBUG>
    )

    if(PT_ENABLE_SANITIZERS)
        target_compile_options(${target} PRIVATE
            $<$<CONFIG:Debug>:-fsanitize=address,undefined>
            $<$<CONFIG:Debug>:-fno-omit-frame-pointer>
        )
        target_link_options(${target} PRIVATE
            $<$<CONFIG:Debug>:-fsanitize=address,undefined>
        )
    endif()
endfunction()
