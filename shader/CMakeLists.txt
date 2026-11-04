cmake_minimum_required(VERSION 3.15)
project(VkNRC_Shaders LANGUAGES NONE)

find_program(GLSLC "glslc" REQUIRED NO_CMAKE_FIND_ROOT_PATH)

add_library(shader INTERFACE)
if (ANDROID)
    set(SHADER_INCLUDE_DIR "${CMAKE_CURRENT_BINARY_DIR}/include")
    set(SHADER_PLATFORM_FLAGS "-DVKNRC_ANDROID=1")
    if (ANDROID_ABI STREQUAL "x86_64")
        set(VKNRC_EMULATOR TRUE)
        list(APPEND SHADER_PLATFORM_FLAGS "-DNRC_EMULATOR=1")
    endif ()
else ()
    set(SHADER_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/include")
endif ()
file(MAKE_DIRECTORY "${SHADER_INCLUDE_DIR}/shader")
target_include_directories(shader INTERFACE "${SHADER_INCLUDE_DIR}")

set(SHADER_HEADER
        Constant.glsl
        RNG.glsl
        Sample.glsl
        LambertBRDF.glsl
        CookTorranceBRDF.glsl
        NRCRecord.glsl
        Scene.glsl
        TraceScene.glsl
        NN_nv.glsl
)
if (ANDROID)
    list(APPEND SHADER_HEADER NN_khr.glsl)
endif ()
list(TRANSFORM SHADER_HEADER PREPEND ${CMAKE_CURRENT_SOURCE_DIR}/src/)

function(add_shader INPUT_FILE OUTPUT_FILE OPTION)
    set(output "${SHADER_INCLUDE_DIR}/shader/${OUTPUT_FILE}")
    add_custom_command(OUTPUT "${output}"
            COMMAND "${GLSLC}" src/${INPUT_FILE} -o "${output}" -mfmt=num --target-env=vulkan1.3 ${SHADER_PLATFORM_FLAGS} ${OPTION}
            WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
            DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/src/${INPUT_FILE} ${SHADER_HEADER} VERBATIM
    )
    add_custom_target(shader_${OUTPUT_FILE} ALL DEPENDS "${output}")
    add_dependencies(shader shader_${OUTPUT_FILE})
endfunction()

add_shader(vbuffer.frag vbuffer.frag.u32 "-O")
add_shader(vbuffer.vert vbuffer.vert.u32 "-O")
if (VKNRC_EMULATOR)
    add_shader(path_tracer.comp path_tracer_emulator.comp.u32 "-O")
else ()
    add_shader(path_tracer.comp path_tracer.comp.u32 "-O")
endif ()
add_shader(screen.frag screen.frag.u32 "-O")
add_shader(screen_paused.frag screen_paused.frag.u32 "-O")
add_shader(screen.vert screen.vert.u32 "-O")
add_shader(nrc_indirect.comp nrc_indirect.comp.u32 "-O")
add_shader(nrc_train_prepare.comp nrc_train_prepare.comp.u32 "-O")
add_shader(nrc_optimize.comp nrc_optimize.comp.u32 "-O")
add_shader(nrc_optimize.comp nrc_optimize_use.comp.u32 "-O;-DWRITE_USE_WEIGHTS")
if (VKNRC_EMULATOR)
    foreach(stage inference gradient)
        add_shader(nrc_${stage}.comp nrc_${stage}_scalar.comp.u32 "-O;-DNRC_KHR=1;-DNRC_SCALAR=1")
    endforeach()
elseif (ANDROID)
    foreach(stage inference gradient)
        add_shader(nrc_${stage}.comp nrc_${stage}_khr_f16.comp.u32 "-DNRC_KHR=1")
        add_shader(nrc_${stage}.comp nrc_${stage}_khr_f32.comp.u32 "-DNRC_KHR=1;-DNRC_KHR_FP32=1")
    endforeach()
else ()
    foreach(stage inference gradient)
        add_shader(nrc_${stage}.comp nrc_${stage}_16.comp.u32 "-O;-DSUBGROUP_SIZE=16")
        add_shader(nrc_${stage}.comp nrc_${stage}_32.comp.u32 "-O;-DSUBGROUP_SIZE=32")
    endforeach()
endif ()
