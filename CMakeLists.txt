cmake_minimum_required(VERSION 3.22)
project(VkNRC)

set(CMAKE_CXX_STANDARD 20)

set(CMAKE_CXX_VISIBILITY_PRESET hidden)

if (CMAKE_SYSTEM_NAME STREQUAL Windows)
    add_definitions(-D_USE_MATH_DEFINES)
    add_definitions(-DNOMINMAX)
    add_definitions(-DWIN32_LEAN_AND_MEAN)
endif ()

if (MINGW)
    set(CMAKE_EXE_LINKER_FLAGS "-static-libgcc -static-libstdc++ -static")
endif ()

if (CMAKE_BUILD_TYPE STREQUAL "Release" AND NOT MINGW AND NOT ANDROID)
    include(CheckIPOSupported)
    check_ipo_supported(RESULT IPO_SUPPORTED)
    if (IPO_SUPPORTED)
        message("Enable IPO")
        set(CMAKE_INTERPROCEDURAL_OPTIMIZATION TRUE)
    endif ()
endif ()

add_subdirectory(dep)
add_subdirectory(shader)
# add_subdirectory(test)

set(VKNRC_RENDER_SOURCES
        src/Scene.cpp
        src/VkScene.cpp
        src/VkSceneBLAS.cpp
        src/VkSceneTLAS.cpp
        src/VkNRCState.cpp

        src/rg/NRCRenderGraph.cpp
        src/rg/PathTracerPass.cpp
        src/rg/VBufferPass.cpp
        src/rg/ScreenPass.cpp
        src/rg/NNInference.cpp
        src/rg/NNInferenceShader.cpp
        src/rg/NNTrain.cpp
        src/rg/NNGradientShader.cpp
)
if (ANDROID)
    set(NATIVE_GLUE "${CMAKE_ANDROID_NDK}/sources/android/native_app_glue")
    add_library(vknrc SHARED
            src/android/AndroidMain.cpp src/android/AndroidDevice.cpp src/android/EmulatorPreview.cpp
            src/android/AndroidFrameTools.cpp src/RenderCapture.cpp
            ${VKNRC_RENDER_SOURCES} "${NATIVE_GLUE}/android_native_app_glue.c")
    target_include_directories(vknrc PRIVATE "${NATIVE_GLUE}")
    target_link_libraries(vknrc PRIVATE dep shader android log)
    target_link_options(vknrc PRIVATE "-Wl,-u,ANativeActivity_onCreate")
    return()
endif ()

add_executable(VkNRC src/main.cpp src/RenderCapture.cpp src/Camera.cpp src/SceneCatalog.cpp ${VKNRC_RENDER_SOURCES})
target_link_libraries(VkNRC PRIVATE dep shader)
target_compile_definitions(VkNRC PRIVATE VKNRC_ASSET_DIRECTORY="${CMAKE_CURRENT_SOURCE_DIR}/assets")
set_target_properties(VkNRC PROPERTIES
        C_VISIBILITY_PRESET hidden
        CXX_VISIBILITY_PRESET hidden
        VISIBILITY_INLINES_HIDDEN ON
)

install(TARGETS VkNRC RUNTIME DESTINATION .)
install(DIRECTORY assets/ DESTINATION assets)

# Optional offscreen regression test; build explicitly with --target VkNRCRollingTest.
add_executable(VkNRCRollingTest EXCLUDE_FROM_ALL
        test/rolling_accumulation.cpp
        src/VkNRCState.cpp
        src/RenderCapture.cpp
        src/rg/ScreenPass.cpp
)
target_link_libraries(VkNRCRollingTest PRIVATE dep shader)
