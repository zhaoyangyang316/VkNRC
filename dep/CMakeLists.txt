cmake_minimum_required(VERSION 3.15)
project(SE3306_Final)

add_subdirectory(MyVK)
add_subdirectory(glm)
add_subdirectory(stb_image)
add_subdirectory(tinyobjloader)
add_subdirectory(spdlog)
add_subdirectory(half)

add_library(dep INTERFACE)
target_link_libraries(dep INTERFACE
        myvk::vulkan myvk::rg
        glm
        stb_image
        tinyobjloader
        spdlog::spdlog
        half::half
)
if (NOT ANDROID)
    target_link_libraries(dep INTERFACE myvk::glfw myvk::imgui)
endif ()
