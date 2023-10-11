cmake_minimum_required(VERSION 3.1 FATAL_ERROR)

project("einops" VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17")

option(ENABLE_EINOPS_TORCH_BACKEND "Enable Torch backend" ON)

if (ENABLE_EINOPS_TORCH_BACKEND)
    find_package(Torch REQUIRED)
    add_compile_definitions(EINOPS_TORCH_BACKEND)
endif()

include_directories("${PROJECT_BINARY_DIR}/include")

add_library(${PROJECT_NAME} INTERFACE)

target_compile_features(${PROJECT_NAME} INTERFACE cxx_std_17)

target_include_directories(${PROJECT_NAME} INTERFACE
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)

option(ENABLE_EINOPS_TESTING "Build einops test suite" ON)
if (ENABLE_EINOPS_TESTING)
    add_subdirectory("test")
endif()