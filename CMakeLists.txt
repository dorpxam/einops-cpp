cmake_minimum_required(VERSION 3.1 FATAL_ERROR)

project("einops" VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++20")

option(ENABLE_EINOPS_TORCH_BACKEND "Enable Torch backend" ON)
option(ENABLE_EINOPS_TENSORFLOW_BACKEND "Enable Tensorflow backend" OFF)

if (ENABLE_EINOPS_TORCH_BACKEND)
    find_package(Torch REQUIRED)
    add_compile_definitions(EINOPS_TORCH_BACKEND)
endif()

if (ENABLE_EINOPS_TENSORFLOW_BACKEND)
    add_compile_definitions(EINOPS_TENSORFLOW_BACKEND)
endif()

include_directories("${PROJECT_BINARY_DIR}/include")

add_library(${PROJECT_NAME} INTERFACE)

target_compile_features(${PROJECT_NAME} INTERFACE cxx_std_20)

target_include_directories(${PROJECT_NAME} INTERFACE
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)

#option(ENABLE_EINOPS_TESTING "Build einops test suite" ON)
#if (ENABLE_EINOPS_TESTING)
    add_subdirectory("test")
#endif()