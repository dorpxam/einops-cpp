cmake_minimum_required(VERSION 3.1 FATAL_ERROR)

add_executable(einops_test main.cpp)

include_directories(${PROJECT_SOURCE_DIR}/test/include ${PROJECT_SOURCE_DIR}/include)

target_include_directories(einops_test INTERFACE ${PROJECT_SOURCE_DIR}/include)

if (ENABLE_EINOPS_TORCH_BACKEND)
    target_link_libraries(einops_test ${TORCH_LIBRARIES})
endif()

#if (EINOPS_TENSORFLOW_BACKEND)
#    target_link_libraries(einops_test ${TENSORFLOW_LIBRARIES})
#endif()