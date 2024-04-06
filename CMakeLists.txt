# add_ops_compile_options(ALL OPTIONS -DASCENDC_DUMP)
cmake_minimum_required(VERSION 3.16)
set(CMAKE_SYSTEM_NAME Linux)
set(CMAKE_CXX_STANDARD 17)

set(CCE_CMAKE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake/Modules)

message(STATUS "The value of MY_VARIABLE is: ${CMAKE_CURRENT_SOURCE_DIR}")

list(APPEND CMAKE_MODULE_PATH ${CCE_CMAKE_PATH})
project(kernel_samples LANGUAGES CCE CXX)


# add_subdirectory(cmake/cpu)
add_subdirectory(cmake/npu)
