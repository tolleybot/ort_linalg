project(custom_op_library)

set(CMAKE_BUILD_TYPE Debug)

cmake_minimum_required(VERSION 3.13)

option(ONNXRUNTIME_ROOTDIR "onnxruntime root dir")

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include_directories(
        ${ONNXRUNTIME_ROOTDIR}/include/onnxruntime/core/session/
)

add_library(custom_op_library SHARED custom_op_library.cc)

# Link flags adapted from
# https://github.com/microsoft/onnxruntime/blob/3c2d52a995954cd9ce24b2831060c4326ab18e1c/cmake/onnxruntime_unittests.cmake#L1465
set(ONNXRUNTIME_CUSTOM_OP_LIB_LINK_FLAG "-Xlinker --version-script=custom_op_library.lds -Xlinker --no-undefined -Xlinker --gc-sections -z noexecstack")
set_property(TARGET custom_op_library APPEND_STRING PROPERTY LINK_FLAGS ${ONNXRUNTIME_CUSTOM_OP_LIB_LINK_FLAG})