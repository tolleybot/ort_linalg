project(custom_op_library)

set(CMAKE_BUILD_TYPE Debug)

cmake_minimum_required(VERSION 3.13)

option(ONNXRUNTIME_ROOTDIR "onnxruntime root dir")

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)


# Onnx runtime uses MS GSL
# Can't seem to get this to find include dir properly
# find_package(Microsoft.GSL CONFIG REQUIRED)
# target_link_libraries(foobar PRIVATE Microsoft.GSL::GSL) # MS GSL linkage

# To find and add the vcpkg include dir
# Look for GSL and add the parent
find_path(GSL_INCLUDE_DIR NAMES gsl_assert PATH_SUFFIXES gsl)
include_directories(${GSL_INCLUDE_DIR}/..)

include_directories(
        ${ONNXRUNTIME_ROOTDIR}/include/onnxruntime/core/session/
        # ${CMAKE_BINARY_DIR}/vcpkg_installed/x64-linux/include
)

add_library(custom_op_library SHARED custom_op_library.cc)

# Link flags adapted from
# https://github.com/microsoft/onnxruntime/blob/3c2d52a995954cd9ce24b2831060c4326ab18e1c/cmake/onnxruntime_unittests.cmake#L1465
set(ONNXRUNTIME_CUSTOM_OP_LIB_LINK_FLAG "-Xlinker --version-script=${PROJECT_SOURCE_DIR}/custom_op_library.lds -Xlinker --no-undefined -Xlinker --gc-sections -z noexecstack")
set_property(TARGET custom_op_library APPEND_STRING PROPERTY LINK_FLAGS ${ONNXRUNTIME_CUSTOM_OP_LIB_LINK_FLAG})