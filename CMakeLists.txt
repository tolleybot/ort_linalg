# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

# Adapted from https://github.com/microsoft/onnxruntime-inference-examples/blob/main/c_cxx/CMakeLists.txt

cmake_minimum_required(VERSION 3.13)

# Project
project(onnxruntime_samples C CXX)
if (WIN32)
 string(APPEND CMAKE_CXX_FLAGS " /W4")
else()
 string(APPEND CMAKE_CXX_FLAGS " -Wall -Wextra")
 string(APPEND CMAKE_C_FLAGS " -Wall -Wextra")
endif()

#onnxruntime providers
option(onnxruntime_USE_CUDA "Build with CUDA support" OFF)
option(onnxruntime_USE_TENSORRT "Build with TensorRT support" OFF)
option(LIBPNG_ROOTDIR "libpng root dir")
option(ONNXRUNTIME_ROOTDIR "onnxruntime root dir")
include(FetchContent)

set(CMAKE_CXX_STANDARD 17)

if(NOT ONNXRUNTIME_ROOTDIR)
  if(WIN32)
    set(ONNXRUNTIME_ROOTDIR "C:/Program Files/onnxruntime")
else()
    #set(ONNXRUNTIME_ROOTDIR "/usr/local")
    set(ONNXRUNTIME_ROOTDIR "/home/cjoy/src/onnxruntime")
  endif()
endif()

# The ORT package has a different include directory structure to a local install via cmake.
# We added the path for the pre-built package above. Add the path for a local install to support either usage.
# TODO: If we want to support additional EPs being loadable from a local install we also need to add EP specific
# directories under /include/onnxruntime/core/providers
include_directories("${ONNXRUNTIME_ROOTDIR}/include"                           # Pre-built package
                    "${ONNXRUNTIME_ROOTDIR}/include/onnxruntime")              # Linux local install to /usr/local

# link_directories("${ONNXRUNTIME_ROOTDIR}/lib")

link_directories("${ONNXRUNTIME_ROOTDIR}/build/Linux/Debug")


if(onnxruntime_USE_CUDA)
  add_definitions(-DUSE_CUDA)
endif()
if(onnxruntime_USE_TENSORRT)
  add_definitions(-DUSE_TENSORRT)
endif()
if(onnxruntime_USE_DML)
  message("Enabling DML")
  add_definitions(-DUSE_DML)
endif()

add_subdirectory(linalg_op)
