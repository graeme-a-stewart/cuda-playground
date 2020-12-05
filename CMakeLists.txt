# SPDX-FileCopyrightText: 2020 CERN
# SPDX-License-Identifier: Apache-2.0

# Relatively high to get good CUDA support
cmake_minimum_required(VERSION 3.18)

project(Cuda-Playground
  VERSION 0.1.0
  DESCRIPTION "Toy and test programs in CUDA"
  LANGUAGES C CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CUDA_STANDARD ${CMAKE_CXX_STANDARD})
set(CMAKE_CUDA_STANDARD_REQUIRED ${CMAKE_CXX_STANDARD_REQUIRED})
set(CMAKE_CUDA_EXTENSIONS OFF)
set(CMAKE_INCLUDE_DIRECTORIES_PROJECT_BEFORE ON)

add_executable(array-add array-add.cu)
add_executable(array-2d-add array-2d-add.cu)
