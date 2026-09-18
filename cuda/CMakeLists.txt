# Build PTX assets: cmake -S cuda -B build/ptx && cmake --build build/ptx.

cmake_minimum_required(VERSION 3.18)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES AND "$ENV{CUDAARCHS}" STREQUAL "")
  set(CMAKE_CUDA_ARCHITECTURES 75 CACHE STRING "CUDA architectures for PTX")
endif()

project(fdtdz LANGUAGES CXX CUDA)

# Needed to link against CUDA runtime/driver APIs.
find_package(CUDAToolkit REQUIRED)

enable_language(CUDA)
set(CMAKE_CXX_STANDARD 17)

add_custom_target(all_ptx_kernels)

# Copy a file after the target is build.
macro(copy_after)
  add_custom_target(${ARGV0}_copy ALL
    COMMAND ${CMAKE_COMMAND} -E copy ${ARGV1} ${ARGV2})
  add_dependencies(${ARGV0}_copy ${ARGV0})
endmacro()

# Generate ptx code into the "ptx/" directory.
macro(kernelgen_single)
  # Generate ptx code.
  add_compile_options(-Wno-deprecated-gpu-targets)
  add_library(${ARGV0} OBJECT kernel_ptx.cu)
  target_compile_definitions(${ARGV0} PRIVATE 
    INTERNALTYPE=${ARGV2} EXTERNALTYPE=${ARGV3} NPML=${ARGV4}
    WITHGLOBAL=${ARGV5} WITHSHARED=${ARGV6} WITHUPDATE=${ARGV7} ${ARGV8}) 
  set_property(TARGET ${ARGV0} PROPERTY CUDA_PTX_COMPILATION ON)
  set_property(TARGET ${ARGV0} PROPERTY CUDA_ARCHITECTURES ${ARGV1})

  # Copy the ptx code into the ptx directory.
  set(SRC_PATH "${CMAKE_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/${ARGV0}.dir/kernel_ptx.ptx")
  set(DST_PATH "${CMAKE_SOURCE_DIR}/ptx/${ARGV0}.ptx")
  copy_after(${ARGV0} ${SRC_PATH} ${DST_PATH})

  # Dependencies for dependent kernels.
  add_dependencies(all_ptx_kernels ${ARGV0}_copy)
endmacro()

# Generate ptx kernels for 0-10 threads dedicated to pml.
macro(kern16gen_npml)
  foreach(NPML RANGE 10)
    kernelgen_single(kernel_16_${ARGV0}_${NPML}_111
      ${ARGV0} half2 float ${NPML} true true true)
  endforeach()
endmacro()

macro(kern32gen_npml)
  foreach(NPML RANGE 10)
    kernelgen_single(kernel_32_${ARGV0}_${NPML}_111
      ${ARGV0} float float ${NPML} true true true __OMIT_HALF2__)
  endforeach()
endmacro()

# Generate ptx kernels for benchmarks.
macro(kerngen_bmark)
  kernelgen_single(kernel_16_${ARGV0}_${ARGV1}_001 
    ${ARGV0} half2 float ${ARGV1} false false true)
  kernelgen_single(kernel_16_${ARGV0}_${ARGV1}_010 
    ${ARGV0} half2 float ${ARGV1} false true false)
  kernelgen_single(kernel_16_${ARGV0}_${ARGV1}_100 
    ${ARGV0} half2 float ${ARGV1} true false false)
  kernelgen_single(kernel_16_${ARGV0}_${ARGV1}_011 
    ${ARGV0} half2 float ${ARGV1} false true true)
  kernelgen_single(kernel_16_${ARGV0}_${ARGV1}_101 
    ${ARGV0} half2 float ${ARGV1} true false true)
  kernelgen_single(kernel_16_${ARGV0}_${ARGV1}_110 
    ${ARGV0} half2 float ${ARGV1} true true false)
endmacro()
  
# Generate ptx code for the following "GPUs: compute_capability".
#
# K80: 3.7
# P100: 6.0 (P4: 6.1)
# V100: 7.0
# T4: 7.5 
# A100: 8.0
#

kern32gen_npml(37)
kern32gen_npml(60)
kern32gen_npml(70)
kern32gen_npml(75)
kern32gen_npml(80)
kern16gen_npml(60)
kern16gen_npml(70)
kern16gen_npml(75)
kern16gen_npml(80)
kerngen_bmark(75 7)



