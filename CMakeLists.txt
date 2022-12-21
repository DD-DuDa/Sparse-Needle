cmake_minimum_required(VERSION 3.9)
project(needle C CXX CUDA)

# find correct version of Python
execute_process(COMMAND python3-config --prefix OUTPUT_VARIABLE Python_ROOT_DIR)
find_package(Python COMPONENTS Development Interpreter REQUIRED)
include_directories(${Python_INCLUDE_DIRS})

# find pybind
execute_process(COMMAND python3 -m pybind11 --cmakedir
  RESULT_VARIABLE __pybind_exit_code
  OUTPUT_VARIABLE __pybind_path
  OUTPUT_STRIP_TRAILING_WHITESPACE)
find_package(pybind11 PATHS ${__pybind_path})



if(NOT MSVC)
  set(CMAKE_CXX_FLAGS "-std=c++14 -O2 -march=native ${CMAKE_CXX_FLAGS}")
  # set(CMAKE_CUDA_STANDARD 14)
else()
  set(CMAKE_CXX_FLAGS "/std:c++14 -O2 -march=native ${CMAKE_CXX_FLAGS}")
  # set(CMAKE_CUDA_STANDARD 14)
endif()

set(CMAKE_CUDA_STANDARD          14)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_EXTENSIONS        OFF)

include_directories(SYSTEM ${pybind11_INCLUDE_DIRS})
list(APPEND LINKER_LIBS ${pybind11_LIBRARIES})


###################
### CPU BACKEND ###
###################
add_library(ndarray_backend_cpu MODULE src/ndarray_backend_cpu.cc)
target_link_libraries(ndarray_backend_cpu PUBLIC ${LINKER_LIBS})
pybind11_extension(ndarray_backend_cpu)
pybind11_strip(ndarray_backend_cpu)


# directly output to ffi folder
set_target_properties(ndarray_backend_cpu
  PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/python/needle/backend_ndarray
  CXX_VISIBILITY_PRESET "hidden"
)

if(${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
  set_property(TARGET ndarray_backend_cpu PROPERTY LINK_OPTIONS -undefined dynamic_lookup)
endif()



####################
### CUDA BACKEND ###
####################
find_package(CUDA)
if(CUDA_FOUND)
  message(STATUS "Found cuda, building cuda backend")

  #######################
  ### CUSPARSELt BACKEND ###
  #######################
  string(REPLACE "/bin/nvcc" "" CUDA_TOOLKIT_PATH ${CMAKE_CUDA_COMPILER})
  if (${CMAKE_HOST_SYSTEM_PROCESSOR} STREQUAL "aarch64" AND
      ${CMAKE_SYSTEM_NAME} STREQUAL "Linux")
      set(OS_ARCH_NVRTC "sbsa-linux")
  elseif (${CMAKE_HOST_SYSTEM_PROCESSOR} STREQUAL "x86_64" AND
          ${CMAKE_SYSTEM_NAME} STREQUAL "Linux")
      set(OS_ARCH_NVRTC "x86_64-linux")
  endif()
  set(NVRTC_SHARED ${CUDA_TOOLKIT_PATH}/targets/${OS_ARCH_NVRTC}/lib/libnvrtc.so)

  # cusparselt_example check-env utilities
  function(set_with_fallback VARIABLE FALLBACK)
  if (NOT DEFINED ${VARIABLE} OR ${VARIABLE} STREQUAL "")
    set(${VARIABLE} $ENV{${VARIABLE}} CACHE INTERNAL ${VARIABLE})
    if (${VARIABLE} STREQUAL "")
      if (NOT ${FALLBACK} STREQUAL "")
        set(${VARIABLE} $ENV{${FALLBACK}} CACHE INTERNAL ${VARIABLE})
      endif ()
    endif ()
  endif ()
  endfunction()

  set_with_fallback(CUSPARSELT_PATH CUSPARSELT_DIR)
  if (CUSPARSELT_PATH STREQUAL "")
      message(FATAL_ERROR "Please set the environment variables CUSPARSELT_DIR to the path of the cuSPARSELt installation.")
  endif ()

  message(STATUS "Using CUSPARSELT_PATH = ${CUSPARSELT_DIR}")

  ####################


  include_directories(SYSTEM ${CUDA_INCLUDE_DIRS})
  list(APPEND LINKER_LIBS ${CUDA_CUDART_LIBRARY})

  # invoke nvidia smi to detect if we really have a GPU
  execute_process(COMMAND "nvidia-smi" ERROR_QUIET  RESULT_VARIABLE NV_RET)
  if(NV_RET EQUAL "0")
    CUDA_SELECT_NVCC_ARCH_FLAGS(ARCH_FLAGS Auto)
  else()
    # set to 3.7 the flag of K80
    CUDA_SELECT_NVCC_ARCH_FLAGS(ARCH_FLAGS 3.7)
  endif()

  # set arch flags properly
  CUDA_ADD_LIBRARY(ndarray_backend_cuda MODULE src/ndarray_backend_cuda.cu OPTIONS ${ARCH_FLAGS})

  # CUSPARSELt BACKEND
  target_include_directories(ndarray_backend_cuda
    PUBLIC ${CUDA_TOOLKIT_PATH}/include
    PUBLIC ${CUSPARSELT_PATH}/include
  )

  target_link_directories(ndarray_backend_cuda
    PUBLIC ${CUSPARSELT_PATH}/lib64
    PUBLIC ${CUDA_TOOLKIT_PATH}/lib64
  )

  target_link_libraries(ndarray_backend_cuda
    cudart
    cusparse
    libcusparseLt.so
    ${NVRTC_SHARED}
    ${LINKER_LIBS}
  )


  # target_link_libraries(ndarray_backend_cuda ${LINKER_LIBS})
  pybind11_extension(ndarray_backend_cuda)
  pybind11_strip(ndarray_backend_cuda)

  # directly output to ffi folder
  set_target_properties(ndarray_backend_cuda
    PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/python/needle/backend_ndarray
    CXX_VISIBILITY_PRESET "hidden"
    CUDA_VISIBILITY_PRESET "hidden"
)




endif()

