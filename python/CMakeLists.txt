cmake_minimum_required(VERSION 3.2)
project(AutoSparseBackend C CXX)


option(USE_ICC "Use Intel compiler (ICC)" OFF)

# find correct version of Python
execute_process(COMMAND python -c "import sys; print(sys.executable)" 
    OUTPUT_VARIABLE Python_EXECUTABLE
    OUTPUT_STRIP_TRAILING_WHITESPACE)
find_package(Python COMPONENTS Development Interpreter REQUIRED)
include_directories(${Python_INCLUDE_DIRS})

# find pybind11 lib
execute_process(COMMAND python3 -m pybind11 --cmakedir
  RESULT_VARIABLE __pybind_exit_code
  OUTPUT_VARIABLE __pybind_path
  OUTPUT_STRIP_TRAILING_WHITESPACE)
find_package(pybind11 PATHS ${__pybind_path})


if(USE_ICC)
    # user select icpc intel compiler
    set(CMAKE_CXX_COMPILER "icpc")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17 -O3 -qopenmp -DICC -Wextra -fPIC")
    message(STATUS "Using Intel compiler (ICC)")
else()
    # user select gcc compiler
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17 -O3 -fopenmp -DGCC -Wextra -fPIC")
    message(STATUS "Using GCC compiler (g++)")
endif()
set(CMAKE_CXX_STANDARD 17)

include_directories(SYSTEM ${pybind11_INCLUDE_DIRS})
list(APPEND LINKER_LIBS ${pybind11_LIBRARIES})

include(ProcessorCount)
ProcessorCount(NUMCORE)
if(NUMCORE EQUAL 0)
    set(NUMCORE 8)
endif()

add_definitions(-DNUMCORE=${NUMCORE})


add_library(auto_sparse_backend MODULE codegen/backend_api_bindings.cpp)
target_link_libraries(auto_sparse_backend PUBLIC ${LINKER_LIBS})
pybind11_extension(auto_sparse_backend)
pybind11_strip(auto_sparse_backend)


# directly output to ffi folder
set_target_properties(auto_sparse_backend
  PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/AutoSparse/backend
  CXX_VISIBILITY_PRESET "hidden"
)