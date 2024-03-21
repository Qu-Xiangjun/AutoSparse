cmake_minimum_required(VERSION 3.10)

project(AutoSparseBackend C CXX)

set(CMAKE_CXX_STANDARD 17)

option(USE_ICC "Use Intel compiler (ICC)" OFF)

# find correct version of Python
set(PYBIND11_PYTHON_VERSION 3.8)

# find pybind11 lib
execute_process(COMMAND python3 -m pybind11 --cmakedir
  RESULT_VARIABLE __pybind_exit_code
  OUTPUT_VARIABLE __pybind_path
  OUTPUT_STRIP_TRAILING_WHITESPACE)
find_package(pybind11 PATHS ${__pybind_path})


if(USE_ICC)
    # user select icpc intel compiler
    set(CMAKE_CXX_COMPILER "icpc")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17 -O3 -march=native -qopenmp -DICC -Wextra -fPIC")
    message(STATUS "Using Intel compiler (ICC)")
else()
    # user select gcc compiler
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17 -O3 -march=native -fopenmp -DGCC -Wextra -fPIC")
    message(STATUS "Using GCC compiler (g++)")
endif()

include(ProcessorCount)
ProcessorCount(NUMCORE)
if(NUMCORE EQUAL 0)
    set(NUMCORE 8)
endif()

add_definitions(-DNUMCORE=${NUMCORE})

pybind11_add_module(${PROJECT_NAME} backend_api_bindings.cpp)

target_compile_definitions(${PROJECT_NAME} PRIVATE VERSION_INFO="1.0.0")

target_include_directories(${PROJECT_NAME} PUBLIC ${PYBIND11_INCLUDE_DIRS})

target_link_libraries(${PROJECT_NAME} PRIVATE m dl)

# directly output to ffi folder
set_target_properties(${PROJECT_NAME}
  PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/../AutoSparse/backend/
  CXX_VISIBILITY_PRESET "hidden"
)