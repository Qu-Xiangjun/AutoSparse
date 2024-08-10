cmake_minimum_required(VERSION 3.4.0 FATAL_ERROR)
if(POLICY CMP0048)
  cmake_policy(SET CMP0048 NEW)
endif()
if(POLICY CMP0054)
  cmake_policy(SET CMP0054 NEW)
endif()
project(taco
  VERSION 0.1
  LANGUAGES C CXX
)
option(CUDA "Build for NVIDIA GPU (CUDA must be preinstalled)" OFF) # 构建了一个CUDA构建选项
option(PYTHON "Build TACO for python environment" OFF) # 构建了一个PYTHON构建选项
option(OPENMP "Build with OpenMP execution support" OFF) # 构建了一个OPENMP构建选项
option(COVERAGE "Build with code coverage analysis" OFF) # 构建了一个COVERAGE构建选项，用于代码覆盖分析
set(TACO_FEATURE_CUDA 0) # 设置变量为0
set(TACO_FEATURE_OPENMP 0)
set(TACO_FEATURE_PYTHON 0)
if(CUDA)
  message("-- Searching for CUDA Installation") # 输出信息
  find_package(CUDA REQUIRED) # 在系统中查找并加载 CUDA 相关的 CMake 配置文件，REQUIRED 表示找不到会导致构建失败。
  add_definitions(-DCUDA_BUILT) # 向编译器添加一个预定义的宏 CUDA_BUILT。这样做的目的是为了标记代码中使用了 CUDA 相关的功能 
  set(TACO_FEATURE_CUDA 1) # 这一行代码将变量 TACO_FEATURE_CUDA 的值设置为 1，表示 CUDA 相关的功能已经被启用。
endif(CUDA)
if(OPENMP)
  message("-- Will use OpenMP for parallel execution")
  add_definitions(-DUSE_OPENMP)
  set(TACO_FEATURE_OPENMP 1)
endif(OPENMP)

if(PYTHON)
  message("-- Will build Python extension")
  add_definitions(-DPYTHON)
  set(TACO_FEATURE_PYTHON 1)
endif(PYTHON)

# 指定了 CMake 构建系统所使用的配置类型，也就是生成的工程或项目的构建类型。
# Release: 用于生成优化后的、用于发布的最终版本。
# Debug: 用于生成带有调试信息的版本，通常用于调试和查找错误。
# MinSizeRel: 用于生成尽可能小且仍保持性能的版本。
# RelWithDebInfo: 生成带有调试信息的优化版本，通常用于性能分析和调试。
# CMake 构建项目时可以通过参数来选择不同的构建类型，例如：
#       cmake --build . --config Release
SET(CMAKE_CONFIGURATION_TYPES "Release;Debug;MinSizeRel;RelWithDebInfo")


# 根据不同的构建类型做出相应的配置
if(NOT CMAKE_BUILD_TYPE) # 未设置默认debug
  set(CMAKE_BUILD_TYPE "Debug" CACHE STRING  # 缓存解释
    "Choose the type of build, options are: Debug Release RelWithDebInfo MinSizeRel."
    FORCE) # 强制一定设置CMAKE_BUILD_TYPE 为 debug
endif(NOT CMAKE_BUILD_TYPE)

if (CMAKE_BUILD_TYPE MATCHES Debug)
  message("-- Debug Build") # 输出调试信息
  # TACO_DEBUG 和 TACO_ASSERTS 的预定义宏定义，并将 TACO_DEBUG 设置为 1，用于标记代码处于调试模式。
  add_definitions(-DTACO_DEBUG) 
  add_definitions(-DTACO_ASSERTS)
  set(TACO_DEBUG 1)
elseif (CMAKE_BUILD_TYPE MATCHES RelWithDebInfo) # 带有调试信息的优化版本
  message("-- Release Build with Debug Information")
  add_definitions(-DTACO_DEBUG)
  add_definitions(-DTACO_ASSERTS)
  set(TACO_DEBUG 1)
elseif (CMAKE_BUILD_TYPE MATCHES Release)
  message("-- Release Build")
elseif (CMAKE_BUILD_TYPE MATCHES MinSizeRel) # 尽可能小且仍保持性能的版本
  message("-- Release Build with Minimal Size")
endif ()

# 检查名为 TACO_ASSERTS 的环境变量是否被设置。如果该环境变量存在且非空，
# 添加一个预处理宏 TACO_ASSERTS 到项目中，用于标记启用了断言（assertions）
if ($ENV{TACO_ASSERTS})
  add_definitions(-DTACO_ASSERTS)
endif ()

# 系统类型检查
if (${CMAKE_SYSTEM_NAME} MATCHES "Windows")
  set(WINDOWS TRUE)
  add_definitions(-DTACO_WINDOWS)
elseif (${CMAKE_SYSTEM_NAME} MATCHES "Linux")
  set(LINUX TRUE)
  add_definitions(-DTACO_LINUX)
elseif (${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
  set(DARWIN TRUE)
  add_definitions(-DTACO_DARWIN)
  set(CMAKE_MACOSX_RPATH 1)
endif()

option(TACO_SHARED_LIBRARY "Build as a shared library" ON) # 创建一个构建选项

# 设置了 CMake 的全局属性，启用了项目文件夹组织。这使得在一些集成开发环境（IDE）中能够更好地组织和显示项目文件。
set_property(GLOBAL PROPERTY USE_FOLDERS ON)

# 静态库\共享库文件\可执行文件输出的目录为 ${CMAKE_BINARY_DIR}/lib，意味着生成的静态库文件将会放在 build/lib 
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")

set(OPTIMIZE "-O3" CACHE STRING "Optimization level") # 03 级别优化
# 设置了一系列 C/C++ 编译器标志（flags），这些标志用于开启各种编译器警告，并且在编译过程中提醒或报告一些潜在的问题。
set(C_CXX_FLAGS "-Wall -Wextra -Wno-unused-parameter -Wno-missing-field-initializers -Wmissing-declarations -Woverloaded-virtual -pedantic-errors -Wno-deprecated")
if(OPENMP) # 如果开启了 OPENMP 构建选项，则会在编译标志中加入 -fopenmp，启用 OpenMP 并行编程。
  # 如果 OPENMP 选项为 ON，则会将 -fopenmp 添加到 C_CXX_FLAGS 中。
  set(C_CXX_FLAGS "-fopenmp ${C_CXX_FLAGS}")
endif(OPENMP)

if(COVERAGE) # 如果开启代码覆盖的选项
  find_program(PATH_TO_GCOVR gcovr REQUIRED)
  # add coverage tooling to build flags
  set(C_CXX_FLAGS "${C_CXX_FLAGS} -g -fprofile-arcs -ftest-coverage")
  # name the coverage files "foo.gcno", not "foo.cpp.gcno"
  set(CMAKE_CXX_OUTPUT_EXTENSION_REPLACE 1)
  message("-- Code coverage analysis (gcovr) enabled")
endif(COVERAGE)

set(C_CXX_FLAGS "${C_CXX_FLAGS}") # 这行代码可能是用于保留之前设置的标志内容。
set(CMAKE_C_FLAGS "${C_CXX_FLAGS}") # C 语言代码将会使用 C_CXX_FLAGS 中定义的编译器标志。
set(CMAKE_CXX_FLAGS "${C_CXX_FLAGS} -std=c++14") # 指定 C++ 的语言标准为 C++14 版本

set(TACO_PROJECT_DIR ${CMAKE_CURRENT_LIST_DIR}) # 为当前 CMakeLists.txt 文件所在的目录

set(TACO_SRC_DIR     ${TACO_PROJECT_DIR}/src)
set(TACO_TEST_DIR    ${TACO_PROJECT_DIR}/test)
set(TACO_TOOLS_DIR   ${TACO_PROJECT_DIR}/tools)
set(TACO_INCLUDE_DIR ${TACO_PROJECT_DIR}/include)

enable_testing() #  启用 CTest 测试框架，允许你在项目中进行测试。
include_directories(${TACO_INCLUDE_DIR}) # 将 TACO_INCLUDE_DIR 添加到项目的包含目录中，以便编译器可以找到项目中的头文件。

# CMAKE_LIBRARY_OUTPUT_DIRECTORY: 这是一个 CMake 预定义的变量，用于指定生成的共享库（Shared Library）文件的输出目录。‘
# 设置 TACO_LIBRARY_DIR 为该值
set(TACO_LIBRARY_DIR ${CMAKE_LIBRARY_OUTPUT_DIRECTORY})

# 定义了安装规则，将头文件复制到安装目录。它会将 ${TACO_INCLUDE_DIR} 目录下的所有 .h 文件复制到安装目录的 include 文件夹中。
install(DIRECTORY ${TACO_INCLUDE_DIR}/ DESTINATION include FILES_MATCHING PATTERN "*.h")

# 将各个子目录（比如源代码、测试代码、工具等）添加到 CMake 构建中，以便 CMake 进入这些目录并执行相应的构建。
add_subdirectory(src)
add_subdirectory(test)
add_subdirectory(tools)
add_subdirectory(apps)

# 这行代码将 CMAKE_CXX_FLAGS 中的 -Wmissing-declarations 替换为空字符串，即移除了 -Wmissing-declarations 标志。
# 可能是为了在某些情况下移除特定的编译器标志。
string(REPLACE " -Wmissing-declarations" "" CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}") 

# 检查项目是否位于 Git 仓库中，并在构建过程中更新子模块（如果有），并获取当前 Git 提交的简短哈希值（short hash）。
find_package(Git QUIET) # 尝试查找 Git 的可执行文件。如果找到 Git 并且项目位于 Git 仓库中，将进行后续操作。
if(GIT_FOUND AND EXISTS "${TACO_PROJECT_DIR}/.git") # 是否找到了 Git 可执行文件，并且项目目录中存在 .git 文件夹（即项目是一个 Git 仓库）。
  # Update submodules as needed
  option(GIT_SUBMODULE "Check submodules during build" ON) # 检查构建过程中是否需要更新子模块（如果项目包含子模块）。
  if(GIT_SUBMODULE)
    message(STATUS "Submodule update")
    execute_process(COMMAND ${GIT_EXECUTABLE} submodule update --init --recursive # 执行 Git 命令来初始化或更新子模块，如果需要的话。
            WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
            RESULT_VARIABLE GIT_SUBMOD_RESULT)
    if(NOT GIT_SUBMOD_RESULT EQUAL "0")
      message(FATAL_ERROR "git submodule update --init failed with ${GIT_SUBMOD_RESULT}, please checkout submodules")
    endif()
  endif()
  # get git revision
  execute_process( # 获取当前 Git 提交的简短哈希值并将其存储在 TACO_GIT_SHORTHASH 变量中。
    COMMAND ${GIT_EXECUTABLE} rev-parse --short HEAD
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    RESULT_VARIABLE GIT_REVPARSE_RESULT
    OUTPUT_VARIABLE TACO_GIT_SHORTHASH
    OUTPUT_STRIP_TRAILING_WHITESPACE
  )
  if(NOT GIT_REVPARSE_RESULT EQUAL "0")
    message(NOTICE "'git rev-parse --short HEAD' failed with ${GIT_REVPARSE_RESULT}, git version info will be unavailable.")
    set(TACO_GIT_SHORTHASH "")
  endif()
else()
  set(TACO_GIT_SHORTHASH "")
endif()

# 检查名为 pybind11 的子模块是否已经被下载。如果未下载或不存在该子模块的 CMakeLists.txt 文件，会输出错误信息并终止构建。
if(NOT EXISTS "${TACO_PROJECT_DIR}/python_bindings/pybind11/CMakeLists.txt")
  message(FATAL_ERROR "The submodules were not downloaded! GIT_SUBMODULE was turned off or failed. Please update submodules and try again.")
endif()

if(PYTHON) # 如果设置了 PYTHON 构建选项
  add_subdirectory(python_bindings)
  message("-- Will build Python extension")
  add_definitions(-DPYTHON)
endif(PYTHON)

# 代码将编译器标志 CMAKE_CXX_FLAGS 中添加了 -Wmissing-declarations，用于启用编译器警告相关的设置。
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wmissing-declarations")
# 创建了一个名为 src 的自定义构建目标，它依赖于 apps。这个构建目标可能会用于在构建过程中执行一些自定义的操作或确保 apps 目标被构建。
add_custom_target(src DEPENDS apps)

if(COVERAGE)
  # code coverage analysis target
  add_custom_target(gcovr
    COMMAND mkdir -p coverage
    COMMAND ${CMAKE_MAKE_PROGRAM} test
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
  )
  add_custom_command(TARGET gcovr
    COMMAND echo "Running gcovr..."
    COMMAND ${PATH_TO_GCOVR} -r ${CMAKE_SOURCE_DIR} --html --html-details -o coverage/index.html ${CMAKE_BINARY_DIR}
    COMMAND echo "See coverage/index.html for coverage information."
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
  )
  add_dependencies(gcovr taco-test)
  if(PYTHON)
    add_dependencies(gcovr core_modules)
  endif(PYTHON)
  set_property(DIRECTORY APPEND PROPERTY ADDITIONAL_MAKE_CLEAN_FILES coverage)
endif(COVERAGE)

string(TIMESTAMP TACO_BUILD_DATE "%Y-%m-%d") # 生成构建日期信息
# 配置一个头文件。它将源文件 include/taco/version.h.in 中的内容进行处理，其中可能包含一些类似 @VAR@ 这样的占位符，
# 然后将其写入到输出文件 include/taco/version.h 中。
# @ONLY 参数表示只替换形如 @VAR@ 这样的变量，而不处理其他 CMake 变量。
configure_file("include/taco/version.h.in" "include/taco/version.h" @ONLY)
# 这行代码将生成的 version.h 文件安装到指定目录。${CMAKE_BINARY_DIR}/include/taco/version.h 是生成的文件路径，
# DESTINATION "include/taco" 表示将文件安装到项目目录中的 include/taco 子目录下。
install(FILES "${CMAKE_BINARY_DIR}/include/taco/version.h" DESTINATION "include/taco")
