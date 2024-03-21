# 指定 CMake 的最低版本要求
cmake_minimum_required(VERSION 3.10)

# 设置项目名称
project(MyLib)

# 指定 C++ 标准
set(CMAKE_CXX_STANDARD 14)

# 设置 Pybind11 所需的 Python 版本
set(PYBIND11_PYTHON_VERSION 3.6)

# 设置编译选项，包括警告和生成位置无关代码
set(CMAKE_CXX_FLAGS "-Wall -Wextra -fPIC")

# 查找 Pybind11 库
# find pybind
execute_process(COMMAND python3 -m pybind11 --cmakedir
  RESULT_VARIABLE __pybind_exit_code
  OUTPUT_VARIABLE __pybind_path
  OUTPUT_STRIP_TRAILING_WHITESPACE)
find_package(pybind11 PATHS ${__pybind_path})

# 查找 Eigen3 库
find_package(Eigen3 REQUIRED)

# 添加 Python 扩展模块
pybind11_add_module(${PROJECT_NAME} pywrap.cpp)

# 添加版本信息宏定义
target_compile_definitions(${PROJECT_NAME} PRIVATE VERSION_INFO="1.0.0")

# 将 Pybind11 的包含目录添加到目标中
target_include_directories(${PROJECT_NAME} PUBLIC ${PYBIND11_INCLUDE_DIRS})

# 将 Eigen3 库链接到目标中
target_link_libraries(${PROJECT_NAME} PRIVATE Eigen3::Eigen)

# directly output to ffi folder
set_target_properties(${PROJECT_NAME}
  PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/lib
  CXX_VISIBILITY_PRESET "hidden"
)