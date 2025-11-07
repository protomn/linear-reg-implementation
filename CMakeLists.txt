cmake_minimum_required(VERSION 3.12)
project(linear_regression_cpp)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Set build type if not specified
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# Compiler-specific flags
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    set(CMAKE_CXX_FLAGS_RELEASE "-O3 -march=native -ffast-math")
    set(CMAKE_CXX_FLAGS_DEBUG "-g -O0 -Wall -Wextra -pedantic")
elseif(MSVC)
    set(CMAKE_CXX_FLAGS_RELEASE "/O2")
    set(CMAKE_CXX_FLAGS_DEBUG "/Od /Wall")
endif()

# Find required packages
find_package(Eigen3 REQUIRED)
if(NOT Eigen3_FOUND)
    message(FATAL_ERROR "Eigen3 library not found. Please install Eigen3.")
endif()

find_package(pybind11 REQUIRED)
if(NOT pybind11_FOUND)
    message(FATAL_ERROR "pybind11 not found. Install via: pip install pybind11")
endif()

# Python module
pybind11_add_module(_core 
    bindings.cpp
    linreg.cpp
)

# Link libraries
target_link_libraries(_core PRIVATE Eigen3::Eigen)

# Include directories
target_include_directories(_core PRIVATE
    ${EIGEN3_INCLUDE_DIR}
    ${pybind11_INCLUDE_DIRS}
)

# Compiler optimizations
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(_core PRIVATE
        -O3
        -march=native
        -ffast-math
        -fvisibility=hidden
    )
endif()

# Install target (optional)
install(TARGETS _core
    LIBRARY DESTINATION .
    RUNTIME DESTINATION .
)

# Print configuration summary
message(STATUS "")
message(STATUS "Configuration Summary:")
message(STATUS "  Build Type:    ${CMAKE_BUILD_TYPE}")
message(STATUS "  C++ Compiler:  ${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}")
message(STATUS "  C++ Standard:  ${CMAKE_CXX_STANDARD}")
message(STATUS "  Eigen3:        ${EIGEN3_INCLUDE_DIR}")
message(STATUS "  pybind11:      ${pybind11_VERSION}")
message(STATUS "")