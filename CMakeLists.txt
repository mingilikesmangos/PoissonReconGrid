cmake_minimum_required(VERSION 3.12)

project(PoissonReconGrid)

# Find Python components
find_package(Python 3.11 COMPONENTS Interpreter Development NumPy REQUIRED)

# Set PYTHON_EXECUTABLE for nanobind
set(PYTHON_EXECUTABLE ${Python_EXECUTABLE})
set(PYTHON_INCLUDE_DIR ${Python_INCLUDE_DIRS})

# Include nanobind as a subdirectory
add_subdirectory(nanobind)

# Include directories
include_directories(${Python_INCLUDE_DIRS})
include_directories(${Python_NumPy_INCLUDE_DIRS})
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/PoissonRecon/Src)  # Include PoissonRecon headers
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/PoissonRecon)  # Include PoissonRecon headers

# Add PoissonRecon source files (adjust as necessary)
file(GLOB PoissonRecon_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/PoissonRecon/Src/*.cpp
)

file(GLOB PoissonRecon_HEADERS
    ${CMAKE_CURRENT_SOURCE_DIR}/PoissonRecon/Src/*.h
)

# Add the library
add_library(poissonrecongrid MODULE PoissonReconGrid.cpp ${PoissonRecon_SOURCES})

# Link libraries
target_link_libraries(poissonrecongrid PRIVATE nanobind)

# Set the module properties
set_target_properties(poissonrecongrid PROPERTIES
    PREFIX "${Python_MODULE_PREFIX}"
    SUFFIX "${Python_MODULE_EXTENSION}"
)

# Ensure C++17 standard
set_target_properties(poissonrecongrid PROPERTIES
    CXX_STANDARD 20
    CXX_STANDARD_REQUIRED YES
)

# Define necessary macros
target_compile_definitions(poissonrecongrid PRIVATE
    #-DDEFAULT_DIMENSION=3
    # -DUSE_DOUBLE  # Uncomment if you want to use double precision
)

target_include_directories(poissonrecongrid PRIVATE
    ${Python_INCLUDE_DIRS}
    ${Python_NumPy_INCLUDE_DIRS}
    ${CMAKE_CURRENT_SOURCE_DIR}/PoissonRecon/Src
    ${CMAKE_CURRENT_SOURCE_DIR}/nanobind/include
)