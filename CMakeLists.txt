cmake_minimum_required(VERSION 3.12)

project(PoissonReconGrid)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_BUILD_TYPE "Release")

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

# Find Python components
find_package(Python 3.11 COMPONENTS Interpreter Development NumPy REQUIRED)

# Set PYTHON_EXECUTABLE for nanobind
set(PYTHON_EXECUTABLE ${Python_EXECUTABLE})
set(PYTHON_INCLUDE_DIR ${Python_INCLUDE_DIRS})

# Include nanobind as a subdirectory
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/nanobind)

# Include PoissonRecon
include(PoissonRecon)
add_library(PoissonRecon INTERFACE)
target_include_directories(PoissonRecon INTERFACE "${CMAKE_BINARY_DIR}/_deps/poissonrecon-src/Src/")

# List of all libraries to link
set(LIBRARIES_TO_LINK PoissonRecon)

# Create library for what we have
add_library(poissonrecongrid
	SHARED
	"src/poisson_utils.h"
    "src/poisson_utils.cpp"
)
target_link_libraries(poissonrecongrid ${LIBRARIES_TO_LINK})
set(LIBRARIES_TO_LINK poissonrecongrid ${LIBRARIES_TO_LINK})

# Nanobind module
nanobind_add_module(poissonrecongrid_bindings
    "${CMAKE_CURRENT_SOURCE_DIR}/src/poisson_bindings.cpp"
)
target_link_libraries(poissonrecongrid_bindings PUBLIC ${LIBRARIES_TO_LINK})
# target_include_directories(poissonrecongrid_bindings PUBLIC "PoissonReconGrid.cpp")