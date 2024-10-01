cmake_minimum_required(VERSION 3.12)

project(PoissonReconGrid)

# Find Python components
find_package(Python 3.11 COMPONENTS Interpreter Development NumPy REQUIRED)

# Set PYTHON_EXECUTABLE for nanobind
set(PYTHON_EXECUTABLE ${Python_EXECUTABLE})
set(PYTHON_INCLUDE_DIR ${Python_INCLUDE_DIRS})

# Include nanobind as a subdirectory
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/nanobind)

# Include directories
include_directories(
    ${Python_INCLUDE_DIRS}
    ${Python_NumPy_INCLUDE_DIRS}
    ${CMAKE_CURRENT_SOURCE_DIR}/PoissonRecon/Src  # Include PoissonRecon headers
    ${CMAKE_CURRENT_SOURCE_DIR}/PoissonRecon  # Include PoissonRecon headers
    ${CMAKE_CURRENT_SOURCE_DIR}/nanobind/include  # Include nanobind headers
)

# List of minimal PoissonRecon source files required
set(PoissonRecon_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/PoissonRecon/Src/PoissonRecon.cpp
)

# Build the core parts of nanobind once
#nanobind_build_library(nanobind)

# Create a static library for PoissonRecon
add_library(PoissonReconLib STATIC ${PoissonRecon_SOURCES})

# Add the library for your extension
add_library(poissonrecongrid MODULE PoissonReconGrid.cpp)

# Link PoissonReconLib and nanobind to your extension
target_link_libraries(poissonrecongrid PRIVATE PoissonReconLib nanobind)

# Get the Python SOABI (e.g., "cpython-311-x86_64-linux-gnu")
execute_process(
    COMMAND "${Python_EXECUTABLE}" -c "import sysconfig; print(sysconfig.get_config_var('SOABI'))"
    OUTPUT_VARIABLE PYTHON_SOABI
    OUTPUT_STRIP_TRAILING_WHITESPACE
)

# Set the module properties
set_target_properties(poissonrecongrid PROPERTIES
    OUTPUT_NAME "poissonrecongrid.${PYTHON_SOABI}"
    PREFIX ""
    SUFFIX ".so"
    CXX_STANDARD 20
    CXX_STANDARD_REQUIRED YES
)

# Define necessary macros
target_compile_definitions(poissonrecongrid PRIVATE
    #-DNB_INLINE
    #-DDEFAULT_DIMENSION=3
    # -DUSE_DOUBLE  # Uncomment if you want to use double precision
)

# Include directories for the target
target_include_directories(poissonrecongrid PRIVATE
    ${Python_INCLUDE_DIRS}
    ${Python_NumPy_INCLUDE_DIRS}
    ${CMAKE_CURRENT_SOURCE_DIR}/PoissonRecon/Src
    ${CMAKE_CURRENT_SOURCE_DIR}/PoissonRecon
    ${CMAKE_CURRENT_SOURCE_DIR}/nanobind/include
)

execute_process(
    COMMAND ${Python_EXECUTABLE} -c "import site; print(site.getsitepackages()[0])"
    OUTPUT_VARIABLE PYTHON_SITE_PACKAGES
    OUTPUT_STRIP_TRAILING_WHITESPACE
)

install(TARGETS poissonrecongrid
        LIBRARY DESTINATION ${PYTHON_SITE_PACKAGES})