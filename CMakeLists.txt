cmake_minimum_required(VERSION 3.21)
project(Parallel-SSSP)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Use Homebrew LLVM and METIS on macOS
if(APPLE)
    # Point to Homebrew's LLVM
    set(HOMEBREW_LLVM /opt/homebrew/opt/llvm)
    # Point to Homebrew's METIS
    set(HOMEBREW_METIS /opt/homebrew/opt/metis)

    # Set compilers
    set(CMAKE_C_COMPILER ${HOMEBREW_LLVM}/bin/clang CACHE STRING "" FORCE)
    set(CMAKE_CXX_COMPILER ${HOMEBREW_LLVM}/bin/clang++ CACHE STRING "" FORCE)

    # Add LLVM and METIS to CMake path
    list(APPEND CMAKE_PREFIX_PATH ${HOMEBREW_LLVM} ${HOMEBREW_METIS})

    # Compiler and linker flags for OpenMP
    set(OpenMP_C_FLAGS "-Xpreprocessor -fopenmp -I${HOMEBREW_LLVM}/include")
    set(OpenMP_C_LIB_NAMES "omp")
    set(OpenMP_CXX_FLAGS "-Xpreprocessor -fopenmp -I${HOMEBREW_LLVM}/include")
    set(OpenMP_CXX_LIB_NAMES "omp")
    set(OpenMP_omp_LIBRARY ${HOMEBREW_LLVM}/lib/libomp.dylib)
endif()

# Find OpenMP
find_package(OpenMP REQUIRED)

# Find MPI
find_package(MPI REQUIRED)

# Find METIS components
find_path(METIS_INCLUDE_DIR metis.h
        HINTS ${HOMEBREW_METIS}/include
        PATHS /usr/local/include /usr/include)
find_library(METIS_LIBRARY
        NAMES metis
        HINTS ${HOMEBREW_METIS}/lib
        PATHS /usr/local/lib /usr/lib)

if(NOT METIS_INCLUDE_DIR)
    message(FATAL_ERROR "METIS include directory not found. Install METIS (e.g., 'brew install metis')")
endif()
if(NOT METIS_LIBRARY)
    message(FATAL_ERROR "METIS library not found. Install METIS (e.g., 'brew install metis')")
endif()

# Add include directories
include_directories(include ${METIS_INCLUDE_DIR})

# Add executable for the main SSSP program
add_executable(SSSP
    src/main.cpp
    src/mpi/sssp_parallel_mpi.cpp       # MPI implementation logic
    src/parallel/sssp_parallel_openmp.cpp # OpenMP implementation logic
    src/serial/sssp_sequential.cpp    # Sequential implementation logic
    src/utils.cpp
    src/graph.cpp
    # Add back source files defining the mode-specific entry points
    src/parallel/main_parallel.cpp
    src/mpi/main_mpi.cpp
)

# Add executable for the update generator program
add_executable(generate_updates
        tools/generate_updates.cpp
        src/utils.cpp # Needs utils for load_graph
        src/graph.cpp # Needs graph implementation used by utils and the tool itself
)

# Link libraries
# Link SSSP executable
target_link_libraries(SSSP
    PRIVATE
    OpenMP::OpenMP_CXX # Link OpenMP
    MPI::MPI_CXX       # Link MPI
    ${METIS_LIBRARY} # Corrected variable name
)

# Link generate_updates executable
target_link_libraries(generate_updates
    PRIVATE
    ${METIS_LIBRARY} # Corrected variable name
)

# Add include directories (already present, just ensuring target_include_directories is considered)
# Using target_include_directories is often preferred over include_directories
target_include_directories(SSSP PRIVATE include ${METIS_INCLUDE_DIR} ${MPI_CXX_INCLUDE_PATH})
target_include_directories(generate_updates PRIVATE include ${METIS_INCLUDE_DIR})

# Comment out or remove the old commented-out executables if SSSP is the unified one
# add_executable(Parallel_SSSP ...)
# add_executable(Parallel_SSSP_MPI ...)
# add_executable(SSSP_SERIAL ...)

# Ensure OpenMP flags are correctly applied (already handled by find_package and target_link_libraries)

# Ensure C++ standard is set (already done)
# set(CMAKE_CXX_STANDARD 17)
# set(CMAKE_CXX_STANDARD_REQUIRED True)

