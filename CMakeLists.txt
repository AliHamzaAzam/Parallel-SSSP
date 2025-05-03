cmake_minimum_required(VERSION 3.21)
project(Parallel-SSSP)

set(CMAKE_CXX_STANDARD 23)
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
add_executable(Parallel_SSSP
        src/main.cpp
        src/sssp_sequential.cpp
        src/sssp_parallel_openmp.cpp
        src/utils.cpp
        src/graph.cpp # Ensure this line is present
)

# Add executable for the update generator program
add_executable(generate_updates
        src/generate_updates.cpp
        src/utils.cpp # Needs utils for load_graph
)

# Add executable for the MPI version
add_executable(Parallel_SSSP_MPI
        src/main_mpi.cpp
        src/sssp_parallel_mpi.cpp
        src/utils.cpp
        src/graph.cpp
)

# Add executable for the Serial version
add_executable(Parallel_SSSP_SERIAL
        src/main_serial.cpp
        src/sssp_sequential.cpp
        src/utils.cpp
        src/graph.cpp
)
# Link OpenMP to the main program
target_link_libraries(Parallel_SSSP PRIVATE OpenMP::OpenMP_CXX)

# Link METIS to all executables that might need graph loading/partitioning
target_link_libraries(Parallel_SSSP PRIVATE ${METIS_LIBRARY})
target_link_libraries(generate_updates PRIVATE ${METIS_LIBRARY})
target_link_libraries(Parallel_SSSP_MPI PRIVATE ${METIS_LIBRARY})

# Link MPI libraries to the MPI executable
target_link_libraries(Parallel_SSSP_MPI PRIVATE MPI::MPI_CXX)
