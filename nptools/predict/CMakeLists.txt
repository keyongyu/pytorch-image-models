# Build the reference ncnn open-set client (predict_ncnn.cpp).
#
#   cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
#   cmake --build build -j
#   ./build/predict_ncnn <run>/openset.ncnn.param <run>/openset.ncnn.bin \
#                        ../../posmlvx/class_84.txt photo.jpg 128
#   ./build/bench_ncnn   <run>/openset.ncnn.param <run>/openset.ncnn.bin \
#                        ../../posmlvx/train_256/posm_33 128 1 5 area   # dir size threads reps resize
#   ./build/opts_ncnn    <run>/openset.ncnn.param <run>/openset.ncnn.bin \
#                        ../../posmlvx/train_256/posm_33                # flag sweep
#
# ncnn is fetched and built as a subproject, pinned to NCNN_TAG below, so no ncnn install is
# needed: it is not packaged for Ubuntu, and the pip `ncnn` wheel ships only the Python extension
# module (no headers, no CMake config) so it cannot satisfy a C++ build. First configure clones and
# compiles it (~35s at -j20, then cached in the build dir). Build against a local checkout instead
# with -DFETCHCONTENT_SOURCE_DIR_NCNN=/path/to/ncnn.
#
# OpenCV 4 must be installed: `sudo apt install libopencv-dev`, or -DCMAKE_PREFIX_PATH=/opt/opencv.

cmake_minimum_required(VERSION 3.16)
project(openset_predict_ncnn LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()

# --- ncnn (fetched) -------------------------------------------------------------------------------
# Everything but the library is off: tools/examples/benchmarks/tests roughly triple the build and
# this client needs none of them. Vulkan off too -- the head is a normalize, a matmul and two
# subtractions, so a GPU upload costs more than it saves, and it keeps glslang out of the picture
# (hence GIT_SUBMODULES ""). Static lib: one self-contained binary, no LD_LIBRARY_PATH.
set(NCNN_TAG "20260526" CACHE STRING "ncnn release tag to fetch and build")
set(NCNN_BUILD_TOOLS      OFF CACHE BOOL "" FORCE)
set(NCNN_BUILD_EXAMPLES   OFF CACHE BOOL "" FORCE)
set(NCNN_BUILD_BENCHMARK  OFF CACHE BOOL "" FORCE)
set(NCNN_BUILD_TESTS      OFF CACHE BOOL "" FORCE)
set(NCNN_PYTHON           OFF CACHE BOOL "" FORCE)
set(NCNN_VULKAN           OFF CACHE BOOL "" FORCE)
set(NCNN_SHARED_LIB       OFF CACHE BOOL "" FORCE)
set(NCNN_INSTALL_SDK      OFF CACHE BOOL "" FORCE)

include(FetchContent)
FetchContent_Declare(
    ncnn
    GIT_REPOSITORY https://github.com/Tencent/ncnn.git
    GIT_TAG        ${NCNN_TAG}
    GIT_SHALLOW    TRUE
    GIT_SUBMODULES ""          # none needed with Vulkan/python off
    GIT_PROGRESS   TRUE
)
FetchContent_MakeAvailable(ncnn)

# --- OpenCV 4 -------------------------------------------------------------------------------------
# Only imgcodecs (decode) and imgproc (the INTER_AREA resize the preprocessing contract requires)
# are used -- naming them keeps a static link from dragging in highgui/videoio, which this client
# never touches. OpenCV 4 is what the distro ships (`sudo apt install libopencv-dev` -> 4.6 on
# noble); the Python side pins opencv-python-headless>=5 separately, and the two do not need to
# match because nothing here touches an API that changed between 4 and 5 -- verified by the C++ and
# Python clients returning identical distances.
find_package(OpenCV 4 REQUIRED COMPONENTS imgcodecs imgproc)

# --- OpenMP ---------------------------------------------------------------------------------------
# ncnn parallelises its layers with OpenMP. An installed ncnn usually propagates this through its
# imported target, but link it explicitly so a build against a static ncnn does not fail at the
# link step with undefined omp_* symbols. Optional: single-threaded still runs correctly.
find_package(OpenMP)

# predict_ncnn: the reference client. bench_ncnn: throughput, split into resize / Mat+normalize /
# forward, because which of the three dominates decides where optimising is worth anything.
# opts_ncnn: sweeps ncnn's optimisation flags for speed AND numerical drift -- meant to be run on
# the deployment device, since the fp16 flags are ARM-only and inert on x86.
foreach(tgt predict_ncnn bench_ncnn opts_ncnn)
    add_executable(${tgt} ${tgt}.cpp)
    target_link_libraries(${tgt} PRIVATE ncnn ${OpenCV_LIBS})
    target_include_directories(${tgt} PRIVATE ${OpenCV_INCLUDE_DIRS})
    if(OpenMP_CXX_FOUND)
        target_link_libraries(${tgt} PRIVATE OpenMP::OpenMP_CXX)
    endif()
endforeach()
if(NOT OpenMP_CXX_FOUND)
    message(WARNING "OpenMP not found -- ncnn will run single-threaded")
endif()

message(STATUS "ncnn   : ${NCNN_TAG} (fetched)")
message(STATUS "OpenCV : ${OpenCV_VERSION}")
message(STATUS "OpenMP : ${OpenMP_CXX_FOUND}")
