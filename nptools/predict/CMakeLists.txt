# Build the reference ncnn open-set client (predict_ncnn.cpp).
#
#   cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
#   cmake --build build -j
#   ./build/predict_ncnn <run>/openset.ncnn.param <run>/openset.ncnn.bin \
#                        ../../posmlvx/class_84.txt photo.jpg 128
#
# ncnn is fetched and built as a subproject, pinned to tag NCNN_TAG below, so the build does not
# depend on a system install -- ncnn is not packaged for Ubuntu, and the pip `ncnn` wheel ships only
# the Python extension module (no headers, no CMake config), so it cannot satisfy a C++ build.
# The first configure clones and compiles ncnn (a few minutes); after that it is cached in the build
# directory. Set -DFETCHCONTENT_SOURCE_DIR_NCNN=/path/to/ncnn to build against a local checkout
# instead of cloning.
#
# OpenCV must be installed (headers + libs): `sudo apt install libopencv-dev`, or point CMake at a
# local prefix with -DCMAKE_PREFIX_PATH=/opt/opencv.

cmake_minimum_required(VERSION 3.16)
project(openset_predict_ncnn LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()

set(NCNN_TAG "20260526" CACHE STRING "ncnn release tag to fetch and build")

# --- ncnn (fetched) -------------------------------------------------------------------------------
# Everything but the library itself is turned off: tools/examples/benchmarks/tests roughly triple
# the build, and this client needs none of them. Vulkan is off too -- the head is a normalize, a
# matmul and two subtractions, so a GPU upload costs more than it saves, and it keeps the glslang
# submodule out of the picture (hence GIT_SUBMODULES ""). Static lib: one binary, nothing to install
# or set LD_LIBRARY_PATH for.
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

# --- OpenCV ---------------------------------------------------------------------------------------
# Only imgcodecs (decode) and imgproc (the INTER_AREA resize the preprocessing contract requires)
# are used; naming them keeps a static link from pulling in highgui/videoio. OpenCV 5 is what the
# Python side pins (opencv-python-headless>=5.0.0.93), but 4.x works -- nothing here touches an API
# that changed between them.
find_package(OpenCV 5 QUIET COMPONENTS imgcodecs imgproc)
if(NOT OpenCV_FOUND)
    find_package(OpenCV REQUIRED COMPONENTS imgcodecs imgproc)
    message(STATUS "OpenCV 5 not found; building against ${OpenCV_VERSION}")
endif()

# --- OpenMP ---------------------------------------------------------------------------------------
# ncnn parallelises its layers with OpenMP. Linked explicitly because a static ncnn does not
# propagate it, and the link then fails on undefined omp_* symbols. Optional: without it ncnn still
# runs correctly, single-threaded.
find_package(OpenMP)

add_executable(predict_ncnn predict_ncnn.cpp)
target_link_libraries(predict_ncnn PRIVATE ncnn ${OpenCV_LIBS})
target_include_directories(predict_ncnn PRIVATE ${OpenCV_INCLUDE_DIRS})
if(OpenMP_CXX_FOUND)
    target_link_libraries(predict_ncnn PRIVATE OpenMP::OpenMP_CXX)
else()
    message(WARNING "OpenMP not found -- ncnn will run single-threaded")
endif()

message(STATUS "ncnn   : ${NCNN_TAG} (fetched)")
message(STATUS "OpenCV : ${OpenCV_VERSION}")
message(STATUS "OpenMP : ${OpenMP_CXX_FOUND}")
