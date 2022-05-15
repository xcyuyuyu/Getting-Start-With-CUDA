cmake_minimum_required(VERSION 3.10)

add_definitions(-w)
# set the project name
project(RGB2GRAY)

set(CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAFS} "-std=c++11 -pthread")

# opencv
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS}) 
set(OpenCV_LIBRARY_DIRS /usr/local/lib)
link_directories(${OpenCV_LIBRARY_DIRS})

message(STATUS "LIBS: " ${OpenCV_LIBS})
message(STATUS "LIB_DIRS: " ${OpenCV_LIBRARY_DIRS})
message(STATUS "INCLUDE_DIRS: " ${OpenCV_INCLUDE_DIRS})

# cuda
find_package(CUDA REQUIRED)
include_directories(${CUDA_INCLUDE_DIRS}) 
set(CUDA_LIBRARY_DIRS /usr/local/cuda/lib64)
link_directories(${CUDA_LIBRARY_DIRS})
# 
include_directories(./)

# add the executable
cuda_add_executable(RGB2GRAY main.cu rgb2gray.cpp reference_calc.cpp compare.cpp)
target_link_libraries(RGB2GRAY ${OpenCV_LIBS} ${CUDA_LIBRARIES})
