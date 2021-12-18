cmake_minimum_required(VERSION 3.21)
project(aruco_detection)

set(CMAKE_CXX_STANDARD 23)

find_package(OpenCV REQUIRED)
find_package(Eigen3 REQUIRED)

include_directories( ${OpenCV_INCLUDE_DIRS} ${EIGEN3_INCLUDE_DIR} )

add_executable(aruco_detection main.cpp)
