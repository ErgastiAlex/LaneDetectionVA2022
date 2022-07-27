cmake_minimum_required(VERSION 3.16)

set(CMAKE_CXX_FLAGS_DEBUG "-DDEBUG")
project(linedetector)

find_package( OpenCV REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS} )

find_package( Eigen3 REQUIRED )
find_package( Boost )

#--- Line detector executable---
add_executable(linedetector linedetector.cpp linedetector.hh)
target_link_libraries(linedetector ${OpenCV_LIBS} )
target_link_libraries(linedetector Eigen3::Eigen )
target_link_libraries(linedetector Boost::boost)

target_include_directories(linedetector PUBLIC  ${CMAKE_CURRENT_SOURCE_DIR})


#--- Evaluator executable ---
add_executable(evaluator evaluator.cpp)
target_link_libraries(evaluator ${OpenCV_LIBS} )
