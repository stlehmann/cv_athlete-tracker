cmake_minimum_required(VERSION 3.5.0)

project(athlete_tracker)

# OpenCV libs
find_package( OpenCV REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS} )

# DLib librs
add_subdirectory(./lib/dlib-19.24/dlib)

# Boost
SET(BOOST_INCLUDEDIR "D:/lib/boost_1_83_0")
find_package(Boost 1.83.0 COMPONENTS program_options REQUIRED)
include_directories(${Boost_INCLUDE_DIR})

add_executable(athlete_tracker athlete_tracker.cpp)
target_link_libraries( athlete_tracker ${OpenCV_LIBS})
target_link_libraries( athlete_tracker dlib::dlib)
target_link_libraries( athlete_tracker "D:/lib/boost_1_83_0/lib64-msvc-14.3/libboost_program_options-vc143-mt-x64-1_83.lib")