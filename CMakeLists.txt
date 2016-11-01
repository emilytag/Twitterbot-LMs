PROJECT(disambiguation)
CMAKE_MINIMUM_REQUIRED(VERSION 2.8)

set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)
set(CMAKE_CXX_FLAGS "-Wall -std=c++11 -Ofast -g")
#Change -O3 to -Ofast to run faster

enable_testing()

# look for Boost
if(DEFINED ENV{BOOST_ROOT})
  set(Boost_NO_SYSTEM_PATHS ON)
endif()
set(Boost_REALPATH ON)
find_package(Boost COMPONENTS program_options serialization REQUIRED)
include_directories(${Boost_INCLUDE_DIR})
set(LIBS ${LIBS} ${Boost_LIBRARIES})

# look for CNN
#find_package(cnn REQUIRED)
#include_directories(${cnn_INCLUDE_DIR})

# look for Eigen
find_package(Eigen3 REQUIRED)
include_directories(${EIGEN3_INCLUDE_DIR})
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/cnn)

FIND_PACKAGE(Threads REQUIRED)
set(LIBS ${LIBS} ${CMAKE_THREAD_LIBS_INIT})

add_subdirectory(cnn)
ADD_EXECUTABLE(charlm charlm.cc)
ADD_EXECUTABLE(wordlm wordlm.cc)
ADD_EXECUTABLE(sample sample.cc)
target_link_libraries(charlm cnn ${LIBS})
target_link_libraries(wordlm cnn ${LIBS})
target_link_libraries(sample cnn ${LIBS})

enable_testing()
