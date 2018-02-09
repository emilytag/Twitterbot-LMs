PROJECT(languagemodels)
CMAKE_MINIMUM_REQUIRED(VERSION 2.8)

set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)
set(CMAKE_CXX_FLAGS "-Wall -std=c++11 -Ofast -g")
#Change -O3 to -Ofast to run faster

enable_testing()

if(ENABLE_BOOST)
  message("-- Enabling Boost")
  if(DEFINED ENV{BOOST_ROOT})
    set(Boost_NO_SYSTEM_PATHS ON)
    if(DEFINED ${Boost_INCLUDE_DIR})
      get_filename_component(Boost_INCLUDE_DIR "${Boost_INCLUDE_DIR}" REALPATH BASE_DIR "${CMAKE_BINARY_DIR}")
    endif()
  endif()
  set(Boost_REALPATH ON)
  find_package(Boost COMPONENTS program_options regex serialization REQUIRED)
  message("-- Boost dir is " ${Boost_INCLUDE_DIR})
  include_directories(${Boost_INCLUDE_DIR})
  if(MSVC)
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} /LIBPATH:${Boost_LIBRARY_DIRS}")
    set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} /LIBPATH:${Boost_LIBRARY_DIRS}")
  else()
    set(LIBS ${LIBS} ${Boost_LIBRARIES})
  endif()
endif()

# look for Eigen
find_package(Eigen3 REQUIRED)
include_directories(${EIGEN3_INCLUDE_DIR})
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/dynet)

FIND_PACKAGE(Threads REQUIRED)
set(LIBS ${LIBS} ${CMAKE_THREAD_LIBS_INIT})

add_subdirectory(dynet)
ADD_EXECUTABLE(charlm charlm.cc)
ADD_EXECUTABLE(wordlm wordlm.cc)

target_link_libraries(charlm dynet ${LIBS})
target_link_libraries(wordlm dynet ${LIBS})

enable_testing()
