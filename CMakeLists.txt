cmake_minimum_required(VERSION 3.14)
project(
  minalg
  VERSION 0.1
  DESCRIPTION "A minimal algebra library"
  LANGUAGES CXX
  )

if(CMAKE_PROJECT_NAME STREQUAL PROJECT_NAME)
    set(CMAKE_CXX_STANDARD 17)
    set(CMAKE_CXX_STANDARD_REQUIRED ON)
    set(CMAKE_CXX_EXTENSIONS OFF)
    set_property(GLOBAL PROPERTY USE_FOLDERS ON)
    include(CTest)
endif()

include(FetchContent)

add_subdirectory(src)

if (BUILD_TESTING)
    message(STATUS "Testing")
    add_subdirectory(tests)
endif()