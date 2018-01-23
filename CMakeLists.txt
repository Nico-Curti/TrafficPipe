cmake_minimum_required (VERSION 3.1)

set(default_build_type "Release")
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  message(STATUS "Setting build type to '${default_build_type}' as none was specified.")
  set(CMAKE_BUILD_TYPE "${default_build_type}" CACHE
      STRING "Choose the type of build." FORCE)
  # Set the possible values of build type for cmake-gui
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS
    "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif()

######################################## macro for Visual Studio warnings ===========================
macro(configure_link_flags)
  set(MSVC_C_CXX_FLAGS
    CMAKE_C_FLAGS_DEBUG
    CMAKE_C_FLAGS_MINSIZEREL
    CMAKE_C_FLAGS_RELEASE
    CMAKE_C_FLAGS_RELWITHDEBINFO
    CMAKE_CXX_FLAGS_DEBUG
    CMAKE_CXX_FLAGS_MINSIZEREL
    CMAKE_CXX_FLAGS_RELEASE
    CMAKE_CXX_FLAGS_RELWITHDEBINFO
  )
  if(${VCPKG_TARGET_TRIPLET} MATCHES "static")
    message(STATUS
      "VCPKG: static link"
    )
    foreach(flag ${MSVC_C_CXX_FLAGS})
      if(${flag} MATCHES "/MD")
        string(REGEX REPLACE "/MD" "/MT" ${flag} "${${flag}}")
      endif()
    endforeach()
    set(VCPKG_CRT_LINKAGE "static")
    set(VCPKG_LIBRARY_LINKAGE "static")
  else()
    message(STATUS
      "VCPKG: dynamic link"
    )
    foreach(flag ${MSVC_C_CXX_FLAGS})
      if(${flag} MATCHES "/MT")
        string(REGEX REPLACE "/MT" "/MD" ${flag} "${${flag}}")
      endif()
    endforeach()
    set(VCPKG_CRT_LINKAGE "dynamic")
    set(VCPKG_LIBRARY_LINKAGE "dynamic")
  endif()
endmacro()


if(MSVC)
  add_definitions(-D_SCL_SECURE_NO_WARNINGS)
  add_definitions(-D_CRT_SECURE_NO_WARNINGS)
  add_definitions(-DBOOST_NO_RVALUE_REFERENCES)
  configure_link_flags()
endif()

if (APPLE)
  add_definitions(-Wno-deprecated-declarations)
endif()

###########################################################################################

### Set output directories on a per-configuration base
# Single configuration
set( CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR} )
# Multi configuration
foreach( OUTPUTCONFIG ${CMAKE_CONFIGURATION_TYPES} )
    string( TOUPPER ${OUTPUTCONFIG} OUTPUTCONFIG )
    set( CMAKE_RUNTIME_OUTPUT_DIRECTORY_${OUTPUTCONFIG} ${CMAKE_SOURCE_DIR} )
endforeach( OUTPUTCONFIG CMAKE_CONFIGURATION_TYPES )


project (traffic)
enable_language (CXX)
set (CMAKE_CXX_STANDARD 11)
set (SRC_DIR ${CMAKE_SOURCE_DIR}/cpp)
set (HPP_DIR ${CMAKE_SOURCE_DIR}/cpp/Data-Analysis/lib) # directory of hpp
include_directories(${SRC_DIR})
include_directories(${HPP_DIR})

find_package(OpenMP REQUIRED)
if (OpenMP_CXX_VERSION_MAJOR LESS 4.0)
  message(FATAL_ERROR " Your OpenMP is too old. Required OpenMP 4.0. Please upgrade.")
endif()
if (OpenMP_FOUND)
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
endif()
if (CYGWIN)
  set (CMAKE_CXX_FLAGS "-std=gnu++14 ${CMAKE_CXX_FLAGS}")
endif()
if (NOT MSVC)
  set (CMAKE_CXX_FLAGS "-ffast-math ${CMAKE_CXX_FLAGS}") # perform more aggressive floating-point optimizations
  set (CMAKE_C_FLAGS "-ffast-math ${CMAKE_C_FLAGS}")
endif()

add_executable( fbp_ga 
                ${SRC_DIR}/fbp_ga.cpp
                ${HPP_DIR}/parse_args.hpp
                ${HPP_DIR}/grid_search.hpp
                ${HPP_DIR}/cv_classifier.hpp
                ${HPP_DIR}/classifier.hpp
                ${HPP_DIR}/neural_network.hpp
                ${HPP_DIR}/hyperparams.hpp
                ${HPP_DIR}/fstat.hpp
                ${HPP_DIR}/fmath.hpp
                ${HPP_DIR}/patterns.hpp
                ${HPP_DIR}/os.hpp
                ${HPP_DIR}/ReplicatedFBP.hpp
              )

add_executable( fbp_val 
                ${SRC_DIR}/fbp_val.cpp
                ${HPP_DIR}/parse_args.hpp
                ${HPP_DIR}/os.hpp
                ${HPP_DIR}/fstat.hpp
                ${HPP_DIR}/fmath.hpp
                ${HPP_DIR}/ReplicatedFBP.hpp
                ${HPP_DIR}/classifier.hpp
                ${HPP_DIR}/hyperparams.hpp
              )
