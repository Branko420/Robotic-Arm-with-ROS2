# generated from ament/cmake/core/templates/nameConfig.cmake.in

# prevent multiple inclusion
if(_deptrum-ros-driver-aurora930_CONFIG_INCLUDED)
  # ensure to keep the found flag the same
  if(NOT DEFINED deptrum-ros-driver-aurora930_FOUND)
    # explicitly set it to FALSE, otherwise CMake will set it to TRUE
    set(deptrum-ros-driver-aurora930_FOUND FALSE)
  elseif(NOT deptrum-ros-driver-aurora930_FOUND)
    # use separate condition to avoid uninitialized variable warning
    set(deptrum-ros-driver-aurora930_FOUND FALSE)
  endif()
  return()
endif()
set(_deptrum-ros-driver-aurora930_CONFIG_INCLUDED TRUE)

# output package information
if(NOT deptrum-ros-driver-aurora930_FIND_QUIETLY)
  message(STATUS "Found deptrum-ros-driver-aurora930: 0.0.0 (${deptrum-ros-driver-aurora930_DIR})")
endif()

# warn when using a deprecated package
if(NOT "" STREQUAL "")
  set(_msg "Package 'deptrum-ros-driver-aurora930' is deprecated")
  # append custom deprecation text if available
  if(NOT "" STREQUAL "TRUE")
    set(_msg "${_msg} ()")
  endif()
  # optionally quiet the deprecation message
  if(NOT deptrum-ros-driver-aurora930_DEPRECATED_QUIET)
    message(DEPRECATION "${_msg}")
  endif()
endif()

# flag package as ament-based to distinguish it after being find_package()-ed
set(deptrum-ros-driver-aurora930_FOUND_AMENT_PACKAGE TRUE)

# include all config extra files
set(_extras "ament_cmake_export_libraries-extras.cmake")
foreach(_extra ${_extras})
  include("${deptrum-ros-driver-aurora930_DIR}/${_extra}")
endforeach()
