#----------------------------------------------------------------
# Generated CMake target import file for configuration "Release".
#----------------------------------------------------------------

# Commands may need to know the format version.
set(CMAKE_IMPORT_FILE_VERSION 1)

# Import target "proj" for configuration "Release"
set_property(TARGET proj APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(proj PROPERTIES
  IMPORTED_LINK_INTERFACE_LIBRARIES_RELEASE "-lm"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libproj.so.12.0.0"
  IMPORTED_SONAME_RELEASE "libproj.so.12"
  )

list(APPEND _cmake_import_check_targets proj )
list(APPEND _cmake_import_check_files_for_proj "${_IMPORT_PREFIX}/lib/libproj.so.12.0.0" )

# Commands beyond this point should not need to know the version.
set(CMAKE_IMPORT_FILE_VERSION)
