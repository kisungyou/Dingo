#----------------------------------------------------------------
# Generated CMake target import file.
#----------------------------------------------------------------

# Commands may need to know the format version.
set(CMAKE_IMPORT_FILE_VERSION 1)

# Import target "dingo::eigen_blas_static" for configuration ""
set_property(TARGET dingo::eigen_blas_static APPEND PROPERTY IMPORTED_CONFIGURATIONS NOCONFIG)
set_target_properties(dingo::eigen_blas_static PROPERTIES
  IMPORTED_LINK_INTERFACE_LANGUAGES_NOCONFIG "CXX"
  IMPORTED_LOCATION_NOCONFIG "${_IMPORT_PREFIX}/lib/libeigen_blas_static.a"
  )

list(APPEND _cmake_import_check_targets dingo::eigen_blas_static )
list(APPEND _cmake_import_check_files_for_dingo::eigen_blas_static "${_IMPORT_PREFIX}/lib/libeigen_blas_static.a" )

# Import target "dingo::eigen_lapack_static" for configuration ""
set_property(TARGET dingo::eigen_lapack_static APPEND PROPERTY IMPORTED_CONFIGURATIONS NOCONFIG)
set_target_properties(dingo::eigen_lapack_static PROPERTIES
  IMPORTED_LINK_INTERFACE_LANGUAGES_NOCONFIG "CXX;Fortran"
  IMPORTED_LOCATION_NOCONFIG "${_IMPORT_PREFIX}/lib/libeigen_lapack_static.a"
  )

list(APPEND _cmake_import_check_targets dingo::eigen_lapack_static )
list(APPEND _cmake_import_check_files_for_dingo::eigen_lapack_static "${_IMPORT_PREFIX}/lib/libeigen_lapack_static.a" )

# Commands beyond this point should not need to know the version.
set(CMAKE_IMPORT_FILE_VERSION)
