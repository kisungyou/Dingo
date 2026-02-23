cmake_minimum_required(VERSION 3.20)

project(Dingo VERSION 0.1.0 LANGUAGES CXX)

option(DINGO_USE_SYSTEM_EIGEN "Use system-installed Eigen instead of vendored snapshot." OFF)
option(DINGO_BUILD_TESTS "Build Dingo tests." ON)
option(DINGO_BUILD_BENCHMARKS "Build Dingo runtime benchmarks." ON)
option(DINGO_ENABLE_EIGEN_BLAS "Build and link bundled Eigen BLAS from third_party/eigen/blas." OFF)
option(DINGO_ENABLE_EIGEN_LAPACK "Build and link bundled Eigen LAPACK from third_party/eigen/lapack." OFF)
option(DINGO_FETCH_TEST_DEPS "Allow tests to fetch dependencies (e.g., Catch2) from the network." ON)
option(DINGO_USE_EIGEN_BLAS_BACKEND "Define EIGEN_USE_BLAS for Dingo consumers (experimental)." OFF)
option(DINGO_USE_EIGEN_LAPACKE_BACKEND "Define EIGEN_USE_LAPACKE for Dingo consumers (requires LAPACKE headers/lib)." OFF)
option(DINGO_EIGEN_MPL2_ONLY "Define EIGEN_MPL2_ONLY to reject any non-MPL2 Eigen components at compile time." ON)

if(DINGO_ENABLE_EIGEN_LAPACK AND NOT DINGO_ENABLE_EIGEN_BLAS)
  message(FATAL_ERROR "DINGO_ENABLE_EIGEN_LAPACK requires DINGO_ENABLE_EIGEN_BLAS=ON.")
endif()
if(DINGO_USE_EIGEN_LAPACKE_BACKEND AND NOT DINGO_ENABLE_EIGEN_LAPACK)
  message(FATAL_ERROR "DINGO_USE_EIGEN_LAPACKE_BACKEND requires DINGO_ENABLE_EIGEN_LAPACK=ON.")
endif()

add_library(dingo INTERFACE)
add_library(dingo::dingo ALIAS dingo)

target_compile_features(dingo INTERFACE cxx_std_17)

target_include_directories(
  dingo
  INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include/dingo>
    $<INSTALL_INTERFACE:include>)

if(DINGO_USE_SYSTEM_EIGEN)
  find_package(Eigen3 3.4 REQUIRED NO_MODULE)
  target_link_libraries(dingo INTERFACE Eigen3::Eigen)
  if(DINGO_EIGEN_MPL2_ONLY)
    target_compile_definitions(dingo INTERFACE EIGEN_MPL2_ONLY)
  endif()
  if(DINGO_ENABLE_EIGEN_BLAS OR DINGO_ENABLE_EIGEN_LAPACK)
    message(FATAL_ERROR
      "Bundled Eigen BLAS/LAPACK is only supported with vendored Eigen. "
      "Set DINGO_USE_SYSTEM_EIGEN=OFF.")
  endif()
else()
  if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/third_party/eigen/Eigen/Dense")
    target_include_directories(dingo INTERFACE $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/third_party/eigen>)
  else()
    message(FATAL_ERROR "Vendored Eigen snapshot not found. Populate third_party/eigen/Eigen or configure -DDINGO_USE_SYSTEM_EIGEN=ON.")
  endif()

  # Provide Eigen3::Eigen target expected by Eigen's blas/lapack subprojects.
  if(NOT TARGET Eigen3::Eigen)
    add_library(Eigen3::Eigen INTERFACE IMPORTED GLOBAL)
    set_target_properties(Eigen3::Eigen PROPERTIES
      INTERFACE_INCLUDE_DIRECTORIES "${CMAKE_CURRENT_SOURCE_DIR}/third_party/eigen")
  endif()

  if(DINGO_EIGEN_MPL2_ONLY)
    target_compile_definitions(dingo INTERFACE EIGEN_MPL2_ONLY)
  endif()

  if(DINGO_ENABLE_EIGEN_BLAS)
    set(EIGEN_BUILD_BLAS ON CACHE BOOL "" FORCE)
    set(EIGEN_BUILD_LAPACK ${DINGO_ENABLE_EIGEN_LAPACK} CACHE BOOL "" FORCE)
    set(EIGEN_BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
    set(EIGEN_BUILD_TESTING OFF CACHE BOOL "" FORCE)
    set(EIGEN_LEAVE_TEST_IN_ALL_TARGET OFF CACHE BOOL "" FORCE)
    set(EIGEN_STANDARD_LIBRARIES_TO_LINK_TO "" CACHE STRING "" FORCE)

    add_subdirectory("${CMAKE_CURRENT_SOURCE_DIR}/third_party/eigen/blas" "${CMAKE_CURRENT_BINARY_DIR}/eigen-blas")
    target_link_libraries(dingo INTERFACE eigen_blas_static)
    if(DINGO_USE_EIGEN_BLAS_BACKEND)
      target_compile_definitions(dingo INTERFACE EIGEN_USE_BLAS)
    endif()

    if(DINGO_ENABLE_EIGEN_LAPACK)
      add_subdirectory("${CMAKE_CURRENT_SOURCE_DIR}/third_party/eigen/lapack" "${CMAKE_CURRENT_BINARY_DIR}/eigen-lapack")
      target_link_libraries(dingo INTERFACE eigen_lapack_static)
      if(DINGO_USE_EIGEN_LAPACKE_BACKEND)
        target_compile_definitions(dingo INTERFACE EIGEN_USE_LAPACKE)
      endif()
    endif()
  endif()
endif()

include(CMakePackageConfigHelpers)

configure_package_config_file(
  "${CMAKE_CURRENT_SOURCE_DIR}/cmake/DingoConfig.cmake.in"
  "${CMAKE_CURRENT_BINARY_DIR}/DingoConfig.cmake"
  INSTALL_DESTINATION "lib/cmake/Dingo")

write_basic_package_version_file(
  "${CMAKE_CURRENT_BINARY_DIR}/DingoConfigVersion.cmake"
  VERSION "${PROJECT_VERSION}"
  COMPATIBILITY SameMajorVersion)

install(TARGETS dingo EXPORT DingoTargets)
if(TARGET eigen_blas_static)
  install(TARGETS eigen_blas_static EXPORT DingoTargets
          RUNTIME DESTINATION bin
          LIBRARY DESTINATION lib
          ARCHIVE DESTINATION lib)
endif()
if(TARGET eigen_lapack_static)
  install(TARGETS eigen_lapack_static EXPORT DingoTargets
          RUNTIME DESTINATION bin
          LIBRARY DESTINATION lib
          ARCHIVE DESTINATION lib)
endif()
install(EXPORT DingoTargets NAMESPACE dingo:: DESTINATION "lib/cmake/Dingo")
install(FILES "${CMAKE_CURRENT_BINARY_DIR}/DingoConfig.cmake"
              "${CMAKE_CURRENT_BINARY_DIR}/DingoConfigVersion.cmake"
        DESTINATION "lib/cmake/Dingo")
install(FILES "${CMAKE_CURRENT_SOURCE_DIR}/LICENSE"
              "${CMAKE_CURRENT_SOURCE_DIR}/THIRD_PARTY_NOTICES.md"
        DESTINATION "share/dingo")
install(DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/LICENSES/" DESTINATION "share/dingo/licenses")

install(FILES "${CMAKE_CURRENT_SOURCE_DIR}/cmake/dingo_install_header" DESTINATION "include" RENAME "dingo")
install(DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/include/dingo/" DESTINATION "include/dingo_lib")
if(NOT DINGO_USE_SYSTEM_EIGEN AND EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/third_party/eigen/Eigen")
  install(DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/third_party/eigen/Eigen" DESTINATION "include")
  if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/third_party/eigen/licenses")
    install(DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/third_party/eigen/licenses/" DESTINATION "share/dingo/third_party/eigen/licenses")
  endif()
endif()

if(DINGO_BUILD_TESTS)
  enable_testing()
  add_subdirectory(tests)
endif()

if(DINGO_BUILD_BENCHMARKS)
  add_subdirectory(benchmarks)
endif()
