include(FetchContent)

find_package(Catch2 3 QUIET CONFIG)

if(NOT Catch2_FOUND)
  if(DINGO_FETCH_TEST_DEPS)
    FetchContent_Declare(
      Catch2
      GIT_REPOSITORY https://github.com/catchorg/Catch2.git
      GIT_TAG v3.5.3)
    FetchContent_MakeAvailable(Catch2)
  else()
    message(WARNING
      "Catch2 not found and DINGO_FETCH_TEST_DEPS=OFF. "
      "Skipping test target creation.")
    return()
  endif()
endif()

add_executable(
  dingo-tests
  test_core.cpp
  test_linalg.cpp)

target_link_libraries(dingo-tests PRIVATE dingo::dingo Catch2::Catch2WithMain)
target_compile_features(dingo-tests PRIVATE cxx_std_17)

include(Catch)
catch_discover_tests(dingo-tests)
