add_executable(dingo-bench runtime_compare.cpp)
target_link_libraries(dingo-bench PRIVATE dingo::dingo)
target_compile_features(dingo-bench PRIVATE cxx_std_17)

