FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/03597a01ee50ed33e9dfd640b249b4be3799d395.zip
)
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

enable_testing()

add_executable(
  minalg_test
  matrix_test.cpp
)
target_compile_features(minalg_test PRIVATE cxx_std_17)
target_compile_options(minalg_test PRIVATE -Werror -Wall -Wextra)

target_link_libraries(
  minalg_test
  PRIVATE
  minalg
  GTest::gtest_main
)

add_test(NAME minalg_test COMMAND minalg_test)