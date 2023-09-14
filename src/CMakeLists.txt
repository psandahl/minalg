file(GLOB HEADER_LIST CONFIGURE_DEPENDS "${minalg_SOURCE_DIR}/include/minalg/*.hpp")

add_library(minalg matrix.cpp ${HEADER_LIST})
target_include_directories(minalg PUBLIC ../include)
target_compile_features(minalg PUBLIC cxx_std_17)
target_compile_options(minalg PRIVATE -Werror -Wall -Wextra)

source_group(
  TREE "${PROJECT_SOURCE_DIR}/include"
  PREFIX "Header Files"
  FILES ${HEADER_LIST})
