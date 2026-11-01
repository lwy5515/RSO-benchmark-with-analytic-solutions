cmake_minimum_required(VERSION 3.16)

set(TARGET evaluate_f2)
set(TARGET_SRC cpp/evaluate_f2.cc)

# Looking for deal.II library
find_package(deal.II 9.4 REQUIRED
  HINTS ${DEAL_II_DIR} $ENV{DEAL_II_DIR})
deal_ii_initialize_cached_variables()
project(${TARGET} LANGUAGES CXX)

# Looking for GSL library for double precision computation
find_package(GSL REQUIRED)

deal_ii_invoke_autopilot()

target_link_libraries(${TARGET} GSL::gsl GSL::gslcblas)
target_compile_features(${TARGET} PRIVATE cxx_std_17)
target_compile_definitions(${TARGET} PRIVATE
  RSO_F2_SOURCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}/cpp")

# Clang-format
find_program(CLANG_FORMAT_EXECUTABLE NAMES clang-format)
if(CLANG_FORMAT_EXECUTABLE)
  file(GLOB f2_sources CONFIGURE_DEPENDS
    "${CMAKE_CURRENT_SOURCE_DIR}/cpp/*.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/cpp/*.cc")
  add_custom_target(indent
    COMMAND "${CLANG_FORMAT_EXECUTABLE}" -i ${f2_sources}
    WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
    COMMENT "Format f_2 evaluation sources using the deal.II style"
    VERBATIM)
endif()
