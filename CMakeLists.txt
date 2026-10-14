cmake_minimum_required(VERSION 3.15)
project(potsynth VERSION 0.1.0 LANGUAGES C)

option(POTSYNTH_OPENMP "Use OpenMP threads" ON)
option(POTSYNTH_NATIVE "Optimise for the build machine (-march=native)" ON)
option(POTSYNTH_TESTS "Build the tests" ON)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)

add_library(potsynth
  src/alf.c
  src/direct.c
  src/eval.c
  src/mag.c
  src/radial.c
  src/setup.c
  src/util.c
  src/window.c
  extern/pocketfft/pocketfft.c)

target_include_directories(potsynth
  PUBLIC  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include> $<INSTALL_INTERFACE:include>
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src ${CMAKE_CURRENT_SOURCE_DIR}/extern/pocketfft)

set_target_properties(potsynth PROPERTIES
  VERSION ${PROJECT_VERSION}
  SOVERSION ${PROJECT_VERSION_MAJOR}
  C_VISIBILITY_PRESET hidden)

target_compile_definitions(potsynth PRIVATE POTSYNTH_BUILDING)
if(BUILD_SHARED_LIBS)
  target_compile_definitions(potsynth PUBLIC POTSYNTH_SHARED)
endif()

if(CMAKE_C_COMPILER_ID MATCHES "GNU|Clang")
  # -ffp-contract=fast: allow fused multiply-add (strict ISO modes disable it,
  # which halves the speed of every inner loop)
  target_compile_options(potsynth PRIVATE -Wall -Wextra -ffp-contract=fast)
  if(POTSYNTH_NATIVE)
    target_compile_options(potsynth PRIVATE -march=native)
  endif()
elseif(MSVC)
  target_compile_options(potsynth PRIVATE /fp:contract)
endif()

find_library(MATH_LIBRARY m)
if(MATH_LIBRARY)
  target_link_libraries(potsynth PUBLIC ${MATH_LIBRARY})
endif()

if(POTSYNTH_OPENMP)
  find_package(OpenMP COMPONENTS C)
  if(OpenMP_C_FOUND)
    target_link_libraries(potsynth PRIVATE OpenMP::OpenMP_C)
  endif()
endif()
if(NOT OpenMP_C_FOUND AND CMAKE_C_COMPILER_ID MATCHES "GNU|Clang")
  target_compile_options(potsynth PRIVATE -Wno-unknown-pragmas)
endif()

include(GNUInstallDirs)
install(TARGETS potsynth
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
install(FILES include/potsynth.h include/potsynth_mag.h
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

if(POTSYNTH_TESTS)
  enable_testing()
  add_subdirectory(tests)
endif()
