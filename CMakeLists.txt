cmake_minimum_required(VERSION 3.15)
project(potsynth VERSION 0.1.0 LANGUAGES C)

option(POTSYNTH_OPENMP "Use OpenMP threads" ON)
option(POTSYNTH_NATIVE "Optimize for the build machine (-march=native)" ON)
option(POTSYNTH_TESTS "Build the tests" ON)
option(POTSYNTH_EXAMPLES "Build the example programs" ON)
option(POTSYNTH_NO_VECEXT "Use the portable scalar kernels instead of vector extensions" OFF)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)

set(POTSYNTH_SOURCES
  ${PROJECT_SOURCE_DIR}/src/alf.c
  ${PROJECT_SOURCE_DIR}/src/direct.c
  ${PROJECT_SOURCE_DIR}/src/error.c
  ${PROJECT_SOURCE_DIR}/src/eval.c
  ${PROJECT_SOURCE_DIR}/src/fast.c
  ${PROJECT_SOURCE_DIR}/src/gfc.c
  ${PROJECT_SOURCE_DIR}/src/grav.c
  ${PROJECT_SOURCE_DIR}/src/mag.c
  ${PROJECT_SOURCE_DIR}/src/radial.c
  ${PROJECT_SOURCE_DIR}/src/setup.c
  ${PROJECT_SOURCE_DIR}/src/shc.c
  ${PROJECT_SOURCE_DIR}/src/util.c
  ${PROJECT_SOURCE_DIR}/src/window.c
  ${PROJECT_SOURCE_DIR}/extern/pocketfft/pocketfft.c)

find_library(MATH_LIBRARY m)

if(POTSYNTH_OPENMP)
  find_package(OpenMP COMPONENTS C)
endif()

# compile options shared by the library and by targets built from its sources (tests)
function(potsynth_target_options target)
  target_include_directories(${target} PRIVATE
    ${PROJECT_SOURCE_DIR}/include ${PROJECT_SOURCE_DIR}/src ${PROJECT_SOURCE_DIR}/extern/pocketfft)
  target_compile_definitions(${target} PRIVATE POTSYNTH_BUILDING)

  if(POTSYNTH_NO_VECEXT)
    target_compile_definitions(${target} PRIVATE POTSYNTH_NO_VECEXT)
  endif()

  if(CMAKE_C_COMPILER_ID MATCHES "GNU|Clang")
    # -ffp-contract=fast: allow fused multiply-add (strict ISO modes disable it,
    # which halves the speed of every inner loop)
    target_compile_options(${target} PRIVATE -Wall -Wextra -ffp-contract=fast)
    if(POTSYNTH_NATIVE)
      target_compile_options(${target} PRIVATE -march=native)
    endif()
  elseif(MSVC)
    target_compile_options(${target} PRIVATE /fp:contract)
  endif()

  if(OpenMP_C_FOUND)
    target_link_libraries(${target} PRIVATE OpenMP::OpenMP_C)
  elseif(CMAKE_C_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(${target} PRIVATE -Wno-unknown-pragmas)
  endif()

  if(MATH_LIBRARY)
    target_link_libraries(${target} PUBLIC ${MATH_LIBRARY})
  endif()
endfunction()

add_library(potsynth ${POTSYNTH_SOURCES})
potsynth_target_options(potsynth)

target_include_directories(potsynth PUBLIC
  $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include> $<INSTALL_INTERFACE:include>)

set_target_properties(potsynth PROPERTIES
  VERSION ${PROJECT_VERSION}
  SOVERSION ${PROJECT_VERSION_MAJOR}
  C_VISIBILITY_PRESET hidden)

if(BUILD_SHARED_LIBS)
  target_compile_definitions(potsynth PUBLIC POTSYNTH_SHARED)
endif()

include(GNUInstallDirs)
install(TARGETS potsynth
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
install(FILES include/potsynth.h include/potsynth_mag.h include/potsynth_grav.h
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

if(POTSYNTH_EXAMPLES)
  add_executable(example_mag examples/example_mag.c)
  target_link_libraries(example_mag PRIVATE potsynth)
  add_executable(example_grav examples/example_grav.c)
  target_link_libraries(example_grav PRIVATE potsynth)
endif()

if(POTSYNTH_TESTS)
  enable_testing()
  add_subdirectory(tests)
endif()
