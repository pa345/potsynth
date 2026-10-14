set(POTSYNTH_REFDATA
  ${CMAKE_CURRENT_SOURCE_DIR}/data/ref_L13_s1.bin
  ${CMAKE_CURRENT_SOURCE_DIR}/data/ref_L60_s2.bin
  ${CMAKE_CURRENT_SOURCE_DIR}/data/ref_L185_s3.bin
  ${CMAKE_CURRENT_SOURCE_DIR}/data/ref_L185_s4.bin)

# public API
add_executable(test_core test_core.c)
target_link_libraries(test_core PRIVATE potsynth)
add_test(NAME core COMMAND test_core)

add_executable(test_mag_ref test_mag_ref.c)
target_link_libraries(test_mag_ref PRIVATE potsynth)
add_test(NAME mag_reference COMMAND test_mag_ref ${POTSYNTH_REFDATA})

add_executable(test_threads test_threads.c)
target_link_libraries(test_threads PRIVATE potsynth)
find_package(Threads)
if(CMAKE_USE_PTHREADS_INIT)
  target_compile_definitions(test_threads PRIVATE POTSYNTH_TEST_PTHREADS)
  target_link_libraries(test_threads PRIVATE Threads::Threads)
endif()
add_test(NAME threads COMMAND test_threads)

# internal components (white-box: compiled from the library sources)
set(POTSYNTH_INTERNAL_SRC ../src/alf.c ../src/util.c ../src/radial.c ../src/window.c
  ../extern/pocketfft/pocketfft.c)
add_executable(test_internal test_internal.c ${POTSYNTH_INTERNAL_SRC})
target_include_directories(test_internal PRIVATE ../src ../include ../extern/pocketfft)
target_link_libraries(test_internal PRIVATE ${MATH_LIBRARY})
if(CMAKE_C_COMPILER_ID MATCHES "GNU|Clang")
  target_compile_options(test_internal PRIVATE -ffp-contract=fast)
endif()
add_test(NAME internal COMMAND test_internal)

add_executable(bench bench.c)
target_link_libraries(bench PRIVATE potsynth)
target_compile_definitions(bench PRIVATE _POSIX_C_SOURCE=200809L)

# optional: Legendre functions against GSL (an independent implementation)
find_package(GSL QUIET)
if(GSL_FOUND)
  add_executable(test_alf test_alf.c ../src/alf.c ../src/util.c)
  target_include_directories(test_alf PRIVATE ../src ../include)
  target_link_libraries(test_alf PRIVATE GSL::gsl ${MATH_LIBRARY})
  add_test(NAME alf_vs_gsl COMMAND test_alf)
else()
  message(STATUS "GSL not found: skipping the Legendre oracle test")
endif()

# Python interface (needs the shared library)
find_package(Python3 COMPONENTS Interpreter)
if(BUILD_SHARED_LIBS AND Python3_FOUND)
  add_test(NAME python COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/test_python.py)
  set_tests_properties(python PROPERTIES ENVIRONMENT "POTSYNTH_LIBRARY=$<TARGET_FILE:potsynth>")
endif()
