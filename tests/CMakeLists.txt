# test suite: compiled from the library sources (white-box tests of internal
# components as well as the public interface); test.c includes test_*.c
add_executable(test_potsynth test.c test_common.c ${POTSYNTH_SOURCES})
potsynth_target_options(test_potsynth)
target_compile_definitions(test_potsynth PRIVATE TEST_DATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/data")

# concurrency test with pthreads (understood by ThreadSanitizer)
find_package(Threads)
if(CMAKE_USE_PTHREADS_INIT)
  target_compile_definitions(test_potsynth PRIVATE HAVE_PTHREADS)
  target_link_libraries(test_potsynth PRIVATE Threads::Threads)
endif()

# optional: Legendre functions against GSL (an independent implementation)
find_package(GSL QUIET)
if(GSL_FOUND)
  target_compile_definitions(test_potsynth PRIVATE HAVE_GSL)
  target_link_libraries(test_potsynth PRIVATE GSL::gsl)
else()
  message(STATUS "GSL not found: skipping the Legendre comparison with GSL")
endif()

add_test(NAME potsynth COMMAND test_potsynth)

# Python interface (needs the shared library)
find_package(Python3 COMPONENTS Interpreter)
if(BUILD_SHARED_LIBS AND Python3_FOUND)
  add_test(NAME python COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/test_python.py)
  set_tests_properties(python PROPERTIES ENVIRONMENT "POTSYNTH_LIBRARY=$<TARGET_FILE:potsynth>")
endif()

# benchmark
add_executable(bench bench.c)
target_link_libraries(bench PRIVATE potsynth)
