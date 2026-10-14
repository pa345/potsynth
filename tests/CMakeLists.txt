set(POTSYNTH_REFDATA
  ${CMAKE_CURRENT_SOURCE_DIR}/data/ref_L13_s1.bin
  ${CMAKE_CURRENT_SOURCE_DIR}/data/ref_L60_s2.bin
  ${CMAKE_CURRENT_SOURCE_DIR}/data/ref_L185_s3.bin
  ${CMAKE_CURRENT_SOURCE_DIR}/data/ref_L185_s4.bin)

add_executable(test_core test_core.c)
target_link_libraries(test_core PRIVATE potsynth)
add_test(NAME core COMMAND test_core)

add_executable(test_mag_ref test_mag_ref.c)
target_link_libraries(test_mag_ref PRIVATE potsynth)
add_test(NAME mag_reference COMMAND test_mag_ref ${POTSYNTH_REFDATA})

add_executable(bench bench.c)
target_link_libraries(bench PRIVATE potsynth)
target_compile_definitions(bench PRIVATE _POSIX_C_SOURCE=200809L)

# optional: Legendre functions against GSL (an independent implementation);
# white-box test, compiled from the library sources
find_package(GSL QUIET)
if(GSL_FOUND)
  add_executable(test_alf test_alf.c ../src/alf.c ../src/util.c)
  target_include_directories(test_alf PRIVATE ../src ../include)
  target_link_libraries(test_alf PRIVATE GSL::gsl ${MATH_LIBRARY})
  add_test(NAME alf_vs_gsl COMMAND test_alf)
else()
  message(STATUS "GSL not found: skipping the Legendre oracle test")
endif()
