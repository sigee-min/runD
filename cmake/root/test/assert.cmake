get_filename_component(RUND_TEST_ASSERT_ROOT
  "${CMAKE_CURRENT_LIST_DIR}/../../.." ABSOLUTE)

function(rund_test_assertion)
  if(TARGET rund-test-assertion)
    return()
  endif()
  add_library(rund-test-assertion STATIC EXCLUDE_FROM_ALL
    "${RUND_TEST_ASSERT_ROOT}/tools/test/assert.cpp")
  target_include_directories(rund-test-assertion PUBLIC
    "${RUND_TEST_ASSERT_ROOT}/tools")
  target_compile_features(rund-test-assertion PUBLIC cxx_std_20)
endfunction()
