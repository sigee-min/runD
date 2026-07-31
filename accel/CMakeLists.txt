add_library(accel INTERFACE)
target_compile_features(accel INTERFACE cxx_std_20)
target_include_directories(accel
  INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)
target_link_libraries(accel INTERFACE kernel)
