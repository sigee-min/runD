list(APPEND NODE_SOURCES
  src/compute/backend/accel.cpp
  src/compute/batch/run.cpp
  src/compute/job/accel.cpp
  src/compute/open/accel.cpp
)

if(RUND_TEST_NODE)
  set_source_files_properties(src/compute/open/accel.cpp PROPERTIES
    COMPILE_DEFINITIONS RUND_NODE_OPEN_PROBE=1)
endif()
