list(APPEND NODE_SOURCES
  src/compute/backend/accel.cpp
  src/compute/backend/accel/buffer.cpp
  src/compute/backend/accel/residency.cpp
  src/compute/backend/accel/transfer/download.cpp
  src/compute/backend/accel/transfer/prepare.cpp
  src/compute/backend/accel/transfer/upload.cpp
  src/compute/batch/run.cpp
  src/compute/job/accel.cpp
  src/compute/open/accel.cpp
)

if(RUND_TEST_NODE)
  set_source_files_properties(src/compute/open/accel.cpp PROPERTIES
    COMPILE_DEFINITIONS RUND_NODE_OPEN_PROBE=1)
endif()
