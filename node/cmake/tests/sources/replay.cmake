set(NODE_TEST_REPLAY_ACCEL_TEST_SOURCES
  tests/contract/runtime/task/replay/accel.cpp)

set(NODE_TEST_REPLAY_HOST_TEST_SOURCES
  tests/contract/runtime/task/replay/host.cpp
  tests/contract/runtime/task/replay/evidence.cpp
  tests/contract/runtime/task/replay/evidence/capacity.cpp
  tests/contract/runtime/task/replay/evidence/commit.cpp
  tests/contract/runtime/task/replay/evidence/net.cpp
  tests/contract/runtime/task/replay/evidence/reactor.cpp
  tests/contract/runtime/task/replay/evidence/reject.cpp)

set(NODE_TEST_REPLAY_RECORD_TEST_SOURCES
  ${RUND_NODE_TEST_TASK_ALLOCATION_SUPPORT}
  tests/contract/runtime/task/replay.cpp
  tests/contract/runtime/task/replay/decode.cpp
  tests/contract/runtime/task/replay/diff.cpp
  tests/contract/runtime/task/replay/failure.cpp
  tests/contract/runtime/task/replay/record.cpp
  tests/contract/runtime/task/replay/run.cpp
  tests/contract/runtime/task/replay/run/local/model.cpp
  tests/contract/runtime/task/replay/run/capacity.cpp
  tests/contract/runtime/task/replay/run/surface.cpp
  tests/contract/runtime/task/replay/run/scenario.cpp
  tests/contract/runtime/task/replay/run/lifetime.cpp
  tests/contract/runtime/task/replay/run/history.cpp)

set(NODE_TEST_REPLAY_KERNEL_TEST_SOURCES
  tests/contract/runtime/task/replay/kernel.cpp)

set(NODE_TEST_REPLAY_TELEMETRY_TEST_SOURCES
  tests/contract/runtime/task/replay/telemetry/parity.cpp)

set(NODE_TEST_REPLAY_PAYLOAD_TEST_SOURCES
  tests/contract/runtime/task/replay/payload.cpp)

set(NODE_TEST_REPLAY_PAYLOAD_CODEC_TEST_SOURCES
  ${RUND_NODE_TEST_TASK_ALLOCATION_SUPPORT}
  tests/contract/runtime/task/replay/payload/artifact.cpp
  tests/contract/runtime/task/replay/payload/codec.cpp)

set(NODE_TEST_REPLAY_STORAGE_CONFIG_TEST_SOURCES
  tests/contract/runtime/task/replay/storage/config.cpp)

set(NODE_TEST_REPLAY_PAYLOAD_STORE_TEST_SOURCES
  ${RUND_NODE_TEST_TASK_ALLOCATION_SUPPORT}
  tests/contract/runtime/task/replay/payload/store/suite.cpp
  tests/contract/runtime/task/replay/payload/store/local/model.cpp
  tests/contract/runtime/task/replay/payload/store/memory.cpp
  tests/contract/runtime/task/replay/payload/store/archive.cpp
  tests/contract/runtime/task/replay/payload/store/publish.cpp
  tests/contract/runtime/task/replay/payload/store/input.cpp
  tests/contract/runtime/task/replay/payload/store/diagnostic.cpp
  tests/contract/runtime/task/replay/payload/store/materialization.cpp)

set(NODE_TEST_REPLAY_SPILL_STORAGE_TEST_SOURCES
  tests/contract/runtime/task/replay/spill/storage.cpp
  tests/contract/runtime/task/replay/spill/cache.cpp
  tests/contract/runtime/task/replay/spill/local/model.cpp
  tests/contract/runtime/task/replay/spill/reject.cpp
  tests/contract/runtime/task/replay/spill/segments/generation.cpp
  tests/contract/runtime/task/replay/spill/segments/lifetime.cpp
  tests/contract/runtime/task/replay/spill/segments/budget.cpp
  tests/contract/runtime/task/replay/spill/segments/append.cpp
  tests/contract/runtime/task/replay/spill/segments/artifact.cpp
  tests/contract/runtime/task/replay/spill/segments/layout.cpp)
