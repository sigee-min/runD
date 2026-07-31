#include "test/assert.hpp"

#include <rund/session.hpp>
#include <rund/session/memory.hpp>
#include <rund/task/stats.hpp>
#include <rund/task/api.hpp>

#include <limits>

int RunRuntimeMemoryContract() {
  const rund::PreparedMemory snapshot{
      .capacity =
          {
              .code = rund::ReasonCode::Ok,
              .worker_count = 2u,
              .requested_lane_capacity = 4u,
              .available_lane_capacity = 4u,
              .requested_reduction_input_capacity = 4u,
              .available_reduction_input_capacity = 4u,
              .requested_reduction_output_capacity = 2u,
              .available_reduction_output_capacity = 2u,
              .requested_epoch_workers = 2u,
              .available_epoch_workers = 2u,
          },
      .lane_capacity = 4u,
      .lane_high_water = 4u,
      .reduction_input_capacity = 4u,
      .reduction_output_capacity = 2u,
      .reduction_input_high_water = 4u,
      .reduction_output_high_water = 2u,
      .epoch_current = 1u,
  };
  TEST_ASSERT(snapshot.capacity.ok());
  TEST_ASSERT(snapshot.capacity.valid());
  TEST_ASSERT(snapshot.capacity.error().empty());
  TEST_ASSERT(snapshot.lane_capacity == 4u);
  TEST_ASSERT(snapshot.lane_high_water == 4u);
  TEST_ASSERT(snapshot.reduction_output_high_water == 2u);

  rund::Session runtime{};
  rund::SessionConfig options{};
  options.id = 77u;
  options.workers = 1u;
  options.scheduler.task_workers = 1u;
  TEST_ASSERT(runtime.open(options));
  bool recorded = false;
  bool worker_recorded = true;
  rund::task::Status worker_join{};
  rund::PreparedMemory failed_snapshot = snapshot;
  failed_snapshot.capacity.code =
      rund::ReasonCode::PreparedGenerationMismatch;
  TEST_ASSERT(failed_snapshot.capacity.valid());
  TEST_ASSERT(!failed_snapshot.capacity.ok());
  TEST_ASSERT(failed_snapshot.capacity.error() ==
              "prepared_generation_mismatch");
  const rund::Session::Result scope = runtime.scope([&] {
    rund::PreparedMemory invalid_snapshot = snapshot;
    invalid_snapshot.capacity.code = rund::ReasonCode::TaskFailed;
    TEST_ASSERT(!invalid_snapshot.capacity.valid());
    TEST_ASSERT(!rund::record_memory(invalid_snapshot));
    invalid_snapshot.capacity.code = static_cast<rund::ReasonCode>(
        std::numeric_limits<std::uint16_t>::max());
    TEST_ASSERT(!invalid_snapshot.capacity.valid());
    TEST_ASSERT(!rund::record_memory(invalid_snapshot));
    recorded = rund::record_memory(snapshot);
    const rund::task::Handle worker =
        rund::task::spawn("prepared-memory-record-worker", [&] {
          worker_recorded = rund::record_memory(failed_snapshot);
        });
    worker_join = rund::task::join(worker);
  });
  TEST_ASSERT(recorded);
  TEST_ASSERT(worker_join);
  TEST_ASSERT(!worker_recorded);
  TEST_ASSERT(scope);
  TEST_ASSERT(scope.memory().capacity.ok());
  TEST_ASSERT(scope.memory().capacity.error().empty());
  TEST_ASSERT(scope.memory().lane_high_water == snapshot.lane_high_water);
  TEST_ASSERT(runtime.close());

  return 0;
}
