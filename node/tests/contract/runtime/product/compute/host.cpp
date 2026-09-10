#include "../../../compute/allocation.hpp"
#include "../support.hpp"
#include "host/check.hpp"
#include "host/local.hpp"

#include <rund/compute.hpp>
#include <rund/compute/session.hpp>

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <string_view>
#include <vector>

int RunRuntimeComputeHostContract() {
  const runtime_compute_host_detail::ReadyOrder two_workers =
      runtime_compute_host_detail::CheckReadyOrder(2u);
  const runtime_compute_host_detail::ReadyOrder four_workers =
      runtime_compute_host_detail::CheckReadyOrder(4u);
  TEST_ASSERT(two_workers.ok);
  TEST_ASSERT(four_workers.ok);
  TEST_ASSERT(two_workers.allocations == 0u);
  TEST_ASSERT(four_workers.allocations == 0u);

  constexpr std::size_t kCount = 64u * 1024u + 17u;
  std::vector<std::int32_t> input(kCount);
  for (std::size_t index = 0u; index < input.size(); ++index) {
    input[index] = static_cast<std::int32_t>(index % 97u);
  }

  auto program =
      rund::compute::on(rund::compute::Target::cpu(2u))
          .map<std::int32_t>("node-host-twice", input.size(),
                             [](auto value) { return value * 2 + 5; })
          .compile();
  TEST_ASSERT(program);
  auto job = program->resident(input);
  TEST_ASSERT(job);

  rund::node::test_contract::TelemetryProbe telemetry{};
  rund::Session runtime{};
  TEST_ASSERT(
      runtime.open(rund::node::test_contract::Options(&telemetry)).ok());

  auto task = runtime.compute(*job).submit();
  const rund::compute::Poll submitted = task.poll();
  if (!submitted.submitted) {
    std::fprintf(stderr, "node cpu compute admission failed: %.*s\n",
                 static_cast<int>(submitted.error().size()),
                 submitted.error().data());
  }
  TEST_ASSERT(submitted.submitted);
  TEST_ASSERT(submitted.reason() == rund::compute::Reason::Ok);
  TEST_ASSERT(submitted.code() == rund::compute::Code::Ok);
  TEST_ASSERT(submitted.error().empty());
  const rund::compute::Completion result = task.wait();
  if (!result) {
    std::fprintf(stderr, "node cpu compute failed: %.*s\n",
                 static_cast<int>(result.error().size()),
                 result.error().data());
  }
  TEST_ASSERT(result);
  const rund::compute::Poll completed = task.poll();
  TEST_ASSERT(completed.submitted);
  TEST_ASSERT(completed.backend_submitted);
  TEST_ASSERT(completed.completed);
  TEST_ASSERT(completed.reason() == rund::compute::Reason::Ok);
  TEST_ASSERT(completed.code() == rund::compute::Code::Ok);
  TEST_ASSERT(completed.error().empty());
  TEST_ASSERT(result.reason() == rund::compute::Reason::Ok);
  TEST_ASSERT(result.error().empty());
  TEST_ASSERT(result.stats().backend == rund::compute::Backend::Cpu);
  TEST_ASSERT(result.stats().worker_count == 2u);
  TEST_ASSERT(result.stats().participating_workers >= 2u);
  TEST_ASSERT(result.stats().pipeline_compiles == 0u);
  TEST_ASSERT(result.stats().buffer_allocations == 0u);
  TEST_ASSERT(result.stats().download_events == 0u);
  TEST_ASSERT(telemetry.events == 1u);
  TEST_ASSERT(telemetry.event.source == rund::telemetry::Source::Compute);
  TEST_ASSERT(telemetry.event.level == rund::telemetry::Level::Basic);
  TEST_ASSERT(telemetry.event.session == 1u);
  TEST_ASSERT(telemetry.event.compute.backend == rund::compute::Backend::Cpu);
  TEST_ASSERT(telemetry.event.compute.code == rund::compute::Code::Ok);
  TEST_ASSERT(telemetry.event.compute.graph == result.stats().graph_hash);
  TEST_ASSERT(telemetry.event.compute.workers == 2u);
  TEST_ASSERT(telemetry.event.compute.active_workers >= 2u);
  TEST_ASSERT(telemetry.event.compute.tiles == result.stats().tile_count);
  TEST_ASSERT(telemetry.event.compute.dispatches == 1u);
  TEST_ASSERT(telemetry.event.compute.buffer_allocations == 0u);
  TEST_ASSERT(telemetry.event.detail.prepare_ns == 0u);
  TEST_ASSERT(telemetry.event.detail.work_ns == 0u);
  TEST_ASSERT(telemetry.event.detail.finish_ns == 0u);
  TEST_ASSERT(telemetry.event.replay.mode == rund::telemetry::Mode::None);
  TEST_ASSERT(telemetry.event.replay.plan ==
              rund::telemetry::Preparation::None);
  TEST_ASSERT(telemetry.event.replay.input_rows == 0u);
  TEST_ASSERT(telemetry.event.replay.input_bytes == 0u);
  TEST_ASSERT(telemetry.event.replay.choices == 0u);
  TEST_ASSERT(telemetry.event.replay.evidence_rows == 0u);
  TEST_ASSERT(telemetry.event.replay.evidence_bytes == 0u);
  TEST_ASSERT(telemetry.event.replay.retained_bytes == 0u);
  TEST_ASSERT(telemetry.event.replay.copied_bytes == 0u);
  TEST_ASSERT(telemetry.event.replay.physical_bytes == 0u);
  TEST_ASSERT(telemetry.event.replay.allocated_bytes == 0u);
  TEST_ASSERT(telemetry.event.replay.reserved_bytes == 0u);
  TEST_ASSERT(telemetry.event.replay.storage_growths == 0u);
  TEST_ASSERT(telemetry.event.replay.result_hash == 0u);
  TEST_ASSERT(telemetry.event.error().empty());
  TEST_ASSERT(telemetry.event.replay.code == rund::replay::Code::Ok);

  node_compute_allocation::Start();
  const rund::compute::Completion warm = runtime.compute(*job).submit().wait();
  node_compute_allocation::Stop();
  if (!warm || node_compute_allocation::Count() != 0u) {
    std::fprintf(
        stderr, "node cpu warm submit allocations=%llu reason=%.*s\n",
        static_cast<unsigned long long>(node_compute_allocation::Count()),
        static_cast<int>(warm.error().size()), warm.error().data());
  }
  TEST_ASSERT(warm);
  TEST_ASSERT(node_compute_allocation::Count() == 0u);

  TEST_ASSERT(compute_host_test::OutputMatches(*job, input));

  const std::vector<std::int32_t> empty;
  auto empty_program =
      rund::compute::on(rund::compute::Target::cpu(2u))
          .map<std::int32_t>("node-host-empty", 0u,
                             [](auto value) { return value + 1; })
          .compile();
  TEST_ASSERT(empty_program);
  auto empty_job = empty_program->resident(empty);
  TEST_ASSERT(empty_job);
  auto empty_task = runtime.compute(*empty_job).submit();
  const rund::compute::Completion empty_result = empty_task.wait();
  TEST_ASSERT(empty_result);
  const rund::compute::Poll empty_poll = empty_task.poll();
  TEST_ASSERT(empty_poll.submitted);
  TEST_ASSERT(!empty_poll.backend_submitted);
  TEST_ASSERT(empty_poll.completed);
  TEST_ASSERT(empty_result.stats().backend == rund::compute::Backend::Cpu);
  TEST_ASSERT(empty_result.stats().graph_hash != 0u);
  TEST_ASSERT(empty_result.stats().pipeline_compiles == 0u);
  TEST_ASSERT(empty_result.stats().buffer_allocations == 0u);
  TEST_ASSERT(empty_result.stats().download_events == 0u);
  const auto empty_output = empty_job->read();
  TEST_ASSERT(empty_output);
  TEST_ASSERT(empty_output->empty());
  TEST_ASSERT(empty_job->stats().output_hash != 0u);

  constexpr std::array<std::uint32_t, 2u> overflow{
      std::numeric_limits<std::uint32_t>::max(), 1u};
  constexpr std::array<std::uint32_t, 2u> recovered_input{1u, 2u};
  auto failure_program =
      rund::compute::on(rund::compute::Target::cpu(2u))
          .map<std::uint32_t>("node-host-overflow", overflow.size(),
                              [](auto value) { return value; })
          .scan(rund::compute::Scan::InclusiveSum)
          .compile();
  TEST_ASSERT(failure_program);
  auto failure_job = failure_program->resident(overflow);
  TEST_ASSERT(failure_job);
  const rund::compute::Completion overflow_result =
      runtime.compute(*failure_job).submit().wait();
  TEST_ASSERT(!overflow_result);
  TEST_ASSERT(overflow_result.reason() ==
              rund::compute::Reason::ScanSumOverflow);
  TEST_ASSERT(overflow_result.error() ==
              std::string_view{"compute_scan_sum_overflow"});
  TEST_ASSERT(failure_job->write(recovered_input));
  TEST_ASSERT(runtime.compute(*failure_job).submit().wait());
  const auto recovered = failure_job->read();
  TEST_ASSERT(recovered);
  TEST_ASSERT(*recovered == std::vector<std::uint32_t>({1u, 3u}));
  TEST_ASSERT(runtime_compute_host_detail::CheckCpuStepParity(runtime));
  TEST_ASSERT(runtime_compute_host_detail::CheckServerReplay(runtime));

  const rund::compute::Status completed_cancel = task.cancel();
  TEST_ASSERT(!completed_cancel);
  TEST_ASSERT(completed_cancel.error() ==
              std::string_view{"compute_already_completed"});

  TEST_ASSERT(compute_host_test::HasComputeTrace(runtime));

  const rund::compute::Poll invalid = rund::compute::Submission{}.poll();
  TEST_ASSERT(!invalid.submitted);
  TEST_ASSERT(!invalid.backend_submitted);
  TEST_ASSERT(invalid.completed);
  TEST_ASSERT(invalid.reason() == rund::compute::Reason::TaskInvalid);
  TEST_ASSERT(invalid.code() == rund::compute::Code::Invalid);
  TEST_ASSERT(invalid.error() == std::string_view{"compute_task_invalid"});

  TEST_ASSERT(runtime.close().ok());
  return 0;
}
