#include "../../compute/allocation.hpp"
#include "test/assert.hpp"

#include <rund/compute.hpp>
#include <rund/compute/session.hpp>
#include <rund/session.hpp>

#include <array>
#include <cstdint>
#include <cstdio>
#include <string_view>
#include <vector>

int RunRuntimeComputeResidentStressContract() {
  std::array<std::int32_t, 5> input{1, 2, 3, 4, 5};
  auto program =
      rund::compute::on(rund::compute::Target::cpu(2u))
          .map<std::int32_t>("resident-stress", input.size(),
                             [](auto value) { return value * 3 + 1; })
          .compile();
  TEST_ASSERT(program);
  auto job = program->resident(input);
  TEST_ASSERT(job);

  rund::SessionConfig options{};
  options.id = 1u;
  options.workers = 2u;
  options.scheduler.task_workers = 2u;
  options.scheduler.task_capacity = 8u;
  options.scheduler.ready_queue_capacity = 8u;
  rund::Session runtime{};
  TEST_ASSERT(runtime.open(options));
  TEST_ASSERT(job->write(input));
  TEST_ASSERT(runtime.compute(*job).submit().wait());
  constexpr std::uint32_t resident_iterations = 10'000u;
  node_compute_allocation::Start();
  for (std::uint32_t iteration = 0u; iteration < resident_iterations;
       ++iteration) {
    input[0] = static_cast<std::int32_t>(iteration);
    input[4] = static_cast<std::int32_t>(iteration + 4u);
    TEST_ASSERT(job->write(input));
    auto task = runtime.compute(*job).submit();
    const auto status = task.wait();
    if (!status) {
      const auto poll = task.poll();
      const auto error = status.error();
      const auto reason = poll.error();
      std::fprintf(
          stderr,
          "resident stress iteration %u: %.*s, completion=%u, "
          "completion_reason=%.*s\n",
          static_cast<unsigned>(iteration), static_cast<int>(error.size()),
          error.empty() ? "" : error.data(), poll.completed ? 1u : 0u,
          static_cast<int>(reason.size()), reason.empty() ? "" : reason.data());
    }
    TEST_ASSERT(status);
    const auto stats = job->stats();
    TEST_ASSERT(stats.pipeline_compiles == 0u);
    TEST_ASSERT(stats.buffer_allocations == 0u);
    TEST_ASSERT(stats.download_events == 0u);
  }
  node_compute_allocation::Stop();
  TEST_ASSERT(node_compute_allocation::Count() == 0u);
  const auto output = job->read();
  TEST_ASSERT(output);
  TEST_ASSERT((*output)[0] ==
              static_cast<std::int32_t>((resident_iterations - 1u) * 3u + 1u));
  TEST_ASSERT((*output)[4] ==
              static_cast<std::int32_t>((resident_iterations + 3u) * 3u + 1u));

  constexpr std::size_t large_count = 1u << 20u;
  std::vector<std::int32_t> large_input(large_count, 9);
  auto large_program =
      rund::compute::on(rund::compute::Target::cpu(2u))
          .map<std::int32_t>("drain-running", large_count,
                             [](auto value) { return value * 7 + 3; })
          .compile();
  TEST_ASSERT(large_program);
  auto large_job = large_program->resident(large_input);
  TEST_ASSERT(large_job);
  auto running = runtime.compute(*large_job).submit();
  TEST_ASSERT(runtime.drain());
  const auto rejected = runtime.compute(*job).submit().wait();
  TEST_ASSERT(!rejected);
  TEST_ASSERT(rejected.error() == std::string_view{"compute_runtime_draining"});
  TEST_ASSERT(running.wait());
  const auto large_output = large_job->read();
  TEST_ASSERT(large_output);
  TEST_ASSERT((*large_output)[0] == 66);
  TEST_ASSERT(runtime.close());
  return 0;
}
