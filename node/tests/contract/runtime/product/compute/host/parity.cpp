#include "local.hpp"

#include <rund/compute.hpp>

#include <chrono>
#include <cstddef>
#include <cstdio>
#include <vector>

namespace runtime_compute_host_detail {

[[nodiscard]] bool CheckCpuStepParity(rund::Session &runtime) {
  using namespace rund::compute;
  constexpr std::size_t count = 4u * 1024u + 17u;
  const std::vector<std::int32_t> input(count, 0);
  auto program = on(Target::cpu(2u))
                     .map<std::int32_t>("cpu-step-parity", count,
                                        [](auto value) { return value + 1; })
                     .filter([](auto value) { return value < 0; })
                     .scan(Scan::InclusiveSum)
                     .compile();
  if (!program) {
    return false;
  }
  auto blocking = program->resident(input);
  auto submitted = program->resident(input);
  if (!blocking || !submitted || !blocking->run()) {
    return false;
  }
  const Stats blocking_stats = blocking->stats();
  auto submission = runtime.compute(*submitted).submit();
  const rund::compute::Poll progress =
      submission.wait_for(std::chrono::nanoseconds::zero());
  if (!progress.submitted ||
      (progress.completed && progress.reason() != Reason::Ok)) {
    return false;
  }
  const rund::compute::Poll settled =
      progress.completed ? progress
                         : submission.wait_for(std::chrono::seconds{30});
  if (!settled.completed || settled.reason() != Reason::Ok) {
    std::fprintf(stderr,
                 "cpu step parity settle completed=%u submitted=%u "
                 "reason=%.*s\n",
                 settled.completed ? 1u : 0u, settled.submitted ? 1u : 0u,
                 static_cast<int>(settled.error().size()),
                 settled.error().data());
    return false;
  }
  const rund::compute::Completion completion = submission.wait();
  if (!completion) {
    return false;
  }
  const Stats submitted_stats = completion.stats();
  auto blocking_output = blocking->read();
  auto submitted_output = submitted->read();
  const bool same =
      blocking_output && submitted_output && blocking_output->empty() &&
      submitted_output->empty() &&
      blocking_stats.graph_hash == submitted_stats.graph_hash &&
      blocking_stats.graph_read_bytes == submitted_stats.graph_read_bytes &&
      blocking_stats.dispatches == submitted_stats.dispatches &&
      blocking_stats.worker_count == submitted_stats.worker_count &&
      blocking_stats.tile_count == submitted_stats.tile_count &&
      blocking_stats.tile_size == submitted_stats.tile_size &&
      blocking_stats.vector_chunks == submitted_stats.vector_chunks &&
      blocking_stats.tail_chunks == submitted_stats.tail_chunks;
  if (!same) {
    std::fprintf(
        stderr,
        "cpu step parity output=%u/%u graph=%llu/%llu read=%llu/%llu "
        "dispatches=%llu/%llu workers=%u/%u tiles=%llu/%llu "
        "tile_size=%llu/%llu vectors=%llu/%llu tails=%llu/%llu\n",
        blocking_output && blocking_output->empty() ? 1u : 0u,
        submitted_output && submitted_output->empty() ? 1u : 0u,
        static_cast<unsigned long long>(blocking_stats.graph_hash),
        static_cast<unsigned long long>(submitted_stats.graph_hash),
        static_cast<unsigned long long>(blocking_stats.graph_read_bytes),
        static_cast<unsigned long long>(submitted_stats.graph_read_bytes),
        static_cast<unsigned long long>(blocking_stats.dispatches),
        static_cast<unsigned long long>(submitted_stats.dispatches),
        blocking_stats.worker_count, submitted_stats.worker_count,
        static_cast<unsigned long long>(blocking_stats.tile_count),
        static_cast<unsigned long long>(submitted_stats.tile_count),
        static_cast<unsigned long long>(blocking_stats.tile_size),
        static_cast<unsigned long long>(submitted_stats.tile_size),
        static_cast<unsigned long long>(blocking_stats.vector_chunks),
        static_cast<unsigned long long>(submitted_stats.vector_chunks),
        static_cast<unsigned long long>(blocking_stats.tail_chunks),
        static_cast<unsigned long long>(submitted_stats.tail_chunks));
  }
  return same;
}

} // namespace runtime_compute_host_detail
