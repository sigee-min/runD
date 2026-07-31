#pragma once

#include <rund/compute.hpp>

#include <array>
#include <cstdint>
#include <vector>

namespace package_compute {

inline int Write() {
  const std::array<std::int32_t, 4> first{1, 2, 3, 4};
  const std::array<std::int32_t, 4> second{5, 6, 7, 8};
  auto program = rund::compute::on(rund::compute::Target::cpu())
                     .map<std::int32_t>("twice", first.size(),
                                        [](auto value) { return value * 2; })
                     .compile();
  if (!program) {
    return program.exit_code();
  }
  auto job = program->resident(first);
  if (!job) {
    return job.exit_code();
  }
  const auto first_run = job->run();
  if (!first_run) {
    return first_run.exit_code();
  }
  const auto graph = job->stats().graph_hash;
  const auto replaced = job->write(second);
  if (!replaced) {
    return replaced.exit_code();
  }
  const auto write = job->write_stats();
  if (write.bytes != second.size() * sizeof(std::int32_t) ||
      write.copies != 1u || write.uploads != 0u) {
    return 2;
  }
  const auto second_run = job->run();
  if (!second_run) {
    return second_run.exit_code();
  }
  const auto warm = job->stats();
  auto output = job->read();
  if (!output) {
    return output.exit_code();
  }
  return *output == std::vector<std::int32_t>{10, 12, 14, 16} &&
                 warm.graph_hash == graph && warm.pipeline_compiles == 0u &&
                 warm.buffer_allocations == 0u && warm.download_events == 0u
             ? 0
             : 2;
}

} // namespace package_compute
