#pragma once

#include <rund/compute.hpp>

#include <array>
#include <cstdint>
#include <vector>

namespace package_compute {

inline int Reuse() {
  const std::array<std::int32_t, 4> input{1, 2, 3, 4};
  const std::vector<std::int32_t> expected{7, 9, 11, 13};
  auto program =
      rund::compute::on(rund::compute::Target::cpu())
          .map<std::int32_t>("twice", input.size(),
                             [](auto x) { return x * 2 + 5; })
          .compile();
  if (!program) {
    return program.exit_code();
  }
  auto job = program->resident(input);
  if (!job) {
    return job.exit_code();
  }
  const auto first = job->run();
  if (!first) {
    return first.exit_code();
  }
  const auto second = job->run();
  if (!second) {
    return second.exit_code();
  }
  auto output = job->read();
  if (!output) {
    return output.exit_code();
  }
  return *output == expected ? 0 : 2;
}

} // namespace package_compute
