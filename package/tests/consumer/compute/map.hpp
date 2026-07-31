#pragma once

#include <rund/compute.hpp>

#include <array>
#include <cstdint>
#include <vector>

namespace package_compute {

inline constexpr std::int32_t ImmutableHost = 1;

inline int Map() {
  std::array<std::int32_t, 4> input{1, 2, 3, 4};
  auto branch = rund::compute::capture(
      [](auto value, auto selector) {
        return rund::compute::select(selector != 0, value + ImmutableHost,
                                     value);
      },
      std::int32_t{1});
  auto output =
      rund::compute::on(rund::compute::Target::cpu(), input)
          .map("branched-constant", branch)
          .collect();
  if (!output) {
    return output.exit_code();
  }
  return *output == std::vector<std::int32_t>{2, 3, 4, 5} ? 0 : 2;
}

} // namespace package_compute
