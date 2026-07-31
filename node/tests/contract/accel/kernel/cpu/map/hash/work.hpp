#pragma once

#include "../../local.hpp"

#include <array>

namespace node_accel_contract::cpu_context {

inline constexpr std::size_t kMapHashCount = 8u;

struct MapHashWork {
  std::array<rund::kernel::i32, kMapHashCount> input{};
  std::array<rund::kernel::i32, kMapHashCount> expected{};
  std::array<rund::kernel::i32, kMapHashCount> map_input{};
  std::array<rund::kernel::i32, kMapHashCount> map_output{};
};

[[nodiscard]] inline MapHashWork MakeMapHashWork() {
  MapHashWork work{};
  work.input = {-3, 0, 4, 9, 12, -8, 7, 21};
  for (std::size_t index = 0u; index < kMapHashCount; ++index) {
    work.expected[index] = work.input[index] * 2 + 5;
  }
  return work;
}

}  // namespace node_accel_contract::cpu_context
