#pragma once

#include <kernel/program/compute/sort/reference.hpp>

#include "../identity.hpp"

namespace node_accel_contract::sort_identity {

struct Expected {
  std::array<rund::kernel::u32, 8u> keys{};
  std::array<rund::kernel::u32, 8u> values{};
  bool valid{false};
};

[[nodiscard]] inline Expected
MakeExpected(const std::array<rund::kernel::u32, 8u> &input_keys) {
  std::array<rund::kernel::u32, 8u> identity_values{};
  for (std::size_t index = 0u; index < identity_values.size(); ++index) {
    identity_values[index] = static_cast<rund::kernel::u32>(index);
  }

  Expected expected{};
  std::array<rund::kernel::u64, 8u> expected_indices{};
  expected.valid =
      rund::kernel::ReferenceStableSortU32(
          input_keys.data(), identity_values.data(), expected.keys.data(),
          expected.values.data(), expected_indices.data(), input_keys.size())
          .ok;
  return expected;
}

} // namespace node_accel_contract::sort_identity
