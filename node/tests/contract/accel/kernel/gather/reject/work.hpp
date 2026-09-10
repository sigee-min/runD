#pragma once

#include <kernel/program/compute/gather/reference.hpp>

#include "../local.hpp"

#include <array>
#include <vector>

namespace node_accel_contract::gather::reject {

struct Work {
  std::array<rund::kernel::u32, 2u> values{11u, 13u};
  std::vector<rund::kernel::u32> indices;
  std::vector<rund::kernel::u32> output_sentinel;
  rund::kernel::u32 overflowing_count{};

  explicit Work(const std::size_t count = 2u)
      : indices(count, 0u), output_sentinel(count, 17u),
        overflowing_count(static_cast<rund::kernel::u32>(count + 1u)) {
    indices[count > 257u ? count - 257u : count - 1u] = 2u;
    indices.back() = 2u;
  }
};

[[nodiscard]] inline bool ReferenceRejectsOutOfRangeIndex(const Work &work) {
  auto expected = work.output_sentinel;
  return !rund::kernel::ReferenceGatherU32(
              work.values.data(), work.indices.data(), expected.data(),
              work.indices.size(), work.values.size())
              .ok &&
         expected == work.output_sentinel;
}

} // namespace node_accel_contract::gather::reject
