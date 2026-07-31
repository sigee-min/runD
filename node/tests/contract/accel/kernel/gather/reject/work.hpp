#pragma once

#include <kernel/program/compute/gather/reference.hpp>

#include "../local.hpp"

#include <array>

namespace node_accel_contract::gather::reject {

struct Work {
  std::array<rund::kernel::u32, 2u> values{11u, 13u};
  std::array<rund::kernel::u32, 2u> indices{0u, 2u};
  std::array<rund::kernel::u32, 2u> output_sentinel{17u, 19u};
  rund::kernel::u32 overflowing_count{3u};
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
