#pragma once

#include <kernel/program/compute/scatter/reference.hpp>

#include "../local.hpp"

#include <array>

namespace node_accel_contract::scatter::reject {

struct Work {
  std::array<rund::kernel::u32, 3u> values{11u, 13u, 17u};
  std::array<rund::kernel::u32, 3u> indices{0u, 2u, 0u};
  std::array<rund::kernel::u32, 4u> output{1u, 2u, 3u, 4u};
};

[[nodiscard]] inline bool ReferenceRejectsDuplicateIndex(const Work &work) {
  std::array<rund::kernel::u32, 4u> expected = work.output;
  return !rund::kernel::ReferenceScatterU32(
              work.values.data(), work.indices.data(), expected.data(),
              work.indices.size(), expected.size())
              .ok;
}

} // namespace node_accel_contract::scatter::reject
