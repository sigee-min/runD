#pragma once

#include "local.hpp"

namespace node_accel_contract::metal_stats {

struct Work {
  std::array<rund::kernel::u32, 4u> lhs{3u, 5u, 8u, 13u};
  std::array<rund::kernel::u32, 4u> rhs{2u, 4u, 6u, 8u};
  std::array<rund::kernel::u32, 4u> first{};
  std::array<rund::kernel::u32, 4u> second{};
  std::array<rund::kernel::u32, 4u> expected{12u, 16u, 21u, 28u};
  std::array<rund::kernel::u32, 1u> params{7u};
  std::array<rund::kernel::u64, 4u> sequence_tiles{0u, 1u, 2u, 3u};
};

[[nodiscard]] inline std::array<rund::kernel::BufferSpan, 2u>
InputBuffers(Work& work) {
  return {rund::kernel::BufferSpan::contiguous(work.lhs.data(),
                                                      work.lhs.size()),
          rund::kernel::BufferSpan::contiguous(work.rhs.data(),
                                                      work.rhs.size())};
}

}  // namespace node_accel_contract::metal_stats
