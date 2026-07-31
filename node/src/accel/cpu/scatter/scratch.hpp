#pragma once

#include <kernel/program/compute/model.hpp>

#include <vector>

namespace rund::node::accel::detail {
using rund::kernel::u32;
using rund::kernel::u64;

struct CpuScatterScratch {
  std::vector<u32> keys{};
  std::vector<u32> marks{};
  u32 epoch = 0u;
};
} // namespace rund::node::accel::detail
