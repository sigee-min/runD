#pragma once

#include <kernel/program/compute/model.hpp>

#include <span>
#include <vector>

namespace rund::node::accel::detail {
using rund::kernel::u32;
using rund::kernel::u64;

struct CpuScatterScratch {
  std::vector<u32> keys{};
  std::vector<u32> marks{};
  u32 epoch = 0u;

  [[nodiscard]] constexpr bool scatter_ready() const noexcept { return true; }
  [[nodiscard]] u32 &scatter_epoch() noexcept { return epoch; }
  [[nodiscard]] std::span<u32> scatter_marks() noexcept { return marks; }
};
} // namespace rund::node::accel::detail
