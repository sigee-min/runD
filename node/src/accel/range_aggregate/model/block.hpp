#pragma once

#include "traits.hpp"

namespace rund::node::accel::detail {

// One physical block geometry for candidate admission, frozen stages and
// resident-count projection. CPU keeps its sequential stage owner.
[[nodiscard]] constexpr rund::kernel::u64
RangeBlockGroups(const RangeSource source, const rund::kernel::u64 blocks,
                 const rund::kernel::u32 width) noexcept {
  if (blocks == 0u) {
    return 0u;
  }
  if (source == RangeSource::Cpu) {
    return 1u;
  }
  return source == RangeSource::Metal
             ? blocks
             : blocks / width +
                   static_cast<rund::kernel::u64>(blocks % width != 0u);
}

// The admitted affine span proves (outputs-1)*stride cannot overflow.
[[nodiscard]] constexpr rund::kernel::u64
RangeBlockQueryGroups(const RangeSource source, const rund::kernel::u64 outputs,
                      const rund::kernel::u64 stride,
                      const rund::kernel::u64 window,
                      const rund::kernel::u32 width) noexcept {
  if (outputs == 0u) {
    return 0u;
  }
  return source == RangeSource::Metal ? (outputs - 1u) * stride / window + 1u
         : source == RangeSource::Cpu
             ? 1u
             : outputs / width + (outputs % width != 0u);
}

} // namespace rund::node::accel::detail
