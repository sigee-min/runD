#pragma once

#include "op.hpp"

namespace node_accel_contract::collective {

[[nodiscard]] constexpr rund::kernel::ScanDesc ScanDesc(
    const std::uint64_t count,
    const rund::kernel::ScanElement element =
        rund::kernel::ScanElement::U32) noexcept {
  return rund::kernel::ScanDesc{
      .op = rund::kernel::ScanOp::ExclusiveSum,
      .element = element,
      .element_count = count,
      .block_size = 4u,
  };
}

}  // namespace node_accel_contract::collective
