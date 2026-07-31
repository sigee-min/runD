#pragma once

#include <kernel/program/compute/stencil/reference.hpp>

#include <algorithm>
#include <bit>
#include <type_traits>

namespace rund::node::accel::detail {

#include "reference/signed.hpp"

[[nodiscard]] inline rund::kernel::StencilResult ExecuteStencilReference(
    const rund::kernel::StencilOp op, const rund::kernel::u32 *const input,
    rund::kernel::u32 *const output, const rund::kernel::u64 element_count,
    const rund::kernel::u64 radius) {
  if (op == rund::kernel::StencilOp::Min) {
    return rund::kernel::ReferenceStencilMinU32(input, output, element_count,
                                                radius);
  }
  if (op == rund::kernel::StencilOp::Max) {
    return rund::kernel::ReferenceStencilMaxU32(input, output, element_count,
                                                radius);
  }
  return rund::kernel::ReferenceStencilSumU32(input, output, element_count,
                                              radius);
}

[[nodiscard]] inline rund::kernel::StencilResult ExecuteStencilReference(
    const rund::kernel::StencilOp op, const rund::kernel::u64 *const input,
    rund::kernel::u64 *const output, const rund::kernel::u64 element_count,
    const rund::kernel::u64 radius) {
  if (op == rund::kernel::StencilOp::Min) {
    return rund::kernel::ReferenceStencilMinU64(input, output, element_count,
                                                radius);
  }
  if (op == rund::kernel::StencilOp::Max) {
    return rund::kernel::ReferenceStencilMaxU64(input, output, element_count,
                                                radius);
  }
  return rund::kernel::ReferenceStencilSumU64(input, output, element_count,
                                              radius);
}

} // namespace rund::node::accel::detail
