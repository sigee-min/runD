#pragma once

#include "../local.hpp"

namespace rund::node::accel::detail {

[[nodiscard]] inline const char *
MetalReduceOpName(const rund::kernel::ReduceOp op) noexcept {
  if (op == rund::kernel::ReduceOp::CountNonzero) {
    return "count_nonzero";
  }
  if (op == rund::kernel::ReduceOp::Min) {
    return "min";
  }
  if (op == rund::kernel::ReduceOp::Max) {
    return "max";
  }
  return "sum";
}

} // namespace rund::node::accel::detail
