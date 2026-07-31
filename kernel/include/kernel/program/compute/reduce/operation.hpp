#pragma once

#include <kernel/program/compute/reduce/model.hpp>

namespace rund::kernel::reduce {

[[nodiscard]] constexpr bool valid(const ReduceOp op) noexcept {
  return op == ReduceOp::Sum || op == ReduceOp::CountNonzero ||
         op == ReduceOp::Min || op == ReduceOp::Max;
}

} // namespace rund::kernel::reduce
