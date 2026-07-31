#pragma once

#include "test/assert.hpp"

#include <kernel/program/compute/reduce/model.hpp>
#include <kernel/program/compute/reduce/identity.hpp>
#include <kernel/program/compute/reduce/plan.hpp>
#include <kernel/program/compute/reduce/reference.hpp>

#include <array>
#include <limits>
#include <string_view>

namespace program_compute_contract {

[[nodiscard]] constexpr rund::kernel::ReduceDesc U32Reduce() noexcept {
  return rund::kernel::ReduceDesc{
      .op = rund::kernel::ReduceOp::Sum,
      .element = rund::kernel::ReduceElement::U32,
      .element_count = 1025u,
      .block_size = 256u,
  };
}

[[nodiscard]] constexpr rund::kernel::ReduceDesc ReduceWithOp(
    const rund::kernel::ReduceOp op) noexcept {
  rund::kernel::ReduceDesc desc = U32Reduce();
  desc.op = op;
  return desc;
}

int ReduceReject();
int ReduceIdentity();
int ReduceShape();
int ReduceShapeOps();
int ReduceReference();

} // namespace program_compute_contract
