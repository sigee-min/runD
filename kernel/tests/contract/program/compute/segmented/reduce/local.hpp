#pragma once

#include "test/assert.hpp"

#include <kernel/program/compute/segmented/reduce/identity.hpp>
#include <kernel/program/compute/segmented/reduce/plan.hpp>
#include <kernel/program/compute/segmented/reduce/reference.hpp>
#include <kernel/program/compute/segmented/reduce/model.hpp>

#include <array>
#include <limits>
#include <string_view>

namespace program_compute_contract {

[[nodiscard]] constexpr rund::kernel::SegmentedReduceDesc
U32SegmentedReduce() noexcept {
  return rund::kernel::SegmentedReduceDesc{
      .op = rund::kernel::ReduceOp::Sum,
      .element = rund::kernel::ReduceElement::U32,
      .element_count = 8u,
      .block_size = 4u,
  };
}

int SegmentedReduceReject();
int SegmentedReduceIdentity();
int SegmentedReduceShape();
int SegmentedReduceReference();

}  // namespace program_compute_contract
