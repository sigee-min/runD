#pragma once

#include <kernel/program/compute/model.hpp>

#include <cstddef>
#include <type_traits>

namespace rund::node::accel::detail {

struct ReducePassParams {
  rund::kernel::u64 input_offset = 0u;
  rund::kernel::u64 output_offset = 0u;
  rund::kernel::u64 input_count = 0u;
  rund::kernel::u64 grid_size = 0u;
  rund::kernel::u32 final_pass = 0u;
  rund::kernel::u32 initial_pass = 0u;
  rund::kernel::u32 count_words = 0u;
};

static_assert(std::is_standard_layout_v<ReducePassParams>);
static_assert(sizeof(ReducePassParams) == 48u);
static_assert(offsetof(ReducePassParams, input_offset) == 0u);
static_assert(offsetof(ReducePassParams, output_offset) == 8u);
static_assert(offsetof(ReducePassParams, input_count) == 16u);
static_assert(offsetof(ReducePassParams, grid_size) == 24u);
static_assert(offsetof(ReducePassParams, final_pass) == 32u);
static_assert(offsetof(ReducePassParams, initial_pass) == 36u);
static_assert(offsetof(ReducePassParams, count_words) == 40u);

} // namespace rund::node::accel::detail
