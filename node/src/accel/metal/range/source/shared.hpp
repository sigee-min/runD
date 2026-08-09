#pragma once

#include "algebra.hpp"

namespace rund::node::accel::detail {

template <typename Sink>
inline void
AppendMetalRangeSharedKernel(Sink &source, const RangeOp op,
                             const bool saturating, const RangeExec &shape,
                             const char *const type, const char *const suffix) {
  source += "kernel void rund_range_";
  source += MetalRangeOpName(op);
  source += "_";
  source += suffix;
  source += R"MSL((
    device const )MSL";
  source += type;
  source += R"MSL(* input [[buffer(0)]],
    device )MSL";
  source += type;
  source += R"MSL(* output [[buffer(1)]],
    constant RangeParams& params [[buffer(2)]],
    uint tid [[thread_index_in_threadgroup]],
    uint group [[threadgroup_position_in_grid]]) {
  threadgroup )MSL";
  source += type;
  source += " tile[";
  (void)source.decimal(shape.shared_element_capacity());
  source += R"MSL(];
  const ulong group_base = ulong(group) * )MSL";
  (void)source.decimal(shape.width());
  source += R"MSL(ul;
  const ulong active_lanes = group_base >= params.input_count
                                 ? 0ul
                                 : min(params.input_count - group_base, )MSL";
  (void)source.decimal(shape.width());
  source += R"MSL(ul);
  const ulong group_end = group_base + active_lanes;
    const uint left_inputs = uint(min(group_base, params.padding));
    const uint right_inputs = group_end >= params.input_count
                                  ? 0u
                                  : uint(min(params.input_count - group_end,
                                             params.padding));
    if (ulong(tid) < active_lanes) {
      const )MSL";
  source += type;
  source += R"MSL( center_value = input[group_base + ulong(tid)];
      tile[)MSL";
  (void)source.decimal(shape.shared_radius_capacity());
  source += R"MSL(u + tid] = center_value;
      if (left_inputs == 0u && tid == 0u) {
        for (uint slot = 0u; ulong(slot) < params.padding; ++slot) {
          tile[)MSL";
  (void)source.decimal(shape.shared_radius_capacity());
  source += R"MSL(u - uint(params.padding) + slot] = center_value;
        }
      }
      if (right_inputs == 0u && ulong(tid) + 1ul == active_lanes) {
        for (uint slot = 0u; ulong(slot) < params.padding; ++slot) {
          tile[)MSL";
  (void)source.decimal(shape.shared_radius_capacity());
  source += R"MSL(u + uint(active_lanes) + slot] = center_value;
        }
      }
    }
    if (tid < left_inputs) {
      const )MSL";
  source += type;
  source +=
      R"MSL( left_value = input[group_base - ulong(left_inputs) + ulong(tid)];
      tile[)MSL";
  (void)source.decimal(shape.shared_radius_capacity());
  source += R"MSL(u - left_inputs + tid] = left_value;
      if (tid == 0u && ulong(left_inputs) < params.padding) {
        for (uint slot = 0u;
             ulong(slot) < params.padding - ulong(left_inputs); ++slot) {
          tile[)MSL";
  (void)source.decimal(shape.shared_radius_capacity());
  source += R"MSL(u - uint(params.padding) + slot] = left_value;
        }
      }
    }
    if (tid < right_inputs) {
      const )MSL";
  source += type;
  source += R"MSL( right_value = input[group_end + ulong(tid)];
      tile[)MSL";
  (void)source.decimal(shape.shared_radius_capacity());
  source += R"MSL(u + uint(active_lanes) + tid] = right_value;
      if (tid + 1u == right_inputs && ulong(right_inputs) < params.padding) {
        for (uint slot = right_inputs; ulong(slot) < params.padding; ++slot) {
          tile[)MSL";
  (void)source.decimal(shape.shared_radius_capacity());
  source += R"MSL(u + uint(active_lanes) + slot] = right_value;
        }
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (ulong(tid) >= active_lanes) { return; }
    const ulong i = group_base + ulong(tid);
    const uint center = )MSL";
  (void)source.decimal(shape.shared_radius_capacity());
  source += R"MSL(u + tid;
    const uint first = center - uint(params.padding);
    )MSL";
  source += type;
  source += R"MSL( value = tile[first];
    for (ulong slot = 1ul; slot < params.window_size; ++slot) {
      const )MSL";
  source += type;
  source += R"MSL( sample = tile[first + uint(slot)];
)MSL";
  source += MetalRangeUpdate(op, saturating);
  source += R"MSL(    }
    output[i] = value;
  }
)MSL";
}

} // namespace rund::node::accel::detail
