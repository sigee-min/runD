#pragma once

#include "algebra.hpp"

namespace rund::node::accel::detail {

template <typename Sink>
inline void
AppendMetalRangeDirectKernel(Sink &source, const RangeOp op,
                             const RangeBoundary boundary,
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
  const ulong group_base = ulong(group) * )MSL";
  (void)source.decimal(shape.width());
  source += R"MSL(ul;
  const ulong i = group_base + ulong(tid);
  if (i >= params.output_count) { return; }
  const ulong anchor = i * params.stride;
  )MSL";
  source += type;
  source += R"MSL( value = )MSL";
  source += type;
  source += R"MSL((0);
  bool seeded = false;
  for (ulong slot = 0ul; slot < params.window_size; ++slot) {
    ulong input_index = 0ul;
    bool valid = true;
    if (slot < params.padding) {
      const ulong delta = params.padding - slot;
      if (anchor < delta) {
)MSL";
  if (boundary == RangeBoundary::Clamp) {
    source += "        input_index = 0ul;\n";
  } else {
    source += "        valid = false;\n";
  }
  source += R"MSL(      } else {
        input_index = anchor - delta;
)MSL";
  if (boundary == RangeBoundary::Clamp) {
    source += R"MSL(        if (input_index >= params.input_count) {
          input_index = params.input_count - 1ul;
        }
)MSL";
  } else {
    source += R"MSL(        if (input_index >= params.input_count) {
          valid = false;
        }
)MSL";
  }
  source += R"MSL(      }
    } else {
      const ulong delta = slot - params.padding;
      if (anchor >= params.input_count ||
          delta >= params.input_count - anchor) {
)MSL";
  if (boundary == RangeBoundary::Clamp) {
    source += "        input_index = params.input_count - 1ul;\n";
  } else {
    source += "        valid = false;\n";
  }
  source += R"MSL(      } else {
        input_index = anchor + delta;
      }
    }
    if (!valid) { continue; }
    const )MSL";
  source += type;
  source += R"MSL( sample = input[input_index];
    if (!seeded) {
      value = sample;
      seeded = true;
    } else {
)MSL";
  source += MetalRangeUpdate(op, saturating);
  source += R"MSL(    }
  }
  output[i] = value;
}
)MSL";
}

} // namespace rund::node::accel::detail
