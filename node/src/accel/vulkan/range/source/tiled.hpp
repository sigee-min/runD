#pragma once
#include "../../../kernel/backend/source/sink.hpp"
namespace rund::node::accel::detail {
template <typename Sink>
[[nodiscard]] bool EmitVulkanTiledDifferenceBody(Sink &sink, const bool wide,
                                                 const RangeExec &shape) {
  backend_source_recipe::SourceBuilder<Sink> source{sink};
  const char *const type = wide ? "uint64_t" : "uint";
  source += R"RANGE(const uint W = )RANGE";
  (void)source.decimal(shape.width());
  source += R"RANGE(u, R = )RANGE";
  (void)source.decimal(kRangeTileOutputsPerLane);
  source += R"RANGE(u;
shared )RANGE";
  source += type;
  source += R"RANGE( deltas[W];
shared )RANGE";
  source += type;
  source += R"RANGE( anchors[W];
void main() {
  const uint tid = gl_LocalInvocationID.x;
  const uint group = gl_WorkGroupID.x;
  const uint64_t base = uint64_t(group) * uint64_t(W * R);
  if (base >= params.input_count) { return; }
  const uint64_t radius = params.padding;
  const uint64_t left = base < radius ? uint64_t(0) : base - radius;
  const uint64_t right = radius >= params.input_count - base
      ? params.input_count - uint64_t(1) : base + radius;
  )RANGE";
  source += type;
  source += R"RANGE( anchor_sum = 0;
  for (uint64_t offset = tid; offset < right - left + uint64_t(1); offset += W) {
    anchor_sum += input_values[uint(left + offset)];
  }
  if (tid == 0u) {
    if (base < radius) { anchor_sum += )RANGE";
  source += type;
  source += R"RANGE(((radius - base) * input_values[uint(0)]); }
    if (radius >= params.input_count - base) {
      anchor_sum += )RANGE";
  source += type;
  source += R"RANGE(((radius - (params.input_count - base) + uint64_t(1)) *
          input_values[uint(params.input_count - uint64_t(1))]);
    }
  }
  )RANGE";
  source += type;
  source += R"RANGE( prefixes[R];
  )RANGE";
  source += type;
  source += R"RANGE( total = 0;
  for (uint j = 0u; j < R; ++j) {
    const uint64_t i = base + uint64_t(tid) * R + j;
    if (i < params.input_count && i != base) {
      const uint64_t right_index = radius >= params.input_count - i
          ? params.input_count - uint64_t(1) : i + radius;
      const uint64_t left_index = i <= radius ? uint64_t(0) : i - radius - uint64_t(1);
      total += input_values[uint(right_index)] - input_values[uint(left_index)];
    }
    prefixes[j] = total;
  }
  deltas[tid] = total;
  anchors[tid] = anchor_sum;
  barrier();
  for (uint offset = 1u; offset < W; offset <<= 1u) {
    const uint tree = (tid + 1u) * 2u * offset - 1u;
    if (tree < W) {
      deltas[tree] += deltas[tree - offset];
      anchors[tree] += anchors[tree - offset];
    }
    barrier();
  }
  if (tid == 0u) { anchors[0] = anchors[W - 1u]; deltas[W - 1u] = 0; }
  barrier();
  for (uint offset = W / 2u; offset != 0u; offset >>= 1u) {
    const uint tree = (tid + 1u) * 2u * offset - 1u;
    if (tree < W) {
      const )RANGE";
  source += type;
  source += R"RANGE( prior = deltas[tree - offset];
      deltas[tree - offset] = deltas[tree];
      deltas[tree] += prior;
    }
    barrier();
  }
  const )RANGE";
  source += type;
  source += R"RANGE( start = anchors[0] + deltas[tid];
  for (uint j = 0u; j < R; ++j) {
    const uint64_t i = base + uint64_t(tid) * R + j;
    if (i < params.output_count) { output_values[uint(i)] = start + prefixes[j]; }
  }
}
)RANGE";
  return source.valid();
}
} // namespace rund::node::accel::detail
