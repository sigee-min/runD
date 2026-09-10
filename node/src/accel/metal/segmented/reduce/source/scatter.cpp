#include "scatter.hpp"

namespace rund::node::accel::detail {

#if defined(__APPLE__) && defined(RUND_NODE_HAVE_METAL_SDK)
namespace {

template <typename Sink>
[[nodiscard]] bool EmitScatter(Sink &source) noexcept(
    noexcept(source += std::string_view{})) {
  source += R"MSL(kernel void rund_compute_segmented_reduce_scatter(
    device const uint* heads [[buffer(0)]],
    device const ulong* offsets [[buffer(1)]],
    device ulong* starts [[buffer(2)]],
    constant RundSegmentedReduceParams& params [[buffer(5)]],
    uint lane [[thread_index_in_threadgroup]],
    uint3 group [[threadgroup_position_in_grid]]) {
  threadgroup uint partial[RUND_SEGMENT_INDEX_WIDTH];
  const ulong groups = min(params.block_count, ulong(RUND_SEGMENT_MAX_GROUPS));
  for (ulong block = ulong(group.x); block < params.block_count;
       block += groups) {
    const ulong index = block * ulong(RUND_SEGMENT_INDEX_WIDTH) + ulong(lane);
    const uint head = index < params.count ? heads[index] : 0u;
    partial[lane] = head != 0u ? 1u : 0u;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = 1u; stride < RUND_SEGMENT_INDEX_WIDTH; stride <<= 1u) {
      const uint offset = (lane + 1u) * (stride << 1u) - 1u;
      if (offset < RUND_SEGMENT_INDEX_WIDTH) {
        partial[offset] += partial[offset - stride];
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (lane == 0u) { partial[RUND_SEGMENT_INDEX_WIDTH - 1u] = 0u; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = RUND_SEGMENT_INDEX_WIDTH >> 1u; stride > 0u;
         stride >>= 1u) {
      const uint offset = (lane + 1u) * (stride << 1u) - 1u;
      if (offset < RUND_SEGMENT_INDEX_WIDTH) {
        const uint left = partial[offset - stride];
        partial[offset - stride] = partial[offset];
        partial[offset] += left;
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (head != 0u) { starts[offsets[block] + ulong(partial[lane])] = index; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
}
)MSL";
  return source.valid();
}

} // namespace

bool EmitMetalSegmentedReduceScatterSource(
    backend_source_recipe::CountSink &sink) noexcept {
  return EmitScatter(sink);
}

bool EmitMetalSegmentedReduceScatterSource(
    backend_source_recipe::StringSink &sink) {
  return EmitScatter(sink);
}

#endif

} // namespace rund::node::accel::detail
