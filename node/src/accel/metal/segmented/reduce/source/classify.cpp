#include "classify.hpp"

namespace rund::node::accel::detail {

#if defined(__APPLE__) && defined(RUND_NODE_HAVE_METAL_SDK)
namespace {

template <typename Sink>
[[nodiscard]] bool EmitClassify(Sink &source) noexcept(
    noexcept(source += std::string_view{})) {
  source += R"MSL(
kernel void rund_compute_segmented_reduce_classify(
    device const uint* heads [[buffer(0)]],
    device ulong* counts [[buffer(1)]],
    device atomic_uint* status [[buffer(4)]],
    constant RundSegmentedReduceParams& params [[buffer(5)]],
    uint lane [[thread_index_in_threadgroup]],
    uint3 group [[threadgroup_position_in_grid]]) {
  threadgroup uint partial[RUND_SEGMENT_INDEX_WIDTH];
  const ulong groups = min(params.block_count, ulong(RUND_SEGMENT_MAX_GROUPS));
  for (ulong block = ulong(group.x); block < params.block_count;
       block += groups) {
    const ulong index = block * ulong(RUND_SEGMENT_INDEX_WIDTH) + ulong(lane);
    const uint head = index < params.count ? heads[index] : 0u;
    if (index == 0ul && head != 1u) {
      atomic_fetch_or_explicit(&status[0], RUND_SEGMENT_INVALID,
                               memory_order_relaxed);
    }
    if (index < params.count && head > 1u) {
      atomic_fetch_or_explicit(&status[0], RUND_SEGMENT_INVALID,
                               memory_order_relaxed);
    }
    partial[lane] = head != 0u ? 1u : 0u;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = RUND_SEGMENT_INDEX_WIDTH >> 1u; stride > 0u;
         stride >>= 1u) {
      if (lane < stride) { partial[lane] += partial[lane + stride]; }
      threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (lane == 0u) { counts[block] = ulong(partial[0]); }
  }
}
)MSL";
  return source.valid();
}

} // namespace

bool EmitMetalSegmentedReduceClassifySource(
    backend_source_recipe::CountSink &sink) noexcept {
  return EmitClassify(sink);
}

bool EmitMetalSegmentedReduceClassifySource(
    backend_source_recipe::StringSink &sink) {
  return EmitClassify(sink);
}

#endif

} // namespace rund::node::accel::detail
