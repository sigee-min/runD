#include "prefix.hpp"

namespace rund::node::accel::detail {

#if defined(__APPLE__) && defined(RUND_NODE_HAVE_METAL_SDK)
namespace {

template <typename Sink>
[[nodiscard]] bool EmitPrefix(Sink &source) noexcept(
    noexcept(source += std::string_view{})) {
  source += R"MSL(kernel void rund_compute_segmented_reduce_prefix(
    device const ulong* counts [[buffer(0)]],
    device ulong* offsets [[buffer(1)]],
    device ulong* segment_count [[buffer(2)]],
    device uint* dispatch [[buffer(3)]],
    constant RundSegmentedReduceParams& params [[buffer(5)]],
    uint lane [[thread_index_in_threadgroup]]) {
  threadgroup ulong partial[RUND_SEGMENT_INDEX_WIDTH];
  const ulong width = ulong(RUND_SEGMENT_INDEX_WIDTH);
  const ulong quotient = params.block_count / width;
  const ulong remainder = params.block_count % width;
  const ulong begin = quotient * ulong(lane) + min(ulong(lane), remainder);
  const ulong end = begin + quotient + (ulong(lane) < remainder ? 1ul : 0ul);
  ulong local = 0ul;
  for (ulong block = begin; block < end; ++block) {
    offsets[block] = local;
    local += counts[block];
  }
  partial[lane] = local;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint stride = 1u; stride < RUND_SEGMENT_INDEX_WIDTH; stride <<= 1u) {
    const uint index = (lane + 1u) * (stride << 1u) - 1u;
    if (index < RUND_SEGMENT_INDEX_WIDTH) {
      partial[index] += partial[index - stride];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  if (lane == 0u) { partial[RUND_SEGMENT_INDEX_WIDTH - 1u] = 0ul; }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint stride = RUND_SEGMENT_INDEX_WIDTH >> 1u; stride > 0u;
       stride >>= 1u) {
    const uint index = (lane + 1u) * (stride << 1u) - 1u;
    if (index < RUND_SEGMENT_INDEX_WIDTH) {
      const ulong left = partial[index - stride];
      partial[index - stride] = partial[index];
      partial[index] += left;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  const ulong base = partial[lane];
  for (ulong block = begin; block < end; ++block) { offsets[block] += base; }
  threadgroup_barrier(mem_flags::mem_device);
  if (lane == 0u) {
    const ulong last = params.block_count - 1ul;
    const ulong segments = offsets[last] + counts[last];
    segment_count[0] = segments;
    const ulong groups =
        segments / params.segments_per_group +
        (segments % params.segments_per_group != 0ul ? 1ul : 0ul);
    dispatch[0] = uint(min(groups, ulong(RUND_SEGMENT_MAX_GROUPS)));
    dispatch[1] = 1u;
    dispatch[2] = 1u;
  }
}
)MSL";
  return source.valid();
}

} // namespace

bool EmitMetalSegmentedReducePrefixSource(
    backend_source_recipe::CountSink &sink) noexcept {
  return EmitPrefix(sink);
}

bool EmitMetalSegmentedReducePrefixSource(
    backend_source_recipe::StringSink &sink) {
  return EmitPrefix(sink);
}

#endif

} // namespace rund::node::accel::detail
