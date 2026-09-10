#include "prefix.hpp"

namespace rund::node::accel::detail {

#if defined(RUND_NODE_HAVE_VULKAN_SDK)
namespace {

template <typename Sink>
[[nodiscard]] bool
EmitHeader(Sink &sink) noexcept(noexcept(sink.append(std::string_view{}))) {
  return sink.append("#version 450\n") &&
         sink.append("#extension "
                     "GL_EXT_shader_explicit_arithmetic_types_int64 : "
                     "require\n") &&
         AppendSegmentedReduceShaderModel(sink);
}

template <typename Sink>
[[nodiscard]] bool EmitPrefixSource(Sink &sink) noexcept(
    noexcept(sink.append(std::string_view{}))) {
  return EmitHeader(sink) && sink.append(R"GLSL(
layout(local_size_x = RUND_SEGMENT_INDEX_WIDTH) in;
layout(set = 0, binding = 0, std430) readonly buffer Params {
  uint64_t count;
  uint64_t block_count;
} params;
layout(set = 0, binding = 1, std430) readonly buffer Counts {
  uint counts[];
};
layout(set = 0, binding = 2, std430) buffer Offsets { uint offsets[]; };
layout(set = 0, binding = 3, std430) buffer SegmentCount { uint segments; };
layout(set = 0, binding = 4, std430) buffer Dispatch { uint dispatch[]; };
shared uint partial[RUND_SEGMENT_INDEX_WIDTH];
void main() {
  const uint lane = gl_LocalInvocationID.x;
  const uint blocks = uint(params.block_count);
  const uint quotient = blocks / RUND_SEGMENT_INDEX_WIDTH;
  const uint remainder = blocks % RUND_SEGMENT_INDEX_WIDTH;
  const uint begin = quotient * lane + min(lane, remainder);
  const uint end = begin + quotient + (lane < remainder ? 1u : 0u);
  uint local = 0u;
  for (uint block = begin; block < end; ++block) {
    offsets[block] = local;
    local += counts[block];
  }
  partial[lane] = local;
  barrier();
  for (uint stride = 1u; stride < RUND_SEGMENT_INDEX_WIDTH; stride <<= 1u) {
    const uint index = (lane + 1u) * (stride << 1u) - 1u;
    if (index < RUND_SEGMENT_INDEX_WIDTH) {
      partial[index] += partial[index - stride];
    }
    barrier();
  }
  if (lane == 0u) { partial[RUND_SEGMENT_INDEX_WIDTH - 1u] = 0u; }
  barrier();
  for (uint stride = RUND_SEGMENT_INDEX_WIDTH >> 1u; stride > 0u;
       stride >>= 1u) {
    const uint index = (lane + 1u) * (stride << 1u) - 1u;
    if (index < RUND_SEGMENT_INDEX_WIDTH) {
      const uint left = partial[index - stride];
      partial[index - stride] = partial[index];
      partial[index] += left;
    }
    barrier();
  }
  const uint base = partial[lane];
  for (uint block = begin; block < end; ++block) { offsets[block] += base; }
  memoryBarrierBuffer();
  barrier();
  if (lane == 0u) {
    const uint last = blocks - 1u;
    segments = offsets[last] + counts[last];
    const uint groups =
        segments / RUND_SEGMENT_TEAMS_PER_GROUP +
        (segments % RUND_SEGMENT_TEAMS_PER_GROUP != 0u ? 1u : 0u);
    dispatch[0] = min(groups, RUND_SEGMENT_MAX_GROUPS);
    dispatch[1] = 1u;
    dispatch[2] = 1u;
  }
}
)GLSL");
}

} // namespace

bool EmitVulkanSegmentedReducePrefixSource(
    backend_source_recipe::CountSink &sink) noexcept {
  return EmitPrefixSource(sink);
}

bool EmitVulkanSegmentedReducePrefixSource(
    backend_source_recipe::StringSink &sink) {
  return EmitPrefixSource(sink);
}

#endif

} // namespace rund::node::accel::detail
