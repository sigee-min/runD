#include "scatter.hpp"

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
[[nodiscard]] bool EmitScatterSource(Sink &sink) noexcept(
    noexcept(sink.append(std::string_view{}))) {
  return EmitHeader(sink) && sink.append(R"GLSL(
layout(local_size_x = RUND_SEGMENT_INDEX_WIDTH) in;
layout(set = 0, binding = 0, std430) readonly buffer Params {
  uint64_t count;
  uint64_t block_count;
} params;
layout(set = 0, binding = 1, std430) readonly buffer Heads { uint heads[]; };
layout(set = 0, binding = 2, std430) readonly buffer Offsets {
  uint offsets[];
};
layout(set = 0, binding = 3, std430) buffer Starts { uint starts[]; };
shared uint partial[RUND_SEGMENT_INDEX_WIDTH];
void main() {
  const uint lane = gl_LocalInvocationID.x;
  const uint blocks = uint(params.block_count);
  const uint groups = min(blocks, RUND_SEGMENT_MAX_GROUPS);
  for (uint block = gl_WorkGroupID.x; block < blocks; block += groups) {
    const uint index = block * RUND_SEGMENT_INDEX_WIDTH + lane;
    const uint head = uint64_t(index) < params.count ? heads[index] : 0u;
    partial[lane] = head != 0u ? 1u : 0u;
    barrier();
    for (uint stride = 1u; stride < RUND_SEGMENT_INDEX_WIDTH; stride <<= 1u) {
      const uint offset = (lane + 1u) * (stride << 1u) - 1u;
      if (offset < RUND_SEGMENT_INDEX_WIDTH) {
        partial[offset] += partial[offset - stride];
      }
      barrier();
    }
    if (lane == 0u) { partial[RUND_SEGMENT_INDEX_WIDTH - 1u] = 0u; }
    barrier();
    for (uint stride = RUND_SEGMENT_INDEX_WIDTH >> 1u; stride > 0u;
         stride >>= 1u) {
      const uint offset = (lane + 1u) * (stride << 1u) - 1u;
      if (offset < RUND_SEGMENT_INDEX_WIDTH) {
        const uint left = partial[offset - stride];
        partial[offset - stride] = partial[offset];
        partial[offset] += left;
      }
      barrier();
    }
    if (head != 0u) { starts[offsets[block] + partial[lane]] = index; }
    barrier();
  }
}
)GLSL");
}

} // namespace

bool EmitVulkanSegmentedReduceScatterSource(
    backend_source_recipe::CountSink &sink) noexcept {
  return EmitScatterSource(sink);
}

bool EmitVulkanSegmentedReduceScatterSource(
    backend_source_recipe::StringSink &sink) {
  return EmitScatterSource(sink);
}

#endif

} // namespace rund::node::accel::detail
