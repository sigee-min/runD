#include "local.hpp"

#include "../../kernel/backend/source_recipe.hpp"

namespace rund::node::accel::detail {

#if defined(RUND_NODE_HAVE_VULKAN_SDK)
namespace {

template <typename Sink>
[[nodiscard]] bool EmitVulkanHistogramSource(Sink &sink, const bool clear)
    noexcept(noexcept(sink.append(std::string_view{}))) {
  if (!sink.append(R"glsl(#version 450
#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require
layout(local_size_x = 256) in;
layout(set = 0, binding = 0, std430) readonly buffer Params {
  uint64_t element_count;
  uint64_t bin_count;
} params;
layout(set = 0, binding = 1, std430) readonly buffer Bins {
  uint bins[];
};
layout(set = 0, binding = 2, std430) buffer Counts {
  uint counts[];
};
layout(set = 0, binding = 3, std430) buffer Status {
  uint status_words[];
};
void main() {
  const uint gid = gl_GlobalInvocationID.x;
)glsl")) {
    return false;
  }
  if (clear) {
    return sink.append(R"glsl(  if (uint64_t(gid) < params.bin_count) { counts[gid] = 0u; }
  if (gid == 0u) { status_words[0] = 0xffffffffu; }
}
)glsl");
  }
  return sink.append(R"glsl(  if (uint64_t(gid) >= params.element_count) { return; }
  const uint bin = bins[gid];
  if (uint64_t(bin) >= params.bin_count) {
    atomicExchange(status_words[0], 0u);
    return;
  }
  atomicAdd(counts[bin], 1u);
}
)glsl");
}

} // namespace

std::string VulkanHistogramSource(const bool clear) {
  return backend_source_recipe::materialize(
      [&](auto &sink) { return EmitVulkanHistogramSource(sink, clear); });
}

bool VulkanHistogramSourceBytes(const bool clear,
                                std::uint64_t &bytes) noexcept {
  return backend_source_recipe::bytes(
      [&](backend_source_recipe::CountSink &sink) noexcept {
        return EmitVulkanHistogramSource(sink, clear);
      },
      bytes);
}
#endif

} // namespace rund::node::accel::detail
