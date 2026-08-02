#include "local.hpp"

#include "../../kernel/backend/source_recipe.hpp"

namespace rund::node::accel::detail {

#if defined(RUND_NODE_HAVE_VULKAN_SDK)
namespace {

template <typename Sink>
[[nodiscard]] bool EmitVulkanGatherSource(
    Sink &sink, const rund::kernel::GatherElement element, const bool control)
    noexcept(noexcept(sink.append(std::string_view{}))) {
  const bool wide = element == rund::kernel::GatherElement::U64;
  if (!sink.append(R"glsl(#version 450
#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require
layout(local_size_x = 256) in;
layout(set = 0, binding = 0, std430) readonly buffer Params {
  uint64_t element_count;
  uint64_t source_count;
  uint count_source;
  uint reserved;
} params;
layout(set = 0, binding = 1, std430) readonly buffer Values {
)glsl") ||
      !sink.append(wide ? "  uint64_t values[];\n" : "  uint values[];\n") ||
      !sink.append(R"glsl(};
layout(set = 0, binding = 2, std430) readonly buffer Indices {
  uint indices[];
};
layout(set = 0, binding = 3, std430) buffer Output {
)glsl") ||
      !sink.append(wide ? "  uint64_t output_values[];\n"
                        : "  uint output_values[];\n") ||
      !sink.append(R"glsl(};
layout(set = 0, binding = 4, std430) buffer Status {
  uint status[];
};
layout(set = 0, binding = 5, std430) readonly buffer Count {
  uint count_words[];
};
layout(set = 0, binding = 6, std430) buffer Indirect {
  uint indirect_args[];
};
)glsl")) {
    return false;
  }
  if (!control) {
    return sink.append(R"glsl(void main() {
  const uint gid = gl_GlobalInvocationID.x;
  if (gid >= indirect_args[3]) { return; }
  const uint source_index = indices[gid];
  output_values[gid] = values[source_index];
}
)glsl");
  }
  return sink.append(R"glsl(uint64_t logical_count() {
  if (params.count_source == 0u) { return params.element_count; }
  if (params.count_source == 1u) { return uint64_t(count_words[0]); }
  return uint64_t(count_words[0]) | (uint64_t(count_words[1]) << 32u);
}
shared uint invalids[256];
void main() {
  const uint tid = gl_LocalInvocationID.x;
  const uint64_t logical = logical_count();
  uint local_invalid = 0xffffffffu;
  if (logical <= params.element_count) {
    for (uint ordinal = tid; uint64_t(ordinal) < logical;) {
      if (uint64_t(indices[ordinal]) >= params.source_count) { local_invalid = min(local_invalid, ordinal); }
      if (ordinal > 0xffffffffu - 256u) { break; }
      ordinal += 256u;
    }
  }
  invalids[tid] = local_invalid;
  barrier();
  for (uint stride = 128u; stride != 0u; stride >>= 1u) {
    if (tid < stride) { invalids[tid] = min(invalids[tid], invalids[tid + stride]); }
    barrier();
  }
  if (tid != 0u) { return; }
  status[0] = 0u; status[1] = uint(min(logical, uint64_t(0xffffffffu)));
  indirect_args[0] = 0u; indirect_args[1] = 1u; indirect_args[2] = 1u; indirect_args[3] = 0u;
  if (logical > params.element_count) { status[0] = 1u; status[1] = uint(params.element_count); return; }
  if (invalids[0] != 0xffffffffu) { status[0] = 2u; status[1] = invalids[0]; return; }
  indirect_args[0] = uint((logical + 255u) / 256u); indirect_args[3] = uint(logical);
}
)glsl");
}

} // namespace

std::string VulkanGatherSource(const rund::kernel::GatherElement element,
                               const bool control) {
  return backend_source_recipe::materialize([&](auto &sink) {
    return EmitVulkanGatherSource(sink, element, control);
  });
}

bool VulkanGatherSourceBytes(const rund::kernel::GatherElement element,
                             const bool control,
                             std::uint64_t &bytes) noexcept {
  return backend_source_recipe::bytes(
      [&](backend_source_recipe::CountSink &sink) noexcept {
        return EmitVulkanGatherSource(sink, element, control);
      },
      bytes);
}
#endif

} // namespace rund::node::accel::detail
