#include "local.hpp"

#include "../../kernel/backend/source_recipe.hpp"

namespace rund::node::accel::detail {

#if defined(RUND_NODE_HAVE_VULKAN_SDK)
namespace {

template <typename Sink>
[[nodiscard]] bool EmitVulkanScatterSource(
    Sink &sink, const rund::kernel::ScatterElement element)
    noexcept(noexcept(sink.append(std::string_view{}))) {
  const bool wide = element == rund::kernel::ScatterElement::U64;
  return sink.append(R"glsl(#version 450
#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require
layout(local_size_x = 256) in;
layout(set = 0, binding = 0, std430) readonly buffer Params {
  uint64_t element_count;
  uint64_t output_count;
} params;
layout(set = 0, binding = 1, std430) readonly buffer Values {
)glsl") &&
         sink.append(wide ? "  uint64_t values[];\n" : "  uint values[];\n") &&
         sink.append(R"glsl(};
layout(set = 0, binding = 2, std430) readonly buffer Indices {
  uint indices[];
};
layout(set = 0, binding = 3, std430) buffer Output {
)glsl") &&
         sink.append(wide ? "  uint64_t output_values[];\n"
                          : "  uint output_values[];\n") &&
         sink.append(R"glsl(};
layout(set = 0, binding = 4, std430) buffer Status {
  uint status[];
};
void record_failure(uint gid, uint reason) {
  atomicMin(status[0], (gid << 1u) | reason);
}
bool claim_target(uint gid, uint target) {
  const uint prior = atomicMin(status[target + 1u], gid);
  if (prior == 0xffffffffu) { return true; }
  const uint duplicate = prior < gid ? gid : prior;
  record_failure(duplicate, 1u);
  return false;
}
void main() {
  const uint gid = gl_GlobalInvocationID.x;
  if (uint64_t(gid) >= params.element_count) { return; }
  const uint target = indices[gid];
  if (uint64_t(target) >= params.output_count) {
    record_failure(gid, 0u);
    return;
  }
  if (claim_target(gid, target)) {
    output_values[target] = values[gid];
  }
}
)glsl");
}

} // namespace

std::string VulkanScatterSource(const rund::kernel::ScatterElement element) {
  return backend_source_recipe::materialize(
      [&](auto &sink) { return EmitVulkanScatterSource(sink, element); });
}

bool VulkanScatterSourceBytes(const rund::kernel::ScatterElement element,
                              std::uint64_t &bytes) noexcept {
  return backend_source_recipe::bytes(
      [&](backend_source_recipe::CountSink &sink) noexcept {
        return EmitVulkanScatterSource(sink, element);
      },
      bytes);
}
#endif

} // namespace rund::node::accel::detail
