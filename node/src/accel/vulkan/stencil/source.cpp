#include "local.hpp"
#include "../../domain.hpp"

#include "../../kernel/backend/source_recipe.hpp"
#include "source/body.hpp"

namespace rund::node::accel::detail {

#if defined(RUND_NODE_HAVE_VULKAN_SDK)
namespace {

template <typename Sink>
[[nodiscard]] bool EmitVulkanStencilSource(
    Sink &sink, const rund::kernel::StencilOp op,
    const rund::kernel::StencilElement element,
    const rund::kernel::ComputeDomain domain)
    noexcept(noexcept(sink.append(std::string_view{}))) {
  const bool wide = element == rund::kernel::StencilElement::U64;
  const bool signed_extrema =
      IsSignedDomain(domain) && op != rund::kernel::StencilOp::Sum;
  return sink.append(R"glsl(#version 450
#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require
layout(local_size_x = 256) in;
layout(set = 0, binding = 0, std430) readonly buffer Params {
  uint64_t element_count;
  uint64_t radius;
} params;
layout(set = 0, binding = 1, std430) readonly buffer Input {
)glsl") &&
         sink.append(wide ? "  uint64_t input_values[];\n"
                          : "  uint input_values[];\n") &&
         sink.append(R"glsl(};
layout(set = 0, binding = 2, std430) buffer Output {
)glsl") &&
         sink.append(wide ? "  uint64_t output_values[];\n"
                          : "  uint output_values[];\n") &&
         sink.append("};\n") &&
         EmitVulkanStencilBody(sink, op, wide, signed_extrema);
}

} // namespace

std::string VulkanStencilSource(const rund::kernel::StencilOp op,
                                const rund::kernel::StencilElement element,
                                const rund::kernel::ComputeDomain domain) {
  return backend_source_recipe::materialize([&](auto &sink) {
    return EmitVulkanStencilSource(sink, op, element, domain);
  });
}

bool VulkanStencilSourceBytes(const rund::kernel::StencilOp op,
                              const rund::kernel::StencilElement element,
                              const rund::kernel::ComputeDomain domain,
                              std::uint64_t &bytes) noexcept {
  return backend_source_recipe::bytes(
      [&](backend_source_recipe::CountSink &sink) noexcept {
        return EmitVulkanStencilSource(sink, op, element, domain);
      },
      bytes);
}
#endif

} // namespace rund::node::accel::detail
