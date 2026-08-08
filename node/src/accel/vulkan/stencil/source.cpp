#include "../../domain.hpp"
#include "local.hpp"

#include "../../kernel/backend/source_recipe.hpp"
#include "../../source/hash.hpp"
#include "source/body.hpp"

namespace rund::node::accel::detail {

#if defined(RUND_NODE_HAVE_VULKAN_SDK)
namespace {

class MatchSink final {
public:
  explicit MatchSink(const std::string_view source) noexcept
      : remaining_(source) {}

  [[nodiscard]] bool append(const std::string_view value) noexcept {
    if (value.size() > remaining_.size() ||
        remaining_.substr(0u, value.size()) != value) {
      return false;
    }
    remaining_.remove_prefix(value.size());
    return true;
  }

  [[nodiscard]] bool complete() const noexcept { return remaining_.empty(); }

private:
  std::string_view remaining_{};
};

template <typename Sink>
[[nodiscard]] bool EmitVulkanStencilSource(
    Sink &sink, const rund::kernel::StencilOp op,
    const rund::kernel::StencilElement element,
    const rund::kernel::ComputeDomain domain,
    const StencilGpuShape
        shape) noexcept(noexcept(sink.append(std::string_view{}))) {
  if (!shape.valid()) {
    return false;
  }
  const bool wide = element == rund::kernel::StencilElement::U64;
  const bool signed_extrema =
      IsSignedDomain(domain) && op != rund::kernel::StencilOp::Sum;
  if (!sink.append(R"glsl(#version 450
#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require
layout(local_size_x = )glsl") ||
      !backend_source_recipe::append_decimal(sink, shape.width()) ||
      !sink.append(R"glsl() in;
layout(set = 0, binding = 0, std430) readonly buffer Params {
  uint64_t element_count;
  uint64_t radius;
} params;
layout(set = 0, binding = 1, std430) readonly buffer Input {
)glsl") ||
      !sink.append(wide ? "  uint64_t input_values[];\n"
                        : "  uint input_values[];\n") ||
      !sink.append(R"glsl(};
layout(set = 0, binding = 2, std430) buffer Output {
)glsl") ||
      !sink.append(wide ? "  uint64_t output_values[];\n"
                        : "  uint output_values[];\n") ||
      !sink.append("};\n")) {
    return false;
  }
  if (shape.uses_shared_memory() &&
      (!sink.append("shared ") || !sink.append(wide ? "uint64_t" : "uint") ||
       !sink.append(" stencil_tile[") ||
       !backend_source_recipe::append_decimal(
           sink, shape.shared_element_capacity()) ||
       !sink.append("];\n"))) {
    return false;
  }
  return EmitVulkanStencilBody(sink, op, wide, signed_extrema, shape);
}

} // namespace

std::string VulkanStencilSource(const rund::kernel::StencilOp op,
                                const rund::kernel::StencilElement element,
                                const rund::kernel::ComputeDomain domain,
                                const StencilGpuShape shape) {
  return backend_source_recipe::materialize([&](auto &sink) {
    return EmitVulkanStencilSource(sink, op, element, domain, shape);
  });
}

bool VulkanStencilSourceBytes(const rund::kernel::StencilOp op,
                              const rund::kernel::StencilElement element,
                              const rund::kernel::ComputeDomain domain,
                              const StencilGpuShape shape,
                              std::uint64_t &bytes) noexcept {
  return backend_source_recipe::bytes(
      [&](backend_source_recipe::CountSink &sink) noexcept {
        return EmitVulkanStencilSource(sink, op, element, domain, shape);
      },
      bytes);
}

bool VulkanStencilSourceMatches(const rund::kernel::StencilOp op,
                                const rund::kernel::StencilElement element,
                                const rund::kernel::ComputeDomain domain,
                                const StencilGpuShape shape,
                                const std::string_view source,
                                const std::uint64_t source_hash) noexcept {
  if (SourceHash(source) != source_hash) {
    return false;
  }
  MatchSink sink{source};
  return EmitVulkanStencilSource(sink, op, element, domain, shape) &&
         sink.complete();
}
#endif

} // namespace rund::node::accel::detail
