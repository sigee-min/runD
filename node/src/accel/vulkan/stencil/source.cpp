#include "local.hpp"
#include "../../domain.hpp"

namespace rund::node::accel::detail {

#if defined(RUND_NODE_HAVE_VULKAN_SDK)
#include "source/body.hpp"

std::string VulkanStencilSource(const rund::kernel::StencilOp op,
                                const rund::kernel::StencilElement element,
                                const rund::kernel::ComputeDomain domain) {
  const bool u64 = element == rund::kernel::StencilElement::U64;
  const bool signed_extrema =
      IsSignedDomain(domain) && op != rund::kernel::StencilOp::Sum;
  std::string source;
  source += "#version 450\n";
  source += "#extension GL_EXT_shader_explicit_arithmetic_types_int64 : ";
  source += "require\n";
  source += "layout(local_size_x = 256) in;\n";
  source += "layout(set = 0, binding = 0, std430) readonly buffer Params {\n";
  source += "  uint64_t element_count;\n";
  source += "  uint64_t radius;\n";
  source += "} params;\n";
  source += "layout(set = 0, binding = 1, std430) readonly buffer Input {\n";
  source += u64 ? "  uint64_t input_values[];\n" : "  uint input_values[];\n";
  source += "};\n";
  source += "layout(set = 0, binding = 2, std430) buffer Output {\n";
  source += u64 ? "  uint64_t output_values[];\n" : "  uint output_values[];\n";
  source += "};\n";
  AppendVulkanStencilBody(source, op, u64, signed_extrema);
  return source;
}
#endif

} // namespace rund::node::accel::detail
