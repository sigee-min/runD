#include "local.hpp"

namespace rund::node::accel::detail {

#if defined(RUND_NODE_HAVE_VULKAN_SDK)
std::string VulkanHistogramSource(const bool clear) {
  std::string source;
  source += "#version 450\n";
  source += "#extension GL_EXT_shader_explicit_arithmetic_types_int64 : ";
  source += "require\n";
  source += "layout(local_size_x = 256) in;\n";
  source += "layout(set = 0, binding = 0, std430) readonly buffer Params {\n";
  source += "  uint64_t element_count;\n";
  source += "  uint64_t bin_count;\n";
  source += "} params;\n";
  source += "layout(set = 0, binding = 1, std430) readonly buffer Bins {\n";
  source += "  uint bins[];\n";
  source += "};\n";
  source += "layout(set = 0, binding = 2, std430) buffer Counts {\n";
  source += "  uint counts[];\n";
  source += "};\n";
  source += "layout(set = 0, binding = 3, std430) buffer Status {\n";
  source += "  uint status_words[];\n";
  source += "};\n";
  source += "void main() {\n";
  source += "  const uint gid = gl_GlobalInvocationID.x;\n";
  if (clear) {
    source += "  if (uint64_t(gid) < params.bin_count) { counts[gid] = 0u; }\n";
    source += "  if (gid == 0u) { status_words[0] = 0xffffffffu; }\n";
  } else {
    source += "  if (uint64_t(gid) >= params.element_count) { return; }\n";
    source += "  const uint bin = bins[gid];\n";
    source += "  if (uint64_t(bin) >= params.bin_count) {\n";
    source += "    atomicExchange(status_words[0], 0u);\n";
    source += "    return;\n";
    source += "  }\n";
    source += "  atomicAdd(counts[bin], 1u);\n";
  }
  source += "}\n";
  return source;
}
#endif

}  // namespace rund::node::accel::detail
