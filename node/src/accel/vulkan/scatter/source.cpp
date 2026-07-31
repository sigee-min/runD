#include "local.hpp"

namespace rund::node::accel::detail {

#if defined(RUND_NODE_HAVE_VULKAN_SDK)
std::string VulkanScatterSource(const rund::kernel::ScatterElement element) {
  const bool u64 = element == rund::kernel::ScatterElement::U64;
  std::string source;
  source += "#version 450\n";
  source += "#extension GL_EXT_shader_explicit_arithmetic_types_int64 : ";
  source += "require\n";
  source += "layout(local_size_x = 256) in;\n";
  source += "layout(set = 0, binding = 0, std430) readonly buffer Params {\n";
  source += "  uint64_t element_count;\n";
  source += "  uint64_t output_count;\n";
  source += "} params;\n";
  source += "layout(set = 0, binding = 1, std430) readonly buffer Values {\n";
  source += u64 ? "  uint64_t values[];\n" : "  uint values[];\n";
  source += "};\n";
  source += "layout(set = 0, binding = 2, std430) readonly buffer Indices {\n";
  source += "  uint indices[];\n";
  source += "};\n";
  source += "layout(set = 0, binding = 3, std430) buffer Output {\n";
  source += u64 ? "  uint64_t output_values[];\n" : "  uint output_values[];\n";
  source += "};\n";
  source += "layout(set = 0, binding = 4, std430) buffer Status {\n";
  source += "  uint status[];\n";
  source += "};\n";
  source += "void record_failure(uint gid, uint reason) {\n";
  source += "  atomicMin(status[0], (gid << 1u) | reason);\n";
  source += "}\n";
  source += "bool claim_target(uint gid, uint target) {\n";
  source += "  const uint prior = atomicMin(status[target + 1u], gid);\n";
  source += "  if (prior == 0xffffffffu) { return true; }\n";
  source += "  const uint duplicate = prior < gid ? gid : prior;\n";
  source += "  record_failure(duplicate, 1u);\n";
  source += "  return false;\n";
  source += "}\n";
  source += "void main() {\n";
  source += "  const uint gid = gl_GlobalInvocationID.x;\n";
  source += "  if (uint64_t(gid) >= params.element_count) { return; }\n";
  source += "  const uint target = indices[gid];\n";
  source += "  if (uint64_t(target) >= params.output_count) {\n";
  source += "    record_failure(gid, 0u);\n";
  source += "    return;\n";
  source += "  }\n";
  source += "  if (claim_target(gid, target)) {\n";
  source += "    output_values[target] = values[gid];\n";
  source += "  }\n";
  source += "}\n";
  return source;
}
#endif

} // namespace rund::node::accel::detail
