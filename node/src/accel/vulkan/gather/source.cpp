#include "local.hpp"

namespace rund::node::accel::detail {

#if defined(RUND_NODE_HAVE_VULKAN_SDK)
std::string VulkanGatherSource(const rund::kernel::GatherElement element,
                               const bool control) {
  const bool u64 = element == rund::kernel::GatherElement::U64;
  std::string source;
  source += "#version 450\n";
  source += "#extension GL_EXT_shader_explicit_arithmetic_types_int64 : ";
  source += "require\n";
  source += "layout(local_size_x = 256) in;\n";
  source += "layout(set = 0, binding = 0, std430) readonly buffer Params {\n";
  source += "  uint64_t element_count;\n";
  source += "  uint64_t source_count;\n";
  source += "  uint count_source;\n";
  source += "  uint reserved;\n";
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
  source += "layout(set = 0, binding = 5, std430) readonly buffer Count {\n";
  source += "  uint count_words[];\n";
  source += "};\n";
  source += "layout(set = 0, binding = 6, std430) buffer Indirect {\n";
  source += "  uint indirect_args[];\n";
  source += "};\n";
  if (control) {
    source += "uint64_t logical_count() {\n";
    source += "  if (params.count_source == 0u) { return params.element_count; }\n";
    source += "  if (params.count_source == 1u) { return uint64_t(count_words[0]); }\n";
    source += "  return uint64_t(count_words[0]) | (uint64_t(count_words[1]) << 32u);\n";
    source += "}\n";
    source += "shared uint invalids[256];\n";
    source += "void main() {\n";
    source += "  const uint tid = gl_LocalInvocationID.x;\n";
    source += "  const uint64_t logical = logical_count();\n";
    source += "  uint local_invalid = 0xffffffffu;\n";
    source += "  if (logical <= params.element_count) {\n";
    source += "    for (uint ordinal = tid; uint64_t(ordinal) < logical;) {\n";
    source += "      if (uint64_t(indices[ordinal]) >= params.source_count) { local_invalid = min(local_invalid, ordinal); }\n";
    source += "      if (ordinal > 0xffffffffu - 256u) { break; }\n";
    source += "      ordinal += 256u;\n";
    source += "    }\n";
    source += "  }\n";
    source += "  invalids[tid] = local_invalid;\n";
    source += "  barrier();\n";
    source += "  for (uint stride = 128u; stride != 0u; stride >>= 1u) {\n";
    source += "    if (tid < stride) { invalids[tid] = min(invalids[tid], invalids[tid + stride]); }\n";
    source += "    barrier();\n";
    source += "  }\n";
    source += "  if (tid != 0u) { return; }\n";
    source += "  status[0] = 0u; status[1] = uint(min(logical, uint64_t(0xffffffffu)));\n";
    source += "  indirect_args[0] = 0u; indirect_args[1] = 1u; indirect_args[2] = 1u; indirect_args[3] = 0u;\n";
    source += "  if (logical > params.element_count) { status[0] = 1u; status[1] = uint(params.element_count); return; }\n";
    source += "  if (invalids[0] != 0xffffffffu) { status[0] = 2u; status[1] = invalids[0]; return; }\n";
    source += "  indirect_args[0] = uint((logical + 255u) / 256u); indirect_args[3] = uint(logical);\n";
    source += "}\n";
    return source;
  }
  source += "void main() {\n";
  source += "  const uint gid = gl_GlobalInvocationID.x;\n";
  source += "  if (gid >= indirect_args[3]) { return; }\n";
  source += "  const uint source_index = indices[gid];\n";
  source += "  output_values[gid] = values[source_index];\n";
  source += "}\n";
  return source;
}
#endif

}  // namespace rund::node::accel::detail
