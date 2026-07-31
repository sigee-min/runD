#include "local.hpp"

namespace rund::node::accel::detail {

#if defined(RUND_NODE_HAVE_VULKAN_SDK)
namespace {

void AppendCompactBase(std::string &source) {
  source += "#version 450\n";
  source += "#extension GL_EXT_shader_explicit_arithmetic_types_int64 : ";
  source += "require\n";
  source += "layout(local_size_x = ";
  source += std::to_string(block::VulkanCompact);
  source += ") in;\n";
  source += "const uint kCompactBlockSize = ";
  source += std::to_string(block::VulkanCompact);
  source += "u;\n";
  source += "layout(set = 0, binding = 0, std430) readonly buffer Params {\n";
  source += "  uint64_t element_count;\n";
  source += "  uint64_t output_capacity;\n";
  source += "} params;\n";
  source += "layout(set = 0, binding = 5, std430) buffer Status {\n";
  source += "  uint status;\n};\n";
}

void AppendCompactBindings(std::string &source, const bool flags,
                           const bool counts, const bool offsets,
                           const bool output) {
  if (flags) {
    source += "layout(set = 0, binding = 1, std430) readonly buffer Flags {\n";
    source += "  uint flags[];\n};\n";
  }
  if (counts) {
    source += "layout(set = 0, binding = 2, std430) buffer Counts {\n";
    source += "  uint counts[];\n};\n";
  }
  if (offsets) {
    source += "layout(set = 0, binding = 3, std430) buffer Offsets {\n";
    source += "  uint offsets[];\n};\n";
  }
  if (output) {
    source += "layout(set = 0, binding = 4, std430) buffer Output {\n";
    source += "  uint output_ids[];\n};\n";
  }
}

void AppendClassify(std::string &source) {
  AppendCompactBindings(source, true, true, false, false);
  source += "shared uint local_values[kCompactBlockSize];\n";
  source += "void main() {\n";
  source += "  const uint lane = gl_LocalInvocationID.x;\n";
  source += "  const uint index = gl_GlobalInvocationID.x;\n";
  source += "  uint selected = 0u;\n";
  source += "  if (uint64_t(index) < params.element_count) {\n";
  source += "    selected = flags[index] != 0u ? 1u : 0u;\n";
  source += "  }\n";
  source += "  local_values[lane] = selected;\n";
  source += "  barrier();\n";
  source += "  for (uint width = kCompactBlockSize >> 1u; width != 0u; "
            "width >>= 1u) {\n";
  source += "    if (lane < width) { local_values[lane] += "
            "local_values[lane + width]; }\n";
  source += "    barrier();\n";
  source += "  }\n";
  source += "  if (lane == 0u) { counts[gl_WorkGroupID.x] = "
            "local_values[0]; }\n";
  source += "}\n";
}

void AppendPrefix(std::string &source) {
  AppendCompactBindings(source, false, true, true, false);
  source += "shared uint local_values[kCompactBlockSize];\n";
  source += "shared uint local_carry;\n";
  source += "shared uint local_base;\n";
  source += "void main() {\n";
  source += "  const uint lane = gl_LocalInvocationID.x;\n";
  source += "  const uint block_count = uint((params.element_count - "
            "uint64_t(1)) / uint64_t(kCompactBlockSize)) + 1u;\n";
  source += "  if (lane == 0u) { local_carry = 0u; }\n";
  source += "  barrier();\n";
  source += "  for (uint tile = 0u; tile < block_count; "
            "tile += kCompactBlockSize) {\n";
  source += "    const uint block = tile + lane;\n";
  source += "    const uint value = block < block_count ? counts[block] : "
            "0u;\n";
  source += "    local_values[lane] = value;\n";
  source += "    if (lane == 0u) { local_base = local_carry; }\n";
  source += "    barrier();\n";
  source += "    for (uint step = 1u; step < kCompactBlockSize; "
            "step <<= 1u) {\n";
  source += "      const uint addend = lane >= step ? "
            "local_values[lane - step] : 0u;\n";
  source += "      barrier();\n";
  source += "      local_values[lane] += addend;\n";
  source += "      barrier();\n";
  source += "    }\n";
  source += "    if (block < block_count) {\n";
  source += "      offsets[block] = local_base + local_values[lane] - value;\n";
  source += "    }\n";
  source += "    barrier();\n";
  source += "    if (lane == 0u) {\n";
  source += "      const uint remaining = block_count - tile;\n";
  source += "      const uint last = min(remaining, kCompactBlockSize) - 1u;\n";
  source += "      local_carry = local_base + local_values[last];\n";
  source += "    }\n";
  source += "    barrier();\n";
  source += "  }\n";
  source += "  if (lane == 0u) {\n";
  source += "    status = uint64_t(local_carry) > params.output_capacity "
            "? 1u : 0u;\n";
  source += "  }\n";
  source += "}\n";
}

void AppendScatter(std::string &source) {
  AppendCompactBindings(source, true, false, true, true);
  source += "shared uint local_values[kCompactBlockSize];\n";
  source += "void main() {\n";
  source += "  const uint lane = gl_LocalInvocationID.x;\n";
  source += "  const uint index = gl_GlobalInvocationID.x;\n";
  source += "  uint selected = 0u;\n";
  source += "  if (uint64_t(index) < params.element_count) {\n";
  source += "    selected = flags[index] != 0u ? 1u : 0u;\n";
  source += "  }\n";
  source += "  local_values[lane] = selected;\n";
  source += "  barrier();\n";
  source += "  for (uint step = 1u; step < kCompactBlockSize; "
            "step <<= 1u) {\n";
  source += "    const uint addend = lane >= step ? "
            "local_values[lane - step] : 0u;\n";
  source += "    barrier();\n";
  source += "    local_values[lane] += addend;\n";
  source += "    barrier();\n";
  source += "  }\n";
  source += "  if (selected != 0u) {\n";
  source += "    const uint target = offsets[gl_WorkGroupID.x] + "
            "local_values[lane] - 1u;\n";
  source += "    if (uint64_t(target) < params.output_capacity) {\n";
  source += "      output_ids[target] = index;\n";
  source += "    }\n";
  source += "  }\n";
  source += "}\n";
}

} // namespace

std::string VulkanCompactSource(const CompactStage stage) {
  std::string source;
  AppendCompactBase(source);
  switch (stage) {
  case CompactStage::Classify:
    AppendClassify(source);
    break;
  case CompactStage::Prefix:
    AppendPrefix(source);
    break;
  case CompactStage::Scatter:
    AppendScatter(source);
    break;
  }
  return source;
}
#endif

} // namespace rund::node::accel::detail
