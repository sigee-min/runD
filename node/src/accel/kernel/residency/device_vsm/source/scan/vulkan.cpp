#include "internal.hpp"

#include <kernel/program/compute/lowering/format.hpp>

namespace rund::node::accel::detail::device_vsm_scan_source {

std::string vulkan_source_u64(const rund::kernel::ArtifactKey &key,
                              const rund::kernel::ScanOp operation,
                              const DeviceVsmScanMap map) {
  if (map.kind != DeviceVsmScanMapKind::None &&
      map.kind != DeviceVsmScanMapKind::AddWrapU64Immediate) {
    return {};
  }
  namespace lowering = rund::kernel::compute_lowering_detail;
  auto emitted_key = key;
  emitted_key.variant = rund::kernel::LoweringArtifactVariant::DeviceVsm;
  std::string source =
      "#version 450\n#extension "
      "GL_EXT_shader_explicit_arithmetic_types_int64 : require\n";
  source += "// rund.compute.device_vsm.scan\n";
  lowering::AppendKey(source, emitted_key, "// ");
  source +=
      "layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;\n"
      "layout(push_constant) uniform RundDeviceVsmDispatch { uint "
      "logical_elements; uint payload_elements; uint page_count; uint width; "
      "} rund_vsm;\n"
      "layout(set = 0, binding = 0, std430) readonly buffer Params { uint "
      "rund_params[]; };\n"
      "layout(set = 0, binding = 1, std430) readonly buffer Input { uint64_t "
      "rund_input[]; };\n"
      "layout(set = 0, binding = 2, std430) buffer Output { uint64_t "
      "rund_output[]; };\n"
      "layout(set = 0, binding = 3, std430) buffer Result { uint "
      "rund_result[]; };\n"
      "shared uint64_t low[256]; shared uint64_t high[256];\n"
      "shared uint64_t carry_low; shared uint64_t carry_high;\n"
      "shared uint overflow;\n"
      "shared uint failed_page;\n"
      "void main() {\n"
      "  const uint local = gl_LocalInvocationID.x;\n"
      "  uint64_t lane_low = 0ul; uint64_t lane_high = 0ul;\n"
      "  for (uint index = local; index < rund_vsm.logical_elements; index += "
      "256u) {\n"
      "    const uint64_t mapped = rund_input[index] + " +
      std::to_string(map.immediate) +
      "ul;\n"
      "    const uint64_t next = lane_low + mapped;\n"
      "    lane_high += uint64_t(next < lane_low); lane_low = next;\n"
      "  }\n"
      "  low[local] = lane_low; high[local] = lane_high; barrier();\n"
      "  for (uint stride = 128u; stride != 0u; stride >>= 1u) {\n"
      "    if (local < stride) {\n"
      "      const uint64_t left = low[local];\n"
      "      const uint64_t combined = left + low[local + stride];\n"
      "      high[local] += high[local + stride] + uint64_t(combined < left);\n"
      "      low[local] = combined;\n"
      "    }\n"
      "    barrier();\n"
      "  }\n"
      "  if (local == 0u) { overflow = uint(high[0] != 0ul); } barrier();\n"
      "  if (local == 0u && overflow != 0u) {\n"
      "    uint64_t prefix = 0ul; failed_page = 0u;\n"
      "    for (uint index = 0u; index < rund_vsm.logical_elements; "
      "++index) {\n"
      "      const uint64_t mapped = rund_input[index] + " +
      std::to_string(map.immediate) +
      "ul;\n"
      "      const uint64_t next = prefix + mapped;\n"
      "      if (next < prefix) { failed_page = index / "
      "rund_vsm.payload_elements; break; }\n"
      "      prefix = next;\n"
      "    }\n"
      "  }\n"
      "  barrier();\n"
      "  if (local == 0u) { carry_low = 0ul; carry_high = 0ul; } barrier();\n"
      "  for (uint chunk = 0u; chunk < rund_vsm.logical_elements; chunk += "
      "256u) {\n"
      "    const uint index = chunk + local;\n"
      "    low[local] = index < rund_vsm.logical_elements ? "
      "rund_input[index] + " +
      std::to_string(map.immediate) +
      "ul : 0ul; high[local] = 0ul; barrier();\n"
      "    for (uint offset = 1u; offset < 256u; offset <<= 1u) {\n"
      "      uint64_t next_low = low[local]; uint64_t next_high = "
      "high[local];\n"
      "      if (local >= offset) {\n"
      "        const uint64_t left = low[local - offset];\n"
      "        const uint64_t combined = left + next_low;\n"
      "        next_high += high[local - offset] + uint64_t(combined < left);\n"
      "        next_low = combined;\n"
      "      }\n"
      "      barrier(); low[local] = next_low; high[local] = next_high; "
      "barrier();\n"
      "    }\n"
      "    if (index < rund_vsm.logical_elements && overflow == 0u) {\n"
      "      ";
  source += operation == rund::kernel::ScanOp::InclusiveSum
                ? "const uint64_t value = low[local] + carry_low;\n"
                : "const uint64_t value = local == 0u ? carry_low : "
                  "low[local - 1u] + carry_low;\n";
  source +=
      "      rund_output[index] = value;\n"
      "    }\n"
      "    barrier();\n"
      "    if (local == 0u) {\n"
      "      const uint count = min(256u, rund_vsm.logical_elements - "
      "chunk);\n"
      "      const uint64_t prior = carry_low;\n"
      "      carry_low = prior + low[count - 1u];\n"
      "      carry_high += high[count - 1u] + uint64_t(carry_low < prior);\n"
      "    }\n"
      "    barrier();\n"
      "  }\n"
      "  if (local == 0u && overflow != 0u) {\n"
      "    atomicExchange(rund_result[0], rund_vsm.page_count);\n"
      "    atomicExchange(rund_result[1], rund_vsm.page_count);\n"
      "    atomicExchange(rund_result[2], rund_vsm.page_count);\n"
      "    atomicExchange(rund_result[3], failed_page);\n"
      "    atomicExchange(rund_result[6], 1u);\n"
      "    atomicExchange(rund_result[7], failed_page);\n"
      "  } else if (local == 0u) {\n"
      "    for (uint word = 0u; word < 6u; ++word) { "
      "atomicExchange(rund_result[word], rund_vsm.page_count); }\n"
      "    atomicExchange(rund_result[6], 0u);\n"
      "    atomicExchange(rund_result[7], 0xffffffffu);\n"
      "  }\n"
      "}\n";
  return source;
}

} // namespace rund::node::accel::detail::device_vsm_scan_source
