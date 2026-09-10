#include "../internal.hpp"

#include <kernel/program/compute/lowering/format.hpp>

namespace rund::node::accel::detail::device_vsm_scan_source {

std::string vulkan_source_u32(const rund::kernel::ArtifactKey &key,
                              const rund::kernel::ScanOp operation,
                              const DeviceVsmScanMap map) {
  if (map.active()) {
    return {};
  }
  namespace lowering = rund::kernel::compute_lowering_detail;
  auto emitted_key = key;
  emitted_key.variant = rund::kernel::LoweringArtifactVariant::DeviceVsm;
  std::string source =
      "#version 450\n#extension "
      "GL_EXT_shader_explicit_arithmetic_types_int64 : require\n";
  source += "// rund.compute.device_vsm.scan.u32\n";
  lowering::AppendKey(source, emitted_key, "// ");
  source +=
      "layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;\n"
      "layout(push_constant) uniform RundDeviceVsmDispatch { uint "
      "logical_elements; uint payload_elements; uint page_count; uint width; "
      "} rund_vsm;\n"
      "layout(set = 0, binding = 0, std430) readonly buffer Params { uint "
      "rund_params[]; };\n"
      "layout(set = 0, binding = 1, std430) readonly buffer Input { uint "
      "rund_input[]; };\n"
      "layout(set = 0, binding = 2, std430) buffer Output { uint "
      "rund_output[]; };\n"
      "layout(set = 0, binding = 3, std430) buffer Result { uint "
      "rund_result[]; };\n"
      "shared uint64_t partial[256];\n"
      "shared uint64_t carry;\n"
      "shared uint overflow;\n"
      "shared uint failed_page;\n"
      "void main() {\n"
      "  const uint local = gl_LocalInvocationID.x;\n"
      "  uint64_t lane_total = 0ul;\n"
      "  for (uint index = local; index < rund_vsm.logical_elements; index += "
      "256u) { lane_total += uint64_t(rund_input[index]); }\n"
      "  partial[local] = lane_total; barrier();\n"
      "  for (uint stride = 128u; stride != 0u; stride >>= 1u) {\n"
      "    if (local < stride) { partial[local] += partial[local + stride]; }\n"
      "    barrier();\n"
      "  }\n"
      "  if (local == 0u) { overflow = uint(partial[0] > 0xfffffffful); }\n"
      "  barrier();\n"
      "  if (local == 0u && overflow != 0u) {\n"
      "    uint64_t prefix = 0ul; failed_page = 0u;\n"
      "    for (uint index = 0u; index < rund_vsm.logical_elements; "
      "++index) {\n"
      "      const uint64_t next = prefix + uint64_t(rund_input[index]);\n"
      "      if (next > 0xfffffffful) { failed_page = index / "
      "rund_vsm.payload_elements; break; }\n"
      "      prefix = next;\n"
      "    }\n"
      "  }\n"
      "  barrier();\n"
      "  if (local == 0u) { carry = 0ul; } barrier();\n"
      "  for (uint chunk = 0u; chunk < rund_vsm.logical_elements; chunk += "
      "256u) {\n"
      "    const uint index = chunk + local;\n"
      "    partial[local] = index < rund_vsm.logical_elements ? "
      "uint64_t(rund_input[index]) : 0ul;\n"
      "    barrier();\n"
      "    for (uint offset = 1u; offset < 256u; offset <<= 1u) {\n"
      "      uint64_t next = partial[local];\n"
      "      if (local >= offset) { next += partial[local - offset]; }\n"
      "      barrier(); partial[local] = next; barrier();\n"
      "    }\n"
      "    if (index < rund_vsm.logical_elements && overflow == 0u) {\n      ";
  source += operation == rund::kernel::ScanOp::InclusiveSum
                ? "const uint64_t value = partial[local] + carry;\n"
                : "const uint64_t value = local == 0u ? carry : "
                  "partial[local - 1u] + carry;\n";
  source += "      rund_output[index] = uint(value);\n"
            "    }\n"
            "    barrier();\n"
            "    if (local == 0u) {\n"
            "      const uint count = min(256u, rund_vsm.logical_elements - "
            "chunk);\n"
            "      carry += partial[count - 1u];\n"
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
