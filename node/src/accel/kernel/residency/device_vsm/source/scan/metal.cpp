#include "internal.hpp"

#include <kernel/program/compute/lowering/format.hpp>

namespace rund::node::accel::detail::device_vsm_scan_source {

std::string metal_source_u64(const rund::kernel::ArtifactKey &key,
                             const rund::kernel::ScanOp operation,
                             const DeviceVsmScanMap map) {
  if (map.kind != DeviceVsmScanMapKind::None &&
      map.kind != DeviceVsmScanMapKind::AddWrapU64Immediate) {
    return {};
  }
  namespace lowering = rund::kernel::compute_lowering_detail;
  auto emitted_key = key;
  emitted_key.variant = rund::kernel::LoweringArtifactVariant::DeviceVsm;
  std::string source;
  source += "#include <metal_stdlib>\nusing namespace metal;\n";
  source += "// rund.compute.device_vsm.scan\n";
  lowering::AppendKey(source, emitted_key, "// ");
  source += "struct RundDeviceVsmConfig { ulong logical_elements; uint "
            "payload_elements; uint page_count; uint width; };\n"
            "kernel void rund_compute_map_";
  lowering::AppendHex64Digits(source, key.op_hash_hi);
  source += "_";
  lowering::AppendHex64Digits(source, key.op_hash_lo);
  source +=
      "_device_vsm(\n"
      "    constant uchar* rund_params [[buffer(0)]],\n"
      "    const device ulong* rund_input [[buffer(1)]],\n"
      "    device ulong* rund_output [[buffer(2)]],\n"
      "    constant RundDeviceVsmConfig& rund_vsm [[buffer(3)]],\n"
      "    device atomic_uint* rund_result [[buffer(4)]],\n"
      "    uint local [[thread_index_in_threadgroup]]) {\n"
      "  (void)rund_params;\n"
      "  threadgroup ulong low[256];\n"
      "  threadgroup ulong high[256];\n"
      "  threadgroup ulong carry_low;\n"
      "  threadgroup ulong carry_high;\n"
      "  threadgroup uint overflow;\n"
      "  threadgroup uint failed_page;\n"
      "  ulong lane_low = 0ul;\n"
      "  ulong lane_high = 0ul;\n"
      "  for (ulong index = ulong(local); index < rund_vsm.logical_elements; "
      "index += 256ul) {\n"
      "    const ulong mapped = rund_input[index] + " +
      std::to_string(map.immediate) +
      "ul;\n"
      "    const ulong next = lane_low + mapped;\n"
      "    lane_high += ulong(next < lane_low);\n"
      "    lane_low = next;\n"
      "  }\n"
      "  low[local] = lane_low; high[local] = lane_high;\n"
      "  threadgroup_barrier(mem_flags::mem_threadgroup);\n"
      "  for (uint stride = 128u; stride != 0u; stride >>= 1u) {\n"
      "    if (local < stride) {\n"
      "      const ulong left = low[local];\n"
      "      const ulong combined = left + low[local + stride];\n"
      "      high[local] += high[local + stride] + ulong(combined < left);\n"
      "      low[local] = combined;\n"
      "    }\n"
      "    threadgroup_barrier(mem_flags::mem_threadgroup);\n"
      "  }\n"
      "  if (local == 0u) { overflow = uint(high[0] != 0ul); }\n"
      "  threadgroup_barrier(mem_flags::mem_threadgroup);\n"
      "  if (local == 0u && overflow != 0u) {\n"
      "    ulong prefix = 0ul; failed_page = 0u;\n"
      "    for (ulong index = 0ul; index < rund_vsm.logical_elements; "
      "++index) {\n"
      "      const ulong mapped = rund_input[index] + " +
      std::to_string(map.immediate) +
      "ul;\n"
      "      const ulong next = prefix + mapped;\n"
      "      if (next < prefix) { failed_page = uint(index / "
      "ulong(rund_vsm.payload_elements)); break; }\n"
      "      prefix = next;\n"
      "    }\n"
      "  }\n"
      "  threadgroup_barrier(mem_flags::mem_threadgroup);\n"
      "  if (local == 0u) { carry_low = 0ul; carry_high = 0ul; }\n"
      "  threadgroup_barrier(mem_flags::mem_threadgroup);\n"
      "  for (ulong chunk = 0ul; chunk < rund_vsm.logical_elements; "
      "chunk += 256ul) {\n"
      "    const ulong index = chunk + ulong(local);\n"
      "    low[local] = index < rund_vsm.logical_elements ? "
      "rund_input[index] + " +
      std::to_string(map.immediate) +
      "ul : 0ul;\n"
      "    high[local] = 0ul;\n"
      "    threadgroup_barrier(mem_flags::mem_threadgroup);\n"
      "    for (uint offset = 1u; offset < 256u; offset <<= 1u) {\n"
      "      ulong next_low = low[local]; ulong next_high = high[local];\n"
      "      if (local >= offset) {\n"
      "        const ulong left = low[local - offset];\n"
      "        const ulong combined = left + next_low;\n"
      "        next_high += high[local - offset] + ulong(combined < left);\n"
      "        next_low = combined;\n"
      "      }\n"
      "      threadgroup_barrier(mem_flags::mem_threadgroup);\n"
      "      low[local] = next_low; high[local] = next_high;\n"
      "      threadgroup_barrier(mem_flags::mem_threadgroup);\n"
      "    }\n"
      "    if (index < rund_vsm.logical_elements && overflow == 0u) {\n"
      "      ";
  source += operation == rund::kernel::ScanOp::InclusiveSum
                ? "const ulong value = low[local] + carry_low;\n"
                : "const ulong value = local == 0u ? carry_low : "
                  "low[local - 1u] + carry_low;\n";
  source +=
      "      rund_output[index] = value;\n"
      "    }\n"
      "    threadgroup_barrier(mem_flags::mem_threadgroup);\n"
      "    if (local == 0u) {\n"
      "      const uint count = uint(min(256ul, rund_vsm.logical_elements - "
      "chunk));\n"
      "      const ulong prior = carry_low;\n"
      "      carry_low = prior + low[count - 1u];\n"
      "      carry_high += high[count - 1u] + ulong(carry_low < prior);\n"
      "    }\n"
      "    threadgroup_barrier(mem_flags::mem_threadgroup);\n"
      "  }\n"
      "  if (local == 0u && overflow != 0u) {\n"
      "    atomic_store_explicit(&rund_result[0], rund_vsm.page_count, "
      "memory_order_relaxed);\n"
      "    atomic_store_explicit(&rund_result[1], rund_vsm.page_count, "
      "memory_order_relaxed);\n"
      "    atomic_store_explicit(&rund_result[2], rund_vsm.page_count, "
      "memory_order_relaxed);\n"
      "    atomic_store_explicit(&rund_result[3], failed_page, "
      "memory_order_relaxed);\n"
      "    atomic_store_explicit(&rund_result[6], 1u, "
      "memory_order_relaxed);\n"
      "    atomic_store_explicit(&rund_result[7], failed_page, "
      "memory_order_relaxed);\n"
      "  } else if (local == 0u) {\n"
      "    for (uint word = 0u; word < 6u; ++word) {\n"
      "      atomic_store_explicit(&rund_result[word], "
      "rund_vsm.page_count, memory_order_relaxed);\n"
      "    }\n"
      "    atomic_store_explicit(&rund_result[6], 0u, "
      "memory_order_relaxed);\n"
      "    atomic_store_explicit(&rund_result[7], 0xffffffffu, "
      "memory_order_relaxed);\n"
      "  }\n"
      "}\n";
  return source;
}

} // namespace rund::node::accel::detail::device_vsm_scan_source
