#include "../internal.hpp"

#include <kernel/program/compute/lowering/format.hpp>

namespace rund::node::accel::detail::device_vsm_scan_source {

std::string metal_source_u32(const rund::kernel::ArtifactKey &key,
                             const rund::kernel::ScanOp operation,
                             const DeviceVsmScanMap map) {
  if (map.active()) {
    return {};
  }
  namespace lowering = rund::kernel::compute_lowering_detail;
  auto emitted_key = key;
  emitted_key.variant = rund::kernel::LoweringArtifactVariant::DeviceVsm;
  std::string source;
  source += "#include <metal_stdlib>\nusing namespace metal;\n";
  source += "// rund.compute.device_vsm.scan.u32\n";
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
      "    const device uint* rund_input [[buffer(1)]],\n"
      "    device uint* rund_output [[buffer(2)]],\n"
      "    constant RundDeviceVsmConfig& rund_vsm [[buffer(3)]],\n"
      "    device atomic_uint* rund_result [[buffer(4)]],\n"
      "    uint local [[thread_index_in_threadgroup]]) {\n"
      "  (void)rund_params;\n"
      "  threadgroup ulong partial[256];\n"
      "  threadgroup ulong carry;\n"
      "  threadgroup uint overflow;\n"
      "  threadgroup uint failed_page;\n"
      "  ulong lane_total = 0ul;\n"
      "  for (ulong index = ulong(local); index < rund_vsm.logical_elements; "
      "index += 256ul) { lane_total += ulong(rund_input[index]); }\n"
      "  partial[local] = lane_total;\n"
      "  threadgroup_barrier(mem_flags::mem_threadgroup);\n"
      "  for (uint stride = 128u; stride != 0u; stride >>= 1u) {\n"
      "    if (local < stride) { partial[local] += partial[local + stride]; }\n"
      "    threadgroup_barrier(mem_flags::mem_threadgroup);\n"
      "  }\n"
      "  if (local == 0u) { overflow = uint(partial[0] > 0xfffffffful); }\n"
      "  threadgroup_barrier(mem_flags::mem_threadgroup);\n"
      "  if (local == 0u && overflow != 0u) {\n"
      "    ulong prefix = 0ul; failed_page = 0u;\n"
      "    for (ulong index = 0ul; index < rund_vsm.logical_elements; "
      "++index) {\n"
      "      const ulong next = prefix + ulong(rund_input[index]);\n"
      "      if (next > 0xfffffffful) { failed_page = uint(index / "
      "ulong(rund_vsm.payload_elements)); break; }\n"
      "      prefix = next;\n"
      "    }\n"
      "  }\n"
      "  threadgroup_barrier(mem_flags::mem_threadgroup);\n"
      "  if (local == 0u) { carry = 0ul; }\n"
      "  threadgroup_barrier(mem_flags::mem_threadgroup);\n"
      "  for (ulong chunk = 0ul; chunk < rund_vsm.logical_elements; "
      "chunk += 256ul) {\n"
      "    const ulong index = chunk + ulong(local);\n"
      "    partial[local] = index < rund_vsm.logical_elements ? "
      "ulong(rund_input[index]) : 0ul;\n"
      "    threadgroup_barrier(mem_flags::mem_threadgroup);\n"
      "    for (uint offset = 1u; offset < 256u; offset <<= 1u) {\n"
      "      ulong next = partial[local];\n"
      "      if (local >= offset) { next += partial[local - offset]; }\n"
      "      threadgroup_barrier(mem_flags::mem_threadgroup);\n"
      "      partial[local] = next;\n"
      "      threadgroup_barrier(mem_flags::mem_threadgroup);\n"
      "    }\n"
      "    if (index < rund_vsm.logical_elements && overflow == 0u) {\n      ";
  source += operation == rund::kernel::ScanOp::InclusiveSum
                ? "const ulong value = partial[local] + carry;\n"
                : "const ulong value = local == 0u ? carry : "
                  "partial[local - 1u] + carry;\n";
  source +=
      "      rund_output[index] = uint(value);\n"
      "    }\n"
      "    threadgroup_barrier(mem_flags::mem_threadgroup);\n"
      "    if (local == 0u) {\n"
      "      const uint count = uint(min(256ul, rund_vsm.logical_elements - "
      "chunk));\n"
      "      carry += partial[count - 1u];\n"
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
      "    atomic_store_explicit(&rund_result[6], 1u, memory_order_relaxed);\n"
      "    atomic_store_explicit(&rund_result[7], failed_page, "
      "memory_order_relaxed);\n"
      "  } else if (local == 0u) {\n"
      "    for (uint word = 0u; word < 6u; ++word) { "
      "atomic_store_explicit(&rund_result[word], rund_vsm.page_count, "
      "memory_order_relaxed); }\n"
      "    atomic_store_explicit(&rund_result[6], 0u, memory_order_relaxed);\n"
      "    atomic_store_explicit(&rund_result[7], 0xffffffffu, "
      "memory_order_relaxed);\n"
      "  }\n"
      "}\n";
  return source;
}

} // namespace rund::node::accel::detail::device_vsm_scan_source
