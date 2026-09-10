#include "../internal.hpp"

#include <kernel/program/compute/lowering/format.hpp>

namespace rund::node::accel::detail::device_vsm_reduce_source {

std::string metal_additive_source(const rund::kernel::ArtifactKey &key,
                                  const rund::kernel::ReduceElement element,
                                  const rund::kernel::ReduceOp operation) {
  namespace lowering = rund::kernel::compute_lowering_detail;
  const bool u32 = element == rund::kernel::ReduceElement::U32;
  const bool sum = operation == rund::kernel::ReduceOp::Sum;
  if ((!u32 && element != rund::kernel::ReduceElement::U64) ||
      (!sum && operation != rund::kernel::ReduceOp::CountNonzero)) {
    return {};
  }
  auto emitted_key = key;
  emitted_key.variant = rund::kernel::LoweringArtifactVariant::DeviceVsm;
  const char *const scalar = u32 ? "uint" : "ulong";
  std::string source;
  source += "#include <metal_stdlib>\nusing namespace metal;\n";
  source += sum ? "// rund.compute.device_vsm.reduce.sum\n"
                : "// rund.compute.device_vsm.reduce.count_nonzero\n";
  lowering::AppendKey(source, emitted_key, "// ");
  source += "struct RundDeviceVsmConfig { ulong logical_elements; uint "
            "payload_elements; uint page_count; uint width; };\n";
  source += "kernel void rund_compute_map_";
  lowering::AppendHex64Digits(source, key.op_hash_hi);
  source += "_";
  lowering::AppendHex64Digits(source, key.op_hash_lo);
  source += "_device_vsm(\n"
            "    constant uchar* params [[buffer(0)]],\n"
            "    const device ";
  source += scalar;
  source += "* input [[buffer(1)]],\n    device ";
  source += scalar;
  source +=
      "* output [[buffer(2)]],\n"
      "    constant RundDeviceVsmConfig& config [[buffer(3)]],\n"
      "    device atomic_uint* result [[buffer(4)]],\n"
      "    uint local [[thread_index_in_threadgroup]]) {\n"
      "  (void)params;\n"
      "  threadgroup ulong partial_low[256];\n"
      "  threadgroup ulong partial_high[256];\n"
      "  ulong low = 0ul; ulong high = 0ul;\n"
      "  for (ulong index = ulong(local); index < config.logical_elements; "
      "index += 256ul) {\n"
      "    const ulong value = ";
  source += sum ? "ulong(input[index]);\n" : "ulong(input[index] != 0);\n";
  source += "    const ulong next = low + value;\n"
            "    high += ulong(next < low); low = next;\n"
            "  }\n"
            "  partial_low[local] = low; partial_high[local] = high;\n"
            "  threadgroup_barrier(mem_flags::mem_threadgroup);\n"
            "  for (uint stride = 128u; stride != 0u; stride >>= 1u) {\n"
            "    if (local < stride) {\n"
            "      const ulong left = partial_low[local];\n"
            "      const ulong combined = left + partial_low[local + stride];\n"
            "      partial_high[local] += partial_high[local + stride] + "
            "ulong(combined < left);\n"
            "      partial_low[local] = combined;\n"
            "    }\n"
            "    threadgroup_barrier(mem_flags::mem_threadgroup);\n"
            "  }\n"
            "  if (local == 0u) {\n";
  if (sum) {
    source += "    const uint overflow = ";
    source +=
        u32 ? "uint(partial_high[0] != 0ul || partial_low[0] > 0xfffffffful);\n"
            : "uint(partial_high[0] != 0ul);\n";
    source +=
        "    if (overflow != 0u) {\n"
        "      uint failed_page = 0u;\n"
        "      ulong prefix_low = 0ul; ulong prefix_high = 0ul;\n"
        "      for (ulong index = 0ul; index < config.logical_elements; "
        "++index) {\n"
        "        const ulong value = ulong(input[index]);\n"
        "        const ulong next = prefix_low + value;\n"
        "        prefix_high += ulong(next < prefix_low); prefix_low = next;\n"
        "        if (";
    source += u32 ? "prefix_high != 0ul || prefix_low > 0xfffffffful"
                  : "prefix_high != 0ul";
    source +=
        ") { failed_page = uint(index / ulong(config.payload_elements)); "
        "break; }\n"
        "      }\n"
        "      atomic_store_explicit(&result[0], config.page_count, "
        "memory_order_relaxed);\n"
        "      atomic_store_explicit(&result[1], config.page_count, "
        "memory_order_relaxed);\n"
        "      atomic_store_explicit(&result[2], config.page_count, "
        "memory_order_relaxed);\n"
        "      atomic_store_explicit(&result[3], failed_page, "
        "memory_order_relaxed);\n"
        "      atomic_store_explicit(&result[6], 1u, memory_order_relaxed);\n"
        "      atomic_store_explicit(&result[7], failed_page, "
        "memory_order_relaxed);\n"
        "      return;\n"
        "    }\n";
  }
  source += "    output[0] = ";
  source += u32 ? "uint(partial_low[0]);\n" : "partial_low[0];\n";
  source += "    for (uint word = 0u; word < 4u; ++word) { "
            "atomic_store_explicit(&result[word], config.page_count, "
            "memory_order_relaxed); }\n"
            "    atomic_store_explicit(&result[4], 1u, memory_order_relaxed);\n"
            "    atomic_store_explicit(&result[5], 1u, memory_order_relaxed);\n"
            "    atomic_store_explicit(&result[6], 0u, memory_order_relaxed);\n"
            "    atomic_store_explicit(&result[7], 0xffffffffu, "
            "memory_order_relaxed);\n"
            "  }\n"
            "}\n";
  return source;
}

} // namespace rund::node::accel::detail::device_vsm_reduce_source
