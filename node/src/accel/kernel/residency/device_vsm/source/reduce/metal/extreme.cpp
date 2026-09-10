#include "../internal.hpp"

#include <kernel/program/compute/lowering/format.hpp>

namespace rund::node::accel::detail::device_vsm_reduce_source {

std::string metal_extreme_source(const rund::kernel::ArtifactKey &key,
                                 const rund::kernel::ReduceElement element,
                                 const rund::kernel::ReduceOp operation) {
  namespace lowering = rund::kernel::compute_lowering_detail;
  const bool u32 = element == rund::kernel::ReduceElement::U32;
  const bool minimum = operation == rund::kernel::ReduceOp::Min;
  if ((!u32 && element != rund::kernel::ReduceElement::U64) ||
      (!minimum && operation != rund::kernel::ReduceOp::Max)) {
    return {};
  }
  auto emitted_key = key;
  emitted_key.variant = rund::kernel::LoweringArtifactVariant::DeviceVsm;
  const char *const scalar = u32 ? "uint" : "ulong";
  const char *const identity =
      minimum ? (u32 ? "0xffffffffu" : "0xfffffffffffffffful") : "0";
  const char *const combine = minimum ? "min" : "max";
  std::string source;
  source += "#include <metal_stdlib>\nusing namespace metal;\n";
  source += minimum ? "// rund.compute.device_vsm.reduce.min\n"
                    : "// rund.compute.device_vsm.reduce.max\n";
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
  source += "* output [[buffer(2)]],\n"
            "    constant RundDeviceVsmConfig& config [[buffer(3)]],\n"
            "    device atomic_uint* result [[buffer(4)]],\n"
            "    uint local [[thread_index_in_threadgroup]]) {\n"
            "  (void)params;\n"
            "  threadgroup ";
  source += scalar;
  source += " partial[256];\n  ";
  source += scalar;
  source += " value = ";
  source += identity;
  source +=
      ";\n"
      "  for (ulong index = ulong(local); index < config.logical_elements; "
      "index += 256ul) { value = ";
  source += combine;
  source += "(value, input[index]); }\n"
            "  partial[local] = value;\n"
            "  threadgroup_barrier(mem_flags::mem_threadgroup);\n"
            "  for (uint stride = 128u; stride != 0u; stride >>= 1u) {\n"
            "    if (local < stride) { partial[local] = ";
  source += combine;
  source += "(partial[local], partial[local + stride]); }\n"
            "    threadgroup_barrier(mem_flags::mem_threadgroup);\n"
            "  }\n"
            "  if (local == 0u) {\n"
            "    output[0] = partial[0];\n"
            "    for (uint word = 0u; word < 4u; ++word) { "
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
