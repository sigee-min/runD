#include "../internal.hpp"

#include <kernel/program/compute/lowering/format.hpp>

namespace rund::node::accel::detail::device_vsm_reduce_source {

std::string vulkan_extreme_source(const rund::kernel::ArtifactKey &key,
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
  const char *const scalar = u32 ? "uint" : "uint64_t";
  const char *const identity =
      minimum ? (u32 ? "0xffffffffu" : "0xfffffffffffffffful") : "0";
  const char *const combine = minimum ? "min" : "max";
  std::string source =
      "#version 450\n#extension "
      "GL_EXT_shader_explicit_arithmetic_types_int64 : require\n";
  source += minimum ? "// rund.compute.device_vsm.reduce.min\n"
                    : "// rund.compute.device_vsm.reduce.max\n";
  lowering::AppendKey(source, emitted_key, "// ");
  source +=
      "layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;\n"
      "layout(push_constant) uniform Dispatch { uint logical_elements; uint "
      "payload_elements; uint page_count; uint width; } config;\n"
      "layout(set = 0, binding = 0, std430) readonly buffer Params { uint "
      "params[]; };\n"
      "layout(set = 0, binding = 1, std430) readonly buffer Input { ";
  source += scalar;
  source += " input_values[]; };\nlayout(set = 0, binding = 2, std430) buffer "
            "Output { ";
  source += scalar;
  source +=
      " output_values[]; };\n"
      "layout(set = 0, binding = 3, std430) buffer Result { uint result[]; "
      "};\n"
      "shared ";
  source += scalar;
  source += " partial[256];\nvoid main() {\n  const uint local = "
            "gl_LocalInvocationID.x;\n  ";
  source += scalar;
  source += " value = ";
  source += identity;
  source +=
      ";\n"
      "  for (uint index = local; index < config.logical_elements; index += "
      "256u) { value = ";
  source += combine;
  source += "(value, input_values[index]); }\n"
            "  partial[local] = value; barrier();\n"
            "  for (uint stride = 128u; stride != 0u; stride >>= 1u) {\n"
            "    if (local < stride) { partial[local] = ";
  source += combine;
  source += "(partial[local], partial[local + stride]); }\n"
            "    barrier();\n"
            "  }\n"
            "  if (local == 0u) {\n"
            "    output_values[0] = partial[0];\n"
            "    for (uint word = 0u; word < 4u; ++word) { "
            "atomicExchange(result[word], config.page_count); }\n"
            "    atomicExchange(result[4], 1u);\n"
            "    atomicExchange(result[5], 1u);\n"
            "    atomicExchange(result[6], 0u);\n"
            "    atomicExchange(result[7], 0xffffffffu);\n"
            "  }\n"
            "}\n";
  return source;
}

} // namespace rund::node::accel::detail::device_vsm_reduce_source
