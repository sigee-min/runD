#include "../internal.hpp"

#include <kernel/program/compute/lowering/format.hpp>

namespace rund::node::accel::detail::device_vsm_reduce_source {

std::string vulkan_additive_source(const rund::kernel::ArtifactKey &key,
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
  const char *const scalar = u32 ? "uint" : "uint64_t";
  std::string source =
      "#version 450\n#extension "
      "GL_EXT_shader_explicit_arithmetic_types_int64 : require\n";
  source += sum ? "// rund.compute.device_vsm.reduce.sum\n"
                : "// rund.compute.device_vsm.reduce.count_nonzero\n";
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
      "shared uint64_t partial_low[256];\n"
      "shared uint64_t partial_high[256];\n"
      "void main() {\n"
      "  const uint local = gl_LocalInvocationID.x;\n"
      "  uint64_t low = 0ul; uint64_t high = 0ul;\n"
      "  for (uint index = local; index < config.logical_elements; index += "
      "256u) {\n"
      "    const uint64_t value = ";
  source += sum ? "uint64_t(input_values[index]);\n"
                : "uint64_t(input_values[index] != 0);\n";
  source +=
      "    const uint64_t next = low + value;\n"
      "    high += uint64_t(next < low); low = next;\n"
      "  }\n"
      "  partial_low[local] = low; partial_high[local] = high; barrier();\n"
      "  for (uint stride = 128u; stride != 0u; stride >>= 1u) {\n"
      "    if (local < stride) {\n"
      "      const uint64_t left = partial_low[local];\n"
      "      const uint64_t combined = left + partial_low[local + stride];\n"
      "      partial_high[local] += partial_high[local + stride] + "
      "uint64_t(combined < left);\n"
      "      partial_low[local] = combined;\n"
      "    }\n"
      "    barrier();\n"
      "  }\n"
      "  if (local == 0u) {\n";
  if (sum) {
    source += "    const uint overflow = ";
    source +=
        u32 ? "uint(partial_high[0] != 0ul || partial_low[0] > 0xfffffffful);\n"
            : "uint(partial_high[0] != 0ul);\n";
    source += "    if (overflow != 0u) {\n"
              "      uint failed_page = 0u;\n"
              "      uint64_t prefix_low = 0ul; uint64_t prefix_high = 0ul;\n"
              "      for (uint index = 0u; index < config.logical_elements; "
              "++index) {\n"
              "        const uint64_t value = uint64_t(input_values[index]);\n"
              "        const uint64_t next = prefix_low + value;\n"
              "        prefix_high += uint64_t(next < prefix_low); prefix_low "
              "= next;\n"
              "        if (";
    source += u32 ? "prefix_high != 0ul || prefix_low > 0xfffffffful"
                  : "prefix_high != 0ul";
    source += ") { failed_page = index / config.payload_elements; break; }\n"
              "      }\n"
              "      atomicExchange(result[0], config.page_count);\n"
              "      atomicExchange(result[1], config.page_count);\n"
              "      atomicExchange(result[2], config.page_count);\n"
              "      atomicExchange(result[3], failed_page);\n"
              "      atomicExchange(result[6], 1u);\n"
              "      atomicExchange(result[7], failed_page);\n"
              "      return;\n"
              "    }\n";
  }
  source += "    output_values[0] = ";
  source += u32 ? "uint(partial_low[0]);\n" : "partial_low[0];\n";
  source += "    for (uint word = 0u; word < 4u; ++word) { "
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
