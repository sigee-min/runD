#pragma once

#include "../../../domain.hpp"

namespace rund::node::accel::detail {

[[nodiscard]] inline const char *
VulkanSegmentedType(const rund::kernel::SegmentedScanElement element) noexcept {
  return element == rund::kernel::SegmentedScanElement::U64 ? "uint64_t"
                                                            : "uint";
}

template <typename Source>
inline void AppendSegmentedPrelude(
    Source &source, const rund::kernel::SegmentedScanElement element,
    const rund::kernel::ComputeDomain domain, const bool chunked) {
  const char *const type = VulkanSegmentedType(element);
  source += "#version 450\n";
  source +=
      "#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require\n";
  source += "layout(local_size_x = 256) in;\n";
  source += "const uint kSegmentedWidth = 256u;\n";
  if (chunked) {
    source += "layout(push_constant) uniform SegmentedDispatch {\n";
    source += "  uint64_t base_block;\n";
    source += "} segmented_dispatch;\n";
  }
  source += "layout(set = 0, binding = 0, std430) readonly buffer Params {\n";
  source += "  uint64_t element_count; uint64_t block_size;\n";
  source += "  uint64_t block_count; uint inclusive; uint reserved;\n";
  source += "} params;\n";
  source += "layout(set = 0, binding = 1, std430) readonly buffer Input { ";
  source += type;
  source += " input_values[]; };\n";
  source += "layout(set = 0, binding = 2, std430) readonly buffer Heads { uint "
            "heads[]; };\n";
  source += "layout(set = 0, binding = 3, std430) buffer Output { ";
  source += type;
  source += " output_values[]; };\n";
  source += "layout(set = 0, binding = 4, std430) buffer Offsets { ";
  source += type;
  source += " offsets[]; };\n";
  source += "layout(set = 0, binding = 5, std430) buffer FirstHeads { uint "
            "first_heads[]; };\n";
  source += "layout(set = 0, binding = 6, std430) buffer Status { uint "
            "status[]; };\n";
  source += "uint SegmentedOverflow(";
  source += type;
  source += " lhs, ";
  source += type;
  source += " rhs, ";
  source += type;
  source += " sum) {\n";
  if (IsSignedDomain(domain)) {
    source += element == rund::kernel::SegmentedScanElement::U64
                  ? "  const uint64_t sign = uint64_t(0x80000000u) << 32u;\n"
                    "  return (((lhs ^ rhs) & sign) == uint64_t(0) && "
                    "((lhs ^ sum) & sign) != uint64_t(0)) ? 1u : 0u;\n"
                  : "  return (((lhs ^ rhs) & 0x80000000u) == 0u && "
                    "((lhs ^ sum) & 0x80000000u) != 0u) ? 1u : 0u;\n";
  } else {
    source += "  return sum < lhs ? 1u : 0u;\n";
  }
  source += "}\n";
  source += "shared ";
  source += type;
  source += " segment_values[2][kSegmentedWidth];\n";
  source += "shared uint segment_flags[2][kSegmentedWidth];\n";
  source += "shared ";
  source += type;
  source += " segment_carry;\n";
  source += "shared uint segment_seen;\n";
  source += "shared uint segment_first;\n";
  source += "shared uint segment_status;\n";
  source += "void main() {\n";
}

} // namespace rund::node::accel::detail
