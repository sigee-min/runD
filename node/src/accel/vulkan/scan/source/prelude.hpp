#pragma once

#include "../../../domain.hpp"
#include "../source.hpp"

namespace rund::node::accel::detail {

[[nodiscard]] inline const char *
VulkanScanScalar(const rund::kernel::ScanElement element) noexcept {
  return element == rund::kernel::ScanElement::U64 ? "uint64_t" : "uint";
}

[[nodiscard]] inline const char *
VulkanScanZero(const rund::kernel::ScanElement element) noexcept {
  return element == rund::kernel::ScanElement::U64 ? "uint64_t(0)" : "0u";
}

template <typename Source>
inline void AppendVulkanScanPrelude(Source &source,
                                    const rund::kernel::ScanElement element,
                                    const rund::kernel::ComputeDomain domain,
                                    const std::uint32_t width,
                                    const bool chunked) {
  const char *const type = VulkanScanScalar(element);
  source += "#version 450\n";
  source += "#extension GL_EXT_shader_explicit_arithmetic_types_int64 : ";
  source += "require\n";
  source += "layout(local_size_x = ";
  source.decimal(width);
  source += ") in;\n";
  source += "const uint kScanWidth = ";
  source.decimal(width);
  source += "u;\n";
  source += "layout(set = 0, binding = 0, std430) readonly buffer Params {\n";
  source += "  uint64_t element_count;\n";
  source += "  uint64_t block_size;\n";
  source += "  uint64_t block_count;\n";
  source += "  uint count_words;\n";
  source += "  uint inclusive;\n";
  source += "} params;\n";
  if (chunked) {
    source += "layout(push_constant) uniform ScanDispatch {\n";
    source += "  uint64_t base_block;\n";
    source += "} scan_dispatch;\n";
  }
  source += "layout(set = 0, binding = 1, std430) readonly buffer Input {\n";
  source += "  ";
  source += type;
  source += " input_values[];\n};\n";
  source += "layout(set = 0, binding = 2, std430) buffer Output {\n";
  source += "  ";
  source += type;
  source += " output_values[];\n};\n";
  source += "layout(set = 0, binding = 3, std430) buffer Totals {\n";
  source += "  ";
  source += type;
  source += " totals[];\n};\n";
  source += "layout(set = 0, binding = 4, std430) buffer Status {\n";
  source += "  uint status[];\n};\n";
  source += "layout(set = 0, binding = 5, std430) readonly buffer "
            "LogicalCount { uint logical_count[]; };\n";
  source += "uint64_t ResidentCount() {\n";
  source +=
      "  if (params.count_words == 0u) { return params.element_count; }\n";
  source += "  uint64_t count = uint64_t(logical_count[0]);\n";
  source += "  if (params.count_words == 2u) { count |= "
            "uint64_t(logical_count[1]) << 32u; }\n";
  source += "  return count;\n";
  source += "}\n";
  source += "uint64_t ActiveCount() { return min(ResidentCount(), "
            "params.element_count); }\n";
  source += "uint CountInvalid() { return ResidentCount() > "
            "params.element_count ? 2u : 0u; }\n";
  source += "uint ScanOverflow(";
  source += type;
  source += " lhs, ";
  source += type;
  source += " rhs, ";
  source += type;
  source += " sum) {\n";
  if (IsSignedDomain(domain)) {
    source += element == rund::kernel::ScanElement::U64
                  ? "  const uint64_t sign = uint64_t(0x80000000u) << 32u;\n"
                    "  return (((lhs ^ rhs) & sign) == uint64_t(0) && "
                    "((lhs ^ sum) & sign) != uint64_t(0)) ? 1u : "
                    "0u;\n"
                  : "  return (((lhs ^ rhs) & 0x80000000u) == 0u && "
                    "((lhs ^ sum) & 0x80000000u) != 0u) ? 1u : 0u;\n";
  } else {
    source += "  return sum < lhs ? 1u : 0u;\n";
  }
  source += "}\n";
}

} // namespace rund::node::accel::detail
