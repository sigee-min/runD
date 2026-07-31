#pragma once

#include <cstring>

namespace rund::node::accel::cpu_simd_detail {
namespace {

[[nodiscard]] const char *
ValidateParamBytes(const ParsedIR &parsed, const BindingPlan &plan,
                   const rund::kernel::BindingSet &bindings) {
  if (bindings.param_bytes != plan.param_bytes ||
      bindings.param_data_bytes != plan.param_bytes) {
    return "cpu_simd_param_size_mismatch";
  }
  if (plan.param_bytes == 0u) {
    return nullptr;
  }
  if (bindings.param_data == nullptr || !FitsSize(plan.param_bytes)) {
    return "cpu_simd_param_data_invalid";
  }

  const auto *const bytes =
      static_cast<const unsigned char *>(bindings.param_data);
  for (std::size_t index = 0u; index < parsed.bindings.size(); ++index) {
    const ParsedBinding &binding = parsed.bindings[index];
    if (binding.kind != kParamBindingKind) {
      continue;
    }
    const u64 offset = plan.slots[index];
    if (!FitsSize(offset) || !FitsSize(binding.element_bytes)) {
      return "cpu_simd_param_data_invalid";
    }
    const auto size = static_cast<std::size_t>(binding.element_bytes);
    const auto begin = static_cast<std::size_t>(offset);
    if (std::memcmp(bytes + begin, binding.value_bytes.data(), size) != 0) {
      return "cpu_simd_param_data_mismatch";
    }
  }
  return nullptr;
}

} // namespace
} // namespace rund::node::accel::cpu_simd_detail
