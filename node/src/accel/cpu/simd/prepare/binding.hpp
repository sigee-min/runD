#pragma once

#include <vector>

namespace rund::node::accel::cpu_simd_detail {
namespace {

inline constexpr u64 kInvalidSlot = ~u64{0u};

struct BindingPlan final {
  std::vector<u64> slots;
  u64 param_bytes = 0u;
  u64 read_count = 0u;
  u64 write_count = 0u;
};

[[nodiscard]] BindingPlan BuildBindingPlan(const ParsedIR &parsed) {
  BindingPlan plan;
  plan.slots.assign(parsed.bindings.size(), kInvalidSlot);
  for (std::size_t index = 0u; index < parsed.bindings.size(); ++index) {
    const ParsedBinding &binding = parsed.bindings[index];
    if (binding.kind == kParamBindingKind) {
      plan.slots[index] = plan.param_bytes;
      plan.param_bytes += binding.element_bytes;
    } else if (binding.kind == kReadBindingKind) {
      plan.slots[index] = plan.read_count;
      ++plan.read_count;
    } else if (binding.kind == kWriteBindingKind) {
      plan.slots[index] = plan.write_count;
      ++plan.write_count;
    }
  }
  return plan;
}

} // namespace
} // namespace rund::node::accel::cpu_simd_detail
