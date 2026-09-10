#include "local.hpp"

#include <kernel/core/checked.hpp>

#include <algorithm>
#include <cstdint>
#include <limits>

namespace rund::node::accel::detail {

// The canonical Map emitter freezes its own exact-IR source upper. Binding
// specialization can replace one base and one stride decimal literal per
// binding. Metal may additionally shrink one kernel-parameter pointee token
// from uchar to uint when the complete binding is four-byte aligned. A U64
// decimal replacement can grow a one-digit canonical literal by at most 19
// bytes; this allocation-free owner is shared by public planning and the
// runtime mutator below.
[[nodiscard]] bool MapSpecializedSourceUpperBytes(
    const std::uint64_t source_bytes, const std::uint64_t source_upper_bytes,
    const rund::kernel::ComputePlan &plan, std::uint64_t &upper) noexcept {
  constexpr std::uint64_t DecimalWidth =
      std::numeric_limits<std::uint64_t>::digits10 + 1u;
  constexpr std::uint64_t LiteralGrowth = DecimalWidth - 1u;
  std::uint64_t binding_count = 0u;
  std::uint64_t growth = 0u;
  upper = std::max(source_bytes, source_upper_bytes);
  return rund::kernel::checked::add(plan.input_buffer_count,
                                    plan.output_buffer_count, binding_count) &&
         rund::kernel::checked::mul(binding_count, 2u * LiteralGrowth,
                                    growth) &&
         rund::kernel::checked::add(upper, growth, upper);
}

[[nodiscard]] bool
MapSpecializedSourceUpperBytes(const rund::kernel::LoweringArtifact &source,
                               const rund::kernel::ComputePlan &plan,
                               std::uint64_t &upper) noexcept {
  return MapSpecializedSourceUpperBytes(
      source.source_text.size(), source.source_text_upper_bytes, plan, upper);
}

} // namespace rund::node::accel::detail
