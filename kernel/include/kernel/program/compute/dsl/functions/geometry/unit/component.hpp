#pragma once

namespace rund::compute_dsl::detail {

[[nodiscard]] inline ComputeValue
UnitComponent(const ComputeValue value,
              const ComputeValue length) noexcept {
  return select(eq(length, 0), fixed_zero(value), div_fixed(value, length));
}

} // namespace rund::compute_dsl::detail
