#pragma once

#include <kernel/program/compute/dsl.hpp>

#include <string_view>

namespace node_accel_contract::policy_case {

inline constexpr rund::kernel::u64 kMaxU64 = ~rund::kernel::u64{0u};

[[nodiscard]] inline bool ReasonIs(const std::string_view lhs,
                                   const char* rhs) noexcept {
  return lhs == std::string_view{rhs};
}

}  // namespace node_accel_contract::policy_case
