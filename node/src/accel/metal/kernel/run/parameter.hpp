#pragma once

#include "../../../kernel/backend/template_plan.hpp"

#include <cstdint>

namespace rund::node::accel::detail {

[[nodiscard]] inline bool
AddAlignedMetalParameterBytes(std::uint64_t &target,
                              const std::uint64_t bytes) noexcept {
  constexpr std::uint64_t Alignment = 16u;
  std::uint64_t padded = bytes;
  return bytes != 0u && backend_template_plan::add(padded, Alignment - 1u) &&
         (padded &= ~(Alignment - 1u), true) &&
         backend_template_plan::add(target, padded);
}

} // namespace rund::node::accel::detail
