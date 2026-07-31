#pragma once

#include "test/assert.hpp"

#include <kernel/program/compute/compact/model.hpp>
#include <kernel/program/compute/compact/identity.hpp>
#include <kernel/program/compute/compact/plan.hpp>
#include <kernel/program/compute/compact/reference.hpp>

#include <array>
#include <limits>
#include <string_view>

namespace program_compute_contract {

[[nodiscard]] constexpr rund::kernel::CompactDesc U32Compact() noexcept {
  return rund::kernel::CompactDesc{
      .element_count = 8u,
      .output_capacity = 8u,
      .flag_bytes = 4u,
      .output_bytes = 4u,
  };
}

[[nodiscard]] inline bool
SameCompactPlan(const rund::kernel::CompactPlan &lhs,
                const rund::kernel::CompactPlan &rhs) noexcept {
  return lhs.element_count == rhs.element_count &&
         lhs.output_capacity == rhs.output_capacity &&
         lhs.flag_bytes == rhs.flag_bytes &&
         lhs.output_bytes == rhs.output_bytes &&
         lhs.scan_temp_bytes == rhs.scan_temp_bytes &&
         lhs.status_bytes == rhs.status_bytes &&
         lhs.temp_bytes == rhs.temp_bytes && lhs.pass_count == rhs.pass_count &&
         lhs.ok == rhs.ok &&
         std::string_view{lhs.reason} == std::string_view{rhs.reason};
}

int CompactReject();
int CompactIdentity();
int CompactShape();
int CompactReference();

} // namespace program_compute_contract
