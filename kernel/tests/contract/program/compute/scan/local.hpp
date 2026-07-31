#pragma once

#include "test/assert.hpp"

#include <kernel/program/compute/scan/model.hpp>
#include <kernel/program/compute/scan/identity.hpp>
#include <kernel/program/compute/scan/plan.hpp>
#include <kernel/program/compute/scan/reference.hpp>

#include <array>
#include <limits>
#include <string_view>

namespace program_compute_contract {

[[nodiscard]] constexpr rund::kernel::ScanDesc U32Scan() noexcept {
  return rund::kernel::ScanDesc{
      .op = rund::kernel::ScanOp::ExclusiveSum,
      .element = rund::kernel::ScanElement::U32,
      .element_count = 8u,
      .block_size = 4u,
  };
}

int ScanReject();
int ScanIdentity();
int ScanShape();
int ScanReference();

} // namespace program_compute_contract
