#pragma once

#include "test/assert.hpp"

#include <kernel/program/compute/segmented/scan/identity.hpp>
#include <kernel/program/compute/segmented/scan/plan.hpp>
#include <kernel/program/compute/segmented/scan/reference.hpp>
#include <kernel/program/compute/segmented/scan/model.hpp>

#include <array>
#include <limits>
#include <string_view>

namespace program_compute_contract {

[[nodiscard]] constexpr rund::kernel::SegmentedScanDesc
U32SegmentedScan() noexcept {
  return rund::kernel::SegmentedScanDesc{
      .op = rund::kernel::SegmentedScanOp::ExclusiveSum,
      .element = rund::kernel::SegmentedScanElement::U32,
      .element_count = 8u,
      .block_size = 4u,
  };
}

int SegmentedScanReject();
int SegmentedScanIdentity();
int SegmentedScanShape();
int SegmentedScanReference();

} // namespace program_compute_contract
