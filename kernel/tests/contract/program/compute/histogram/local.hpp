#pragma once

#include "test/assert.hpp"

#include <kernel/program/compute/histogram/model.hpp>
#include <kernel/program/compute/histogram/identity.hpp>
#include <kernel/program/compute/histogram/plan.hpp>
#include <kernel/program/compute/histogram/reference.hpp>

#include <array>
#include <limits>
#include <string_view>

namespace program_compute_contract {

[[nodiscard]] constexpr rund::kernel::HistogramDesc
U32Histogram() noexcept {
  return rund::kernel::HistogramDesc{
      .index = rund::kernel::HistogramIndex::U32,
      .count = rund::kernel::HistogramCount::U32,
      .element_count = 8u,
      .bin_count = 4u,
  };
}

int HistogramReject();
int HistogramIdentity();
int HistogramShape();
int HistogramReference();

}  // namespace program_compute_contract
