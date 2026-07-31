#pragma once

#include "test/assert.hpp"

#include <kernel/program/compute/scatter.hpp>

#include <array>
#include <limits>
#include <string_view>

namespace program_compute_contract {

[[nodiscard]] constexpr rund::kernel::ScatterDesc U32Scatter() noexcept {
  return rund::kernel::ScatterDesc{
      .element = rund::kernel::ScatterElement::U32,
      .element_count = 4u,
      .output_count = 8u,
  };
}

int ScatterReject();
int ScatterIdentity();
int ScatterShape();
int ScatterReference();

} // namespace program_compute_contract
