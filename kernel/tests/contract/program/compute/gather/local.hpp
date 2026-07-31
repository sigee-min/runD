#pragma once

#include "test/assert.hpp"

#include <kernel/program/compute/gather/model.hpp>
#include <kernel/program/compute/gather/identity.hpp>
#include <kernel/program/compute/gather/plan.hpp>
#include <kernel/program/compute/gather/reference.hpp>

#include <array>
#include <limits>
#include <string_view>

namespace program_compute_contract {

[[nodiscard]] constexpr rund::kernel::GatherDesc U32Gather() noexcept {
  return rund::kernel::GatherDesc{
      .element = rund::kernel::GatherElement::U32,
      .element_count = 8u,
      .source_count = 16u,
  };
}

int GatherReject();
int GatherIdentity();
int GatherShape();
int GatherReference();

} // namespace program_compute_contract
