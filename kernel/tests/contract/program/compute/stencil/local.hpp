#pragma once

#include "test/assert.hpp"

#include <kernel/program/compute/stencil/model.hpp>
#include <kernel/program/compute/stencil/identity.hpp>
#include <kernel/program/compute/stencil/plan.hpp>
#include <kernel/program/compute/stencil/reference.hpp>

#include <array>
#include <limits>
#include <string_view>

namespace program_compute_contract {

[[nodiscard]] constexpr rund::kernel::StencilDesc U32Stencil() noexcept {
  return rund::kernel::StencilDesc{
      .op = rund::kernel::StencilOp::Sum,
      .element = rund::kernel::StencilElement::U32,
      .boundary = rund::kernel::StencilBoundary::Clamp,
      .element_count = 8u,
      .radius = 1u,
  };
}

int StencilReject();
int StencilIdentity();
int StencilShape();
int StencilReference();

}  // namespace program_compute_contract
