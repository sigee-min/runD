#pragma once

#include "test/assert.hpp"

#include <kernel/program/compute/graph/signature.hpp>
#include <kernel/program/compute/window/identity.hpp>
#include <kernel/program/compute/window/model.hpp>
#include <kernel/program/compute/window/plan.hpp>
#include <kernel/program/compute/window/reference.hpp>

#include <array>
#include <bit>
#include <limits>
#include <string_view>

namespace program_compute_contract {

[[nodiscard]] constexpr rund::kernel::WindowDesc U32Window() noexcept {
  return rund::kernel::WindowDesc{
      .op = rund::kernel::WindowOp::Sum,
      .element = rund::kernel::WindowElement::U32,
      .boundary = rund::kernel::WindowBoundary::Clamp,
      .domain = rund::kernel::ComputeDomain::U32,
      .input_count = 6u,
      .output_count = 4u,
      .window_size = 3u,
      .stride = 2u,
      .pad_left = 1u,
  };
}

[[nodiscard]] constexpr rund::kernel::ComputeFixedFormat
Fixed32(const rund::kernel::ComputeOverflow overflow,
        const rund::kernel::ComputeApproximation approximation =
            rund::kernel::ComputeApproximation::Exact) noexcept {
  return rund::kernel::ComputeFixedFormat{
      .integer_bits = 16u,
      .fraction_bits = 16u,
      .rounding = rund::kernel::ComputeRounding::NearestEven,
      .overflow = overflow,
      .approximation = approximation,
  };
}

int WindowReject();
int WindowIdentity();
int WindowShape();
int WindowReference();

} // namespace program_compute_contract
