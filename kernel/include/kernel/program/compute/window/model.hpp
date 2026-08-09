#pragma once

#include <kernel/program/compute/model.hpp>

namespace rund::kernel {

enum class WindowOp : u8 {
  Sum = 1u,
  Min = 2u,
  Max = 3u,
};

enum class WindowElement : u8 {
  U32 = 1u,
  U64 = 2u,
};

enum class WindowBoundary : u8 {
  Clamp = 1u,
  Clip = 2u,
};

// Output j aggregates K positions beginning at the signed logical position
// j * S - P. N and Q are independent authored counts; validation proves that
// every output window intersects the N-element input.
struct WindowDesc final {
  WindowOp op = WindowOp::Sum;
  WindowElement element = WindowElement::U32;
  WindowBoundary boundary = WindowBoundary::Clamp;
  ComputeDomain domain = ComputeDomain::U32;
  ComputeFixedFormat fixed_format{};
  ComputeCountSource count_source = ComputeCountSource::Descriptor;
  u64 input_count = 0u;
  u64 output_count = 0u;
  u64 window_size = 1u;
  u64 stride = 1u;
  u64 pad_left = 0u;
};

struct WindowPlan final {
  WindowOp op = WindowOp::Sum;
  WindowElement element = WindowElement::U32;
  WindowBoundary boundary = WindowBoundary::Clamp;
  ComputeDomain domain = ComputeDomain::U32;
  ComputeFixedFormat fixed_format{};
  ComputeCountSource count_source = ComputeCountSource::Descriptor;
  u64 input_count = 0u;
  u64 output_count = 0u;
  u64 window_size = 0u;
  u64 stride = 0u;
  u64 pad_left = 0u;
  u64 element_bytes = 0u;
  u64 input_bytes = 0u;
  u64 output_bytes = 0u;
  u64 temp_bytes = 0u;
  u64 pass_count = 0u;
  bool ok = false;
  const char *reason = "compute_window_invalid";

  [[nodiscard]] constexpr explicit operator bool() const noexcept { return ok; }
};

struct WindowHash final {
  u64 hi = 0u;
  u64 lo = 0u;
};

struct WindowResult final {
  u64 input_count = 0u;
  u64 output_count = 0u;
  bool ok = false;
  const char *reason = "compute_window_invalid";

  [[nodiscard]] constexpr explicit operator bool() const noexcept { return ok; }
};

} // namespace rund::kernel
