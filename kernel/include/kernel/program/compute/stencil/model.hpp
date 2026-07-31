#pragma once

#include <kernel/core/model.hpp>

namespace rund::kernel {

enum class StencilOp : u8 {
  Sum = 1u,
  Min = 2u,
  Max = 3u,
};

enum class StencilElement : u8 {
  U32 = 1u,
  U64 = 2u,
};

enum class StencilBoundary : u8 {
  Clamp = 1u,
};

struct StencilDesc {
  StencilOp op = StencilOp::Sum;
  StencilElement element = StencilElement::U32;
  StencilBoundary boundary = StencilBoundary::Clamp;
  u64 element_count = 0u;
  u64 radius = 1u;
};

struct StencilPlan {
  StencilOp op = StencilOp::Sum;
  StencilElement element = StencilElement::U32;
  StencilBoundary boundary = StencilBoundary::Clamp;
  u64 element_count = 0u;
  u64 element_bytes = 0u;
  u64 radius = 0u;
  u64 input_bytes = 0u;
  u64 output_bytes = 0u;
  u64 temp_bytes = 0u;
  u64 pass_count = 0u;
  bool ok = false;
  const char* reason = "compute_stencil_invalid";

  [[nodiscard]] constexpr explicit operator bool() const noexcept {
    return ok;
  }
};

struct StencilHash {
  u64 hi = 0u;
  u64 lo = 0u;
};

struct StencilResult {
  u64 element_count = 0u;
  bool ok = false;
  const char* reason = "compute_stencil_invalid";

  [[nodiscard]] constexpr explicit operator bool() const noexcept {
    return ok;
  }
};

}  // namespace rund::kernel
