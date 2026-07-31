#pragma once

#include <kernel/program/compute/stencil/reference/window.hpp>

namespace rund::kernel {

[[nodiscard]] inline StencilResult ReferenceStencilSumU32(
    const u32* const input,
    u32* const output,
    const u64 element_count,
    const u64 radius) noexcept {
  return stencil_reference::ReferenceWindow(
      input, output, element_count, radius, StencilOp::Sum);
}

[[nodiscard]] inline StencilResult ReferenceStencilMinU32(
    const u32* const input,
    u32* const output,
    const u64 element_count,
    const u64 radius) noexcept {
  return stencil_reference::ReferenceWindow(
      input, output, element_count, radius, StencilOp::Min);
}

[[nodiscard]] inline StencilResult ReferenceStencilMinU64(
    const u64* const input,
    u64* const output,
    const u64 element_count,
    const u64 radius) noexcept {
  return stencil_reference::ReferenceWindow(
      input, output, element_count, radius, StencilOp::Min);
}

[[nodiscard]] inline StencilResult ReferenceStencilMaxU32(
    const u32* const input,
    u32* const output,
    const u64 element_count,
    const u64 radius) noexcept {
  return stencil_reference::ReferenceWindow(
      input, output, element_count, radius, StencilOp::Max);
}

[[nodiscard]] inline StencilResult ReferenceStencilMaxU64(
    const u64* const input,
    u64* const output,
    const u64 element_count,
    const u64 radius) noexcept {
  return stencil_reference::ReferenceWindow(
      input, output, element_count, radius, StencilOp::Max);
}

[[nodiscard]] inline StencilResult ReferenceStencilSumU64(
    const u64* const input,
    u64* const output,
    const u64 element_count,
    const u64 radius) noexcept {
  return stencil_reference::ReferenceWindow(
      input, output, element_count, radius, StencilOp::Sum);
}
}  // namespace rund::kernel
