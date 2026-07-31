#pragma once

#include "../../backend/number.hpp"
#include "window.hpp"

namespace rund::node::accel::detail {

[[nodiscard]] inline bool StagedInputRange(
    const rund::kernel::BufferSpan &span,
    const rund::kernel::ComputeDispatchWindow &window,
    const rund::kernel::u64 cursor, const rund::kernel::u64 alignment,
    rund::kernel::u64 &input_offset, rund::kernel::u64 &input_range,
    rund::kernel::u64 &next_cursor) noexcept {
  const rund::kernel::u64 align = alignment == 0u ? 1u : alignment;
  const rund::kernel::u64 remainder = cursor % align;
  input_offset = cursor;
  if (remainder != 0u) {
    const rund::kernel::u64 padding = align - remainder;
    if (!rund::kernel::checked::add(cursor, padding)) {
      return false;
    }
    input_offset = cursor + padding;
  }
  if (!rund::kernel::checked::mul(window.tile_count, span.element_bytes)) {
    return false;
  }
  input_range = window.tile_count * span.element_bytes;
  if (!rund::kernel::checked::add(input_offset, input_range)) {
    return false;
  }
  next_cursor = input_offset + input_range;
  return true;
}

[[nodiscard]] inline bool
StagedInputByteCount(const rund::kernel::BindingSet &bindings,
                     const rund::kernel::ComputeDispatchWindow &window,
                     const std::span<const InputWindowPlan> input_plans,
                     const rund::kernel::u64 alignment,
                     rund::kernel::u64 &out) noexcept {
  if ((bindings.input_buffer_count != 0u &&
       bindings.input_buffers == nullptr) ||
      input_plans.size() != bindings.input_buffer_count) {
    return false;
  }
  rund::kernel::u64 cursor = 0u;
  for (rund::kernel::u64 index = 0u; index < bindings.input_buffer_count;
       ++index) {
    rund::kernel::u64 input_offset = 0u;
    rund::kernel::u64 input_range = 0u;
    rund::kernel::u64 next_cursor = 0u;
    const rund::kernel::ComputeDispatchWindow input_window =
        InputWindow(input_plans[static_cast<std::size_t>(index)], window);
    if (!StagedInputRange(bindings.input_buffers[index], input_window, cursor,
                          alignment, input_offset, input_range, next_cursor)) {
      return false;
    }
    cursor = next_cursor;
  }
  out = cursor;
  return true;
}

} // namespace rund::node::accel::detail
