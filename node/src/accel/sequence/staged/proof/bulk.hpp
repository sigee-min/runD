#pragma once

#include "output.hpp"

namespace rund::node::accel::detail {

[[nodiscard]] inline bool StagedInputsContiguous(
    const rund::kernel::BindingSet& bindings) noexcept {
  if (bindings.input_buffer_count != 0u && bindings.input_buffers == nullptr) {
    return false;
  }
  for (rund::kernel::u64 index = 0u; index < bindings.input_buffer_count;
       ++index) {
    const rund::kernel::BufferSpan& span = bindings.input_buffers[index];
    if (span.stride_bytes != span.element_bytes) { return false; }
  }
  return true;
}

[[nodiscard]] inline bool StagedBulkAllowed(
    const rund::kernel::BindingSet& bindings,
    const rund::kernel::ComputeDispatchWindow& window,
    const bool identity) noexcept {
  const bool contiguous_output =
      bindings.staged_output_stride == bindings.output_bytes_per_tile &&
      StagedInputsContiguous(bindings);
  if (!identity || !contiguous_output) { return false; }
  if (window.tile_count == 0u) { return true; }
  const rund::kernel::u64 last_tile =
      window.begin_sequence + window.tile_count - 1u;
  return last_tile < bindings.staged_output_count;
}

}  // namespace rund::node::accel::detail
