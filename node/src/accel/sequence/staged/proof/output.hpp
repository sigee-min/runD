#pragma once

#include "../../../backend/number.hpp"
#include "range.hpp"

namespace rund::node::accel::detail {

[[nodiscard]] inline bool
StagedOutputTileOk(const rund::kernel::BindingSet &bindings,
                   const rund::kernel::u64 tile) noexcept {
  if (tile >= bindings.tile_count || tile >= bindings.staged_output_count ||
      !rund::kernel::checked::mul(tile, bindings.staged_output_stride)) {
    return false;
  }
  const rund::kernel::u64 output_offset = tile * bindings.staged_output_stride;
  return rund::kernel::checked::add(output_offset,
                                    bindings.output_bytes_per_tile);
}

[[nodiscard]] inline bool
StagedIdentityAndOutputsOk(const rund::kernel::BindingSet &bindings,
                           const rund::kernel::ComputeDispatchWindow &window,
                           bool &identity) noexcept {
  identity = true;
  for (rund::kernel::u64 offset = 0u; offset < window.tile_count; ++offset) {
    const rund::kernel::u64 sequence = window.begin_sequence + offset;
    std::size_t sequence_index = 0u;
    if (!ToSize(sequence, sequence_index)) {
      return false;
    }
    const rund::kernel::u64 tile = bindings.sequence_tiles[sequence_index];
    if (!StagedOutputTileOk(bindings, tile)) {
      return false;
    }
    if (tile != sequence) {
      identity = false;
    }
  }
  return true;
}

} // namespace rund::node::accel::detail
