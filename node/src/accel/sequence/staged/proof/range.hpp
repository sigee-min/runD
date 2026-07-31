#pragma once

#include "../../../backend/number.hpp"
#include "proof.hpp"

namespace rund::node::accel::detail {

[[nodiscard]] inline bool StagedSequenceRangeOk(
    const rund::kernel::BindingSet &bindings,
    const rund::kernel::ComputeDispatchWindow &window) noexcept {
  return bindings.sequence_tiles != nullptr &&
         rund::kernel::checked::add(window.begin_sequence, window.tile_count) &&
         window.begin_sequence + window.tile_count <=
             bindings.sequence_tile_count;
}

} // namespace rund::node::accel::detail
