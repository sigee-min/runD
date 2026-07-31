#pragma once

#include "../../backend/number.hpp"
#include "identity.hpp"

#include <cstdint>

namespace rund::node::accel::detail {

enum class InputAddressMode : std::uint8_t {
  Sequence,
  Identity,
};

[[nodiscard]] inline bool
PackInputBufferRange(const rund::kernel::BufferSpan &span,
                     const rund::kernel::BindingSet &bindings,
                     const rund::kernel::ComputeDispatchWindow &window,
                     std::byte *const packed, const std::size_t packed_size,
                     const bool bulk_window,
                     const InputAddressMode address_mode) {
  if (!rund::kernel::checked::add(window.begin_sequence, window.tile_count) ||
      window.begin_sequence + window.tile_count > span.count ||
      !rund::kernel::checked::mul(window.tile_count, span.element_bytes)) {
    return false;
  }
  const rund::kernel::u64 byte_count = window.tile_count * span.element_bytes;
  std::size_t expected_size = 0u;
  if (!ToSize(byte_count, expected_size) || expected_size != packed_size) {
    return false;
  }
  const auto *const source = static_cast<const std::byte *>(span.data);
  if ((packed == nullptr || source == nullptr) && packed_size != 0u) {
    return false;
  }
  if (PackIdentityInput(
          span, bindings, window, packed, packed_size,
          bulk_window || address_mode == InputAddressMode::Identity)) {
    return true;
  }
  for (rund::kernel::u64 offset = 0u; offset < window.tile_count; ++offset) {
    rund::kernel::u64 tile = 0u;
    if (!rund::kernel::checked::add(window.begin_sequence, offset)) {
      return false;
    }
    const rund::kernel::u64 sequence = window.begin_sequence + offset;
    if (address_mode == InputAddressMode::Identity) {
      tile = sequence;
    } else if (!bindings.sequence_tile_at(sequence, tile)) {
      return false;
    }
    if (tile >= span.count ||
        !rund::kernel::checked::mul(tile, span.stride_bytes) ||
        !rund::kernel::checked::mul(offset, span.element_bytes)) {
      return false;
    }
    const rund::kernel::u64 source_offset = tile * span.stride_bytes;
    std::memcpy(packed + static_cast<std::size_t>(offset * span.element_bytes),
                source + static_cast<std::size_t>(source_offset),
                static_cast<std::size_t>(span.element_bytes));
  }
  return true;
}

} // namespace rund::node::accel::detail
