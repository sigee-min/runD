#pragma once

#include "../backend/number.hpp"
#include "staged/proof/plan.hpp"

#include <cstddef>
#include <cstring>

namespace rund::node::accel::detail {

[[nodiscard]] inline rund::kernel::u64
ScatterIdentityOutput(const rund::kernel::BindingSet &bindings,
                      const rund::kernel::ComputeDispatchWindow &window,
                      const void *const source_bytes,
                      const bool bulk_window) noexcept {
  if (source_bytes == nullptr || bindings.staged_output == nullptr ||
      !bulk_window ||
      !rund::kernel::checked::mul(window.tile_count,
                                  bindings.output_bytes_per_tile) ||
      !rund::kernel::checked::mul(window.begin_sequence,
                                  bindings.staged_output_stride)) {
    return 0u;
  }
  const rund::kernel::u64 byte_count =
      window.tile_count * bindings.output_bytes_per_tile;
  const rund::kernel::u64 output_offset =
      window.begin_sequence * bindings.staged_output_stride;
  if (byte_count == 0u ||
      !rund::kernel::checked::add(output_offset, byte_count)) {
    return 0u;
  }
  auto *const output = static_cast<std::byte *>(bindings.staged_output);
  const auto *const source = static_cast<const std::byte *>(source_bytes);
  std::memcpy(output + static_cast<std::size_t>(output_offset), source,
              static_cast<std::size_t>(byte_count));
  return byte_count;
}

[[nodiscard]] inline bool
ScatterOutputBytes(const rund::kernel::BindingSet &bindings,
                   const rund::kernel::ComputeDispatchWindow &window,
                   const void *const source_bytes,
                   const std::size_t source_size, const StagedProof &staged,
                   rund::kernel::u64 &copied) {
  copied = 0u;
  if (!staged.matches(bindings, window) || bindings.staged_output == nullptr ||
      (source_bytes == nullptr && source_size != 0u) ||
      bindings.staged_output_stride < bindings.output_bytes_per_tile ||
      !rund::kernel::checked::mul(window.tile_count,
                                  bindings.output_bytes_per_tile)) {
    return false;
  }
  const rund::kernel::u64 byte_count =
      window.tile_count * bindings.output_bytes_per_tile;
  if (byte_count > static_cast<rund::kernel::u64>(source_size)) {
    return false;
  }
  const auto *const source = static_cast<const std::byte *>(source_bytes);
  auto *const output = static_cast<std::byte *>(bindings.staged_output);
  if (staged.bulk()) {
    if (const rund::kernel::u64 identity =
            ScatterIdentityOutput(bindings, window, source, true)) {
      copied = identity;
      return true;
    }
  }
  for (rund::kernel::u64 offset = 0u; offset < window.tile_count; ++offset) {
    rund::kernel::u64 tile = 0u;
    if (!bindings.sequence_tile_at(window.begin_sequence + offset, tile) ||
        tile >= bindings.staged_output_count ||
        !rund::kernel::checked::mul(tile, bindings.staged_output_stride) ||
        !rund::kernel::checked::mul(offset, bindings.output_bytes_per_tile) ||
        !rund::kernel::checked::add(copied, bindings.output_bytes_per_tile)) {
      return false;
    }
    const rund::kernel::u64 output_offset =
        tile * bindings.staged_output_stride;
    const rund::kernel::u64 source_offset =
        offset * bindings.output_bytes_per_tile;
    std::size_t output_size = 0u;
    std::size_t source_offset_size = 0u;
    std::size_t bytes = 0u;
    if (!rund::kernel::checked::add(output_offset,
                                    bindings.output_bytes_per_tile) ||
        !rund::kernel::checked::add(source_offset,
                                    bindings.output_bytes_per_tile) ||
        !ToSize(output_offset, output_size) ||
        !ToSize(source_offset, source_offset_size) ||
        !ToSize(bindings.output_bytes_per_tile, bytes)) {
      return false;
    }
    std::memcpy(output + output_size, source + source_offset_size, bytes);
    copied += bindings.output_bytes_per_tile;
  }
  return copied == byte_count;
}

} // namespace rund::node::accel::detail
