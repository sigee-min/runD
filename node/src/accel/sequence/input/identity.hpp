#pragma once

#include "../../backend/number.hpp"
#include "range.hpp"

#include <cstddef>
#include <cstring>
#include <vector>

namespace rund::node::accel::detail {

[[nodiscard]] inline bool
PackIdentityInput(const rund::kernel::BufferSpan &span,
                  const rund::kernel::BindingSet &bindings,
                  const rund::kernel::ComputeDispatchWindow &window,
                  std::byte *const packed, const std::size_t packed_size,
                  const bool bulk_window) noexcept {
  (void)bindings;
  const auto *const source = static_cast<const std::byte *>(span.data);
  if (!bulk_window || span.stride_bytes != span.element_bytes ||
      (packed == nullptr && packed_size != 0u) ||
      (source == nullptr && packed_size != 0u) ||
      !rund::kernel::checked::mul(window.begin_sequence, span.stride_bytes)) {
    return false;
  }
  if (packed_size == 0u) {
    return true;
  }
  const rund::kernel::u64 source_offset =
      window.begin_sequence * span.stride_bytes;
  std::memcpy(packed, source + static_cast<std::size_t>(source_offset),
              packed_size);
  return true;
}

[[nodiscard]] inline bool
PackIdentityInput(const rund::kernel::BufferSpan &span,
                  const rund::kernel::BindingSet &bindings,
                  const rund::kernel::ComputeDispatchWindow &window,
                  std::vector<std::byte> &packed, const std::size_t packed_size,
                  const bool bulk_window) noexcept {
  return PackIdentityInput(span, bindings, window, packed.data(), packed_size,
                           bulk_window);
}

} // namespace rund::node::accel::detail
