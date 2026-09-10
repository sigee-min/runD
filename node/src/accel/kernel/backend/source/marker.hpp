#pragma once

#include <kernel/program/compute/lowering/text.hpp>

#include <cstddef>
#include <string_view>

namespace rund::node::accel::detail::backend_source_marker {

[[nodiscard]] inline bool
consume_fragment(const std::string_view source, std::size_t &cursor,
                 const std::string_view fragment) noexcept {
  if (cursor > source.size() || fragment.size() > source.size() - cursor ||
      source.compare(cursor, fragment.size(), fragment) != 0) {
    return false;
  }
  cursor += fragment.size();
  return true;
}

[[nodiscard]] inline bool
consume_safe_identifier(const std::string_view source, std::size_t &cursor,
                        const std::string_view name) noexcept {
  if (name.empty()) {
    return consume_fragment(source, cursor, "empty");
  }
  for (const char value : name) {
    if (cursor > source.size() || source.size() - cursor < 2u) {
      return false;
    }
    const auto byte =
        static_cast<rund::kernel::u8>(static_cast<unsigned char>(value));
    if (source[cursor] !=
            rund::kernel::compute_lowering_detail::HexDigit(
                static_cast<rund::kernel::u8>((byte >> 4u) & 0x0fu)) ||
        source[cursor + 1u] !=
            rund::kernel::compute_lowering_detail::HexDigit(
                static_cast<rund::kernel::u8>(byte & 0x0fu))) {
      return false;
    }
    cursor += 2u;
  }
  return true;
}

} // namespace rund::node::accel::detail::backend_source_marker
