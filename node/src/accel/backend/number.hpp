#pragma once

#include <kernel/core/model.hpp>

#include <cstddef>
#include <limits>

namespace rund::node::accel::detail {

[[nodiscard]] inline bool ToSize(const rund::kernel::u64 value,
                                 std::size_t &out) noexcept {
  if (value >
      static_cast<rund::kernel::u64>(std::numeric_limits<std::size_t>::max())) {
    return false;
  }
  out = static_cast<std::size_t>(value);
  return true;
}

} // namespace rund::node::accel::detail
