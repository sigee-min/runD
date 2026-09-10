#include "local.hpp"

namespace node_accel_contract::reset_contract {

bool Accepted(const Range range) noexcept { return range.valid(); }

bool Rejected(const Range range) noexcept {
  return !range.valid() && range.offset() == 0u && range.count() == 0u &&
         range.stride() == 0u && range.element() == 0u && range.end() == 0u;
}

} // namespace node_accel_contract::reset_contract
