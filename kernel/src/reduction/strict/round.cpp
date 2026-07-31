#include "local.hpp"

namespace rund::kernel::strict_float {

u128 ShiftRightJam128(const u128 value, const u32 shift) {
  if (shift == 0u) {
    return value;
  }
  if (shift >= 127u) {
    return value == 0u ? 0u : 1u;
  }
  const u128 mask = (static_cast<u128>(1u) << shift) - 1u;
  const bool sticky = (value & mask) != 0u;
  return (value >> shift) | (sticky ? 1u : 0u);
}

} // namespace rund::kernel::strict_float
