#pragma once

#include <math64/soa/status.hpp>

#include <limits>

namespace rund::math64::soa::detail {

inline u32 SatProcessed(const u64 value) noexcept {
  return value > std::numeric_limits<u32>::max() ? std::numeric_limits<u32>::max() : static_cast<u32>(value);
}

}  // namespace rund::math64::soa::detail
