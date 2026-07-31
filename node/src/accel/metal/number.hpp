#pragma once

#include "../backend/number.hpp"

#include <limits>

#if defined(__APPLE__) && defined(RUND_NODE_HAVE_METAL_SDK)
#import <Foundation/Foundation.h>
#endif

namespace rund::node::accel::detail {

#if defined(__APPLE__) && defined(RUND_NODE_HAVE_METAL_SDK)
[[nodiscard]] inline bool ToNSUInteger(const rund::kernel::u64 value,
                                       NSUInteger &out) noexcept {
  if (value > static_cast<rund::kernel::u64>(
                  std::numeric_limits<NSUInteger>::max())) {
    return false;
  }
  out = static_cast<NSUInteger>(value);
  return true;
}
#endif

} // namespace rund::node::accel::detail
