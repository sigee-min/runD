#pragma once

#include <math32/core/model.hpp>

namespace rund::math32::soa {

enum class StatusReason : u8 {
  Ok = 0u,
  SizeMismatch = 1u,
  ComponentOverlap = 2u,
  InputOutputOverlap = 3u,
  NotEvaluated = 4u,
};

struct Status {
  StatusReason reason = StatusReason::NotEvaluated;
  u32 processed = 0u;
  [[nodiscard]] constexpr bool ok() const noexcept {
    return reason == StatusReason::Ok;
  }
};

}  // namespace rund::math32::soa
