#pragma once

#include <kernel/core/model.hpp>

namespace rund::kernel {

struct CompactDesc {
  u64 element_count = 0u;
  u64 output_capacity = 0u;
  u32 flag_bytes = 4u;
  u32 output_bytes = 4u;
};

struct CompactPlan {
  u64 element_count = 0u;
  u64 output_capacity = 0u;
  u64 flag_bytes = 0u;
  u64 output_bytes = 0u;
  u64 scan_temp_bytes = 0u;
  u64 status_bytes = 0u;
  u64 temp_bytes = 0u;
  u64 pass_count = 0u;
  bool ok = false;
  const char* reason = "compute_compact_invalid";

  [[nodiscard]] constexpr explicit operator bool() const noexcept {
    return ok;
  }
};

struct CompactHash {
  u64 hi = 0u;
  u64 lo = 0u;
};

struct CompactResult {
  u64 element_count = 0u;
  u64 output_count = 0u;
  bool ok = false;
  const char* reason = "compute_compact_invalid";

  [[nodiscard]] constexpr explicit operator bool() const noexcept {
    return ok;
  }
};

}  // namespace rund::kernel
