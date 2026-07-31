#pragma once

#include <kernel/core/model.hpp>

namespace rund::kernel {

enum class ScatterElement : u8 {
  U32 = 1u,
  U64 = 2u,
};

struct ScatterDesc {
  ScatterElement element = ScatterElement::U32;
  u64 element_count = 0u;
  u64 output_count = 0u;
};

struct ScatterPlan {
  ScatterElement element = ScatterElement::U32;
  u64 element_count = 0u;
  u64 output_count = 0u;
  u64 element_bytes = 0u;
  u64 index_bytes = 0u;
  u64 status_bytes = 0u;
  u64 temp_bytes = 0u;
  u64 scratch_slots = 0u;
  u64 pass_count = 0u;
  bool ok = false;
  const char *reason = "compute_scatter_invalid";

  [[nodiscard]] constexpr explicit operator bool() const noexcept { return ok; }
};

struct ScatterHash {
  u64 hi = 0u;
  u64 lo = 0u;
};

struct ScatterResult {
  u64 element_count = 0u;
  u64 output_count = 0u;
  u64 first_rejected_index = 0u;
  bool ok = false;
  const char *reason = "compute_scatter_invalid";

  [[nodiscard]] constexpr explicit operator bool() const noexcept { return ok; }
};

} // namespace rund::kernel
