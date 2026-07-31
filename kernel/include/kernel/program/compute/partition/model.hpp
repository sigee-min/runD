#pragma once

#include <kernel/core/model.hpp>

namespace rund::kernel {

struct PartitionDesc {
  u64 element_count = 0u;
  u32 flag_bytes = 4u;
  u32 value_bytes = 4u;
};

struct PartitionPlan {
  u64 element_count = 0u;
  u64 flag_bytes = 0u;
  u64 value_bytes = 0u;
  u64 scan_temp_bytes = 0u;
  u64 temp_bytes = 0u;
  u64 pass_count = 0u;
  bool ok = false;
  const char *reason = "compute_partition_invalid";

  [[nodiscard]] constexpr explicit operator bool() const noexcept { return ok; }
};

struct PartitionHash {
  u64 hi = 0u;
  u64 lo = 0u;
};

struct PartitionResult {
  u64 element_count = 0u;
  u64 false_count = 0u;
  u64 true_count = 0u;
  bool ok = false;
  const char *reason = "compute_partition_invalid";

  [[nodiscard]] constexpr explicit operator bool() const noexcept { return ok; }
};

} // namespace rund::kernel
