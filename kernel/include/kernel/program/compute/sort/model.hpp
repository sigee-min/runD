#pragma once

#include <kernel/program/compute/model.hpp>

namespace rund::kernel {

enum class SortKey : u8 {
  U32 = 1u,
  U64 = 2u,
};

enum class SortValue : u8 {
  U32 = 1u,
  IdentityU32 = 2u,
};

struct SortDesc {
  SortKey key = SortKey::U32;
  SortValue value = SortValue::U32;
  u64 element_count = 0u;
  u32 radix_bits = 8u;
  u32 key_bits = 0u;
  bool stable = true;
  ComputeCountSource count_source = ComputeCountSource::Descriptor;
};

struct SortPlan {
  SortKey key = SortKey::U32;
  SortValue value = SortValue::U32;
  u64 element_count = 0u;
  u64 key_bytes = 0u;
  u64 value_bytes = 0u;
  u32 radix_bits = 0u;
  u32 key_bits = 0u;
  u32 radix_pass_count = 0u;
  u64 bucket_count = 0u;
  u64 temp_key_bytes = 0u;
  u64 temp_value_bytes = 0u;
  u64 temp_count_bytes = 0u;
  u64 temp_rank_bytes = 0u;
  u64 temp_bytes = 0u;
  ComputeCountSource count_source = ComputeCountSource::Descriptor;
  bool stable = false;
  bool ok = false;
  const char* reason = "compute_sort_invalid";

  [[nodiscard]] constexpr explicit operator bool() const noexcept {
    return ok;
  }
};

struct SortHash {
  u64 hi = 0u;
  u64 lo = 0u;
};

struct SortResult {
  u64 element_count = 0u;
  bool ok = false;
  const char* reason = "compute_sort_invalid";

  [[nodiscard]] constexpr explicit operator bool() const noexcept {
    return ok;
  }
};

}  // namespace rund::kernel
