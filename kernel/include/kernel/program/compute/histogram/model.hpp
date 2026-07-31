#pragma once

#include <kernel/core/model.hpp>

namespace rund::kernel {

enum class HistogramIndex : u8 {
  U32 = 1u,
};

enum class HistogramCount : u8 {
  U32 = 1u,
};

struct HistogramDesc {
  HistogramIndex index = HistogramIndex::U32;
  HistogramCount count = HistogramCount::U32;
  u64 element_count = 0u;
  u64 bin_count = 0u;
};

struct HistogramPlan {
  HistogramIndex index = HistogramIndex::U32;
  HistogramCount count = HistogramCount::U32;
  u64 element_count = 0u;
  u64 bin_count = 0u;
  u64 index_bytes = 0u;
  u64 count_bytes = 0u;
  u64 input_bytes = 0u;
  u64 output_bytes = 0u;
  u64 status_bytes = 0u;
  u64 temp_bytes = 0u;
  u64 pass_count = 0u;
  bool ok = false;
  const char* reason = "compute_histogram_invalid";

  [[nodiscard]] constexpr explicit operator bool() const noexcept {
    return ok;
  }
};

struct HistogramHash {
  u64 hi = 0u;
  u64 lo = 0u;
};

struct HistogramResult {
  u64 element_count = 0u;
  u64 bin_count = 0u;
  bool ok = false;
  const char* reason = "compute_histogram_invalid";

  [[nodiscard]] constexpr explicit operator bool() const noexcept {
    return ok;
  }
};

}  // namespace rund::kernel
