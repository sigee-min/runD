#pragma once

#include <kernel/core/model.hpp>

namespace rund::kernel {

enum class SegmentedScanOp : u8 {
  ExclusiveSum = 1u,
  InclusiveSum = 2u,
};

enum class SegmentedScanElement : u8 {
  U32 = 1u,
  U64 = 2u,
};

struct SegmentedScanDesc {
  SegmentedScanOp op = SegmentedScanOp::ExclusiveSum;
  SegmentedScanElement element = SegmentedScanElement::U32;
  u64 element_count = 0u;
  u64 block_size = 0u;
};

struct SegmentedScanPlan {
  SegmentedScanOp op = SegmentedScanOp::ExclusiveSum;
  SegmentedScanElement element = SegmentedScanElement::U32;
  u64 element_count = 0u;
  u64 element_bytes = 0u;
  u64 head_bytes = 0u;
  u64 block_size = 0u;
  u64 block_count = 0u;
  u64 pass_count = 0u;
  u64 temp_value_bytes = 0u;
  u64 temp_head_bytes = 0u;
  u64 temp_bytes = 0u;
  bool ok = false;
  const char *reason = "compute_segmented_scan_invalid";

  [[nodiscard]] constexpr explicit operator bool() const noexcept { return ok; }
};

struct SegmentedScanHash {
  u64 hi = 0u;
  u64 lo = 0u;
};

struct SegmentedScanResult {
  u64 element_count = 0u;
  u64 segment_count = 0u;
  u64 final_segment_total = 0u;
  bool ok = false;
  const char *reason = "compute_segmented_scan_reference_invalid";

  [[nodiscard]] constexpr explicit operator bool() const noexcept { return ok; }
};

} // namespace rund::kernel
