#pragma once

#include <kernel/program/compute/model.hpp>

namespace rund::kernel {

enum class ScanOp : u8 {
  ExclusiveSum = 1u,
  InclusiveSum = 2u,
};

enum class ScanElement : u8 {
  U32 = 1u,
  U64 = 2u,
};

struct ScanDesc {
  ScanOp op = ScanOp::ExclusiveSum;
  ScanElement element = ScanElement::U32;
  u64 element_count = 0u;
  u64 block_size = 0u;
  ComputeCountSource count_source = ComputeCountSource::Descriptor;
};

struct ScanPlan {
  ScanOp op = ScanOp::ExclusiveSum;
  ScanElement element = ScanElement::U32;
  u64 element_count = 0u;
  u64 element_bytes = 0u;
  u64 block_size = 0u;
  u64 block_count = 0u;
  u64 pass_count = 0u;
  u64 temp_bytes = 0u;
  ComputeCountSource count_source = ComputeCountSource::Descriptor;
  bool ok = false;
  const char* reason = "compute_scan_invalid";

  [[nodiscard]] constexpr explicit operator bool() const noexcept {
    return ok;
  }
};

struct ScanHash {
  u64 hi = 0u;
  u64 lo = 0u;
};

struct ScanResult {
  u64 element_count = 0u;
  u64 total = 0u;
  bool ok = false;
  const char* reason = "compute_scan_reference_invalid";

  [[nodiscard]] constexpr explicit operator bool() const noexcept {
    return ok;
  }
};

}  // namespace rund::kernel
