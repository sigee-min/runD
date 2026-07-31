#pragma once

#include <kernel/program/compute/model.hpp>

namespace rund::kernel {

enum class GatherElement : u8 {
  U32 = 1u,
  U64 = 2u,
};

struct GatherDesc {
  GatherElement element = GatherElement::U32;
  u64 element_count = 0u;
  u64 source_count = 0u;
  ComputeCountSource count_source = ComputeCountSource::Descriptor;
};

struct GatherPlan {
  GatherElement element = GatherElement::U32;
  u64 element_count = 0u;
  u64 source_count = 0u;
  u64 element_bytes = 0u;
  u64 index_bytes = 0u;
  u64 status_bytes = 0u;
  u64 temp_bytes = 0u;
  u64 pass_count = 0u;
  ComputeCountSource count_source = ComputeCountSource::Descriptor;
  bool ok = false;
  const char* reason = "compute_gather_invalid";

  [[nodiscard]] constexpr explicit operator bool() const noexcept {
    return ok;
  }
};

struct GatherHash {
  u64 hi = 0u;
  u64 lo = 0u;
};

struct GatherResult {
  u64 element_count = 0u;
  u64 first_invalid_index = 0u;
  bool ok = false;
  const char* reason = "compute_gather_invalid";

  [[nodiscard]] constexpr explicit operator bool() const noexcept {
    return ok;
  }
};

}  // namespace rund::kernel
