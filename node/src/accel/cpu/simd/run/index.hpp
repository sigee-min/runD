#pragma once

#include "../context.hpp"

#include <kernel/program/compute/metadata.hpp>

#include <cstring>
#include <limits>
#include <span>

namespace rund::node::accel::cpu_simd_detail {

struct IndexCheck final {
  const char *reason{};
  rund::kernel::u64 ordinal{std::numeric_limits<rund::kernel::u64>::max()};

  [[nodiscard]] constexpr bool ok() const noexcept { return reason == nullptr; }
};

[[nodiscard]] inline IndexCheck
ValidateIndices(const CpuSimdBindingView &bindings,
                const std::span<const rund::kernel::ReadRoute> routes,
                const rund::kernel::u64 count) noexcept {
  for (const rund::kernel::ReadRoute route : routes) {
    if (route.count == 0u || bindings.reads == nullptr ||
        route.source >= bindings.read_count ||
        route.index >= bindings.read_count) {
      return {.reason = "cpu_simd_input_invalid"};
    }
    const CpuSimdReadBinding &indices = bindings.reads[route.index];
    if (indices.data == nullptr || indices.stride < sizeof(rund::kernel::u32)) {
      return {.reason = "cpu_simd_input_invalid"};
    }
  }
  for (rund::kernel::u64 ordinal = 0u; ordinal < count; ++ordinal) {
    for (const rund::kernel::ReadRoute route : routes) {
      const CpuSimdReadBinding &indices = bindings.reads[route.index];
      if (ordinal > std::numeric_limits<std::size_t>::max() / indices.stride) {
        return {.reason = "cpu_simd_input_bounds_overflow"};
      }
      const std::size_t offset =
          static_cast<std::size_t>(ordinal) * indices.stride;
      rund::kernel::u32 index = 0u;
      std::memcpy(&index, indices.data + offset, sizeof(index));
      if (index >= route.count) {
        return {.reason = "compute_gather_index_out_of_range",
                .ordinal = ordinal};
      }
    }
  }
  return {};
}

} // namespace rund::node::accel::cpu_simd_detail
