#pragma once

#include <kernel/program/compute/scatter/reduce/model.hpp>

#include <string_view>

namespace rund::node::accel::detail {

template <typename Sink>
[[nodiscard]] bool AppendMetalScatterReduceParallelFold(
    Sink &source, const rund::kernel::ScatterReduceOp op,
    const bool signed_value) noexcept(noexcept(source += std::string_view{})) {
  source += R"MSL(  const ulong ordinal = ulong(gid);
  if (ordinal >= logical) { return; }
  const uint target = indices[ordinal];
  device atomic_uint* contributor_counts =
      reinterpret_cast<device atomic_uint*>(counts);
  const uint rank = simd_prefix_exclusive_sum(1u);
  const uint width = simd_sum(1u);
  const bool uniform = simd_all(target == simd_broadcast_first(target));
  uint value = uint(values[ordinal]);
  if (uniform) {
)MSL";
  if (op == rund::kernel::ScatterReduceOp::Sum) {
    source += "    value = simd_sum(value);\n";
  } else {
    source += "    value = ";
    if (signed_value) {
      source += "as_type<uint>(";
    }
    source +=
        op == rund::kernel::ScatterReduceOp::Min ? "simd_min(" : "simd_max(";
    source += signed_value ? "as_type<int>(value))" : "value)";
    source += signed_value ? ");\n" : ";\n";
  }
  source += R"MSL(  }
  uint conflicts = 0u;
  if (!uniform || rank == 0u) {
    const uint count = uniform ? width : 1u;
    const uint prior = atomic_fetch_add_explicit(
        &contributor_counts[target], count, memory_order_relaxed);
    // Exactly one cohort or lane sees zero for each occupied target.
    conflicts = count - uint(prior == 0u);
    device atomic_uint* atomic_output =
        reinterpret_cast<device atomic_uint*>(output);
)MSL";
  if (op == rund::kernel::ScatterReduceOp::Sum) {
    source += R"MSL(    atomic_fetch_add_explicit(
      &atomic_output[target], value, memory_order_relaxed);
)MSL";
  } else if (op == rund::kernel::ScatterReduceOp::Min) {
    source += signed_value ? R"MSL(    device atomic_int* signed_atomic_output =
      reinterpret_cast<device atomic_int*>(output);
    atomic_fetch_min_explicit(
      &signed_atomic_output[target], as_type<int>(value),
      memory_order_relaxed);
)MSL"
                           : R"MSL(    atomic_fetch_min_explicit(
      &atomic_output[target], value, memory_order_relaxed);
)MSL";
  } else {
    source += signed_value ? R"MSL(    device atomic_int* signed_atomic_output =
      reinterpret_cast<device atomic_int*>(output);
    atomic_fetch_max_explicit(
      &signed_atomic_output[target], as_type<int>(value),
      memory_order_relaxed);
)MSL"
                           : R"MSL(    atomic_fetch_max_explicit(
      &atomic_output[target], value, memory_order_relaxed);
)MSL";
  }
  source += R"MSL(  }
  const uint group_conflicts = simd_sum(conflicts);
  if (rank == 0u && group_conflicts != 0u) {
      atomic_fetch_add_explicit(&status[2], group_conflicts, memory_order_relaxed);
  }
)MSL";
  return source.valid();
}

} // namespace rund::node::accel::detail
