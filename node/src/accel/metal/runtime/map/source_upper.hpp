#pragma once

#include <kernel/core/checked.hpp>
#include <kernel/program/compute/artifact.hpp>
#include <kernel/program/compute/backend.hpp>

#include <cstdint>
#include <limits>
#include <string_view>

namespace rund::node::accel::detail {

namespace metal_controlled_map_source_detail {

inline constexpr std::string_view Entry =
    "    uint gid [[thread_position_in_grid]]) {\n";
inline constexpr std::string_view ControlledPrefix =
    "    const device uint* rund_control_active [[buffer(";
inline constexpr std::string_view ControlledSuffix =
    ")]],\n    uint gid [[thread_position_in_grid]]) {\n"
    "  if (gid >= rund_control_active[0]) { return; }\n";
inline constexpr std::string_view FunctionPrefix = "rund_compute_map_";
inline constexpr std::string_view FunctionSuffix = "_controlled";
inline constexpr std::string_view CanonicalVariant =
    "// artifact_variant=canonical";
inline constexpr std::string_view ControlledVariant =
    "// artifact_variant=controlled";

} // namespace metal_controlled_map_source_detail

[[nodiscard]] inline constexpr std::uint64_t
DecimalDigitCount(std::uint64_t value) noexcept {
  std::uint64_t digits = 1u;
  while (value >= 10u) {
    value /= 10u;
    ++digits;
  }
  return digits;
}

[[nodiscard]] inline bool
MetalControlledMapSourceUpperBytes(const rund::kernel::ComputePlan &plan,
                                   const std::uint64_t specialized,
                                   std::uint64_t &upper) noexcept {
  using namespace metal_controlled_map_source_detail;
  std::uint64_t binding = 0u;
  if (!rund::kernel::checked::add(plan.input_buffer_count,
                                  plan.output_buffer_count, binding) ||
      !rund::kernel::checked::add(binding, 1u, binding)) {
    return false;
  }
  const std::uint64_t replacement = ControlledPrefix.size() +
                                    DecimalDigitCount(binding) +
                                    ControlledSuffix.size();
  const std::uint64_t growth =
      replacement - Entry.size() + FunctionSuffix.size() +
      ControlledVariant.size() - CanonicalVariant.size();
  return rund::kernel::checked::add(specialized, growth, upper);
}

[[nodiscard]] inline constexpr std::string_view
MetalMapControlSourceText() noexcept {
  return R"metal(
#include <metal_stdlib>
using namespace metal;

struct ControlWindow {
  ulong begin;
  ulong count;
};

struct ControlConfig {
  uint has_count;
  uint count_u64;
  uint has_predicate;
  uint predicate_u64;
  uint dispatch_width;
  uint checked;
  ulong capacity;
  ulong predicate_expected;
};

kernel void rund_map_control_dispatch(
    device const uchar *count_source [[buffer(0)]],
    device const uchar *predicate_source [[buffer(1)]],
    device uint *dispatch_args [[buffer(2)]],
    constant ControlWindow *windows [[buffer(3)]],
    constant ControlConfig &config [[buffer(4)]],
    device atomic_uint *status [[buffer(5)]],
    uint gid [[thread_position_in_grid]]) {
  ulong logical = config.capacity;
  if (config.has_count != 0u) {
    logical = config.count_u64 != 0u
                  ? *reinterpret_cast<device const ulong *>(count_source)
                  : ulong(*reinterpret_cast<device const uint *>(count_source));
  }
  const bool overflow = logical > config.capacity;
  const uint prior = atomic_load_explicit(&status[0], memory_order_relaxed);
  bool enabled = true;
  if (config.has_predicate != 0u) {
    const ulong observed =
        config.predicate_u64 != 0u
            ? *reinterpret_cast<device const ulong *>(predicate_source)
            : ulong(*reinterpret_cast<device const uint *>(predicate_source));
    enabled = observed == config.predicate_expected;
  }
  const ControlWindow window = windows[gid];
  const ulong remaining = !overflow && logical > window.begin
                              ? logical - window.begin
                              : 0ul;
  const ulong active =
      enabled && !overflow && (config.checked == 0u || prior == 0u)
                           ? min(remaining, window.count)
                           : 0ul;
  if (gid == 0u) {
    if (config.checked == 0u) {
      atomic_store_explicit(&status[0], overflow ? 1u : 0u,
                            memory_order_relaxed);
    }
  }
  const uint base = gid * 4u;
  dispatch_args[base + 0u] =
      uint((active + ulong(config.dispatch_width) - 1ul) /
           ulong(config.dispatch_width));
  dispatch_args[base + 1u] = 1u;
  dispatch_args[base + 2u] = 1u;
  dispatch_args[base + 3u] = uint(active);
}
)metal";
}

namespace metal_map_source_detail {

inline constexpr std::string_view CheckNamePrefix = "rund_compute_map_";
inline constexpr std::string_view CheckPrefix = R"metal(
#include <metal_stdlib>
using namespace metal;

struct ControlConfig {
  uint has_count;
  uint count_u64;
  uint has_predicate;
  uint predicate_u64;
  uint dispatch_width;
  uint checked;
  ulong capacity;
  ulong predicate_expected;
};

kernel void )metal";
inline constexpr std::string_view CheckArguments =
    "(\n"
    "    device const uchar *count_source [[buffer(0)]],\n"
    "    device const uchar *predicate_source [[buffer(1)]],\n";
inline constexpr std::string_view CheckBindingPrefix =
    "    device const uchar *indices";
inline constexpr std::string_view CheckBindingMiddle = " [[buffer(";
inline constexpr std::string_view CheckBindingSuffix = ")]],\n";
inline constexpr std::string_view CheckConfigPrefix =
    "    constant ControlConfig &config [[buffer(";
inline constexpr std::string_view CheckConfigMiddle =
    ")]],\n    device uint *status [[buffer(";
inline constexpr std::string_view CheckBody = R"metal()]],
    uint tid [[thread_index_in_threadgroup]]) {
  ulong logical = config.capacity;
  if (config.has_count != 0u) {
    logical = config.count_u64 != 0u
                  ? *reinterpret_cast<device const ulong *>(count_source)
                  : ulong(*reinterpret_cast<device const uint *>(count_source));
  }
  bool enabled = true;
  if (config.has_predicate != 0u) {
    const ulong observed =
        config.predicate_u64 != 0u
            ? *reinterpret_cast<device const ulong *>(predicate_source)
            : ulong(*reinterpret_cast<device const uint *>(predicate_source));
    enabled = observed == config.predicate_expected;
  }
  uint local_invalid = 0xffffffffu;
  if (enabled && logical <= config.capacity) {
    for (ulong ordinal = ulong(tid); ordinal < logical; ordinal += 256ul) {
)metal";
inline constexpr std::string_view CheckLinePrefix =
    "      if (ulong(*reinterpret_cast<device const uint *>(indices";
inline constexpr std::string_view CheckLineStride = " + ordinal * ";
inline constexpr std::string_view CheckLineLimit = "ul)) >= ";
inline constexpr std::string_view CheckLineSuffix =
    "ul) { local_invalid = min(local_invalid, uint(min(ordinal, "
    "0xfffffffeul))); }\n";
inline constexpr std::string_view CheckTail = R"metal(    }
  }
  threadgroup uint invalids[256];
  invalids[tid] = local_invalid;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint stride = 128u; stride != 0u; stride >>= 1u) {
    if (tid < stride) {
      invalids[tid] = min(invalids[tid], invalids[tid + stride]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  if (tid != 0u) { return; }
  status[1] = uint(min(logical, 0xfffffffful));
  status[0] = logical > config.capacity
                  ? 1u
                  : (invalids[0] == 0xffffffffu ? 0u : 2u);
  if (status[0] == 2u) { status[1] = invalids[0]; }
}
)metal";

} // namespace metal_map_source_detail

[[nodiscard]] inline bool MetalMapCheckSourceUpperBytes(
    const std::uint64_t check_count, const std::uint64_t stride_digit_bytes,
    const std::uint64_t limit_digit_bytes, std::uint64_t &upper) noexcept {
  using rund::kernel::checked::add;
  using rund::kernel::checked::mul;
  constexpr std::uint64_t HexNameDigits = 33u;
  if (check_count > std::numeric_limits<std::uint64_t>::max() - 3u) {
    return false;
  }
  std::uint64_t bytes = metal_map_source_detail::CheckPrefix.size();
  std::uint64_t item = 0u;
  if (!add(bytes, metal_map_source_detail::CheckNamePrefix.size(), bytes) ||
      !add(bytes, HexNameDigits, bytes) ||
      !add(bytes, metal_map_source_detail::CheckArguments.size(), bytes) ||
      !mul(check_count,
           metal_map_source_detail::CheckBindingPrefix.size() +
               metal_map_source_detail::CheckBindingMiddle.size() +
               metal_map_source_detail::CheckBindingSuffix.size() +
               metal_map_source_detail::CheckLinePrefix.size() +
               metal_map_source_detail::CheckLineStride.size() +
               metal_map_source_detail::CheckLineLimit.size() +
               metal_map_source_detail::CheckLineSuffix.size(),
           item) ||
      !add(bytes, item, bytes) ||
      !add(bytes, metal_map_source_detail::CheckConfigPrefix.size(), bytes) ||
      !add(bytes, metal_map_source_detail::CheckConfigMiddle.size(), bytes) ||
      !add(bytes, metal_map_source_detail::CheckBody.size(), bytes) ||
      !add(bytes, metal_map_source_detail::CheckTail.size(), bytes) ||
      !add(bytes, stride_digit_bytes, bytes) ||
      !add(bytes, limit_digit_bytes, bytes)) {
    return false;
  }
  for (std::uint64_t index = 0u; index < check_count; ++index) {
    if (index > std::numeric_limits<std::uint64_t>::max() - 3u ||
        !add(bytes, 2u * DecimalDigitCount(index), bytes) ||
        !add(bytes, DecimalDigitCount(index + 2u), bytes)) {
      return false;
    }
  }
  return add(bytes, DecimalDigitCount(check_count + 2u), bytes) &&
         add(bytes, DecimalDigitCount(check_count + 3u), upper);
}

[[nodiscard]] inline bool
MetalMapCheckSourceUpperBytes(const rund::kernel::LoweringArtifact &artifact,
                              std::uint64_t &upper) noexcept {
  std::uint64_t check_count = 0u;
  std::uint64_t limit_digits = 0u;
  for (std::size_t index = 0u; index < artifact.metadata.read_routes.size();
       ++index) {
    bool first = true;
    for (std::size_t prior = 0u; prior < index; ++prior) {
      if (artifact.metadata.read_routes[prior].index ==
          artifact.metadata.read_routes[index].index) {
        first = false;
        break;
      }
    }
    if (!first) {
      continue;
    }
    if (!rund::kernel::checked::add(
            limit_digits,
            DecimalDigitCount(artifact.metadata.read_routes[index].count),
            limit_digits) ||
        !rund::kernel::checked::add(check_count, 1u, check_count)) {
      return false;
    }
  }
  std::uint64_t stride_digits = 0u;
  return rund::kernel::checked::mul(check_count, 20u, stride_digits) &&
         MetalMapCheckSourceUpperBytes(check_count, stride_digits, limit_digits,
                                       upper);
}

} // namespace rund::node::accel::detail
