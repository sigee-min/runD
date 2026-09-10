#include "model.hpp"

#include <algorithm>

namespace rund::compute::resource::plan_detail {
namespace {

using Wide = __int128_t;

struct Bezout final {
  std::uint64_t gcd{};
  Wide left{};
  Wide right{};
};

[[nodiscard]] Bezout bezout(std::uint64_t left, std::uint64_t right) noexcept {
  Wide old_left = 1;
  Wide next_left = 0;
  Wide old_right = 0;
  Wide next_right = 1;
  while (right != 0u) {
    const std::uint64_t quotient = left / right;
    const std::uint64_t remainder = left % right;
    left = right;
    right = remainder;
    const Wide left_coefficient =
        old_left - static_cast<Wide>(quotient) * next_left;
    old_left = next_left;
    next_left = left_coefficient;
    const Wide right_coefficient =
        old_right - static_cast<Wide>(quotient) * next_right;
    old_right = next_right;
    next_right = right_coefficient;
  }
  return Bezout{.gcd = left, .left = old_left, .right = old_right};
}

[[nodiscard]] Wide floor_div(const Wide value, const Wide divisor) noexcept {
  Wide quotient = value / divisor;
  if (value % divisor < 0) {
    --quotient;
  }
  return quotient;
}

[[nodiscard]] Wide ceil_div(const Wide value, const Wide divisor) noexcept {
  return -floor_div(-value, divisor);
}

[[nodiscard]] bool bounded_solution(const std::uint64_t left_stride,
                                    const std::uint64_t left_count,
                                    const std::uint64_t right_stride,
                                    const std::uint64_t right_count,
                                    const Wide target,
                                    std::uint64_t &left_index,
                                    std::uint64_t &right_index) noexcept {
  const Bezout coefficients = bezout(left_stride, right_stride);
  if (coefficients.gcd == 0u ||
      target % static_cast<Wide>(coefficients.gcd) != 0) {
    return false;
  }
  const Wide scale = target / static_cast<Wide>(coefficients.gcd);
  const Wide first_left = coefficients.left * scale;
  const Wide first_right = -coefficients.right * scale;
  const Wide left_step = static_cast<Wide>(right_stride / coefficients.gcd);
  const Wide right_step = static_cast<Wide>(left_stride / coefficients.gcd);
  const Wide lower = std::max(ceil_div(-first_left, left_step),
                              ceil_div(-first_right, right_step));
  const Wide upper = std::min(
      floor_div(static_cast<Wide>(left_count - 1u) - first_left, left_step),
      floor_div(static_cast<Wide>(right_count - 1u) - first_right, right_step));
  if (lower > upper) {
    return false;
  }
  const Wide solved_left = first_left + left_step * lower;
  const Wide solved_right = first_right + right_step * lower;
  if (solved_left < 0 || solved_right < 0 ||
      solved_left >= static_cast<Wide>(left_count) ||
      solved_right >= static_cast<Wide>(right_count)) {
    return false;
  }
  left_index = static_cast<std::uint64_t>(solved_left);
  right_index = static_cast<std::uint64_t>(solved_right);
  return true;
}

[[nodiscard]] bool enumerated_solution(const PhysicalAccess &small,
                                       const PhysicalAccess &large,
                                       const Wide delta,
                                       const bool small_is_left,
                                       std::uint64_t &left_index,
                                       std::uint64_t &right_index) noexcept {
  for (std::uint64_t index = 0u; index < small.element_count; ++index) {
    const Wide small_start = static_cast<Wide>(small.offset) +
                             static_cast<Wide>(index) * small.stride_bytes;
    const Wide large_start =
        small_is_left ? small_start - delta : small_start + delta;
    if (large_start < static_cast<Wide>(large.offset)) {
      continue;
    }
    const Wide distance = large_start - static_cast<Wide>(large.offset);
    if (distance % static_cast<Wide>(large.stride_bytes) != 0) {
      continue;
    }
    const Wide large_index = distance / large.stride_bytes;
    if (large_index < 0 ||
        large_index >= static_cast<Wide>(large.element_count)) {
      continue;
    }
    if (small_is_left) {
      left_index = index;
      right_index = static_cast<std::uint64_t>(large_index);
    } else {
      left_index = static_cast<std::uint64_t>(large_index);
      right_index = index;
    }
    return true;
  }
  return false;
}

} // namespace

Overlap find_overlap(const PhysicalAccess &left,
                     const PhysicalAccess &right) noexcept {
  if (left.alias_group != right.alias_group ||
      left.offset >= right.envelope_end || right.offset >= left.envelope_end) {
    return {};
  }
  // Contiguous public and exact-strided forms canonicalize to the same full
  // interval when stride equals element width.
  if (left.stride_bytes == left.element_bytes &&
      right.stride_bytes == right.element_bytes) {
    return Overlap{.begin = std::max(left.offset, right.offset),
                   .end = std::min(left.envelope_end, right.envelope_end),
                   .found = true};
  }
  constexpr std::uint64_t EnumerateLimit = 64u;
  const Wide delta_begin = 1 - static_cast<Wide>(left.element_bytes);
  const Wide delta_end = static_cast<Wide>(right.element_bytes) - 1;
  for (Wide delta = delta_begin; delta <= delta_end; ++delta) {
    std::uint64_t left_index = 0u;
    std::uint64_t right_index = 0u;
    bool found = false;
    if (left.element_count <= EnumerateLimit ||
        right.element_count <= EnumerateLimit) {
      const bool left_is_small = left.element_count <= right.element_count;
      found = left_is_small ? enumerated_solution(left, right, delta, true,
                                                  left_index, right_index)
                            : enumerated_solution(right, left, delta, false,
                                                  left_index, right_index);
    } else {
      const Wide target = static_cast<Wide>(right.offset) - left.offset + delta;
      found = bounded_solution(left.stride_bytes, left.element_count,
                               right.stride_bytes, right.element_count, target,
                               left_index, right_index);
    }
    if (!found) {
      continue;
    }
    const std::uint64_t left_start =
        left.offset + left_index * left.stride_bytes;
    const std::uint64_t right_start =
        right.offset + right_index * right.stride_bytes;
    return Overlap{
        .begin = std::max(left_start, right_start),
        .end = std::min(left_start + left.element_bytes,
                        right_start + right.element_bytes),
        .found = true,
    };
  }
  return {};
}

} // namespace rund::compute::resource::plan_detail

namespace rund::compute::resource {

Result<bool> intersects(const Resource &left_resource, const Access &left,
                        const Resource &right_resource,
                        const Access &right) noexcept {
  auto physical_left = plan_detail::physical_access(left_resource, left);
  if (!physical_left) {
    return Result<bool>::fail(physical_left.reason());
  }
  auto physical_right = plan_detail::physical_access(right_resource, right);
  if (!physical_right) {
    return Result<bool>::fail(physical_right.reason());
  }
  return Result<bool>::success(
      plan_detail::find_overlap(*physical_left, *physical_right).found);
}

} // namespace rund::compute::resource
