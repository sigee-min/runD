#pragma once

#include "../model/shape.hpp"
#include "arithmetic.hpp"

#include <algorithm>
#include <cstdint>
#include <optional>

namespace rund::node::accel::detail {
namespace range_plan_detail {

struct BoundaryTraffic final {
  rund::kernel::u128 valid_samples{};
  rund::kernel::u64 left_affected{};
  rund::kernel::u64 right_affected{};
  rund::kernel::u64 left_prefix_reads{};
};

[[nodiscard]] constexpr std::optional<BoundaryTraffic>
AffineBoundaryTraffic(const RangeShape &shape) noexcept {
  const rund::kernel::u64 output_count = shape.output_count();
  const rund::kernel::u64 stride = shape.stride();
  const rund::kernel::u64 padding = shape.padding();
  const rund::kernel::u64 right_extent = shape.right_extent();
  const rund::kernel::u64 left_affected =
      std::min(output_count, padding / stride + static_cast<rund::kernel::u64>(
                                                    padding % stride != 0u));
  const rund::kernel::u64 right_first =
      right_extent >= shape.input_count()
          ? 0u
          : std::min(
                output_count,
                (shape.input_count() - right_extent) / stride +
                    static_cast<rund::kernel::u64>(
                        (shape.input_count() - right_extent) % stride != 0u));
  const rund::kernel::u64 right_affected = output_count - right_first;
  const rund::kernel::u64 first_positive_left =
      std::min(output_count, padding / stride + 1u);

  rund::kernel::u128 all_samples = 0u;
  rund::kernel::u128 left_missing = 0u;
  rund::kernel::u128 right_missing = 0u;
  rund::kernel::u128 anchor_sum = 0u;
  if (!Multiply(output_count, shape.window_size(), all_samples) ||
      !Multiply(left_affected, padding, left_missing) ||
      !Multiply(static_cast<rund::kernel::u128>(left_affected) *
                    (left_affected == 0u ? 0u : left_affected - 1u) / 2u,
                stride, anchor_sum) ||
      left_missing < anchor_sum) {
    return std::nullopt;
  }
  left_missing -= anchor_sum;

  const rund::kernel::u128 first_right_missing =
      right_affected == 0u
          ? 0u
          : static_cast<rund::kernel::u128>(right_first) * stride +
                right_extent + 1u - shape.input_count();
  if (!Multiply(right_affected, first_right_missing, right_missing) ||
      !Multiply(static_cast<rund::kernel::u128>(right_affected) *
                    (right_affected == 0u ? 0u : right_affected - 1u) / 2u,
                stride, anchor_sum) ||
      !Add(right_missing, anchor_sum, right_missing) ||
      !Add(left_missing, right_missing, anchor_sum) ||
      anchor_sum >= all_samples) {
    return std::nullopt;
  }
  return BoundaryTraffic{
      .valid_samples = all_samples - anchor_sum,
      .left_affected = left_affected,
      .right_affected = right_affected,
      .left_prefix_reads = output_count - first_positive_left,
  };
}

} // namespace range_plan_detail
} // namespace rund::node::accel::detail
