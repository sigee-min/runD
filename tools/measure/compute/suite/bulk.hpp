#pragma once

#include "core.hpp"

#include <rund/compute/math.hpp>

#include <bit>
#include <cstdint>

#if defined(RUND_COMPUTE_FOCUS)
#include <kernel/program/compute/matrix/tile.hpp>
#include <kernel/program/compute/transform/stage.hpp>
#endif

namespace rund::measure::compute {

#if defined(RUND_COMPUTE_FOCUS)
struct BulkCost final {
  std::uint64_t count;
  std::uint64_t passes;
  std::uint64_t logical_ops;
  std::uint64_t value_bytes;
  std::uint64_t coefficient_bytes;

  [[nodiscard]] constexpr std::uint64_t logical_bytes() const noexcept {
    return value_bytes + coefficient_bytes;
  }
};

consteval BulkCost MatrixCost(const std::uint64_t side) {
  constexpr std::uint64_t tile = rund::kernel::matrix_tile::Side;
  const std::uint64_t tiles = (side + tile - 1u) / tile;
  const std::uint64_t elements = side * side;
  const std::uint64_t products = elements * side;
  const std::uint64_t tile_values = tile * tile;
  const std::uint64_t loaded_values = tiles * tiles * tiles * 2u * tile_values;
  return BulkCost{
      .count = elements,
      .passes = tiles,
      .logical_ops = products * 2u,
      .value_bytes = (loaded_values + elements) * sizeof(std::int32_t),
      .coefficient_bytes = 0u,
  };
}

consteval BulkCost TransformCost(const std::uint64_t count) {
  const std::uint64_t stages = std::countr_zero(count);
  const std::uint64_t passes = rund::kernel::transform_stage::Dispatches(count);
  return BulkCost{
      .count = count,
      .passes = passes,
      .logical_ops = stages * (count / 2u) * 10u,
      .value_bytes = count * passes * sizeof(Fixed<1, 31>) * 4u,
      .coefficient_bytes = stages * (count / 2u) * sizeof(Fixed<1, 31>) * 2u,
  };
}

inline constexpr BulkCost MatrixBulkCost = MatrixCost(512u);
inline constexpr BulkCost TransformBulkCost = TransformCost(1u << 20u);
static_assert(MatrixBulkCost.logical_ops == 268'435'456u);
static_assert(MatrixBulkCost.value_bytes == 34'603'008u);
static_assert(MatrixBulkCost.coefficient_bytes == 0u);
static_assert(MatrixBulkCost.logical_bytes() == 34'603'008u);
static_assert(TransformBulkCost.passes == 7u);
static_assert(TransformBulkCost.logical_ops == 104'857'600u);
static_assert(TransformBulkCost.value_bytes == 117'440'512u);
static_assert(TransformBulkCost.coefficient_bytes == 83'886'080u);
static_assert(TransformBulkCost.logical_bytes() == 201'326'592u);
#endif

} // namespace rund::measure::compute
