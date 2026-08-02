#pragma once

#include "program/factor.hpp"
#include "program/matrix.hpp"
#include "program/solve.hpp"
#include "program/spectrum.hpp"
#include "program/transform.hpp"

#include <kernel/program/compute/matrix/tile.hpp>
#include <kernel/program/compute/transform/stage.hpp>
#include "../../../kernel/backend/source_recipe.hpp"

#include <string_view>

namespace rund::node::accel::detail {

// Storage width changes representation, never dependency order or topology.
template <typename Sink>
[[nodiscard]] inline bool EmitMetalNumericProgramSource(Sink &out) noexcept(
    noexcept(out.append(std::string_view{}))) {
  const auto define = [&out](const std::string_view name,
                             const kernel::u64 value) {
    return out.append("#define ") && out.append(name) && out.append(" ") &&
           backend_source_recipe::append_decimal(out, value) &&
           out.append("u\n");
  };
  return define("RUND_MATRIX_TILE_SIDE", kernel::matrix_tile::Side) &&
         define("RUND_MATRIX_TILE_CELLS", kernel::matrix_tile::Cells) &&
         define("RUND_MATRIX_TILE_LANES", kernel::matrix_tile::Lanes) &&
         define("RUND_MATRIX_TILE_ROWS", kernel::matrix_tile::RowsPerLane) &&
         define("RUND_MATRIX_TILE_COLS", kernel::matrix_tile::ColsPerLane) &&
         define("RUND_MATRIX_LANE_COLS", kernel::matrix_tile::LaneCols) &&
         define("RUND_TRANSFORM_LANES", kernel::transform_stage::Lanes) &&
         out.append(source::program::Matrix) &&
         out.append(source::program::Transform) &&
         out.append(source::program::Factor) &&
         out.append(source::program::Solve) &&
         out.append(source::program::Spectrum) && out.valid();
}

} // namespace rund::node::accel::detail
