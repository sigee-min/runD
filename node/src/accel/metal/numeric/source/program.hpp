#pragma once

#include "program/factor.hpp"
#include "program/matrix.hpp"
#include "program/solve.hpp"
#include "program/spectrum.hpp"
#include "program/transform.hpp"

#include <kernel/program/compute/matrix/tile.hpp>
#include <kernel/program/compute/transform/stage.hpp>

#include <string>
#include <string_view>

namespace rund::node::accel::detail {

// Storage width changes representation, never dependency order or topology.
[[nodiscard]] inline const std::string &MetalNumericProgramSource() {
  static const std::string source = [] {
    std::string out;
    const auto define = [&out](const std::string_view name,
                               const kernel::u64 value) {
      out += "#define ";
      out += name;
      out += ' ';
      out += std::to_string(value);
      out += "u\n";
    };
    define("RUND_MATRIX_TILE_SIDE", kernel::matrix_tile::Side);
    define("RUND_MATRIX_TILE_CELLS", kernel::matrix_tile::Cells);
    define("RUND_MATRIX_TILE_LANES", kernel::matrix_tile::Lanes);
    define("RUND_MATRIX_TILE_ROWS", kernel::matrix_tile::RowsPerLane);
    define("RUND_MATRIX_TILE_COLS", kernel::matrix_tile::ColsPerLane);
    define("RUND_MATRIX_LANE_COLS", kernel::matrix_tile::LaneCols);
    define("RUND_TRANSFORM_LANES", kernel::transform_stage::Lanes);
    out.reserve(out.size() + source::program::Matrix.size() +
                source::program::Transform.size() +
                source::program::Factor.size() +
                source::program::Solve.size() +
                source::program::Spectrum.size());
    out.append(source::program::Matrix);
    out.append(source::program::Transform);
    out.append(source::program::Factor);
    out.append(source::program::Solve);
    out.append(source::program::Spectrum);
    return out;
  }();
  return source;
}

} // namespace rund::node::accel::detail
