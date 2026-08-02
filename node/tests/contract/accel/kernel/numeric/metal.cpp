#include "src/accel/metal/numeric.hpp"
#include "src/accel/metal/numeric/source.hpp"
#include "src/accel/metal/numeric/source/program.hpp"
#include <kernel/program/compute/matrix/tile.hpp>
#include <kernel/program/compute/transform/stage.hpp>

#include <string>
#include <string_view>

namespace node_accel_contract {
namespace {

[[nodiscard]] bool Contains(const std::string_view source,
                            const std::string_view token) {
  return source.find(token) != std::string_view::npos;
}

} // namespace

[[nodiscard]] bool MetalNumericSourcesUseParallelTopology() {
  namespace source_recipe =
      rund::node::accel::detail::backend_source_recipe;
  using rund::node::accel::detail::kMetalAlgebraLanes;
  using rund::node::accel::detail::EmitMetalNumericFixedLane32Source;
  using rund::node::accel::detail::EmitMetalNumericFixedLane64Source;
  using rund::node::accel::detail::MetalNumericSource;
  using rund::node::accel::detail::MetalNumericSourceUpperBytes;

  const std::string lane32 = source_recipe::materialize(
      [](auto &sink) noexcept(
          noexcept(EmitMetalNumericFixedLane32Source(sink))) {
        return EmitMetalNumericFixedLane32Source(sink);
      });
  const std::string lane64 = source_recipe::materialize(
      [](auto &sink) noexcept(
          noexcept(EmitMetalNumericFixedLane64Source(sink))) {
        return EmitMetalNumericFixedLane64Source(sink);
      });
  const std::string combined = MetalNumericSource();
  std::uint64_t exact_bytes = 0u;
  const std::string_view program = lane32;
  return !lane32.empty() && !lane64.empty() && !combined.empty() &&
         MetalNumericSourceUpperBytes(exact_bytes) &&
         exact_bytes == combined.size() &&
         combined.size() == lane32.size() + lane64.size() &&
         combined.starts_with(lane32) && combined.ends_with(lane64) &&
         rund::kernel::transform_stage::Lanes == 256u &&
         rund::kernel::transform_stage::LocalStages == 8u &&
         kMetalAlgebraLanes == 32u && rund::kernel::matrix_tile::Side == 32u &&
         rund::kernel::matrix_tile::Cells == 1024u &&
         rund::kernel::matrix_tile::RowsPerLane == 2u &&
         rund::kernel::matrix_tile::ColsPerLane == 4u &&
         rund::kernel::matrix_tile::Lanes == 128u &&
         Contains(program, "RUND_KERNEL(rund_numeric_matrix_)") &&
         Contains(program, "#define RUND_MATRIX_TILE_SIDE 32u") &&
         Contains(program, "#define RUND_MATRIX_TILE_LANES 128u") &&
         Contains(program, "threadgroup RUND_SCALAR left_tile["
                           "RUND_MATRIX_TILE_CELLS]") &&
         Contains(program, "threadgroup RUND_SCALAR right_tile["
                           "RUND_MATRIX_TILE_CELLS]") &&
         Contains(program, "for (ulong base = 0ul; base < p.inner;") &&
         Contains(program, "for (ulong offset = 0ul; offset < width;") &&
         Contains(program, "RUND_SCALAR sum13 = RUND_ZERO") &&
         Contains(program, "RUND_KERNEL(rund_numeric_transform_)") &&
         Contains(program, "#define RUND_TRANSFORM_LANES 256u") &&
         Contains(program, "threadgroup RUND_SCALAR block_r["
                           "RUND_TRANSFORM_LANES]") &&
         Contains(program, "if (span == 1ul)") &&
         Contains(program, "for (ulong local_span = 2ul;") &&
         Contains(program, "uint3 gid [[thread_position_in_grid]]") &&
         Contains(program, "ulong span = p.inner") &&
         !Contains(program, "if (group.x != 0u)") &&
         !Contains(program, "for (ulong span = 2ul; span <= n;") &&
         Contains(program, "RUND_KERNEL(rund_numeric_factor_)") &&
         Contains(program, "RUND_KERNEL(rund_numeric_solve_)") &&
         Contains(program, "RUND_KERNEL(rund_numeric_spectrum_)") &&
         Contains(program, "[[thread_index_in_threadgroup]]") &&
         Contains(program, "[[threadgroup_position_in_grid]]") &&
         Contains(program, "[[threads_per_threadgroup]]") &&
         Contains(lane32, "threadgroup_barrier(") &&
         Contains(program, "for (ulong cell = ulong(lane);") &&
         !Contains(program, "if (gid != 0u)") &&
         !Contains(program, "ulong batch = ulong(gid)") &&
         Contains(lane32, "#define RUND_SUFFIX i32") &&
         Contains(lane64, "#define RUND_SUFFIX i64") &&
         Contains(lane64, "RUND_KERNEL(rund_numeric_matrix_)") &&
         Contains(lane64, "RUND_KERNEL(rund_numeric_transform_)") &&
         Contains(lane64, "RUND_KERNEL(rund_numeric_factor_)") &&
         Contains(lane64, "RUND_KERNEL(rund_numeric_solve_)") &&
         Contains(lane64, "RUND_KERNEL(rund_numeric_spectrum_)");
}

} // namespace node_accel_contract
