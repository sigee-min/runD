#include "local.hpp"

namespace program_skeleton_contract {
namespace {

int LinearView() {
  std::array<rund::kernel::i32, 6u> grid_storage{1, 2, 3, 4, 5, 6};
  std::array<rund::kernel::i32, 6u> grid_out_storage{};
  auto grid = rund::kernel::view<rund::kernel::i32>(
      grid_storage.data(), rund::kernel::Index<2u>{2u, 3u});
  auto grid_out = rund::kernel::view<rund::kernel::i32>(
      grid_out_storage.data(), rund::kernel::Index<2u>{2u, 3u});
  TEST_ASSERT(grid);
  TEST_ASSERT(grid_out);
  TEST_ASSERT(grid.access == rund::kernel::ViewAccessPattern::Contiguous);
  const rund::kernel::LinearContiguousView<rund::kernel::i32, 2u> linear_grid =
      rund::kernel::TryLinearContiguousView(grid, rund::kernel::space(2u, 3u));
  TEST_ASSERT(linear_grid);
  TEST_ASSERT(linear_grid.data == grid_storage.data());
  TEST_ASSERT(linear_grid.unit_count == grid_storage.size());
  TEST_ASSERT(linear_grid.data[5u] == grid(1u, 2u));
  const rund::kernel::LinearContiguousView<rund::kernel::i32, 2u>
      mismatched_grid = rund::kernel::TryLinearContiguousView(
          grid, rund::kernel::space(3u, 2u));
  TEST_ASSERT(!mismatched_grid);
  TEST_ASSERT(std::string_view{mismatched_grid.reason} ==
              "linear_view_shape_mismatch");
  const rund::kernel::SkeletonResult grid_result =
      rund::kernel::each(rund::kernel::space(2u, 3u), [&](auto ij) noexcept {
        grid_out(ij) = rund::math32::detail::ScalarAddWrap(grid(ij), 10);
      });
  TEST_ASSERT(grid_result.ok);
  TEST_ASSERT(grid_out(1u, 2u) == 16);
  return 0;
}

int ZeroViews() {
  const rund::kernel::View<rund::kernel::i32, 1u> zero_view =
      rund::kernel::view<rund::kernel::i32>(nullptr,
                                            rund::kernel::Index<1u>{0u});
  TEST_ASSERT(zero_view);
  alignas(64) std::array<rund::kernel::i32, 4u> aligned_storage{};
  const rund::kernel::View<rund::kernel::i32, 1u> zero_misaligned =
      rund::kernel::view<rund::kernel::i32>(aligned_storage.data() + 1u,
                                            rund::kernel::Index<1u>{0u}, 64u);
  TEST_ASSERT(!zero_misaligned);
  TEST_ASSERT(std::string_view{zero_misaligned.reason} ==
              "view_alignment_failed");

  const rund::kernel::View<rund::kernel::i32, 2u> zero_strided_rank2 =
      rund::kernel::view<rund::kernel::i32>(
          nullptr, rund::kernel::Index<2u>{2u, 0u},
          std::array<std::ptrdiff_t, 2u>{2, 1}, 0u);
  TEST_ASSERT(zero_strided_rank2);
  TEST_ASSERT(zero_strided_rank2.access ==
              rund::kernel::ViewAccessPattern::Contiguous);
  const rund::kernel::View<rund::kernel::i32, 2u> zero_broadcast_rank2 =
      rund::kernel::view<rund::kernel::i32>(
          nullptr, rund::kernel::Index<2u>{0u, 3u},
          std::array<std::ptrdiff_t, 2u>{0, 0}, 0u);
  TEST_ASSERT(zero_broadcast_rank2);
  TEST_ASSERT(zero_broadcast_rank2.access ==
              rund::kernel::ViewAccessPattern::Contiguous);
  return 0;
}

int StridedViews() {
  std::array<rund::kernel::i32, 4u> strided_storage{1, 10, 2, 20};
  auto column = rund::kernel::view<rund::kernel::i32>(
      strided_storage.data(), rund::kernel::Index<1u>{2u},
      std::array<std::ptrdiff_t, 1u>{2}, strided_storage.size());
  TEST_ASSERT(column);
  TEST_ASSERT(column.access == rund::kernel::ViewAccessPattern::StridedAffine);
  TEST_ASSERT(column(0u) == 1);
  TEST_ASSERT(column(1u) == 2);
  const rund::kernel::LinearContiguousView<rund::kernel::i32, 1u>
      linear_column = rund::kernel::TryLinearContiguousView(
          column, rund::kernel::space(2u));
  TEST_ASSERT(!linear_column);
  TEST_ASSERT(std::string_view{linear_column.reason} ==
              "linear_view_not_contiguous");

  std::array<rund::kernel::i32, 6u> transposed_storage{1, 2, 3, 4, 5, 6};
  const rund::kernel::View<rund::kernel::i32, 2u> transposed =
      rund::kernel::view<rund::kernel::i32>(
          transposed_storage.data(), rund::kernel::Index<2u>{3u, 2u},
          std::array<std::ptrdiff_t, 2u>{1, 3}, transposed_storage.size());
  TEST_ASSERT(transposed);
  TEST_ASSERT(transposed.access ==
              rund::kernel::ViewAccessPattern::StridedAffine);
  TEST_ASSERT(transposed(2u, 1u) == 6);
  const rund::kernel::LinearContiguousView<rund::kernel::i32, 2u>
      linear_transposed = rund::kernel::TryLinearContiguousView(
          transposed, rund::kernel::space(3u, 2u));
  TEST_ASSERT(!linear_transposed);
  TEST_ASSERT(std::string_view{linear_transposed.reason} ==
              "linear_view_not_contiguous");

  const rund::kernel::View<rund::kernel::i32, 2u> broadcast =
      rund::kernel::view<rund::kernel::i32>(
          transposed_storage.data(), rund::kernel::Index<2u>{3u, 2u},
          std::array<std::ptrdiff_t, 2u>{0, 0}, transposed_storage.size());
  TEST_ASSERT(broadcast);
  TEST_ASSERT(broadcast.access ==
              rund::kernel::ViewAccessPattern::BroadcastZeroStride);
  TEST_ASSERT(broadcast(2u, 1u) == transposed_storage[0]);
  return 0;
}

} // namespace

int SkeletonShapeViews() {
  if (LinearView() != 0) {
    return 1;
  }
  if (ZeroViews() != 0) {
    return 1;
  }
  return StridedViews();
}

} // namespace program_skeleton_contract
