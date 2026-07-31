#include "local.hpp"

namespace program_skeleton_contract {

int SkeletonShapeFold() {
  const std::array<rund::kernel::i32, 6u> matrix_a{1, 2, 3, 4, 5, 6};
  const std::array<rund::kernel::i32, 6u> matrix_b{7, 8, 9, 10, 11, 12};
  std::array<rund::kernel::i32, 4u> matrix_c{};
  auto a = rund::kernel::view<rund::kernel::i32>(
      const_cast<rund::kernel::i32 *>(matrix_a.data()),
      rund::kernel::Index<2u>{2u, 3u});
  auto b = rund::kernel::view<rund::kernel::i32>(
      const_cast<rund::kernel::i32 *>(matrix_b.data()),
      rund::kernel::Index<2u>{3u, 2u});
  auto c = rund::kernel::view<rund::kernel::i32>(
      matrix_c.data(), rund::kernel::Index<2u>{2u, 2u});
  TEST_ASSERT(a);
  TEST_ASSERT(b);
  TEST_ASSERT(c);
  bool inner_fold_ok = true;
  const rund::kernel::SkeletonResult product =
      rund::kernel::each(rund::kernel::space(2u, 2u), [&](auto ij) noexcept {
        rund::kernel::i32 acc = 0;
        const rund::kernel::SkeletonResult folded = rund::kernel::fold(
            rund::kernel::space(3u), acc,
            [&](rund::kernel::i32 sum, auto kk) noexcept {
              const rund::kernel::i32 product =
                  rund::math32::detail::ScalarMulLow(a(ij[0], kk[0]),
                                                     b(kk[0], ij[1]));
              return rund::math32::detail::ScalarAddWrap(sum, product);
            });
        inner_fold_ok = inner_fold_ok && folded.ok;
        c(ij) = acc;
      });
  TEST_ASSERT(product.ok);
  TEST_ASSERT(inner_fold_ok);
  TEST_ASSERT(matrix_c[0] == 58);
  TEST_ASSERT(matrix_c[1] == 64);
  TEST_ASSERT(matrix_c[2] == 139);
  TEST_ASSERT(matrix_c[3] == 154);

  std::array<rund::kernel::i32, 8u> volume{};
  auto volume3 = rund::kernel::view<rund::kernel::i32>(
      volume.data(), rund::kernel::Index<3u>{2u, 2u, 2u});
  TEST_ASSERT(volume3);
  const rund::kernel::SkeletonResult volume_result = rund::kernel::each(
      rund::kernel::space(2u, 2u, 2u), [&](auto xyz) noexcept {
        volume3(xyz) = static_cast<rund::kernel::i32>((xyz[0] * 100u) +
                                                      (xyz[1] * 10u) + xyz[2]);
      });
  TEST_ASSERT(volume_result.ok);
  TEST_ASSERT(volume_result.visited_count == 8u);
  TEST_ASSERT(volume3(1u, 1u, 1u) == 111);
  return 0;
}

} // namespace program_skeleton_contract
