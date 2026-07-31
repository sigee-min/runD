#include "local.hpp"

namespace program_skeleton_contract {

int SkeletonValidationView() {
  std::array<rund::kernel::i32, 4u> strided_storage{1, 10, 2, 20};
  alignas(64) std::array<rund::kernel::i32, 4u> aligned_storage{};
  const rund::kernel::View<rund::kernel::i32, 1u> null_view =
      rund::kernel::view<rund::kernel::i32>(nullptr,
                                            rund::kernel::Index<1u>{1u});
  TEST_ASSERT(!null_view);
  TEST_ASSERT(null_view.access == rund::kernel::ViewAccessPattern::Unsupported);
  TEST_ASSERT(std::string_view{null_view.reason} == "view_null_data");

  const rund::kernel::View<rund::kernel::i32, 1u> small_backing =
      rund::kernel::view<rund::kernel::i32>(
          strided_storage.data(), rund::kernel::Index<1u>{2u},
          std::array<std::ptrdiff_t, 1u>{4}, strided_storage.size());
  TEST_ASSERT(!small_backing);
  TEST_ASSERT(std::string_view{small_backing.reason} ==
              "view_backing_span_too_small");

  const rund::kernel::View<rund::kernel::i32, 1u> negative_offset =
      rund::kernel::view<rund::kernel::i32>(
          strided_storage.data(), rund::kernel::Index<1u>{2u},
          std::array<std::ptrdiff_t, 1u>{-1}, strided_storage.size());
  TEST_ASSERT(!negative_offset);
  TEST_ASSERT(std::string_view{negative_offset.reason} ==
              "view_negative_offset");

  const rund::kernel::View<rund::kernel::i32, 1u> stride_overflow =
      rund::kernel::view<rund::kernel::i32>(
          strided_storage.data(), rund::kernel::Index<1u>{2u},
          std::array<std::ptrdiff_t, 1u>{
              std::numeric_limits<std::ptrdiff_t>::min()},
          strided_storage.size());
  TEST_ASSERT(!stride_overflow);
  TEST_ASSERT(std::string_view{stride_overflow.reason} ==
              "view_stride_range_overflow");

  const rund::kernel::u64 too_large =
      std::numeric_limits<rund::kernel::u64>::max();
  const rund::kernel::View<rund::kernel::i32, 2u> shape_overflow =
      rund::kernel::view<rund::kernel::i32>(
          strided_storage.data(), rund::kernel::Index<2u>{too_large, 2u});
  TEST_ASSERT(!shape_overflow);
  TEST_ASSERT(std::string_view{shape_overflow.reason} == "view_shape_overflow");

  const rund::kernel::View<rund::kernel::i32, 1u> aligned_view =
      rund::kernel::view<rund::kernel::i32>(
          aligned_storage.data(),
          rund::kernel::Index<1u>{aligned_storage.size()}, 64u);
  TEST_ASSERT(aligned_view);
  const rund::kernel::View<rund::kernel::i32, 1u> misaligned_view =
      rund::kernel::view<rund::kernel::i32>(aligned_storage.data() + 1u,
                                            rund::kernel::Index<1u>{1u}, 64u);
  TEST_ASSERT(!misaligned_view);
  TEST_ASSERT(std::string_view{misaligned_view.reason} ==
              "view_alignment_failed");
  return 0;
}

} // namespace program_skeleton_contract
