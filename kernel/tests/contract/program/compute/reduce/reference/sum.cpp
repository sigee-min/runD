#include "../local.hpp"

namespace program_compute_contract {

int ReduceReferenceSumCount();
int ReduceReferenceMinMax();
int ReduceReferenceReject();

int ReduceReference() {
  if (ReduceReferenceSumCount() != 0) {
    return 1;
  }
  if (ReduceReferenceMinMax() != 0) {
    return 1;
  }
  return ReduceReferenceReject();
}

int ReduceReferenceSumCount() {
  const std::array<rund::kernel::u32, 5u> u32_input{3u, 7u, 11u, 13u, 17u};
  rund::kernel::u32 u32_output = 0u;
  const rund::kernel::ReduceResult u32_result =
      rund::kernel::ReferenceReduceSumU32(u32_input.data(), &u32_output,
                                          u32_input.size());
  TEST_ASSERT(u32_result.ok);
  TEST_ASSERT(u32_result.total == 51u);
  TEST_ASSERT(u32_output == 51u);

  const std::array<rund::kernel::u64, 4u> u64_input{5u, 9u, 21u, 34u};
  rund::kernel::u64 u64_output = 0u;
  const rund::kernel::ReduceResult u64_result =
      rund::kernel::ReferenceReduceSumU64(u64_input.data(), &u64_output,
                                          u64_input.size());
  TEST_ASSERT(u64_result.ok);
  TEST_ASSERT(u64_result.total == 69u);
  TEST_ASSERT(u64_output == 69u);

  const std::array<rund::kernel::u32, 5u> count_u32_input{0u, 7u, 0u, 13u,
                                                          17u};
  u32_output = 0u;
  const rund::kernel::ReduceResult count_u32 =
      rund::kernel::ReferenceReduceCountNonzeroU32(
          count_u32_input.data(), &u32_output, count_u32_input.size());
  TEST_ASSERT(count_u32.ok);
  TEST_ASSERT(count_u32.total == 3u);
  TEST_ASSERT(u32_output == 3u);

  const std::array<rund::kernel::u64, 4u> count_u64_input{0u, 9u, 0u, 34u};
  u64_output = 0u;
  const rund::kernel::ReduceResult count_u64 =
      rund::kernel::ReferenceReduceCountNonzeroU64(
          count_u64_input.data(), &u64_output, count_u64_input.size());
  TEST_ASSERT(count_u64.ok);
  TEST_ASSERT(count_u64.total == 2u);
  TEST_ASSERT(u64_output == 2u);
  return 0;
}

} // namespace program_compute_contract
