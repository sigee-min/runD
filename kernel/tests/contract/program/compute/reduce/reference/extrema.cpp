#include "../local.hpp"

namespace program_compute_contract {

int ReduceReferenceMinMax() {
  const std::array<rund::kernel::u32, 5u> u32_input{9u, 2u, 13u, 5u, 7u};
  rund::kernel::u32 u32_output = 0u;
  const rund::kernel::ReduceResult min_u32 =
      rund::kernel::ReferenceReduceMinU32(u32_input.data(), &u32_output,
                                          u32_input.size());
  TEST_ASSERT(min_u32.ok);
  TEST_ASSERT(min_u32.total == 2u);
  TEST_ASSERT(u32_output == 2u);
  const rund::kernel::ReduceResult max_u32 =
      rund::kernel::ReferenceReduceMaxU32(u32_input.data(), &u32_output,
                                          u32_input.size());
  TEST_ASSERT(max_u32.ok);
  TEST_ASSERT(max_u32.total == 13u);
  TEST_ASSERT(u32_output == 13u);

  const std::array<rund::kernel::u64, 4u> u64_input{42u, 8u, 19u, 100u};
  rund::kernel::u64 u64_output = 0u;
  const rund::kernel::ReduceResult min_u64 =
      rund::kernel::ReferenceReduceMinU64(u64_input.data(), &u64_output,
                                          u64_input.size());
  TEST_ASSERT(min_u64.ok);
  TEST_ASSERT(min_u64.total == 8u);
  TEST_ASSERT(u64_output == 8u);
  const rund::kernel::ReduceResult max_u64 =
      rund::kernel::ReferenceReduceMaxU64(u64_input.data(), &u64_output,
                                          u64_input.size());
  TEST_ASSERT(max_u64.ok);
  TEST_ASSERT(max_u64.total == 100u);
  TEST_ASSERT(u64_output == 100u);
  return 0;
}

}  // namespace program_compute_contract
