#include "../local.hpp"

namespace program_compute_contract {

int ReduceReferenceReject() {
  const rund::kernel::ReduceResult missing =
      rund::kernel::ReferenceReduceSumU32(nullptr, nullptr, 4u);
  TEST_ASSERT(!missing.ok);
  TEST_ASSERT(std::string_view{missing.reason} ==
              "compute_reduce_buffer_invalid");

  const std::array<rund::kernel::u32, 2u> overflow_u32_input{
      std::numeric_limits<rund::kernel::u32>::max(), 1u};
  rund::kernel::u32 u32_output = 0u;
  const rund::kernel::ReduceResult overflow_u32 =
      rund::kernel::ReferenceReduceSumU32(
          overflow_u32_input.data(), &u32_output, overflow_u32_input.size());
  TEST_ASSERT(!overflow_u32.ok);
  TEST_ASSERT(std::string_view{overflow_u32.reason} ==
              "compute_reduce_sum_overflow");

  const std::array<rund::kernel::u64, 2u> overflow_u64_input{
      std::numeric_limits<rund::kernel::u64>::max(), 1u};
  rund::kernel::u64 u64_output = 0u;
  const rund::kernel::ReduceResult overflow_u64 =
      rund::kernel::ReferenceReduceSumU64(
          overflow_u64_input.data(), &u64_output, overflow_u64_input.size());
  TEST_ASSERT(!overflow_u64.ok);
  TEST_ASSERT(std::string_view{overflow_u64.reason} ==
              "compute_reduce_sum_overflow");
  return 0;
}

}  // namespace program_compute_contract
