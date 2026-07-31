#include "local.hpp"

namespace program_compute_contract {

int StencilReference() {
  const std::array<rund::kernel::u32, 5u> u32_input{1u, 2u, 4u, 8u,
                                                   16u};
  std::array<rund::kernel::u32, 5u> u32_output{};
  const rund::kernel::StencilResult u32_result =
      rund::kernel::ReferenceStencilSumU32(u32_input.data(),
                                           u32_output.data(),
                                           u32_input.size(), 1u);
  TEST_ASSERT(u32_result.ok);
  TEST_ASSERT(u32_output[0] == 4u);
  TEST_ASSERT(u32_output[1] == 7u);
  TEST_ASSERT(u32_output[2] == 14u);
  TEST_ASSERT(u32_output[3] == 28u);
  TEST_ASSERT(u32_output[4] == 40u);

  const std::array<rund::kernel::u64, 3u> u64_input{
      std::numeric_limits<rund::kernel::u64>::max(), 1u, 2u};
  std::array<rund::kernel::u64, 3u> u64_output{};
  const rund::kernel::StencilResult u64_result =
      rund::kernel::ReferenceStencilSumU64(u64_input.data(),
                                           u64_output.data(),
                                           u64_input.size(), 1u);
  TEST_ASSERT(u64_result.ok);
  TEST_ASSERT(u64_output[0] == std::numeric_limits<rund::kernel::u64>::max());
  TEST_ASSERT(u64_output[1] == 2u);
  TEST_ASSERT(u64_output[2] == 5u);

  std::array<rund::kernel::u32, 1u> single_input{9u};
  std::array<rund::kernel::u32, 1u> single_output{};
  const rund::kernel::StencilResult single_result =
      rund::kernel::ReferenceStencilSumU32(single_input.data(),
                                           single_output.data(), 1u, 1u);
  TEST_ASSERT(single_result.ok);
  TEST_ASSERT(single_output[0] == 27u);

  std::array<rund::kernel::u32, 5u> wide_output{};
  const rund::kernel::StencilResult wide_result =
      rund::kernel::ReferenceStencilSumU32(u32_input.data(),
                                           wide_output.data(),
                                           u32_input.size(), 2u);
  TEST_ASSERT(wide_result.ok);
  TEST_ASSERT(wide_output[0] == 9u);
  TEST_ASSERT(wide_output[1] == 16u);
  TEST_ASSERT(wide_output[2] == 31u);
  TEST_ASSERT(wide_output[3] == 46u);
  TEST_ASSERT(wide_output[4] == 60u);

  const std::array<rund::kernel::u32, 5u> extrema_input{5u, 2u, 9u, 1u,
                                                        7u};
  std::array<rund::kernel::u32, 5u> min_output{};
  const rund::kernel::StencilResult min_result =
      rund::kernel::ReferenceStencilMinU32(extrema_input.data(),
                                           min_output.data(),
                                           extrema_input.size(), 1u);
  TEST_ASSERT(min_result.ok);
  TEST_ASSERT(min_output[0] == 2u);
  TEST_ASSERT(min_output[1] == 2u);
  TEST_ASSERT(min_output[2] == 1u);
  TEST_ASSERT(min_output[3] == 1u);
  TEST_ASSERT(min_output[4] == 1u);

  std::array<rund::kernel::u32, 5u> max_output{};
  const rund::kernel::StencilResult max_result =
      rund::kernel::ReferenceStencilMaxU32(extrema_input.data(),
                                           max_output.data(),
                                           extrema_input.size(), 1u);
  TEST_ASSERT(max_result.ok);
  TEST_ASSERT(max_output[0] == 5u);
  TEST_ASSERT(max_output[1] == 9u);
  TEST_ASSERT(max_output[2] == 9u);
  TEST_ASSERT(max_output[3] == 9u);
  TEST_ASSERT(max_output[4] == 7u);

  const rund::kernel::StencilResult missing =
      rund::kernel::ReferenceStencilSumU32(nullptr, nullptr, 4u, 1u);
  TEST_ASSERT(!missing.ok);
  TEST_ASSERT(std::string_view{missing.reason} ==
              "compute_stencil_buffer_invalid");
  return 0;
}

}  // namespace program_compute_contract
