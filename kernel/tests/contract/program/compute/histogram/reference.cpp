#include "local.hpp"

namespace program_compute_contract {

int HistogramReference() {
  const std::array<rund::kernel::u32, 8u> input{0u, 1u, 1u, 3u,
                                                2u, 1u, 0u, 3u};
  std::array<rund::kernel::u32, 4u> output{9u, 9u, 9u, 9u};
  const rund::kernel::HistogramResult result =
      rund::kernel::ReferenceHistogramU32(input.data(), output.data(),
                                          input.size(), output.size());
  TEST_ASSERT(result.ok);
  TEST_ASSERT(result.element_count == input.size());
  TEST_ASSERT(result.bin_count == output.size());
  TEST_ASSERT(output[0] == 2u);
  TEST_ASSERT(output[1] == 3u);
  TEST_ASSERT(output[2] == 1u);
  TEST_ASSERT(output[3] == 2u);

  const rund::kernel::HistogramResult missing =
      rund::kernel::ReferenceHistogramU32(nullptr, nullptr, input.size(),
                                          output.size());
  TEST_ASSERT(!missing.ok);
  TEST_ASSERT(std::string_view{missing.reason} ==
              "compute_histogram_buffer_invalid");

  const std::array<rund::kernel::u32, 2u> out_of_range{0u, 4u};
  const rund::kernel::HistogramResult bad_bin =
      rund::kernel::ReferenceHistogramU32(out_of_range.data(), output.data(),
                                          out_of_range.size(), output.size());
  TEST_ASSERT(!bad_bin.ok);
  TEST_ASSERT(std::string_view{bad_bin.reason} ==
              "compute_histogram_bin_invalid");
  return 0;
}

}  // namespace program_compute_contract
