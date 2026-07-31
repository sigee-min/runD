#include "cases.hpp"

#include "test/assert.hpp"

#include <kernel/reduction/fold/slots.hpp>

#include <span>
#include <string_view>
#include <vector>

int RunStrictOrderedFloatContract() {
  const std::vector<rund::kernel::u64> float32_values{0x3F800000u, 0x40000000u};
  const rund::kernel::FoldResult default_ordered_float =
      rund::kernel::FoldOrderedSlots(std::span<const rund::kernel::u64>(float32_values.data(),
                                                            float32_values.size()),
                               rund::kernel::FoldOperation::StrictFloat32Add);
  TEST_ASSERT(!default_ordered_float.ok);
  TEST_ASSERT(std::string_view{default_ordered_float.reason} == "floating_point_fold_forbidden");
  const rund::kernel::FoldResult strict_ordered_float =
      rund::kernel::FoldStrictOrderedSlots(std::span<const rund::kernel::u64>(float32_values.data(),
                                                                  float32_values.size()),
                                     rund::kernel::FoldOperation::StrictFloat32Add,
                                     rund::kernel::StrictFloat32ReductionPolicy());
  TEST_ASSERT(strict_ordered_float.ok);
  TEST_ASSERT(strict_ordered_float.value == 0x40400000u);
  return 0;
}
