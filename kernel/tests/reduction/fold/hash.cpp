#include "cases.hpp"

#include "test/assert.hpp"

#include <kernel/reduction/fold/slots.hpp>

#include <span>
#include <string_view>
#include <vector>

int RunFoldHashContract() {
  const std::vector<rund::kernel::u64> values{0x10u, 0x20u, 0x30u, 0x40u, 0x50u};
  rund::kernel::FixedBinaryTreeHashBuilder builder{};
  for (const rund::kernel::u64 value : values) {
    builder.Push(value);
  }
  TEST_ASSERT(builder.Finalize() ==
              rund::kernel::FoldHashFixedBinaryTree(std::span<const rund::kernel::u64>(values.data(), values.size())));
  builder.Reset();
  TEST_ASSERT(builder.Finalize() == rund::kernel::FoldHashFixedBinaryTree(std::span<const rund::kernel::u64>{}));
  const rund::kernel::FoldResult folded =
      rund::kernel::FoldOrderedSlots(std::span<const rund::kernel::u64>(values.data(), values.size()),
                               rund::kernel::FoldOperation::FixedBinaryTreeHash);
  TEST_ASSERT(folded.ok);
  TEST_ASSERT(folded.fold_cost_measured);
  TEST_ASSERT(folded.value ==
              rund::kernel::FoldHashFixedBinaryTree(std::span<const rund::kernel::u64>(values.data(), values.size())));
  const rund::kernel::FoldResult unsupported =
      rund::kernel::FoldOrderedSlots(std::span<const rund::kernel::u64>(values.data(), values.size()),
                               static_cast<rund::kernel::FoldOperation>(255u));
  TEST_ASSERT(!unsupported.ok);
  TEST_ASSERT(std::string_view{unsupported.reason} == "unsupported_fold_operation");
  TEST_ASSERT(unsupported.fold_cost_measured);
  return 0;
}
