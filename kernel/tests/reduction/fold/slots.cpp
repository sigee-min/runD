#include "cases.hpp"

#include "test/assert.hpp"

#include <kernel/reduction/fold/primitive.hpp>
#include <kernel/reduction/fold/slots.hpp>

int RunFoldSlotsContract() {
  rund::kernel::FoldSlots slots{};
  rund::kernel::EnsureFoldSlots(slots, 4u);
  TEST_ASSERT(rund::kernel::StoreFoldSlot(slots, 0u, 0x10u));
  TEST_ASSERT(rund::kernel::StoreFoldSlot(slots, 1u, 0x80u));
  TEST_ASSERT(rund::kernel::StoreFoldSlot(slots, 2u, 0x30u));
  TEST_ASSERT(rund::kernel::StoreFoldSlot(slots, 3u, 0x80u));
  TEST_ASSERT(!rund::kernel::StoreFoldSlot(slots, 4u, 0u));
  TEST_ASSERT(rund::kernel::FoldSlotsMax(rund::kernel::ViewFoldSlots(slots)) ==
              0x80u);
  TEST_ASSERT(rund::kernel::FoldSlotsXor(rund::kernel::ViewFoldSlots(slots)) ==
              (0x10u ^ 0x80u ^ 0x30u ^ 0x80u));
  TEST_ASSERT(rund::kernel::FoldSlotsMin(rund::kernel::ViewFoldSlots(slots)) ==
              0x10u);
  TEST_ASSERT(rund::kernel::FoldSlotsSaturatingAdd(rund::kernel::ViewFoldSlots(
                  slots)) == (0x10u + 0x80u + 0x30u + 0x80u));
  const rund::kernel::FoldResult max_result = rund::kernel::FoldOrderedSlots(
      rund::kernel::ViewFoldSlots(slots), rund::kernel::FoldOperation::Max);
  TEST_ASSERT(max_result.ok);
  TEST_ASSERT(max_result.value == 0x80u);
  TEST_ASSERT(max_result.slot_count == 4u);
  const rund::kernel::FoldPrimitiveSpec max_spec =
      rund::kernel::DescribeFoldOperation(rund::kernel::FoldOperation::Max);
  TEST_ASSERT(max_spec.supported);
  TEST_ASSERT(max_spec.value_domain ==
              rund::kernel::FoldValueDomain::UnsignedInteger);
  TEST_ASSERT(!max_spec.admits_floating_point);
  return 0;
}
