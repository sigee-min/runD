#include "contract/program/compute/graph/local.hpp"
#include <kernel/program/compute/graph/signature.hpp>
#include <kernel/program/compute/reduce/model.hpp>
#include <kernel/program/compute/reduce/identity.hpp>
#include <kernel/program/compute/reduce/plan.hpp>
#include <kernel/program/compute/reduce/reference.hpp>
#include <kernel/program/compute/scan/model.hpp>
#include <kernel/program/compute/scan/identity.hpp>
#include <kernel/program/compute/scan/plan.hpp>
#include <kernel/program/compute/scan/reference.hpp>
#include <kernel/program/compute/sort/model.hpp>
#include <kernel/program/compute/sort/identity.hpp>
#include <kernel/program/compute/sort/plan.hpp>
#include <kernel/program/compute/sort/reference.hpp>

namespace program_compute_contract {

int RunBoundedGraphSignatureContract() {
  const rund::kernel::ScanPlan scan = rund::kernel::PlanScan({
      .op = rund::kernel::ScanOp::InclusiveSum,
      .element = rund::kernel::ScanElement::U32,
      .element_count = 64u,
      .block_size = 16u,
      .count_source = rund::kernel::ComputeCountSource::BufferU32,
  });
  const rund::kernel::GraphSignature scan_signature =
      rund::kernel::GraphSignatureFor(scan);
  TEST_ASSERT(scan.ok && scan_signature.ok);
  TEST_ASSERT(scan_signature.value_count == 3u);
  TEST_ASSERT(scan_signature.values[1u].kind ==
              rund::kernel::GraphValueKind::LogicalCount);
  TEST_ASSERT(scan_signature.values[1u].role ==
              rund::kernel::BufferRole::Read);
  TEST_ASSERT(scan_signature.values[1u].element_bytes ==
              sizeof(rund::kernel::u32));
  TEST_ASSERT(scan_signature.values[1u].count == 1u);

  const rund::kernel::ReducePlan reduce = rund::kernel::PlanReduce({
      .op = rund::kernel::ReduceOp::Sum,
      .element = rund::kernel::ReduceElement::U64,
      .element_count = 64u,
      .block_size = 16u,
      .count_source = rund::kernel::ComputeCountSource::BufferU64,
  });
  const rund::kernel::GraphSignature reduce_signature =
      rund::kernel::GraphSignatureFor(reduce);
  TEST_ASSERT(reduce.ok && reduce_signature.ok);
  TEST_ASSERT(reduce_signature.value_count == 3u);
  TEST_ASSERT(reduce_signature.values[1u].kind ==
              rund::kernel::GraphValueKind::LogicalCount);
  TEST_ASSERT(reduce_signature.values[1u].role ==
              rund::kernel::BufferRole::Read);
  TEST_ASSERT(reduce_signature.values[1u].element_bytes ==
              sizeof(rund::kernel::u64));
  TEST_ASSERT(reduce_signature.values[1u].count == 1u);

  const rund::kernel::SortPlan sort = rund::kernel::PlanSort({
      .key = rund::kernel::SortKey::U32,
      .value = rund::kernel::SortValue::IdentityU32,
      .element_count = 64u,
      .count_source = rund::kernel::ComputeCountSource::BufferU32,
  });
  const rund::kernel::GraphSignature sort_signature =
      rund::kernel::GraphSignatureFor(sort);
  TEST_ASSERT(sort.ok && sort_signature.ok);
  TEST_ASSERT(sort_signature.value_count == 4u);
  TEST_ASSERT(sort_signature.values[1u].kind ==
              rund::kernel::GraphValueKind::LogicalCount);
  TEST_ASSERT(sort_signature.values[1u].element_bytes ==
              sizeof(rund::kernel::u32));
  return 0;
}

} // namespace program_compute_contract
