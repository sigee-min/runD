#include "contract/dispatch/cases.hpp"

#include "contract/support.hpp"
#include "test/assert.hpp"

#include <kernel/internal/dispatch/kernel.hpp>

#include <array>

int RunDispatchFailureContract() {
  using namespace kernel_contract_test;

  VisitFixture fixture{};
  fixture.Reset(10u);
  FakePool pool = BuildStaticPool(2u);
  const std::array<rund::kernel::Partition, 2u> gapped_partitions{
      rund::kernel::Partition{.worker_index = 0u, .begin = 0u, .end = 3u},
      rund::kernel::Partition{.worker_index = 1u, .begin = 4u, .end = 10u},
  };
  const rund::kernel::internal::Plan plan{
      .packet_count = 10u,
      .execution_width = 2u,
      .partitions = gapped_partitions.data(),
      .partition_count =
          static_cast<rund::kernel::u32>(gapped_partitions.size()),
      .worker_backend = MakeFakeBackend(&pool),
      .context = &fixture.context,
      .dispatch = MarkVisitedPackets,
  };
  const rund::kernel::Result result = rund::kernel::internal::Execute(plan);
  TEST_ASSERT(!result.ok);

  FakePool mismatched_pool = BuildStaticPool(4u);
  const std::array<rund::kernel::Partition, 2u> valid_partitions{
      rund::kernel::Partition{.worker_index = 0u, .begin = 0u, .end = 8u},
      rund::kernel::Partition{.worker_index = 1u, .begin = 8u, .end = 16u},
  };
  const rund::kernel::internal::Plan mismatched_plan{
      .packet_count = 16u,
      .execution_width = 2u,
      .partitions = valid_partitions.data(),
      .partition_count =
          static_cast<rund::kernel::u32>(valid_partitions.size()),
      .worker_backend = MakeFakeBackend(&mismatched_pool),
      .context = &fixture.context,
      .dispatch = MarkVisitedPackets,
  };
  const rund::kernel::Result mismatch =
      rund::kernel::internal::Execute(mismatched_plan);
  TEST_ASSERT(!mismatch.ok);
  return 0;
}
