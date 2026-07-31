#include "contract/dispatch/cases.hpp"

#include "contract/support.hpp"
#include "test/assert.hpp"

#include <kernel/internal/dispatch/kernel.hpp>

#include <array>

int RunDispatchPartitionContract() {
  using namespace kernel_contract_test;

  VisitFixture fixture{};
  fixture.Reset(10u);
  FakePool pool = BuildStaticPool(2u);
  const std::array<rund::kernel::Partition, 2u> custom_partitions{
      rund::kernel::Partition{.worker_index = 0u, .begin = 0u, .end = 3u},
      rund::kernel::Partition{.worker_index = 1u, .begin = 3u, .end = 10u},
  };
  const rund::kernel::internal::Plan plan{
      .packet_count = 10u,
      .execution_width = 2u,
      .partitions = custom_partitions.data(),
      .partition_count = static_cast<rund::kernel::u32>(custom_partitions.size()),
      .worker_backend = MakeFakeBackend(&pool),
      .context = &fixture.context,
      .dispatch = MarkVisitedPackets,
  };
  const rund::kernel::Result result = rund::kernel::internal::Execute(plan);
  TEST_ASSERT(result.ok);
  TEST_ASSERT(result.telemetry.total_partitions_executed == 2u);
  TEST_ASSERT(fixture.VisitedOnce(10u));
  return 0;
}
