#include "local.hpp"
#include "test/assert.hpp"

int RunRangePlannerContract() {
  using namespace node_accel_contract::range;
  TEST_ASSERT(ModelContract());
  TEST_ASSERT(PlannerContract());
  TEST_ASSERT(ProjectionContract());
  TEST_ASSERT(ExecutionContract());
  TEST_ASSERT(MemoryContract());
  TEST_ASSERT(SourceContract());
  TEST_ASSERT(BackendContract());
  TEST_ASSERT(CacheContract());
  return 0;
}
