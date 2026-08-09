#include "range/local.hpp"

int RunRangePlannerContract() {
  using namespace node_accel_contract::range;
  return ModelContract() && PlannerContract() && ExecutionContract() &&
                 MemoryContract() && SourceContract() && BackendContract() &&
                 CacheContract()
             ? 0
             : 1;
}
