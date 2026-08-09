#include "range/local.hpp"

int RunRangePlannerContract() {
  using namespace node_accel_contract::range;
  return ModelContract() && PlannerContract() && ProjectionContract() &&
                 ExecutionContract() && MemoryContract() && SourceContract() &&
                 BackendContract() && CacheContract()
             ? 0
             : 1;
}
