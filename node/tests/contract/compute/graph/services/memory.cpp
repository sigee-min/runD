#include "memory/local.hpp"

namespace rund_node_graph_services {

bool ValidMemoryPlan() {
  using namespace memory_detail;
  return CheckBasicMemoryPlan() && CheckWideMemoryPlan() &&
         CheckAliasMemoryPlan() && CheckCapacityMemoryPlan() &&
         CheckRejectedMemoryPlan();
}

} // namespace rund_node_graph_services
