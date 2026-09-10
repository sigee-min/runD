#include "local.hpp"

namespace rund_node_test_pipeline {

[[nodiscard]] int CheckMemory(rund::compute::Device &device) {
  if (const int attribution = memory::CheckAttribution(); attribution != 0) {
    return attribution;
  }
  if (const int shared = memory::CheckSharedAttribution(); shared != 0) {
    return shared;
  }
  if (const int basic = memory::CheckBasic(device); basic != 0) {
    return basic;
  }
  if (const int reset = memory::CheckReset(device); reset != 0) {
    return reset;
  }
  if (const int arena = memory::CheckArena(device); arena != 0) {
    return arena;
  }
  return memory::CheckCapacity(device);
}

} // namespace rund_node_test_pipeline
