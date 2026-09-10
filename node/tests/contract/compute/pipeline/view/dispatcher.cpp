#include "local.hpp"

namespace rund_node_test_pipeline {

[[nodiscard]] int CheckViews(rund::compute::Device &device,
                             const Backend backend) {
  if (const int basic = view::CheckBasic(device, backend); basic != 0) {
    return basic;
  }
  if (const int reduce = view::CheckReduce(device, backend); reduce != 0) {
    return reduce;
  }
  if (const int sort = view::CheckSort(device, backend); sort != 0) {
    return sort;
  }
  if (const int pool = view::CheckPoolReset(device); pool != 0) {
    return pool;
  }
  if (const int primitives = view::CheckPrimitives(device, backend);
      primitives != 0) {
    return primitives;
  }
  if (const int arena = CheckViewArena(device, backend); arena != 0) {
    return 30 + arena;
  }
  if (!view::CheckStridedReset(device, backend) ||
      !view::CheckScatterOffset(device)) {
    return 28;
  }
  return 0;
}

} // namespace rund_node_test_pipeline
