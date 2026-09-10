#include "local.hpp"

namespace rund_node_test_pipeline_residency {

int CheckSlidingModel() {
  int result = CheckSlidingLifecycle();
  if (result != 0) {
    return result;
  }
  result = CheckSlidingGraphProjection();
  if (result != 0) {
    return result;
  }
  result = CheckSlidingTerminalUnknown();
  if (result != 0) {
    return result;
  }
  result = CheckSlidingGraphReplay();
  if (result != 0) {
    return result;
  }
  result = CheckSlidingTerminalFrontier();
  if (result != 0) {
    return result;
  }
  result = CheckSlidingTerminalFailures();
  if (result != 0) {
    return result;
  }
  result = CheckSlidingGraphAdmission();
  if (result != 0) {
    return result;
  }
  result = CheckSlidingTerminalSuffix();
  if (result != 0) {
    return result;
  }
  return CheckSlidingGraphTopology();
}

} // namespace rund_node_test_pipeline_residency
