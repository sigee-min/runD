#include "local.hpp"

namespace rund_node_test_pipeline_residency {

int CheckExecution() {
  if (const int result = execution_test::CheckExecutionPlan(); result != 0) {
    return result;
  }
  if (const int result = execution_test::CheckExecutionAuthority();
      result != 0) {
    return result;
  }
  if (const int result = execution_test::CheckExecutionNative(); result != 0) {
    return result;
  }
  return 0;
}

} // namespace rund_node_test_pipeline_residency
