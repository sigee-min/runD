#include "local.hpp"

#if defined(RUND_NODE_TEST_BACKEND_CPU) || \
    defined(RUND_NODE_TEST_BACKEND_VULKAN)

namespace rund_node_test_pipeline {

int CheckMetalResidencyAdmission() { return 0; }

} // namespace rund_node_test_pipeline

#else

namespace rund_node_test_pipeline {

int CheckMetalResidencyAdmission() {
  const int transactional = CheckTransactionalStageRollback();
  if (transactional != 0) {
    return transactional;
  }
  const int execution = CheckExecutionAdapter();
  if (execution != 0) {
    return 50 + execution;
  }
  for (const std::size_t epochs : {4u, 5u, 9u}) {
    const int staged_loop = CheckStagedLoopExecution(epochs);
    if (staged_loop != 0) {
      return 75 + staged_loop;
    }
  }
  const int capacity = CheckVirtualAdmissionRollback();
  if (capacity != 0) {
    return 100 + capacity;
  }
  const int native_terminal = CheckSelectedNativeTerminalAndDeviceLoss();
  if (native_terminal != 0) {
    return 200 + native_terminal;
  }
  const int terminal_loss = CheckQueueTerminalLossQuarantine();
  if (terminal_loss != 0) {
    return 300 + terminal_loss;
  }
  const int schedule_failure = CheckScheduleKnownFailureRetry();
  if (schedule_failure != 0) {
    return 400 + schedule_failure;
  }
  const int schedule_budget = CheckScheduleBudgetFallback();
  if (schedule_budget != 0) {
    return 500 + schedule_budget;
  }
  const int schedule_loss = CheckScheduleTerminalLossQuarantine();
  return schedule_loss == 0 ? 0 : 600 + schedule_loss;
}

} // namespace rund_node_test_pipeline

#endif
