#include "queue/local.hpp"

int RunRuntimeTaskReadyQueueContract() {
  using Check = int (*)();
  constexpr Check checks[]{
      ready_queue_detail::CheckStorageAndOrder,
      ready_queue_detail::CheckContinuations,
      ready_queue_detail::CheckReuseAndBatch,
      ready_queue_detail::CheckFailureOrder,
  };
  for (const Check check : checks) {
    if (const int result = check(); result != 0) {
      return result;
    }
  }
  return 0;
}
