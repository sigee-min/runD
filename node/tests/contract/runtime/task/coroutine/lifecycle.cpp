#include "local.hpp"

namespace rund::node::test_contract::coroutine {

int RunRuntimeTaskCoroutineLifecycleContract() {
  if (const int rc = CheckCoroutineComplete(); rc != 0) return rc;
  if (const int rc = CheckCoroutineYield(); rc != 0) return rc;
  if (const int rc = CheckCoroutineSleep(); rc != 0) return rc;
  if (const int rc = CheckCoroutineReadyIo(); rc != 0) return rc;
  if (const int rc = CheckCoroutineBlockedIo(); rc != 0) return rc;
  if (const int rc = CheckCoroutineReadyMany(); rc != 0) return rc;
  if (const int rc = CheckDiscardedReadyOps(); rc != 0) return rc;
  if (const int rc = CheckCoroutineChannelSend(); rc != 0) return rc;
  if (const int rc = CheckCoroutineChannelRecv(); rc != 0) return rc;
  if (const int rc = CheckCoroutineChannelRaii(); rc != 0) return rc;
  if (const int rc = CheckDiscardedChannelOps(); rc != 0) return rc;
  if (const int rc = CheckCoroutineJoinAwait(); rc != 0) return rc;
  if (const int rc = CheckCoroutineHandleAwait(); rc != 0) return rc;
  if (const int rc = CheckCoroutineNestedTask(); rc != 0) return rc;
  if (const int rc = CheckNestedResultReuseAndLeafBoundary(); rc != 0) {
    return rc;
  }
  return CheckDiscardedOperations();
}

}  // namespace rund::node::test_contract::coroutine

int RunRuntimeTaskCoroutineContract() {
  if (rund::node::test_contract::coroutine::
          RunRuntimeTaskCoroutineLifecycleContract() != 0) {
    return 1;
  }
  if (rund::node::test_contract::coroutine::
          RunRuntimeTaskCoroutineFailureContract() != 0) {
    return 1;
  }
  return rund::node::test_contract::coroutine::
      RunRuntimeTaskCoroutineCancellationContract();
}
