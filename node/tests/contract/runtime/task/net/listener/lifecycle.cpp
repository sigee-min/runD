#include "test/assert.hpp"

#include "lifecycle/local.hpp"

int RunRuntimeTaskNetListenerLifecycleContract() {
  if (const int rc = ListenerLifecycleOpensBindsListensAndCloses(); rc != 0) {
    return rc;
  }
  if (const int rc = ListenerInvalidInputsFailClosed(); rc != 0) {
    return rc;
  }
  if (const int rc = ListenerBacklogAndStaleHandlesFailClosed(); rc != 0) {
    return rc;
  }
  return ListenerShutdownConnectedSocketSucceeds();
}
