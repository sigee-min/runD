#include "local.hpp"

bool FrameIoBudgetExhaustionReportsIncomplete() {
  FRAMEIO_ASSERT(FrameIoZeroWriteBudgetReportsIncomplete());
  FRAMEIO_ASSERT(FrameIoZeroReadBudgetReportsIncomplete());
  FRAMEIO_ASSERT(FrameIoHeaderOnlyReadBudgetReportsIncomplete());
  return true;
}
